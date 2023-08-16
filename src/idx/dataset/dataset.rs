use itertools::izip;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::cmp::min;
use std::convert::TryInto;
use std::path::Path;
use strength_reduce::StrengthReducedU64;

use super::super::chunk::{Chunk, ULE};
use super::types::*;
use super::{DatasetExt, DatasetExtReader};
use crate::filters::byteorder::Order as ByteOrder;

/// A HDF5 dataset (a single variable).
///
/// > Note to reader implementations: The official HDF5 library uses a 1MB dataset cache by default.
///
#[derive(Debug, Serialize, Deserialize)]
pub struct Dataset<'a, const D: usize> {
    pub dtype: Datatype,
    pub dsize: usize,
    pub order: ByteOrder,

    #[serde(borrow)]
    #[serde(with = "super::super::serde::chunks_u64s")]
    pub chunks: Cow<'a, [Chunk<D>]>,

    #[serde(with = "super::super::serde::arr_u64")]
    pub shape: [u64; D],

    #[serde(with = "super::super::serde::arr_u64")]
    pub chunk_shape: [u64; D],

    #[serde(with = "super::super::serde::sr_u64")]
    chunk_shape_reduced: [StrengthReducedU64; D],

    #[serde(with = "super::super::serde::arr_u64")]
    pub scaled_dim_sz: [u64; D],

    #[serde(with = "super::super::serde::arr_u64")]
    pub dim_sz: [u64; D],

    #[serde(with = "super::super::serde::arr_u64")]
    pub chunk_dim_sz: [u64; D],
    pub shuffle: bool,
    pub gzip: Option<u8>,
}

impl<const D: usize> Dataset<'_, D> {
    pub fn index(ds: &hdf5::Dataset) -> Result<Dataset<'static, D>, anyhow::Error> {
        use hdf5::filters::Filter;

        ensure!(ds.ndim() == D, "Dataset rank does not match.");

        let filters = ds.filters();

        let mut shuffle = false;
        let mut gzip = None;

        for f in filters {
            match f {
                Filter::Shuffle => shuffle = true,
                Filter::Deflate(z) => gzip = Some(z),
                _ => return Err(anyhow!("{}: Unsupported filter", ds.name())),
            }
        }

        let dtype = ds.dtype()?;
        let order = dtype.byte_order();
        let shape: [u64; D] = ds
            .shape()
            .into_iter()
            .map(|u| u as u64)
            .collect::<Vec<u64>>()
            .as_slice()
            .try_into()?;

        let chunk_shape = ds.chunk().map_or_else(
            || shape,
            |cs| {
                cs.into_iter()
                    .map(|u| u as u64)
                    .collect::<Vec<u64>>()
                    .as_slice()
                    .try_into()
                    .unwrap()
            },
        );

        let mut chunks: Vec<Chunk<D>> = match (ds.is_chunked(), ds.offset()) {
            // Continuous
            (false, Some(offset)) => Ok::<_, anyhow::Error>(vec![Chunk {
                offset: [ULE::ZERO; D],
                size: ULE::new(ds.storage_size()),
                addr: ULE::new(offset),
            }]),

            // Chunked
            (true, None) => {
                let n = ds.num_chunks().expect("weird..");

                #[cfg(feature = "fast-index")]
                {
                    let mut v = Vec::with_capacity(n);

                    ds.chunks_visit(|ci| {
                        v.push(Chunk {
                            offset: ci
                                .offset
                                .iter()
                                .copied()
                                .map(ULE::new)
                                .collect::<Vec<_>>()
                                .as_slice()
                                .try_into()
                                .unwrap(),
                            size: ULE::new(ci.size),
                            addr: ULE::new(ci.addr),
                        });

                        0
                    })?;

                    Ok(v)
                }

                #[cfg(not(feature = "fast-index"))]
                {
                    let chunks = (0..n)
                        .map(|i| {
                            let chunk = ds.chunk_info(i).unwrap();

                            ensure!(
                                chunk.filter_mask == 0,
                                "mismatching filter mask with dataset filter mask"
                            );

                            Ok(Chunk {
                                offset: chunk
                                    .offset
                                    .iter()
                                    .cloned()
                                    .map(ULE::new)
                                    .collect::<Vec<_>>()
                                    .as_slice()
                                    .try_into()
                                    .unwrap(),
                                size: ULE::new(chunk.size),
                                addr: ULE::new(chunk.addr),
                            })
                        })
                        .collect();

                    chunks
                }
            }

            _ => Err(anyhow!(
                "{}: Unsupported data layout (chunked: {}, offset: {:?})",
                ds.name(),
                ds.is_chunked(),
                ds.offset()
            )),
        }?;

        chunks.sort();
        let chunks = Cow::from(chunks);

        {
            let expected_chunks = shape
                .iter()
                .zip(&chunk_shape)
                .map(|(s, c)| (s + (c - 1)) / c)
                .product::<u64>() as usize;

            ensure!(
                chunks.len() == expected_chunks,
                "{}: unexpected number of chunks given dataset size (is_chunked: {}, chunks: {} != {} (expected), shape: {:?}, chunk shape: {:?})",
                ds.name(),
                ds.is_chunked(),
                chunks.len(),
                expected_chunks,
                shape,
                chunk_shape);
        }

        Dataset::new(
            dtype.into(),
            order.into(),
            shape,
            chunks,
            chunk_shape,
            shuffle,
            gzip,
        )
    }

    pub fn new<'a, C>(
        dtype: Datatype,
        order: ByteOrder,
        shape: [u64; D],
        chunks: C,
        chunk_shape: [u64; D],
        shuffle: bool,
        gzip: Option<u8>,
    ) -> Result<Dataset<'a, D>, anyhow::Error>
    where
        C: Into<Cow<'a, [Chunk<D>]>>,
    {
        let chunks = chunks.into();
        let dsize = dtype.dsize();

        // optimized divisor for chunk shape
        let chunk_shape_reduced = chunk_shape
            .iter()
            .map(|c| StrengthReducedU64::new(*c))
            .collect::<Vec<_>>()
            .as_slice()
            .try_into()?;

        // scaled dimension size: dimension size of dataset in chunk offset coordinates.
        // the dimension size is rounded up. when the dataset size is not a multiple of
        // chunk size we have a partially filled chunk which is also present in the list of chunks.
        let scaled_dim_sz: [u64; D] = {
            let mut d = shape
                .iter()
                .zip(&chunk_shape)
                .map(|(d, z)| (d + (z - 1)) / z)
                .rev()
                .scan(1, |p, c| {
                    let sz = *p;
                    *p *= c;
                    Some(sz)
                })
                .collect::<Vec<u64>>();
            d.reverse();
            d
        }
        .as_slice()
        .try_into()?;

        // size of dataset dimensions
        let dim_sz: [u64; D] = {
            let mut d = shape
                .iter()
                .rev()
                .scan(1, |p, &c| {
                    let sz = *p;
                    *p *= c;
                    Some(sz)
                })
                .collect::<Vec<u64>>();
            d.reverse();
            d
        }
        .as_slice()
        .try_into()?;

        // size of chunk dimensions
        let chunk_dim_sz: [u64; D] = {
            let mut d = chunk_shape
                .iter()
                .rev()
                .scan(1, |p, &c| {
                    let sz = *p;
                    *p *= c;
                    Some(sz)
                })
                .collect::<Vec<u64>>();
            d.reverse();
            d
        }
        .as_slice()
        .try_into()?;

        Ok(Dataset {
            dtype,
            dsize,
            order,
            chunks,
            shape,
            chunk_shape,
            chunk_shape_reduced,
            scaled_dim_sz,
            dim_sz,
            chunk_dim_sz,
            shuffle,
            gzip,
        })
    }

    /// Number of elements in dataset.
    #[must_use]
    pub fn size(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }

    /// Dataset contains a single scalar value.
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    /// Test whether dataset and chunk layout is valid.
    pub fn valid(&self) -> anyhow::Result<bool> {
        for chunk in self.chunks.iter() {
            let offset = chunk.offset.iter().map(|u| u.get()).collect::<Vec<_>>();
            ensure!(
                chunk.contains(&offset, &self.chunk_shape) == std::cmp::Ordering::Equal,
                "chunk does not contain its offset"
            );
        }

        Ok(true)
    }

    /// Returns an iterator over chunk, offset and size which if joined will make up the specified slice through the
    /// variable.
    pub fn chunk_slices(
        &self,
        indices: Option<&[u64; D]>,
        counts: Option<&[u64; D]>,
    ) -> impl Iterator<Item = (&Chunk<D>, u64, u64)> {
        let indices: [u64; D] = *indices.unwrap_or(&[0; D]);
        let counts: [u64; D] = *counts.unwrap_or(&self.shape);

        if indices
            .iter()
            .zip(counts.iter())
            .map(|(i, c)| i + c)
            .zip(self.shape.iter())
            .any(|(l, &s)| l > s)
            || counts.iter().any(|&c| c == 0)
        {
            // Out of bounds or counts is zero in any dimension.
            ChunkSlicer::empty(self)
        } else {
            ChunkSlicer::new(self, indices, counts)
        }
    }

    /// Returns an Vec with chunks, offset and size grouped by chunk, with segments and
    /// destination offset.
    pub fn group_chunk_slices(
        &self,
        indices: Option<&[u64; D]>,
        counts: Option<&[u64; D]>,
    ) -> Vec<(&Chunk<D>, u64, u64, u64)> {
        // Find chunks and calculate offset in destination vector.
        let mut chunks = self
            .chunk_slices(indices, counts)
            .scan(0u64, |offset, (c, start, end)| {
                let slice_sz = end - start;
                let current = *offset;
                *offset += slice_sz;

                Some((c, current, start, end))
            })
            .collect::<Vec<_>>();

        // Sort by chunk file address, not destination address.
        chunks.sort_unstable_by_key(|(c, _, _, _)| c.addr.get());

        chunks

        // XXX: A Vec of Vec's becomes very slow to de-allocate (a couple of seconds
        // actually on a big file with about 380 chunks). So it is faster to have an
        // expanded vector.
        //
        // Group by chunk
        // let mut groups = Vec::<(&Chunk<D>, Vec<(u64, u64, u64)>)>::new();

        // for (current, c, start, end) in chunks.iter() {
        //     match groups.last_mut() {
        //         Some((group_chunk, segments)) if *group_chunk == *c => {
        //             segments.push((*current, *start, *end));
        //         }
        //         _ => {
        //             groups.push((c, vec![(*current, *start, *end)]));
        //         }
        //     }
        // }

        // debug_assert!(groups.iter().map(|(c, _)| c).all_unique());
        // debug_assert!(groups.iter().map(|(_, s)| s.iter().map(|(current, _, _)| current)).flatten().all_unique());
        //
        // groups
    }

    pub fn chunk_at_coord(&self, indices: &[u64]) -> &Chunk<D> {
        debug_assert_eq!(indices.len(), self.chunk_shape.len());
        debug_assert_eq!(indices.len(), self.scaled_dim_sz.len());

        let offset = indices
            .iter()
            .zip(&self.chunk_shape_reduced)
            .zip(&self.scaled_dim_sz)
            .fold(0, |offset, ((&index, &ch_sh), &sz)| {
                offset + index / ch_sh * sz
            });

        &self.chunks[offset as usize]
    }
}

impl<const D: usize> DatasetExt for Dataset<'_, D> {
    fn size(&self) -> usize {
        self.size()
    }

    fn dtype(&self) -> Datatype {
        self.dtype
    }

    fn dsize(&self) -> usize {
        self.dsize
    }

    fn shape(&self) -> &[u64] {
        self.shape.as_slice()
    }

    fn chunk_shape(&self) -> &[u64] {
        self.chunk_shape.as_slice()
    }

    fn valid(&self) -> anyhow::Result<bool> {
        self.valid()
    }

    fn as_par_reader(&self, p: &dyn AsRef<Path>) -> anyhow::Result<Box<dyn DatasetExtReader + '_>> {
        use crate::reader::direct::Direct;

        Ok(Box::new(Direct::with_dataset(self, p)?))
    }
}

pub struct ChunkSlicer<'a, const D: usize> {
    dataset: &'a Dataset<'a, D>,

    /// The current offset in values from start.
    offset: u64,

    /// The coordinates of the current offset in values from the start of the slice.
    offset_coords: [u64; D],

    /// The slice start indices.
    indices: [u64; D],

    /// The size of the slice.
    counts: [u64; D],
    counts_reduced: [StrengthReducedU64; D],

    /// The end of the slice in values from the start of the dataset. The product of
    /// all the sizes in `counts`.
    end: u64,
}

impl<'a, const D: usize> ChunkSlicer<'a, D> {
    /// Empty slice returned for indices and counts that are out of bounds or of zero size.
    pub fn empty(dataset: &'a Dataset<D>) -> ChunkSlicer<'a, D> {
        ChunkSlicer {
            dataset,
            offset: 0,
            offset_coords: [0; D],
            indices: [0; D],
            counts: [0; D],
            counts_reduced: (0..D)
                .map(|_| StrengthReducedU64::new(1))
                .collect::<Vec<_>>()
                .as_slice()
                .try_into()
                .unwrap(),
            end: 0,
        }
    }

    pub fn new(dataset: &'a Dataset<D>, indices: [u64; D], counts: [u64; D]) -> ChunkSlicer<'a, D> {
        let end = if dataset.is_scalar() {
            // scalar
            assert!(indices.is_empty());
            1
        } else {
            // size of slice dimensions
            counts.iter().product::<u64>()
        };

        debug_assert_eq!(indices.len(), dataset.shape.len());
        debug_assert_eq!(counts.len(), dataset.shape.len());

        // Checked in `Dataset::chunk_slices`.
        debug_assert!(izip!(&indices, &counts, &dataset.shape).all(|(i, c, z)| i + c <= *z));

        ChunkSlicer {
            dataset,
            offset: 0,
            offset_coords: [0; D],
            indices,
            counts_reduced: counts
                .iter()
                .map(|c| StrengthReducedU64::new(*c))
                .collect::<Vec<_>>()
                .as_slice()
                .try_into()
                .unwrap(),
            counts,
            end,
        }
    }

    /// Offset in values from chunk offset coordinates. `dim_sz` is dimension size of chunk
    /// dimensions.
    fn chunk_start(coords: &[u64; D], chunk_offset: &[ULE; D], dim_sz: &[u64; D]) -> u64 {
        debug_assert_eq!(coords.len(), chunk_offset.len());
        debug_assert_eq!(coords.len(), dim_sz.len());

        coords
            .iter()
            .zip(chunk_offset)
            .zip(dim_sz)
            .fold(0, |start, ((&coord, &offset), &sz)| {
                start + (coord - offset.get()) * sz
            })
    }
}

impl<'a, const D: usize> Iterator for ChunkSlicer<'a, D> {
    type Item = (&'a Chunk<D>, u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.end {
            return None;
        }

        // scalar
        if self.dataset.is_scalar() {
            debug_assert!(self.dataset.chunks.len() == 1);
            debug_assert!(self.dataset.shape.is_empty());
            self.offset += 1;

            return Some((&self.dataset.chunks[0], 0, 1));
        }

        let mut start = [0; D];
        for (s, i, o) in izip!(&mut start, &self.indices, &self.offset_coords) {
            *s = i + o;
        }
        let chunk: &Chunk<D> = self.dataset.chunk_at_coord(&start);

        debug_assert!(
            chunk.contains(&start, &self.dataset.chunk_shape) == std::cmp::Ordering::Equal
        );

        // position in chunk of current offset
        let chunk_start = Self::chunk_start(&start, &chunk.offset, &self.dataset.chunk_dim_sz);

        // Number of values to advance in current chunk.
        let mut advance = 0;

        let mut carry = 0;
        let mut di = 0;

        for (idx, offset, count, count_sru, chunk_offset, chunk_sz, chunk_dim_sz, dataset_shape) in
            izip!(
                &self.indices,
                &mut self.offset_coords,
                &self.counts,
                &self.counts_reduced,
                &chunk.offset,
                &self.dataset.chunk_shape,
                &self.dataset.chunk_dim_sz,
                &self.dataset.shape,
            )
            .rev()
        {
            // The chunk size may not align to the dataset size. If the chunk
            // dimension is greater than the end of the dataset, it must be cut
            // so that it ends at the end of the dataset.
            //
            // There are two possibilities:
            // * a) Either the chunk is stored in full on disk, with some bogus data.
            // * b) A cut-down chunk is stored on disk.
            //
            // "a" makes more sense. Let's try that.
            //
            // The dimension size should remain the same, since it only depends on the lower
            // dimensions and we are working our way from the last one (`rev`).
            //
            // This does seem to create a lot of chunks.
            let chunk_dim_end = chunk_offset.get() + chunk_sz;
            let chunk_sz = if chunk_dim_end > *dataset_shape {
                chunk_sz - (chunk_dim_end - *dataset_shape)
            } else {
                *chunk_sz
            };

            // If the chunk size in this dimension is 1, count must also be 1, and we will
            // always carry over to the higher dimension. Unless the dimension size is 1, in which
            // case the offset will be advanced with 1.
            di += 1;
            if chunk_sz == 1 && *chunk_dim_sz != 1 {
                *offset += carry;
                carry = *offset / *count_sru;
                *offset = *offset % *count_sru;
                continue;
            } else {
                // Carry over if previous dimension was exhausted.
                *offset += carry;
                carry = *offset / *count_sru;
                *offset = *offset % *count_sru;

                debug_assert!(*offset < *count);

                // Advance to end of slice (`count`) or to end of dimension
                let current = *offset;
                let count_chunk_end = chunk_offset.get() + chunk_sz - idx;
                *offset = min(*count, count_chunk_end);

                let diff = (*offset - current) * chunk_dim_sz;
                advance += diff;
                self.offset += diff;

                carry += *offset / *count_sru;
                *offset = *offset % *count_sru;

                if *idx == chunk_offset.get() && // slice starts at at chunk start (in this dimension)
                        *count == count_chunk_end && // slice ends at chunk end.
                            self.offset < self.end
                // Reached end of dataset
                {
                    continue;
                } else {
                    debug_assert!(self.offset <= self.end);
                    break;
                }
            }
        }

        // Left-over carry?
        //
        // Calculate new offset and start coords:
        let i = self.indices.len() - di;
        for (offset, count) in izip!(&mut self.offset_coords[..i], &self.counts_reduced[..i]).rev()
        {
            *offset += carry;
            carry = *offset / *count;
            *offset = *offset % *count;
            if carry == 0 {
                break;
            }
        }

        debug_assert!(
            advance > 0,
            "slice iterator not advancing: stuck indefinitely."
        );

        // position in chunk of new offset
        let chunk_end = chunk_start + advance;

        debug_assert!(
            chunk_end as usize <= self.dataset.chunk_shape.iter().product::<u64>() as usize
        );

        Some((chunk, chunk_start, chunk_end))
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::test_dataset;
    use super::*;
    use test::Bencher;

    #[bench]
    fn chunk_start(b: &mut Bencher) {
        let dim_sz = [10, 1];
        let coords = [20, 10];
        let ch_offset = [ULE::new(20), ULE::new(10)];

        b.iter(|| test::black_box(ChunkSlicer::chunk_start(&coords, &ch_offset, &dim_sz)))
    }

    #[test]
    fn serialize_d2() {
        let d = test_dataset();

        let s = bincode::serialize(&d).unwrap();

        let md: Dataset<2> = bincode::deserialize(&s).unwrap();

        for (a, b) in izip!(d.chunk_shape_reduced.iter(), md.chunk_shape_reduced.iter()) {
            assert_eq!(a.get(), b.get());
        }
    }
}
