use itertools::izip;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::cmp::min;
use std::convert::TryInto;
use std::path::Path;
use strength_reduce::StrengthReducedU64;

use super::chunk::{Chunk, ULE};
use crate::filters::byteorder::Order as ByteOrder;
use crate::reader::{Reader, UnifyReader, UnifyStreamer};

/// Dataset in possible dimensions.
#[derive(Debug, Serialize, Deserialize)]
pub enum DatasetD<'a> {
    #[serde(borrow)]
    D0(Dataset<'a, 0>),
    #[serde(borrow)]
    D1(Dataset<'a, 1>),
    #[serde(borrow)]
    D2(Dataset<'a, 2>),
    #[serde(borrow)]
    D3(Dataset<'a, 3>),
    #[serde(borrow)]
    D4(Dataset<'a, 4>),
    #[serde(borrow)]
    D5(Dataset<'a, 5>),
    #[serde(borrow)]
    D6(Dataset<'a, 6>),
    #[serde(borrow)]
    D7(Dataset<'a, 7>),
    #[serde(borrow)]
    D8(Dataset<'a, 8>),
    #[serde(borrow)]
    D9(Dataset<'a, 9>),
}

impl DatasetD<'_> {
    pub fn index(ds: &hdf5::Dataset) -> Result<DatasetD<'static>, anyhow::Error> {
        use DatasetD::*;

        match ds.ndim() {
            0 => Ok(D0(Dataset::<0>::index(ds)?)),
            1 => Ok(D1(Dataset::<1>::index(ds)?)),
            2 => Ok(D2(Dataset::<2>::index(ds)?)),
            3 => Ok(D3(Dataset::<3>::index(ds)?)),
            4 => Ok(D4(Dataset::<4>::index(ds)?)),
            5 => Ok(D5(Dataset::<5>::index(ds)?)),
            6 => Ok(D6(Dataset::<6>::index(ds)?)),
            7 => Ok(D7(Dataset::<7>::index(ds)?)),
            8 => Ok(D8(Dataset::<8>::index(ds)?)),
            9 => Ok(D9(Dataset::<9>::index(ds)?)),
            n => panic!("Dataset only implemented for 0..9 dimensions (not {})", n),
        }
    }

    pub fn as_reader(&self, path: &Path) -> Result<impl Reader + '_, anyhow::Error> {
        use crate::reader::cache::CacheReader;
        use std::fs;
        use DatasetD::*;

        type UReader<'a, R> = UnifyReader<
            CacheReader<'a, R, 0>,
            CacheReader<'a, R, 1>,
            CacheReader<'a, R, 2>,
            CacheReader<'a, R, 3>,
            CacheReader<'a, R, 4>,
            CacheReader<'a, R, 5>,
            CacheReader<'a, R, 6>,
            CacheReader<'a, R, 7>,
            CacheReader<'a, R, 8>,
            CacheReader<'a, R, 9>,
        >;

        Ok(match self {
            D0(ds) => UReader::R0(CacheReader::with_dataset(&ds, fs::File::open(path)?)?),
            D1(ds) => UReader::R1(CacheReader::with_dataset(&ds, fs::File::open(path)?)?),
            D2(ds) => UReader::R2(CacheReader::with_dataset(&ds, fs::File::open(path)?)?),
            D3(ds) => UReader::R3(CacheReader::with_dataset(&ds, fs::File::open(path)?)?),
            D4(ds) => UReader::R4(CacheReader::with_dataset(&ds, fs::File::open(path)?)?),
            D5(ds) => UReader::R5(CacheReader::with_dataset(&ds, fs::File::open(path)?)?),
            D6(ds) => UReader::R6(CacheReader::with_dataset(&ds, fs::File::open(path)?)?),
            D7(ds) => UReader::R7(CacheReader::with_dataset(&ds, fs::File::open(path)?)?),
            D8(ds) => UReader::R8(CacheReader::with_dataset(&ds, fs::File::open(path)?)?),
            D9(ds) => UReader::R9(CacheReader::with_dataset(&ds, fs::File::open(path)?)?),
        })
    }

    pub fn as_streamer(&self, path: &Path) -> Result<UnifyStreamer<'_>, anyhow::Error> {
        use crate::reader::stream::StreamReader;
        use DatasetD::*;

        Ok(match self {
            D0(ds) => UnifyStreamer::R0(StreamReader::with_dataset(&ds, path)?),
            D1(ds) => UnifyStreamer::R1(StreamReader::with_dataset(&ds, path)?),
            D2(ds) => UnifyStreamer::R2(StreamReader::with_dataset(&ds, path)?),
            D3(ds) => UnifyStreamer::R3(StreamReader::with_dataset(&ds, path)?),
            D4(ds) => UnifyStreamer::R4(StreamReader::with_dataset(&ds, path)?),
            D5(ds) => UnifyStreamer::R5(StreamReader::with_dataset(&ds, path)?),
            D6(ds) => UnifyStreamer::R6(StreamReader::with_dataset(&ds, path)?),
            D7(ds) => UnifyStreamer::R7(StreamReader::with_dataset(&ds, path)?),
            D8(ds) => UnifyStreamer::R8(StreamReader::with_dataset(&ds, path)?),
            D9(ds) => UnifyStreamer::R9(StreamReader::with_dataset(&ds, path)?),
        })
    }

    pub fn dsize(&self) -> usize {
        use DatasetD::*;
        match self {
            D0(ds) => ds.dsize,
            D1(ds) => ds.dsize,
            D2(ds) => ds.dsize,
            D3(ds) => ds.dsize,
            D4(ds) => ds.dsize,
            D5(ds) => ds.dsize,
            D6(ds) => ds.dsize,
            D7(ds) => ds.dsize,
            D8(ds) => ds.dsize,
            D9(ds) => ds.dsize,
        }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum Datatype {
    UInt(usize),
    Int(usize),
    Float(usize),
    Unknown,
}

impl From<hdf5::Datatype> for Datatype {
    fn from(dtype: hdf5::Datatype) -> Self {
        match dtype {
            _ if dtype.is::<u8>() => Datatype::UInt(dtype.size()),
            _ if dtype.is::<u32>() => Datatype::UInt(dtype.size()),
            _ if dtype.is::<i32>() => Datatype::Int(dtype.size()),
            _ if dtype.is::<i64>() => Datatype::Int(dtype.size()),
            _ if dtype.is::<f32>() => Datatype::Float(dtype.size()),
            _ if dtype.is::<f64>() => Datatype::Float(dtype.size()),
            _ => Datatype::Unknown,
        }
    }
}

/// A HDF5 dataset (a single variable).
#[derive(Debug, Serialize, Deserialize)]
pub struct Dataset<'a, const D: usize> {
    pub dtype: Datatype,
    pub dsize: usize,
    pub order: ByteOrder,

    #[serde(borrow)]
    #[serde(with = "super::serde::chunks_u64s")]
    pub chunks: Cow<'a, [Chunk<D>]>,

    #[serde(with = "super::serde::arr_u64")]
    pub shape: [u64; D],

    #[serde(with = "super::serde::arr_u64")]
    pub chunk_shape: [u64; D],

    #[serde(with = "super::serde::sr_u64")]
    chunk_shape_reduced: [StrengthReducedU64; D],

    #[serde(with = "super::serde::arr_u64")]
    pub scaled_dim_sz: [u64; D],

    #[serde(with = "super::serde::arr_u64")]
    pub dim_sz: [u64; D],

    #[serde(with = "super::serde::arr_u64")]
    pub chunk_dim_sz: [u64; D],
    pub shuffle: bool,
    pub gzip: Option<u8>,
}

impl<const D: usize> Dataset<'_, D> {
    pub fn index(ds: &hdf5::Dataset) -> Result<Dataset<'static, D>, anyhow::Error> {
        ensure!(ds.ndim() == D, "Dataset rank does not match.");

        let shuffle = ds.filters().get_shuffle();
        let gzip = ds.filters().get_gzip();

        if ds.filters().get_fletcher32()
            || ds.filters().get_scale_offset().is_some()
            || ds.filters().get_szip().is_some()
        {
            return Err(anyhow!("{}: Unsupported filter", ds.name()));
        }

        let dtype = ds.dtype()?;
        let dsize = ds.dtype()?.size();
        let order = dtype.byte_order();
        let shape: [u64; D] = ds
            .shape()
            .into_iter()
            .map(|u| u as u64)
            .collect::<Vec<u64>>()
            .as_slice()
            .try_into()?;

        let chunk_shape = ds.chunks().map_or_else(
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

        let mut chunks: Vec<Chunk<D>> = match (ds.num_chunks().is_some(), ds.offset()) {
            // Continuous
            (false, Some(offset)) => Ok::<_, anyhow::Error>(vec![Chunk {
                offset: [ULE::ZERO; D],
                size: ULE::new(ds.storage_size()),
                addr: ULE::new(offset),
            }]),

            // Chunked
            (true, None) => {
                // TODO: See note in chunks.rs about making this faster.
                //
                // HDF5 internally uses chunk_by_coord on read, so it should be faster in most
                // cases. E.g. the btree version of chunk store is hashed on offset.
                //
                let n = ds.num_chunks().expect("weird..");

                // Avoiding Dataset::chunk_info() because of hdf5-rs Register accumulating
                // Handle's and growing out of hand.
                //
                // See: https://github.com/aldanor/hdf5-rust/issues/76
                use hdf5_sys::h5d::{H5Dget_chunk_info_by_coord, H5Dget_space};
                use hdf5_sys::h5s::H5Sclose;

                let dsid = ds.id();
                let space = unsafe { H5Dget_space(dsid) };

                let mut filter_mask: u32 = 0;
                let mut addr: u64 = 0;
                let mut size: u64 = 0;

                let chunks = hdf5::sync::sync(|| {
                    (0..n)
                        .map(|i| {
                            let offset = Self::chunk_coord_at_index(&chunk_shape, &scaled_dim_sz, i as u64);

                            let e = unsafe {
                                H5Dget_chunk_info_by_coord(
                                    dsid,
                                    offset.as_ptr(),
                                    &mut filter_mask,
                                    &mut addr,
                                    &mut size,
                                )
                            };

                            ensure!(e == 0, "Failed to get chunk: {} in {}", i, ds.name()); // TODO: Maybe ds.name() deadlocks?
                            ensure!(
                                filter_mask == 0,
                                "mismatching filter mask with dataset filter mask"
                            );

                            Ok(Chunk {
                                offset: offset
                                    .iter()
                                    .cloned()
                                    .map(ULE::new)
                                    .collect::<Vec<_>>()
                                    .as_slice()
                                    .try_into()
                                    .unwrap(),
                                size: ULE::new(size),
                                addr: ULE::new(addr),
                            })
                        })
                        .collect()
                });

                unsafe { H5Sclose(space) };

                chunks
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

        Ok(Dataset {
            dtype: dtype.into(),
            dsize,
            order: order.into(),
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

    /// Returns an iterator over chunk, offset and size which if joined will make up the specified slice through the
    /// variable.
    pub fn chunk_slices(
        &self,
        indices: Option<&[u64; D]>,
        counts: Option<&[u64; D]>,
    ) -> impl Iterator<Item = (&Chunk<D>, u64, u64)> {
        let indices: [u64; D] = *indices.unwrap_or(&[0; D]);
        let counts: [u64; D] = *counts.unwrap_or(&self.shape);

        assert!(
            indices
                .iter()
                .zip(counts.iter())
                .map(|(i, c)| i + c)
                .zip(self.shape.iter())
                .all(|(l, &s)| l <= s),
            "out of bounds"
        );

        ChunkSlicer::new(self, indices, counts)
    }

    /// Calculate starting coordinates for chunk at given chunk index.
    pub fn chunk_coord_at_index(chunk_shape: &[u64; D], scaled_dim_sz: &[u64; D], mut idx: u64) -> [u64; D]
    {
        use std::array::FixedSizeArray;

        let mut coords = [0u64; D];

        for (c, scaled, chunksz) in izip!(
                coords.as_mut_slice().iter_mut(),
                scaled_dim_sz.as_slice(),
                chunk_shape.as_slice())
        {
            *c = idx / *scaled;
            idx %= *scaled;
            *c *= chunksz;
        }

        assert_eq!(idx, 0);

        coords
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

pub struct ChunkSlicer<'a, const D: usize> {
    dataset: &'a Dataset<'a, D>,
    offset: u64,
    offset_coords: [u64; D],
    start_coords: [u64; D],
    indices: [u64; D],
    counts: [u64; D],
    counts_reduced: [StrengthReducedU64; D],
    end: u64,
}

impl<'a, const D: usize> ChunkSlicer<'a, D> {
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
        assert!(izip!(&indices, &counts, &dataset.shape).all(|(i, c, z)| i + c <= *z));

        ChunkSlicer {
            dataset,
            offset: 0,
            offset_coords: [0; D],
            start_coords: indices,
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

    /// Offset from chunk offset coordinates. `dim_sz` is dimension size of chunk
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

        let chunk: &Chunk<D> = self.dataset.chunk_at_coord(&self.start_coords);

        debug_assert!(
            chunk.contains(&self.start_coords, &self.dataset.chunk_shape)
                == std::cmp::Ordering::Equal
        );

        // position in chunk of current offset
        let chunk_start = Self::chunk_start(
            &self.start_coords,
            &chunk.offset,
            &self.dataset.chunk_dim_sz,
        );

        // Starting from the last dimension we can advance the offset to the end of the dimension
        // of chunk or to the end of the dimension in the slice. As long as these are the
        // exactly the same, and the start of the slice dimension is the same as the start of the
        // chunk dimension, we can move on to the next dimension and see if it should be advanced
        // as well.
        let mut advanced = 0;
        let mut carry = 0;
        let mut i = 0;

        for (idx, start, offset, count, count_reduced, chunk_offset, chunk_len, chunk_dim_sz) in
            izip!(
                &self.indices,
                &mut self.start_coords,
                &mut self.offset_coords,
                &self.counts,
                &self.counts_reduced,
                &chunk.offset,
                &self.dataset.chunk_shape,
                &self.dataset.chunk_dim_sz
            )
            .rev()
        {
            *offset += carry;
            carry = *offset / *count_reduced;
            *offset = *offset % *count_reduced;
            *start = idx + *offset;

            let last = *offset;

            *offset = min(*count, chunk_offset.get() + chunk_len - idx);

            let diff = (*offset - last) * chunk_dim_sz;

            advanced += diff;
            self.offset += diff;

            carry += *offset / *count_reduced;
            *offset = *offset % *count_reduced;
            *start = idx + *offset;
            i += 1;

            if self.offset >= self.end
                || start != &chunk_offset.get()
                || (*start + count) != (chunk_offset.get() + chunk_len)
                || diff == 0
            {
                break;
            }
        }

        i = self.indices.len() - i;
        for (idx, start, offset, count) in izip!(
            &self.indices[..i],
            &mut self.start_coords[..i],
            &mut self.offset_coords[..i],
            &self.counts_reduced[..i]
        )
        .rev()
        {
            *offset += carry;
            carry = *offset / *count;
            *offset = *offset % *count;
            *start = idx + *offset;
            if carry == 0 {
                break;
            }
        }

        debug_assert!(
            advanced > 0,
            "slice iterator not advancing: stuck indefinitely."
        );

        // position in chunk of new offset
        let chunk_end = chunk_start + advanced;

        Some((chunk, chunk_start, chunk_end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    fn test_dataset() -> Dataset<'static, 2> {
        Dataset::<2> {
            dtype: Datatype::Float(4),
            dsize: 4,
            order: ByteOrder::BE,
            shape: [20, 20],
            chunk_shape: [10, 10],
            chunk_shape_reduced: [10u64, 10]
                .iter()
                .map(|i| StrengthReducedU64::new(*i))
                .collect::<Vec<_>>()
                .as_slice()
                .try_into()
                .unwrap(),
            scaled_dim_sz: [2, 1],
            dim_sz: [20, 1],
            chunk_dim_sz: [10, 1],
            chunks: Cow::from(vec![
                Chunk::new(0, 400, [0, 0]),
                Chunk::new(400, 400, [0, 10]),
                Chunk::new(800, 400, [10, 0]),
                Chunk::new(1200, 400, [10, 10]),
            ]),
            shuffle: false,
            gzip: None,
        }
    }

    #[bench]
    fn chunk_coord_at_index(b: &mut Bencher) {
        let d = test_dataset();

        assert_eq!(Dataset::<2>::chunk_coord_at_index(&d.chunk_shape, &d.scaled_dim_sz, 0), [0, 0]);
        assert_eq!(Dataset::<2>::chunk_coord_at_index(&d.chunk_shape, &d.scaled_dim_sz, 1), [0, 10]);
        assert_eq!(Dataset::<2>::chunk_coord_at_index(&d.chunk_shape, &d.scaled_dim_sz, 2), [10, 0]);
        assert_eq!(Dataset::<2>::chunk_coord_at_index(&d.chunk_shape, &d.scaled_dim_sz, 3), [10, 10]);

        b.iter(|| Dataset::<2>::chunk_coord_at_index(&d.chunk_shape, &d.scaled_dim_sz, 3));
    }

    #[bench]
    fn chunk_slices_range(b: &mut Bencher) {
        let d = test_dataset();

        b.iter(|| d.chunk_slices(None, None).for_each(drop));
    }

    #[bench]
    fn make_chunk_slices_iterator(b: &mut Bencher) {
        let d = test_dataset();

        b.iter(|| test::black_box(d.chunk_slices(None, None)))
    }

    #[bench]
    fn chunk_at_coord(b: &mut Bencher) {
        let d = test_dataset();

        println!("chunks: {:#?}", d.chunks);

        assert_eq!(d.chunk_at_coord(&[0, 0]).offset, [ULE::new(0), ULE::new(0)]);
        assert_eq!(d.chunk_at_coord(&[0, 5]).offset, [ULE::new(0), ULE::new(0)]);
        assert_eq!(d.chunk_at_coord(&[5, 5]).offset, [ULE::new(0), ULE::new(0)]);
        assert_eq!(
            d.chunk_at_coord(&[0, 10]).offset,
            [ULE::new(0), ULE::new(10)]
        );
        assert_eq!(
            d.chunk_at_coord(&[0, 15]).offset,
            [ULE::new(0), ULE::new(10)]
        );
        assert_eq!(
            d.chunk_at_coord(&[10, 0]).offset,
            [ULE::new(10), ULE::new(0)]
        );
        assert_eq!(
            d.chunk_at_coord(&[10, 1]).offset,
            [ULE::new(10), ULE::new(0)]
        );
        assert_eq!(
            d.chunk_at_coord(&[15, 1]).offset,
            [ULE::new(10), ULE::new(0)]
        );

        b.iter(|| test::black_box(d.chunk_at_coord(&[15, 1])))
    }

    #[bench]
    fn chunk_start(b: &mut Bencher) {
        let dim_sz = [10, 1];
        let coords = [20, 10];
        let ch_offset = [ULE::new(20), ULE::new(10)];

        b.iter(|| test::black_box(ChunkSlicer::chunk_start(&coords, &ch_offset, &dim_sz)))
    }

    #[test]
    fn chunk_slices_scenarios() {
        let d = test_dataset();

        assert_eq!(
            d.chunk_slices(Some(&[0, 0]), Some(&[1, 10]))
                .collect::<Vec<(&Chunk<2>, u64, u64)>>(),
            [(&d.chunks[0], 0, 10)]
        );

        assert_eq!(
            d.chunk_slices(Some(&[0, 0]), Some(&[1, 20]))
                .collect::<Vec<(&Chunk<2>, u64, u64)>>(),
            [(&d.chunks[0], 0, 10), (&d.chunks[1], 0, 10)]
        );

        assert_eq!(
            d.chunk_slices(Some(&[0, 5]), Some(&[1, 15]))
                .collect::<Vec<(&Chunk<2>, u64, u64)>>(),
            [(&d.chunks[0], 5, 10), (&d.chunks[1], 0, 10)]
        );

        assert_eq!(
            d.chunk_slices(Some(&[0, 0]), Some(&[2, 10]))
                .collect::<Vec<(&Chunk<2>, u64, u64)>>(),
            [(&d.chunks[0], 0, 20)]
        );

        assert_eq!(
            d.chunk_slices(Some(&[0, 5]), Some(&[2, 10]))
                .collect::<Vec<(&Chunk<2>, u64, u64)>>(),
            [
                (&d.chunks[0], 5, 10),
                (&d.chunks[1], 0, 5),
                (&d.chunks[0], 15, 20),
                (&d.chunks[1], 10, 15)
            ]
        );

        assert_eq!(
            d.chunk_slices(Some(&[0, 0]), Some(&[2, 20]))
                .collect::<Vec<(&Chunk<2>, u64, u64)>>(),
            [
                (&d.chunks[0], 0, 10),
                (&d.chunks[1], 0, 10),
                (&d.chunks[0], 10, 20),
                (&d.chunks[1], 10, 20)
            ]
        );

        assert_eq!(
            d.chunk_slices(Some(&[2, 0]), Some(&[1, 10]))
                .collect::<Vec<(&Chunk<2>, u64, u64)>>(),
            [(&d.chunks[0], 20, 30),]
        );

        assert_eq!(
            d.chunk_slices(Some(&[2, 5]), Some(&[1, 10]))
                .collect::<Vec<(&Chunk<2>, u64, u64)>>(),
            [(&d.chunks[0], 25, 30), (&d.chunks[1], 20, 25),]
        );

        // column
        assert_eq!(
            d.chunk_slices(Some(&[2, 5]), Some(&[4, 1]))
                .collect::<Vec<(&Chunk<2>, u64, u64)>>(),
            [
                (&d.chunks[0], 25, 26),
                (&d.chunks[0], 35, 36),
                (&d.chunks[0], 45, 46),
                (&d.chunks[0], 55, 56),
            ]
        );

        assert_eq!(
            d.chunk_slices(Some(&[2, 15]), Some(&[4, 1]))
                .collect::<Vec<(&Chunk<2>, u64, u64)>>(),
            [
                (&d.chunks[1], 25, 26),
                (&d.chunks[1], 35, 36),
                (&d.chunks[1], 45, 46),
                (&d.chunks[1], 55, 56),
            ]
        );
    }

    #[test]
    fn coads_slice_all() {
        use crate::idx::Index;
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let d = i.dataset("SST").unwrap();
        if let DatasetD::D3(d) = d {
            println!(
                "slices: {}",
                d.chunk_slices(None, None).collect::<Vec<_>>().len()
            );
        } else {
            panic!("wrong dims")
        }
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

    #[test]
    fn serialize_variant_d2() {
        use flexbuffers::FlexbufferSerializer as ser;
        let d = DatasetD::D2(test_dataset());

        println!("serialize");
        let mut s = ser::new();
        d.serialize(&mut s).unwrap();

        println!("deserialize");
        let r = flexbuffers::Reader::get_root(s.view()).unwrap();
        let md = DatasetD::deserialize(r).unwrap();
        if let DatasetD::D2(md) = md {
            if let DatasetD::D2(d) = d {
                for (a, b) in izip!(d.chunk_shape_reduced.iter(), md.chunk_shape_reduced.iter()) {
                    assert_eq!(a.get(), b.get());
                }
            } else {
                panic!("wrong variant");
            }
        } else {
            panic!("wrong variant");
        }
    }
}
