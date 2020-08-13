use itertools::izip;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::cmp::min;
use std::convert::TryInto;
use strength_reduce::StrengthReducedU64;

use super::chunk::Chunk;
use crate::filters::byteorder::Order as ByteOrder;

/// Dataset in possible dimensions.
#[derive(Debug)]
pub enum DatasetD {
    D1(Dataset<1>),
    D2(Dataset<2>),
    D3(Dataset<3>),
    D4(Dataset<4>),
}

impl DatasetD {
    pub fn index(ds: &hdf5::Dataset) -> Result<DatasetD, anyhow::Error> {
        use DatasetD::*;

        match ds.ndim() {
            1 => Ok(D1(Dataset::<1>::index(ds)?)),
            2 => Ok(D2(Dataset::<2>::index(ds)?)),
            3 => Ok(D3(Dataset::<3>::index(ds)?)),
            4 => Ok(D4(Dataset::<4>::index(ds)?)),
            _ => unimplemented!()
        }
    }

    pub fn read(&self) -> () {
    }
}

#[derive(Debug, PartialEq)]
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

/// A Dataset can have a maximum of _32_ dimensions.
#[derive(Debug)]
pub struct Dataset<const D: usize> where [u64; D]: std::array::LengthAtMost32 {
    pub dtype: Datatype,
    pub dsize: usize,
    pub order: ByteOrder,
    pub chunks: Vec<Chunk<D>>,
    pub shape: [u64; D],
    pub chunk_shape: [u64; D],

    // #[serde(
    //     serialize_with = "serialize_sru64",
    //     deserialize_with = "deserialize_sru64"
    // )]
    // chunk_shape_reduced: [StrengthReducedU64; D],

    scaled_dim_sz: [u64; D],
    dim_sz: [u64; D],
    chunk_dim_sz: [u64; D],
    pub shuffle: bool,
    pub gzip: Option<u8>,
}

// fn serialize_sru64<S>(v: &Vec<StrengthReducedU64>, s: S) -> Result<S::Ok, S::Error>
// where
//     S: Serializer,
// {
//     let b: Vec<u64> = v.iter().map(|s| s.get()).collect();
//     b.serialize(s)
// }

// fn deserialize_sru64<'de, D>(d: D) -> Result<Vec<StrengthReducedU64>, D::Error>
// where
//     D: Deserializer<'de>,
// {
//     let v = Vec::<u64>::deserialize(d)?;
//     Ok(v.iter().map(|v| StrengthReducedU64::new(*v)).collect())
// }

impl<const D: usize> Dataset<D> where [u64; D]: std::array::LengthAtMost32 {
    pub fn index(ds: &hdf5::Dataset) -> Result<Dataset<D>, anyhow::Error> {
        ensure!(ds.ndim() == D, "Dataset dimensions does not match!");

        let shuffle = ds.filters().get_shuffle();
        let gzip = ds.filters().get_gzip();

        if ds.filters().get_fletcher32()
            || ds.filters().get_scale_offset().is_some()
            || ds.filters().get_szip().is_some()
        {
            return Err(anyhow!("{}: Unsupported filter", ds.name()));
        }

        let mut chunks: Vec<Chunk<D>> = match (ds.num_chunks().is_some(), ds.offset()) {
            // Continuous
            (false, Some(offset)) => Ok::<_, anyhow::Error>(vec![Chunk {
                offset: [0; D],
                size: ds.storage_size(),
                addr: offset,
            }]),

            // Chunked
            (true, None) => {
                let n = ds.num_chunks().expect("weird..");

                (0..n)
                    .map(|i| {
                        ds.chunk_info(i)
                            .map(|ci| {
                                assert!(ci.filter_mask == 0);
                                Chunk {
                                    offset: ci.offset.as_slice().try_into().unwrap(),
                                    size: ci.size,
                                    addr: ci.addr,
                                }
                            })
                            .ok_or_else(|| anyhow!("{}: Could not get chunk info", ds.name()))
                    })
                    .collect()
            }

            _ => Err(anyhow!(
                "{}: Unsupported data layout (chunked: {}, offset: {:?})",
                ds.name(),
                ds.is_chunked(),
                ds.offset()
            )),
        }?;

        chunks.sort();

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
            || shape.clone(),
            |cs| cs.into_iter().map(|u| u as u64).collect::<Vec<u64>>().as_slice().try_into().unwrap(),
        );

        {
            let expected_chunks = shape
                .iter()
                .zip(&chunk_shape)
                .map(|(s, c)| (s + (c - 1)) / c)
                .product::<u64>() as usize;

            anyhow::ensure!(
                chunks.len() == expected_chunks,
                "{}: unexpected number of chunks given dataset size (is_chunked: {}, chunks: {} != {} (expected), shape: {:?}, chunk shape: {:?})",
                ds.name(),
                ds.is_chunked(),
                chunks.len(),
                expected_chunks,
                shape,
                chunk_shape);
        }

        // optimized divisor for chunk shape
        // let chunk_shape_reduced = chunk_shape
        //     .iter()
        //     .map(|c| StrengthReducedU64::new(*c))
        //     .collect();

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
        }.as_slice().try_into()?;

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
        }.as_slice().try_into()?;

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
        }.as_slice().try_into()?;

        Ok(Dataset {
            dtype: dtype.into(),
            dsize,
            order: order.into(),
            chunks,
            shape,
            chunk_shape,
            // chunk_shape_reduced,
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

    pub fn chunk_at_coord(&self, indices: &[u64]) -> &Chunk<D> {
        debug_assert_eq!(indices.len(), self.chunk_shape.len());
        debug_assert_eq!(indices.len(), self.scaled_dim_sz.len());

        let offset = indices
            .iter()
            .zip(&self.chunk_shape)
            .zip(&self.scaled_dim_sz)
            .fold(0, |offset, ((&index, &ch_sh), &sz)| {
                offset + index / ch_sh * sz
            });

        &self.chunks[offset as usize]
    }
}

pub struct ChunkSlicer<'a, const D: usize> where [u64; D]: std::array::LengthAtMost32 {
    dataset: &'a Dataset<D>,
    offset: u64,
    offset_coords: [u64; D],
    start_coords: [u64; D],
    indices: [u64; D],
    counts: [u64; D],
    // counts_reduced: [StrengthReducedU64; D],
    end: u64,
}

impl<'a, const D: usize> ChunkSlicer<'a, D> where [u64; D]: std::array::LengthAtMost32 {
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
            start_coords: indices.clone(),
            indices: indices,
            // counts_reduced: counts.iter().map(|c| StrengthReducedU64::new(*c)).collect(),
            counts: counts,
            end,
        }
    }

    /// Offset from chunk offset coordinates. `dim_sz` is dimension size of chunk
    /// dimensions.
    fn chunk_start(coords: &[u64; D], chunk_offset: &[u64; D], dim_sz: &[u64; D]) -> u64 {
        debug_assert_eq!(coords.len(), chunk_offset.len());
        debug_assert_eq!(coords.len(), dim_sz.len());

        coords
            .iter()
            .zip(chunk_offset)
            .zip(dim_sz)
            .fold(0, |start, ((&coord, &offset), &sz)| {
                start + (coord - offset) * sz
            })
    }
}

impl<'a, const D: usize> Iterator for ChunkSlicer<'a, D> where [u64; D]: std::array::LengthAtMost32 {
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

        for (idx, start, offset, count /* , count_reduced */, chunk_offset, chunk_len, chunk_dim_sz) in
            izip!(
                &self.indices,
                &mut self.start_coords,
                &mut self.offset_coords,
                &self.counts,
                // &self.counts_reduced,
                &chunk.offset,
                &self.dataset.chunk_shape,
                &self.dataset.chunk_dim_sz
            )
            .rev()
        {
            *offset += carry;
            carry = *offset / *count;
            *offset = *offset % *count;
            *start = idx + *offset;

            let last = *offset;

            *offset = min(*count, chunk_offset + chunk_len - idx);

            let diff = (*offset - last) * chunk_dim_sz;

            advanced += diff;
            self.offset += diff;

            carry += *offset / *count;
            *offset = *offset % *count;
            *start = idx + *offset;
            i += 1;

            if self.offset >= self.end
                || start != chunk_offset
                || (*start + count) != (chunk_offset + chunk_len)
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
            &self.counts[..i]
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

    fn test_dataset() -> Dataset<2> {
        Dataset::<2> {
            dtype: Datatype::Float(4),
            dsize: 4,
            order: ByteOrder::BE,
            shape: [20, 20],
            chunk_shape: [10, 10],
            // chunk_shape_reduced: [10u64, 10]
            //     .iter()
            //     .map(|i| StrengthReducedU64::new(*i))
            //     .collect(),
            scaled_dim_sz: [2, 1],
            dim_sz: [20, 1],
            chunk_dim_sz: [10, 1],
            chunks: vec![
                Chunk::<2> {
                    offset: [0, 0],
                    size: 400,
                    addr: 0,
                },
                Chunk::<2> {
                    offset: [0, 10],
                    size: 400,
                    addr: 400,
                },
                Chunk::<2> {
                    offset: [10, 0],
                    size: 400,
                    addr: 800,
                },
                Chunk::<2> {
                    offset: [10, 10],
                    size: 400,
                    addr: 1200,
                },
            ],
            shuffle: false,
            gzip: None,
        }
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
        let mut d = test_dataset();

        d.chunks.sort();

        println!("chunks: {:#?}", d.chunks);

        assert_eq!(d.chunk_at_coord(&[0, 0]).offset, [0, 0]);
        assert_eq!(d.chunk_at_coord(&[0, 5]).offset, [0, 0]);
        assert_eq!(d.chunk_at_coord(&[5, 5]).offset, [0, 0]);
        assert_eq!(d.chunk_at_coord(&[0, 10]).offset, [0, 10]);
        assert_eq!(d.chunk_at_coord(&[0, 15]).offset, [0, 10]);
        assert_eq!(d.chunk_at_coord(&[10, 0]).offset, [10, 0]);
        assert_eq!(d.chunk_at_coord(&[10, 1]).offset, [10, 0]);
        assert_eq!(d.chunk_at_coord(&[15, 1]).offset, [10, 0]);

        b.iter(|| test::black_box(d.chunk_at_coord(&[15, 1])))
    }

    #[bench]
    fn chunk_start(b: &mut Bencher) {
        let dim_sz = vec![10, 1];
        let coords = vec![20, 10];
        let ch_offset = vec![20, 10];

        b.iter(|| test::black_box(ChunkSlicer::chunk_start(&coords, &ch_offset, &dim_sz)))
    }

    #[test]
    fn chunk_slices_scenarios() {
        let d = test_dataset();

        assert_eq!(
            d.chunk_slices(Some(&[0, 0]), Some(&[1, 10]))
                .collect::<Vec<(&Chunk, u64, u64)>>(),
            [(&d.chunks[0], 0, 10)]
        );

        assert_eq!(
            d.chunk_slices(Some(&[0, 0]), Some(&[1, 20]))
                .collect::<Vec<(&Chunk, u64, u64)>>(),
            [(&d.chunks[0], 0, 10), (&d.chunks[1], 0, 10)]
        );

        assert_eq!(
            d.chunk_slices(Some(&[0, 5]), Some(&[1, 15]))
                .collect::<Vec<(&Chunk, u64, u64)>>(),
            [(&d.chunks[0], 5, 10), (&d.chunks[1], 0, 10)]
        );

        assert_eq!(
            d.chunk_slices(Some(&[0, 0]), Some(&[2, 10]))
                .collect::<Vec<(&Chunk, u64, u64)>>(),
            [(&d.chunks[0], 0, 20)]
        );

        assert_eq!(
            d.chunk_slices(Some(&[0, 5]), Some(&[2, 10]))
                .collect::<Vec<(&Chunk, u64, u64)>>(),
            [
                (&d.chunks[0], 5, 10),
                (&d.chunks[1], 0, 5),
                (&d.chunks[0], 15, 20),
                (&d.chunks[1], 10, 15)
            ]
        );

        assert_eq!(
            d.chunk_slices(Some(&[0, 0]), Some(&[2, 20]))
                .collect::<Vec<(&Chunk, u64, u64)>>(),
            [
                (&d.chunks[0], 0, 10),
                (&d.chunks[1], 0, 10),
                (&d.chunks[0], 10, 20),
                (&d.chunks[1], 10, 20)
            ]
        );

        assert_eq!(
            d.chunk_slices(Some(&[2, 0]), Some(&[1, 10]))
                .collect::<Vec<(&Chunk, u64, u64)>>(),
            [(&d.chunks[0], 20, 30),]
        );

        assert_eq!(
            d.chunk_slices(Some(&[2, 5]), Some(&[1, 10]))
                .collect::<Vec<(&Chunk, u64, u64)>>(),
            [(&d.chunks[0], 25, 30), (&d.chunks[1], 20, 25),]
        );

        // column
        assert_eq!(
            d.chunk_slices(Some(&[2, 5]), Some(&[4, 1]))
                .collect::<Vec<(&Chunk, u64, u64)>>(),
            [
                (&d.chunks[0], 25, 26),
                (&d.chunks[0], 35, 36),
                (&d.chunks[0], 45, 46),
                (&d.chunks[0], 55, 56),
            ]
        );

        assert_eq!(
            d.chunk_slices(Some(&[2, 15]), Some(&[4, 1]))
                .collect::<Vec<(&Chunk, u64, u64)>>(),
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

        println!(
            "slices: {}",
            d.chunk_slices(None, None).collect::<Vec<_>>().len()
        );
    }

    #[test]
    fn serialize() {
        let d = test_dataset();

        let s = serde_json::to_string(&d).unwrap();
        println!("serialized: {}", s);

        let md: Dataset = serde_json::from_str(&s).unwrap();

        for (a, b) in izip!(d.chunk_shape_reduced, md.chunk_shape_reduced) {
            assert_eq!(a.get(), b.get());
        }
    }
}
