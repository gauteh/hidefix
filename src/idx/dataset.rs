use itertools::izip;
use smallvec::{smallvec, SmallVec};
use std::cmp::min;

use super::chunk::Chunk;

use hdf5::Datatype;
use hdf5_sys::h5t::H5T_order_t;

/// Size of coordinate vectors used which do not require allocations. Slicing variables with
/// greater dimensions than this will be slower.
const COORD_SZ: usize = 4;

#[derive(Debug)]
pub struct Dataset {
    pub dtype: Datatype,
    pub order: H5T_order_t,
    pub chunks: Vec<Chunk>,
    pub shape: Vec<u64>,
    pub chunk_shape: Vec<u64>,
    scaled_dim_sz: Vec<u64>,
    dim_sz: Vec<u64>,
    chunk_dim_sz: Vec<u64>,
    pub shuffle: bool,
    pub gzip: Option<u8>,
}

impl Dataset {
    pub fn index(ds: hdf5::Dataset) -> Result<Dataset, anyhow::Error> {
        let shuffle = ds.filters().get_shuffle();
        let gzip = ds.filters().get_gzip();

        if ds.filters().get_fletcher32()
            || ds.filters().get_scale_offset().is_some()
            || ds.filters().get_szip().is_some()
        {
            return Err(anyhow!("Unsupported filter"));
        }

        let mut chunks: Vec<Chunk> = match (ds.is_chunked(), ds.offset()) {
            // Continuous
            (false, Some(offset)) => Ok::<_, anyhow::Error>(vec![Chunk {
                offset: vec![0; ds.ndim()],
                size: ds.storage_size(),
                addr: offset,
            }]),

            // Chunked
            (true, None) => {
                let n = ds.num_chunks().expect("weird..");

                (0..n)
                    .map(|i| {
                        ds.chunk_info(i)
                            .map(|ci| Chunk {
                                offset: ci.offset,
                                size: ci.size,
                                addr: ci.addr,
                            })
                            .ok_or_else(|| anyhow!("Could not get chunk info"))
                    })
                    .collect()
            }

            _ => Err(anyhow!("Unsupported data layout")),
        }?;

        chunks.sort();

        let dtype = ds.dtype()?;
        let order = dtype.byte_order();
        let shape = ds
            .shape()
            .into_iter()
            .map(|u| u as u64)
            .collect::<Vec<u64>>();
        let chunk_shape = ds
            .chunks()
            .map(|cs| cs.into_iter().map(|u| u as u64).collect())
            .unwrap_or(shape.clone());

        // scaled dimensions
        let scaled_dim_sz = {
            let mut d = shape
                .iter()
                .zip(&chunk_shape)
                .map(|(d, z)| d / z)
                .rev()
                .scan(1, |p, c| {
                    let sz = *p;
                    *p *= c;
                    Some(sz)
                })
                .collect::<Vec<u64>>();
            d.reverse();
            d
        };

        // size of dataset dimensions
        let dim_sz = {
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
        };

        // size of chunk dimensions
        let chunk_dim_sz = {
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
        };

        Ok(Dataset {
            dtype,
            order,
            chunks,
            shape,
            chunk_shape,
            scaled_dim_sz,
            dim_sz,
            chunk_dim_sz,
            shuffle,
            gzip,
        })
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }

    /// Returns an iterator over chunk, offset and size which if joined will make up the specified slice through the
    /// variable.
    pub fn chunk_slices(
        &self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
    ) -> impl Iterator<Item = (&Chunk, u64, u64)> {
        let indices: Vec<u64> = indices.unwrap_or(&vec![0; self.shape.len()]).to_vec();
        let counts: Vec<u64> = counts.unwrap_or(&self.shape).to_vec();

        assert!(
            indices
                .iter()
                .zip(&counts)
                .map(|(i, c)| i + c)
                .zip(&self.shape)
                .all(|(l, &s)| l <= s),
            "out of bounds"
        );

        ChunkSlicer::new(self, indices, counts)
    }

    pub fn chunk_at_coord(&self, indices: &[u64]) -> Result<&Chunk, anyhow::Error> {
        // scale coordinates
        let mut scaled = SmallVec::<[u64; COORD_SZ]>::with_capacity(indices.len());
        unsafe {
            scaled.set_len(indices.len());
        }

        for (s, i, csz) in izip!(&mut scaled, indices, &self.chunk_shape) {
            *s = i / csz;
        }

        let scaled_offset = scaled
            .iter()
            .zip(&self.scaled_dim_sz)
            .map(|(c, sz)| c * sz)
            .sum::<u64>();
        Ok(&self.chunks[scaled_offset as usize])
    }
}

pub struct ChunkSlicer<'a> {
    dataset: &'a Dataset,
    offset: u64,
    offset_coords: SmallVec<[u64; COORD_SZ]>,
    start_coords: SmallVec<[u64; COORD_SZ]>,
    indices: SmallVec<[u64; COORD_SZ]>,
    counts: SmallVec<[u64; COORD_SZ]>,
    end: u64,
    slice_sz: SmallVec<[u64; COORD_SZ]>,
}

impl<'a> ChunkSlicer<'a> {
    pub fn new(dataset: &'a Dataset, indices: Vec<u64>, counts: Vec<u64>) -> ChunkSlicer<'a> {
        // size of slice dimensions
        let slice_sz = Self::dim_sz(&counts);
        let end = counts.iter().product::<u64>();

        ChunkSlicer {
            dataset,
            offset: 0,
            offset_coords: smallvec![0; indices.len()],
            start_coords: SmallVec::from_slice(&indices),
            indices: SmallVec::from_vec(indices),
            counts: SmallVec::from_vec(counts),
            end,
            slice_sz,
        }
    }

    fn dim_sz(dims: &[u64]) -> SmallVec<[u64; COORD_SZ]> {
        let mut d = dims
            .iter()
            .rev()
            .scan(1, |p, &c| {
                let sz = *p;
                *p *= c;
                Some(sz)
            })
            .collect::<SmallVec<[u64; COORD_SZ]>>();
        d.reverse();
        d
    }

    fn offset_at_coords(dim_sz: &[u64], coords: &[u64]) -> u64 {
        coords.iter().zip(dim_sz).map(|(i, z)| i * z).sum::<u64>()
    }

    fn coords_at_offset(mut offset: u64, dim_sz: &[u64], coords: &mut [u64]) {
        for (c, sz) in coords.iter_mut().zip(dim_sz) {
            *c = offset / sz;
            offset = offset % sz;
        }
    }

    fn chunk_start(coords: &[u64], chunk_offset: &[u64], dim_sz: &[u64]) -> u64 {
        coords
            .iter()
            .zip(chunk_offset)
            .map(|(o, c)| o - c)
            .zip(dim_sz)
            .map(|(d, sz)| d * sz)
            .sum::<u64>()
    }
}

impl<'a> Iterator for ChunkSlicer<'a> {
    type Item = (&'a Chunk, u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.end {
            return None;
        }

        let chunk: &Chunk = self
            .dataset
            .chunk_at_coord(&self.start_coords)
            .expect("Moved index out of dataset!");

        let chunk_last = chunk.offset.last().unwrap();
        let shape_last = self.dataset.chunk_shape.last().unwrap();

        // position in chunk of current offset
        let chunk_start = Self::chunk_start(
            &self.start_coords,
            &chunk.offset,
            &self.dataset.chunk_dim_sz,
        );

        debug_assert!(
            chunk.contains(&self.start_coords, &self.dataset.chunk_shape)
                == std::cmp::Ordering::Equal
        );

        let mut advanced = 0;

        // we can advance in the last dimension, and any dimensions within the chunk
        // that overlap completely with the slice.

        // determine how far we can advance the offset along in the current chunk.
        while self.offset < self.end
            && chunk.contains(&self.start_coords, &self.dataset.chunk_shape)
            == std::cmp::Ordering::Equal
            && (Self::chunk_start(
                &self.start_coords,
                &chunk.offset,
                &self.dataset.chunk_dim_sz,
            ) - chunk_start)
                == advanced
        {
            // position in last dimension of offset
            let old_last = *self.offset_coords.last().unwrap();

            *self.offset_coords.last_mut().unwrap() = min(
                *self.counts.last().unwrap(),
                chunk_last + shape_last - self.indices.last().unwrap(),
            );

            let diff = self.offset_coords.last().unwrap() - old_last;
            self.offset += diff;
            advanced += diff;

            debug_assert!(diff > 0);

            Self::coords_at_offset(self.offset, &self.slice_sz, &mut self.offset_coords);

            for (i, o, s) in izip!(&self.indices, &self.offset_coords, &mut self.start_coords) {
                *s = *i + *o;
            }
        }

        debug_assert!(advanced > 0);

        // position in chunk of new offset
        let chunk_end = chunk_start + advanced;

        debug_assert!(chunk_end > chunk_start);

        Some((chunk, chunk_start, chunk_end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    fn test_dataset() -> Dataset {
        Dataset {
            dtype: Datatype::from_type::<f32>().unwrap(),
            order: H5T_order_t::H5T_ORDER_LE,
            shape: vec![20, 20],
            chunk_shape: vec![10, 10],
            scaled_dim_sz: vec![2, 1],
            dim_sz: vec![20, 1],
            chunk_dim_sz: vec![10, 1],
            chunks: vec![
                Chunk {
                    offset: vec![0, 0],
                    size: 400,
                    addr: 0,
                },
                Chunk {
                    offset: vec![0, 10],
                    size: 400,
                    addr: 400,
                },
                Chunk {
                    offset: vec![10, 0],
                    size: 400,
                    addr: 800,
                },
                Chunk {
                    offset: vec![10, 10],
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

        assert_eq!(d.chunk_at_coord(&[0, 0]).unwrap().offset, [0, 0]);
        assert_eq!(d.chunk_at_coord(&[0, 5]).unwrap().offset, [0, 0]);
        assert_eq!(d.chunk_at_coord(&[5, 5]).unwrap().offset, [0, 0]);
        assert_eq!(d.chunk_at_coord(&[0, 10]).unwrap().offset, [0, 10]);
        assert_eq!(d.chunk_at_coord(&[0, 15]).unwrap().offset, [0, 10]);
        assert_eq!(d.chunk_at_coord(&[10, 0]).unwrap().offset, [10, 0]);
        assert_eq!(d.chunk_at_coord(&[10, 1]).unwrap().offset, [10, 0]);
        assert_eq!(d.chunk_at_coord(&[15, 1]).unwrap().offset, [10, 0]);

        b.iter(|| test::black_box(d.chunk_at_coord(&[15, 1]).unwrap()))
    }

    #[bench]
    fn offset_at_coords(b: &mut Bencher) {
        let dim_sz = vec![30 * 40, 30, 1];

        let coords = vec![10, 10, 10];

        b.iter(|| test::black_box(ChunkSlicer::offset_at_coords(&dim_sz, &coords)))
    }

    #[bench]
    fn coords_at_offset(b: &mut Bencher) {
        let dim_sz = vec![30 * 40, 30, 1];

        let mut coords = vec![10, 10, 10];
        let offset = 30 * 10 + 40 * 10 + 10;

        b.iter(|| test::black_box(ChunkSlicer::coords_at_offset(offset, &dim_sz, &mut coords)))
    }

    #[bench]
    fn chunk_start(b: &mut Bencher) {
        let dim_sz = vec![10, 1];
        let coords = vec![20, 10];
        let ch_offset = vec![20, 10];

        b.iter(|| test::black_box(ChunkSlicer::chunk_start(&coords, &ch_offset, &dim_sz)))
    }

    #[bench]
    fn dim_sz(b: &mut Bencher) {
        let counts = vec![12, 180, 90];

        b.iter(|| test::black_box(ChunkSlicer::dim_sz(&counts)))
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
        let i = Index::index("../data/coads_climatology.nc4").unwrap();
        let d = i.dataset("SST").unwrap();

        println!(
            "slices: {}",
            d.chunk_slices(None, None).collect::<Vec<_>>().len()
        );
    }
}
