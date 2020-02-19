use itertools::izip;
use std::cmp::min;
use strength_reduce::StrengthReducedU64;

use super::chunk::Chunk;

use hdf5::{ByteOrder, Datatype};

#[derive(Debug)]
pub struct Dataset {
    pub dtype: Datatype,
    pub dsize: usize,
    pub order: ByteOrder,
    pub chunks: Vec<Chunk>,
    pub shape: Vec<u64>,
    pub chunk_shape: Vec<u64>,
    chunk_shape_reduced: Vec<StrengthReducedU64>,
    scaled_dim_sz: Vec<u64>,
    dim_sz: Vec<u64>,
    chunk_dim_sz: Vec<u64>,
    pub shuffle: bool,
    pub gzip: Option<u8>,
}

impl Dataset {
    pub fn index(ds: &hdf5::Dataset) -> Result<Dataset, anyhow::Error> {
        let shuffle = ds.filters().get_shuffle();
        let gzip = ds.filters().get_gzip();

        if ds.filters().get_fletcher32()
            || ds.filters().get_scale_offset().is_some()
            || ds.filters().get_szip().is_some()
        {
            return Err(anyhow!("{}: Unsupported filter", ds.name()));
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
                            .ok_or_else(|| anyhow!("{}: Could not get chunk info", ds.name()))
                    })
                    .collect()
            }

            _ => Err(anyhow!("{}: Unsupported data layout (chunked: {}, offset: {:?})", ds.name(), ds.is_chunked(), ds.offset())),
        }?;

        chunks.sort();

        let dtype = ds.dtype()?;
        let dsize = dtype.size();
        let order = dtype.byte_order();
        let shape = ds
            .shape()
            .into_iter()
            .map(|u| u as u64)
            .collect::<Vec<u64>>();

        let chunk_shape = ds.chunks().map_or_else(
            || shape.clone(),
            |cs| cs.into_iter().map(|u| u as u64).collect(),
        );

        let chunk_shape_reduced = chunk_shape
            .iter()
            .map(|c| StrengthReducedU64::new(*c))
            .collect();

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

    #[must_use]
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

    pub fn chunk_at_coord(&self, indices: &[u64]) -> &Chunk {
        assert_eq!(indices.len(), self.chunk_shape_reduced.len());
        assert_eq!(indices.len(), self.scaled_dim_sz.len());

        let offset = (0..indices.len()).fold(0, |offset, i| {
            offset + (indices[i] / self.chunk_shape_reduced[i]) * self.scaled_dim_sz[i]
        });

        &self.chunks[offset as usize]
    }
}

pub struct ChunkSlicer<'a> {
    dataset: &'a Dataset,
    offset: u64,
    offset_coords: Vec<u64>,
    start_coords: Vec<u64>,
    indices: Vec<u64>,
    counts: Vec<u64>,
    counts_reduced: Vec<StrengthReducedU64>,
    end: u64,
}

impl<'a> ChunkSlicer<'a> {
    pub fn new(dataset: &'a Dataset, indices: Vec<u64>, counts: Vec<u64>) -> ChunkSlicer<'a> {
        // size of slice dimensions
        let end = counts.iter().product::<u64>();

        ChunkSlicer {
            dataset,
            offset: 0,
            offset_coords: vec![0; indices.len()],
            start_coords: indices.clone(),
            indices: indices,
            counts_reduced: counts.iter().map(|c| StrengthReducedU64::new(*c)).collect(),
            counts: counts,
            end,
        }
    }

    /// Offset from chunk offset coordinates. `dim_sz` is dimension size of chunk
    /// dimensions.
    fn chunk_start(coords: &[u64], chunk_offset: &[u64], dim_sz: &[u64]) -> u64 {
        assert_eq!(coords.len(), chunk_offset.len());
        assert_eq!(coords.len(), dim_sz.len());

        (0..coords.len()).fold(0, |start, i| {
            start + (coords[i] - chunk_offset[i]) * dim_sz[i]
        })
    }
}

impl<'a> Iterator for ChunkSlicer<'a> {
    type Item = (&'a Chunk, u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.end {
            return None;
        }

        let chunk: &Chunk = self.dataset.chunk_at_coord(&self.start_coords);

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
            *start = idx + *offset;
            carry = *offset / *count_reduced;

            let last = *offset;

            *offset = min(*count, chunk_offset + chunk_len - idx);

            let diff = (*offset - last) * chunk_dim_sz;

            advanced += diff;
            self.offset += diff;

            carry += *offset / *count_reduced;
            *offset = *offset % *count_reduced;
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

        debug_assert!(advanced > 0);

        // position in chunk of new offset
        let chunk_end = chunk_start + advanced;

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
            dsize: 4,
            order: hdf5::ByteOrder::BigEndian,
            shape: vec![20, 20],
            chunk_shape: vec![10, 10],
            chunk_shape_reduced: [10u64, 10]
                .iter()
                .map(|i| StrengthReducedU64::new(*i))
                .collect(),
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
        let i = Index::index("../data/coads_climatology.nc4").unwrap();
        let d = i.dataset("SST").unwrap();

        println!(
            "slices: {}",
            d.chunk_slices(None, None).collect::<Vec<_>>().len()
        );
    }
}
