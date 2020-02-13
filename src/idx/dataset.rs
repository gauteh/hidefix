use std::cmp::min;

use super::chunk::Chunk;

use hdf5::Datatype;
use hdf5_sys::h5t::H5T_order_t;

#[derive(Debug)]
pub struct Dataset {
    pub dtype: Datatype,
    pub order: H5T_order_t,
    pub chunks: Vec<Chunk>,
    pub shape: Vec<u64>,
    pub chunk_shape: Vec<u64>,
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

        Ok(Dataset {
            dtype,
            order,
            chunks,
            shape,
            chunk_shape,
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

        // NOTE: The collapser adds some overhead (about 1 us on
        // simple tests. But usually saves it in causing fewer
        // reads.

        ChunkSlicerCollapsed::new(
            ChunkSlicer::new(self, indices, counts),
            self.dtype.size() as u64,
        )
    }

    /// Find chunk containing coordinate.
    pub fn chunk_at_coord(&self, indices: &[u64]) -> Result<&Chunk, anyhow::Error> {
        // NOTE: This seems to be faster than the explicit expression
        // (by factor 3x). Perhaps a less convuluted expression can be
        // found. See `1e14162:hidefix/src/idx/dataset.rs`.
        self.chunks
            .binary_search_by(|c| c.contains(indices, self.chunk_shape.as_slice()).reverse())
            .map(|i| &self.chunks[i])
            .map_err(|_| anyhow!("could not find chunk"))
    }
}

pub struct ChunkSlicer<'a> {
    indices: Vec<u64>,
    counts: Vec<u64>,
    offset: Vec<u64>,
    dataset: &'a Dataset,
    chunk_sz: Vec<u64>,
}

impl<'a> ChunkSlicer<'a> {
    pub fn new(dataset: &'a Dataset, indices: Vec<u64>, counts: Vec<u64>) -> ChunkSlicer<'a> {
        // size of chunk dimensions
        let chunk_sz = {
            let mut d = dataset
                .chunk_shape
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

        let offset = vec![0; indices.len()];

        ChunkSlicer {
            indices,
            counts,
            offset,
            dataset,
            chunk_sz,
        }
    }
}

impl<'a> Iterator for ChunkSlicer<'a> {
    type Item = (&'a Chunk, u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        // advance offset
        let mut carry = 0;
        for (o, c) in self.offset.iter_mut().zip(&self.counts).rev() {
            *o += carry;
            carry = *o / c;
            *o %= c;
        }

        if carry > 0 {
            return None;
        }

        let idx: Vec<u64> = self
            .indices
            .iter()
            .zip(self.offset.iter())
            .map(|(i, o)| i + *o)
            .collect();

        let chunk: &Chunk = self
            .dataset
            .chunk_at_coord(&idx)
            .expect("Moved index out of dataset!");

        let chunk_last = chunk.offset.last().unwrap();
        let shape_last = self.dataset.chunk_shape.last().unwrap();

        // position in chunk of current offset
        let chunk_start = idx
            .iter()
            .zip(&chunk.offset)
            .map(|(o, c)| o - c)
            .zip(&self.chunk_sz)
            .map(|(d, sz)| d * sz)
            .sum::<u64>();

        let last = self.offset.last_mut().unwrap();

        // determine how far we can advance the offset along in the current chunk.
        *last = min(
            *self.counts.last().unwrap(),
            chunk_last + shape_last - self.indices.last().unwrap(),
        );

        // position in chunk of new offset
        let chunk_end = self
            .indices
            .iter()
            .zip(&self.offset)
            .map(|(i, o)| i + *o)
            .zip(&chunk.offset)
            .map(|(o, c)| o - c)
            .zip(&self.chunk_sz)
            .map(|(d, sz)| d * sz)
            .sum::<u64>();

        Some((chunk, chunk_start, chunk_end))
    }
}

pub struct ChunkSlicerCollapsed<'a> {
    slicer: std::iter::Fuse<ChunkSlicer<'a>>,
    nxt: Option<(&'a Chunk, u64, u64)>,
    sz: u64,
}

impl<'a> ChunkSlicerCollapsed<'a> {
    pub fn new(slicer: ChunkSlicer<'a>, sz: u64) -> ChunkSlicerCollapsed<'a> {
        ChunkSlicerCollapsed {
            slicer: slicer.fuse(),
            nxt: None,
            sz,
        }
    }
}

impl<'a> Iterator for ChunkSlicerCollapsed<'a> {
    type Item = (&'a Chunk, u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        let (ac, ab, mut ae) = self.nxt.or_else(|| self.slicer.next())?;

        self.nxt = None;

        while let Some((bc, bb, be)) = self.slicer.next() {
            if ac.addr == bc.addr && ac.addr + ae * self.sz == bc.addr + bb * self.sz {
                // chunk slices are adjacent, extend chunk
                ae = ae + (be - bb);
            } else {
                // we found a jump, set next and emit current slice
                self.nxt = Some((bc, bb, be));
                return Some((ac, ab, ae));
            }
        }
        return Some((ac, ab, ae));
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

        b.iter(|| d.chunk_slices(None, None).collect::<Vec<_>>());
    }

    #[test]
    fn chunk_at_coord() {
        let mut d = test_dataset();

        d.chunks.sort();

        assert_eq!(d.chunk_at_coord(&[0, 0]).unwrap().offset, [0, 0]);
        assert_eq!(d.chunk_at_coord(&[0, 5]).unwrap().offset, [0, 0]);
        assert_eq!(d.chunk_at_coord(&[5, 5]).unwrap().offset, [0, 0]);
        assert_eq!(d.chunk_at_coord(&[0, 10]).unwrap().offset, [0, 10]);
        assert_eq!(d.chunk_at_coord(&[0, 15]).unwrap().offset, [0, 10]);
        assert_eq!(d.chunk_at_coord(&[10, 0]).unwrap().offset, [10, 0]);
        assert_eq!(d.chunk_at_coord(&[10, 1]).unwrap().offset, [10, 0]);
        assert_eq!(d.chunk_at_coord(&[15, 1]).unwrap().offset, [10, 0]);
    }

    #[test]
    fn chunk_slices() {
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
    }
}
