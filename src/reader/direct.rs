use crate::filters::byteorder::Order;
use std::path::{Path, PathBuf};

use super::{
    chunk::{decode_chunk, read_chunk, read_chunk_to},
    dataset::Reader,
};
use crate::idx::{Chunk, Dataset};

pub struct Direct<'a, const D: usize> {
    ds: &'a Dataset<'a, D>,
    path: PathBuf,
    chunk_sz: u64,
}

impl<'a, const D: usize> Direct<'a, D> {
    pub fn with_dataset<P: AsRef<Path>>(
        ds: &'a Dataset<D>,
        path: P,
    ) -> Result<Direct<'a, D>, anyhow::Error> {
        let chunk_sz = ds.chunk_shape.iter().product::<u64>() * ds.dsize as u64;

        Ok(Direct {
            ds,
            path: path.as_ref().into(),
            chunk_sz,
        })
    }

    #[cfg(off)]
    pub fn read_to_par(
        &self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
        dst: &mut [u8],
    ) -> Result<u64, anyhow::Error> {
        use rayon::prelude::*;

        let indices: Option<&[u64; D]> = indices
            .map(|i| i.try_into())
            .map_or(Ok(None), |v| v.map(Some))
            .unwrap();

        let counts: Option<&[u64; D]> = counts
            .map(|i| i.try_into())
            .map_or(Ok(None), |v| v.map(Some))
            .unwrap();

        let counts: &[u64; D] = counts.unwrap_or(&self.ds.shape);

        let dsz = self.ds.dsize as u64;
        let vsz = counts.iter().product::<u64>() * dsz;

        ensure!(
            dst.len() >= vsz as usize,
            "destination buffer has insufficient capacity"
        );

        let groups = self.ds.group_chunk_slices(indices, Some(counts));

        let mut fd = std::fs::File::open(&self.path)?;

        // Read all chunks sequentially: maybe this is too big?
        let chunks = groups
            .iter()
            .map(|(c, _segments)| {
                let mut chunk = vec![0u8; c.size.get() as usize];
                read_chunk_to(&mut fd, c.addr.get(), &mut chunk)?;
                Ok(chunk)
            })
            .collect::<Result<Vec<Vec<u8>>, anyhow::Error>>()?;

        // Decode chunks in parallel
        let chunks = groups
            .into_par_iter()
            .zip(chunks)
            .map(|((c, segments), chunk)| {
                Ok((
                    c,
                    segments,
                    decode_chunk(
                        chunk,
                        self.chunk_sz,
                        dsz,
                        self.ds.gzip.is_some(),
                        self.ds.shuffle,
                    )?,
                ))
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?;

        // Extract segments from chunks to destination vector
        //
        // TODO:This can also be done in the parallel operation above, when I figure out
        // hwo to split the destination vector.
        for (_c, segments, chunk) in chunks {
            for (current, start, end) in segments {
                let start = (start * dsz) as usize;
                let end = (end * dsz) as usize;
                let current = (current * dsz) as usize;

                debug_assert!(start <= chunk.len());
                debug_assert!(end <= chunk.len());

                let sz = end - start;
                dst[current..(current + sz)].copy_from_slice(&chunk[start..end]);
            }
        }

        Ok(vsz)
    }
}

impl<'a, const D: usize> Reader for Direct<'a, D> {
    fn order(&self) -> Order {
        self.ds.order
    }

    fn dsize(&self) -> usize {
        self.ds.dsize
    }

    fn shape(&self) -> &[u64] {
        &self.ds.shape
    }

    fn read_to(
        &mut self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
        dst: &mut [u8],
    ) -> Result<usize, anyhow::Error> {
        // let sz = self.read_to_par(indices, counts, dst)?;
        // return Ok(sz as usize);

        let indices: Option<&[u64; D]> = indices
            .map(|i| i.try_into())
            .map_or(Ok(None), |v| v.map(Some))
            .unwrap();

        let counts: Option<&[u64; D]> = counts
            .map(|i| i.try_into())
            .map_or(Ok(None), |v| v.map(Some))
            .unwrap();

        let counts: &[u64; D] = counts.unwrap_or(&self.ds.shape);

        let dsz = self.ds.dsize as u64;
        let vsz = counts.iter().product::<u64>() * dsz;

        ensure!(
            dst.len() >= vsz as usize,
            "destination buffer has insufficient capacity"
        );

        let groups = self.ds.group_chunk_slices(indices, Some(counts));

        let mut fd = std::fs::File::open(&self.path)?;

        let mut i = 0;
        let mut last_chunk: Option<(&Chunk<D>, Vec<u8>)> = None;

        for (c, current, start, end) in groups {
            let cache = match (last_chunk.as_mut(), c) {
                (Some((last, cache)), c) if c.addr == last.addr => {
                    cache // still on same
                },
                _ => {
                    // Read new chunk
                    let cache = read_chunk(
                        &mut fd,
                        c.addr.get(),
                        c.size.get(),
                        self.chunk_sz,
                        dsz,
                        self.ds.gzip.is_some(),
                        self.ds.shuffle,
                        false,
                    )?;
                    i += 1;

                    last_chunk = Some((c, cache));
                    &last_chunk.as_mut().unwrap().1
                }
            };

            let start = (start * dsz) as usize;
            let end = (end * dsz) as usize;
            let current = (current * dsz) as usize;

            debug_assert!(start <= cache.len());
            debug_assert!(end <= cache.len());

            let sz = end - start;

            // TODO: Make sure `dst` and `cache` are aligned: copying could be SIMD-ifyed.

            dst[current..(current + sz)].copy_from_slice(&cache[start..end]);
        }

        println!("chunks read: {}", i);

        Ok(vsz as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idx::DatasetD;
    use crate::prelude::*;

    #[test]
    fn read_coads_sst() {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let ds = if let DatasetD::D3(ds) = i.dataset("SST").unwrap() {
            ds
        } else {
            panic!()
        };
        let mut r = Direct::with_dataset(ds, i.path().unwrap()).unwrap();

        let vs = r.values::<f32>(None, None).unwrap();

        let h = hdf5::File::open(i.path().unwrap()).unwrap();
        let hvs = h.dataset("SST").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }
}
