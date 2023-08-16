use crate::filters::byteorder::Order;
use std::fs::File;
use std::path::{Path, PathBuf};

use super::{
    chunk::{decode_chunk, read_chunk, read_chunk_to},
    dataset::{ParReader, Reader},
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
}

impl<'a, const D: usize> ParReader for Direct<'a, D> {
    fn read_to_par(
        &self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
        dst: &mut [u8],
    ) -> Result<usize, anyhow::Error> {
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
        let groups = groups.group_by(|a, b| a.0.addr == b.0.addr);
        let groups = groups.collect::<Vec<_>>();

        groups.par_iter().try_for_each_init(
            || File::open(&self.path),
            |fd, group| {
                let mut fd = fd
                    .as_mut()
                    .map_err(|_| anyhow::anyhow!("Could not open file."))?;
                let c = group[0].0;

                let mut chunk: Vec<u8> = vec![0; c.size.get() as usize];
                read_chunk_to(&mut fd, c.addr.get(), &mut chunk)?;

                let chunk = decode_chunk(
                    chunk,
                    self.chunk_sz,
                    dsz,
                    self.ds.gzip.is_some(),
                    self.ds.shuffle,
                )?;

                for (_c, current, start, end) in *group {
                    let start = (start * dsz) as usize;
                    let end = (end * dsz) as usize;
                    let current = (current * dsz) as usize;

                    debug_assert!(start <= chunk.len());
                    debug_assert!(end <= chunk.len());

                    let sz = end - start;

                    // Safety: The sub-slices never overlap between threads and segments. But I
                    // cannot find a good way to do this in Rust at the moment. Maybe with a
                    // slice::split_at_indices method or equivalent that gives a new slice of sub-slices.
                    let dptr = dst[current..].as_ptr() as _;
                    let src = chunk[start..end].as_ptr();

                    unsafe {
                        core::ptr::copy_nonoverlapping(src, dptr, sz);
                    }
                }

                Ok::<_, anyhow::Error>(())
            },
        )?;

        Ok(vsz as usize)
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

        let mut last_chunk: Option<(&Chunk<D>, Vec<u8>)> = None;

        for (c, current, start, end) in groups {
            let cache = match (last_chunk.as_mut(), c) {
                (Some((last, cache)), c) if c.addr == last.addr => {
                    cache // still on same
                }
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
