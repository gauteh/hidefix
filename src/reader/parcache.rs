use rayon::prelude::*;
use std::convert::TryInto;
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use lru::LruCache;

use super::{chunk::read_chunk, dataset::Reader};
use crate::filters::byteorder::Order;
use crate::idx::Dataset;

pub struct ParCache<'a, const D: usize> {
    ds: &'a Dataset<'a, D>,
    path: PathBuf,
    cache: Arc<Mutex<LruCache<u64, Vec<u8>>>>,
    chunk_sz: u64,
}

impl<'a, const D: usize> ParCache<'a, D> {
    pub fn with_dataset<P: AsRef<Path>>(
        ds: &'a Dataset<D>,
        path: P,
    ) -> Result<ParCache<'a, D>, anyhow::Error> {
        const CACHE_SZ: u64 = 32 * 1024 * 1024;
        let chunk_sz = ds.chunk_shape.iter().product::<u64>() * ds.dsize as u64;
        let cache_sz = std::cmp::max(CACHE_SZ / chunk_sz, 1);
        let path = path.as_ref().into();

        Ok(Self {
            ds,
            path,
            cache: Arc::new(Mutex::new(LruCache::new(cache_sz as usize))),
            chunk_sz,
        })
    }
}

impl<'a, const D: usize> Reader for ParCache<'a, D> {
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
        mut dst: &mut [u8],
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

        // Collect chunks into Vec so that it can be iterated over in parallel.
        let chunks = self
            .ds
            .chunk_slices(indices, Some(counts))
            .collect::<Vec<_>>();

        chunks.par_iter().map(|(c, start, end)| {
            let start = (start * dsz) as usize;
            let end = (end * dsz) as usize;

            debug_assert!(start <= end);
            let slice_sz = end - start;

            let cache = self.cache.read().unwrap().get(&c.addr.get());



        });

        for (c, start, end) in self.ds.chunk_slices(indices, Some(counts)) {
            let start = (start * dsz) as usize;
            let end = (end * dsz) as usize;

            debug_assert!(start <= end);

            let slice_sz = end - start;

            // if let Some(cache) = self.cache.get(&c.addr.get()) {
            //     debug_assert!(start <= cache.len());
            //     debug_assert!(end <= cache.len());
            //     dst[..slice_sz].copy_from_slice(&cache[start..end]);
            // } else {
            //     let cache = read_chunk(
            //         &mut self.fd,
            //         c.addr.get(),
            //         c.size.get(),
            //         self.chunk_sz,
            //         dsz,
            //         self.ds.gzip.is_some(),
            //         self.ds.shuffle,
            //         false,
            //     )?;

            //     debug_assert!(start <= cache.len());
            //     debug_assert!(end <= cache.len());
            //     dst[..slice_sz].copy_from_slice(&cache[start..end]);
            //     self.cache.put(c.addr.get(), cache);
            // }

            dst = &mut dst[slice_sz..];
        }

        Ok(vsz as usize)
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
}
