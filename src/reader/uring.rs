use std::path::Path;

// use super::{chunk::read_chunk, dataset::Reader};
use super::{chunk::read_chunk, dataset::Reader};
use crate::filters::byteorder::Order;
use crate::idx::Dataset;

pub struct UringReader<'a, const D: usize> {
    ds: &'a Dataset<'a, D>,
    chunk_sz: u64,
}

impl<'a, const D: usize> UringReader<'a, D> {
    pub fn with_dataset<P: AsRef<Path>>(
        ds: &'a Dataset<D>,
        path: P,
    ) -> Result<UringReader<'a, D>, anyhow::Error> {
        let chunk_sz = ds.chunk_shape.iter().product::<u64>() * ds.dsize as u64;

        Ok(UringReader { ds, chunk_sz })
    }
}

impl<'a, const D: usize> Reader for UringReader<'a, D> {
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

        // Find chunks and calculate offset in destination vector.
        let mut chunks = self
            .ds
            .chunk_slices(indices, Some(counts))
            .scan(0u64, |offset, (c, start, end)| {
                let slice_sz = end - start;
                let current = *offset;
                *offset = *offset + slice_sz;

                Some((current, c, start, end))
            }).collect::<Vec<_>>();

        // Sort chunks by chunk order, not destination address.
        chunks.sort_unstable_by_key(|(current, c, start, end)| c.addr.get());

        // for (c, start, end) in self.ds.chunk_slices(indices, Some(counts)) {
        //     let start = (start * dsz) as usize;
        //     let end = (end * dsz) as usize;

        //     debug_assert!(start <= end);

        //     let slice_sz = end - start;

        //     if let Some(cache) = self.cache.get(&c.addr.get()) {
        //         debug_assert!(start <= cache.len());
        //         debug_assert!(end <= cache.len());
        //         dst[..slice_sz].copy_from_slice(&cache[start..end]);
        //     } else {
        //         let cache = read_chunk(
        //             &mut self.fd,
        //             c.addr.get(),
        //             c.size.get(),
        //             self.chunk_sz,
        //             dsz,
        //             self.ds.gzip.is_some(),
        //             self.ds.shuffle,
        //             false,
        //         )?;

        //         debug_assert!(start <= cache.len());
        //         debug_assert!(end <= cache.len());
        //         dst[..slice_sz].copy_from_slice(&cache[start..end]);
        //         self.cache.put(c.addr.get(), cache);
        //     }

        //     dst = &mut dst[slice_sz..];
        // }

        Ok(vsz as usize)
    }
}
