use async_stream::stream;
use futures::{Stream, StreamExt};
use std::convert::TryInto;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::pin::Pin;

use bytes::Bytes;
use lru::LruCache;

use super::Streamer;
use crate::filters;
use crate::filters::byteorder::{self, Order};
use crate::idx::Dataset;

/// The stream reader is intended to be used in network applications. The cache is currently local
/// to each `stream` call.
pub struct StreamReader<'a, const D: usize> {
    ds: &'a Dataset<'a, D>,
    p: PathBuf,
    chunk_sz: u64,
}

/// The maximum cache size in bytes. Will not be lower than the size of one chunk.
const CACHE_SZ: u64 = 32 * 1024 * 1024;

impl<'a, const D: usize> StreamReader<'a, D> {
    pub fn with_dataset<P>(ds: &'a Dataset<D>, p: P) -> Result<StreamReader<'a, D>, anyhow::Error>
    where
        P: AsRef<Path>,
    {
        let chunk_sz = ds.chunk_shape.iter().product::<u64>() * ds.dsize as u64;

        Ok(StreamReader {
            ds,
            p: p.as_ref().into(),
            chunk_sz,
        })
    }
}

impl<'a, const D: usize> Streamer for StreamReader<'a, D> {
    fn dsize(&self) -> usize {
        self.ds.dsize
    }

    /// A stream of bytes from the variable. Always in Big Endian.
    fn stream(
        &self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
    ) -> Pin<Box<dyn Stream<Item = Result<Bytes, anyhow::Error>> + Send + 'static>> {
        let dsz = self.ds.dsize as u64;

        let indices: Option<&[u64; D]> = indices
            .map(|i| i.try_into())
            .map_or(Ok(None), |v| v.map(Some))
            .unwrap();
        let counts: Option<&[u64; D]> = counts
            .map(|i| i.try_into())
            .map_or(Ok(None), |v| v.map(Some))
            .unwrap();

        let slices = self
            .ds
            .chunk_slices(indices, counts)
            .map(|(c, a, b)| (c.addr.get(), c.size.get(), a * dsz, b * dsz))
            .collect::<Vec<_>>();

        let p = self.p.clone();
        let shuffle = self.ds.shuffle;
        let gzip = self.ds.gzip;
        let chunk_sz = self.chunk_sz;
        let order = self.order();
        let cache_sz = std::cmp::max(CACHE_SZ / chunk_sz, 1);

        (stream! {
            let mut fd = File::open(p)?;
            let mut ds_cache = LruCache::<u64, Bytes>::new(cache_sz as usize);

            for (addr, sz, start, end) in slices {
                let start = start as usize;
                let end = end as usize;
                debug_assert!(start <= end);

                if let Some(cache) = ds_cache.get(&addr) {
                    debug_assert!(start <= cache.len());
                    debug_assert!(end <= cache.len());
                    yield Ok((cache.slice(start..end)));
                } else {
                    let mut cache: Vec<u8> = Vec::with_capacity(sz as usize);
                    unsafe {
                        cache.set_len(sz as usize);
                    }

                    fd.seek(SeekFrom::Start(addr))?;
                    fd.read_exact(&mut cache)?;

                    // TODO: Keep buffers around to avoid allocations.

                    let cache = if let Some(_) = gzip {
                        let mut decache: Vec<u8> = Vec::with_capacity(chunk_sz as usize);
                        unsafe {
                            decache.set_len(chunk_sz as usize);
                        }

                        tokio::task::block_in_place(||
                            filters::gzip::decompress(&cache, &mut decache))?;

                        decache
                    } else {
                        cache
                    };

                    let mut cache = if shuffle && dsz > 1 {
                        filters::shuffle::unshuffle_sized(&cache, dsz as usize)
                    } else {
                        cache
                    };

                    // Always output big endian. This code was written for network code that need
                    // to transmit the data network-endian/big-endian in XDR format. However, this
                    // makes the stream inefficient for code that need to stream the values on LE-machines.
                    byteorder::to_big_e_sized(&mut cache, order, dsz as usize)?;

                    let cache = Bytes::from(cache);
                    ds_cache.put(addr, cache.clone());

                    debug_assert!(start <= cache.len());
                    debug_assert!(end <= cache.len());
                    yield Ok(cache.slice(start..end));
                }
            }
        })
        .boxed()
    }

    fn order(&self) -> Order {
        self.ds.order
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use futures::executor::block_on_stream;

    #[test]
    fn read_t_float32() {
        let i = Index::index("tests/data/dmrpp/t_float.h5").unwrap();
        let r = i.streamer("d32_1").unwrap();

        let v = r.stream_values::<f32>(None, None);
        let vs: Vec<f32> = block_on_stream(v).flatten().flatten().collect();

        let h = hdf5::File::open(i.path().unwrap()).unwrap();
        let hvs = h.dataset("d32_1").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }

    #[test]
    fn read_chunked_1d() {
        let i = Index::index("tests/data/dmrpp/chunked_oneD.h5").unwrap();
        let r = i.streamer("d_4_chunks").unwrap();

        let v = r.stream_values::<f32>(None, None);
        let vs: Vec<f32> = block_on_stream(v).flatten().flatten().collect();

        let h = hdf5::File::open(i.path().unwrap()).unwrap();
        let hvs = h.dataset("d_4_chunks").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }

    #[test]
    fn read_chunked_2d() {
        let i = Index::index("tests/data/dmrpp/chunked_twoD.h5").unwrap();
        let r = i.streamer("d_4_chunks").unwrap();

        let v = r.stream_values::<f32>(None, None);
        let vs: Vec<f32> = block_on_stream(v).flatten().flatten().collect();

        let h = hdf5::File::open(i.path().unwrap()).unwrap();
        let hvs = h.dataset("d_4_chunks").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }
}
