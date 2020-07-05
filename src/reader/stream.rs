use async_stream::stream;
use futures::{Stream, StreamExt};
use futures_util::pin_mut;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use byte_slice_cast::{FromByteVec, IntoVecOf};
use bytes::Bytes;
use lru::LruCache;

use crate::filters;
use crate::filters::byteorder::{self, Order, ToNative};
use crate::idx::Dataset;

/// The stream reader is intended to be used in network applications. The cache is currently local
/// to each `stream` call.
pub struct DatasetReader<'a> {
    ds: &'a Dataset,
    p: PathBuf,
    chunk_sz: u64,
}

/// The maximum cache size in bytes. Will not be lower than the size of one chunk.
const CACHE_SZ: u64 = 32 * 1024 * 1024;

impl<'a> DatasetReader<'a> {
    pub fn with_dataset<P>(ds: &'a Dataset, p: P) -> Result<DatasetReader<'a>, anyhow::Error>
    where
        P: AsRef<Path>,
    {
        let chunk_sz = ds.chunk_shape.iter().product::<u64>() * ds.dsize as u64;

        Ok(DatasetReader {
            ds,
            p: p.as_ref().into(),
            chunk_sz,
        })
    }

    /// A stream of bytes from the variable. Always in Big Endian.
    pub fn stream(
        &self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
    ) -> impl Stream<Item = Result<Bytes, anyhow::Error>> {
        let dsz = self.ds.dsize as u64;
        let counts: &[u64] = counts.unwrap_or_else(|| self.ds.shape.as_slice());
        let slices = self
            .ds
            .chunk_slices(indices, Some(&counts))
            .map(|(c, a, b)| (c.addr, c.size, a * dsz, b * dsz))
            .collect::<Vec<_>>();

        let p = self.p.clone();
        let shuffle = self.ds.shuffle;
        let gzip = self.ds.gzip;
        let chunk_sz = self.chunk_sz;
        let order = self.order();
        let cache_sz = std::cmp::max(CACHE_SZ / chunk_sz, 1);

        stream! {
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

                    let cache = if let Some(_) = gzip {
                        let mut decache: Vec<u8> = Vec::with_capacity(chunk_sz as usize);
                        unsafe {
                            decache.set_len(chunk_sz as usize);
                        }

                        let mut dz = flate2::read::ZlibDecoder::new(&cache[..]);
                        dz.read_exact(&mut decache)?;

                        decache
                    } else {
                        cache
                    };

                    let mut cache = if shuffle && dsz > 1 {
                        filters::shuffle::unshuffle_sized(&cache, dsz as usize)
                    } else {
                        cache
                    };

                    byteorder::to_big_e_sized(&mut cache, order, dsz as usize)?;
                    let cache = Bytes::from(cache);

                    ds_cache.put(addr, cache.clone());

                    debug_assert!(start <= cache.len());
                    debug_assert!(end <= cache.len());
                    yield Ok(cache.slice(start..end));
                }
            }
        }
    }

    pub fn order(&self) -> Order {
        self.ds.order
    }

    pub fn stream_values<T>(
        &self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
    ) -> impl Stream<Item = Result<Vec<T>, anyhow::Error>>
    where
        T: FromByteVec + Unpin,
        [T]: ToNative,
    {
        // TODO: use as_slice_of() to avoid copy, or possible values_to(&mut buf) so that
        //       caller keeps ownership of slice too.
        let reader = self.stream(indices, counts);
        let order = Order::BE;

        // XXX: This got a lot slower after always outputing big endian from stream(). Can probably
        // fix that or make customizable. But need Big E for dars.

        stream! {
            pin_mut!(reader);
            while let Some(Ok(b)) = reader.next().await {
                let mut values = b.to_vec().into_vec_of::<T>()
                    .map_err(|_| anyhow!("could not cast to value"))?;

                values.to_native(order);
                yield Ok(values);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idx::Index;
    use futures::executor::block_on_stream;

    #[test]
    fn read_t_float32() {
        let i = Index::index("tests/data/dmrpp/t_float.h5").unwrap();
        let r =
            DatasetReader::with_dataset(i.dataset("d32_1").unwrap(), i.path().unwrap()).unwrap();

        let v = r.stream_values::<f32>(None, None);
        pin_mut!(v);
        let vs: Vec<f32> = block_on_stream(v).flatten().flatten().collect();

        let h = hdf5::File::open(i.path().unwrap()).unwrap();
        let hvs = h.dataset("d32_1").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }

    #[test]
    fn read_chunked_1d() {
        let i = Index::index("tests/data/dmrpp/chunked_oneD.h5").unwrap();
        let r = DatasetReader::with_dataset(i.dataset("d_4_chunks").unwrap(), i.path().unwrap())
            .unwrap();

        let v = r.stream_values::<f32>(None, None);
        pin_mut!(v);
        let vs: Vec<f32> = block_on_stream(v).flatten().flatten().collect();

        let h = hdf5::File::open(i.path().unwrap()).unwrap();
        let hvs = h.dataset("d_4_chunks").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }

    #[test]
    fn read_chunked_2d() {
        let i = Index::index("tests/data/dmrpp/chunked_twoD.h5").unwrap();
        let r = DatasetReader::with_dataset(i.dataset("d_4_chunks").unwrap(), i.path().unwrap())
            .unwrap();

        let v = r.stream_values::<f32>(None, None);
        pin_mut!(v);
        let vs: Vec<f32> = block_on_stream(v).flatten().flatten().collect();

        let h = hdf5::File::open(i.path().unwrap()).unwrap();
        let hvs = h.dataset("d_4_chunks").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }
}
