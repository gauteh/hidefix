use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use byte_slice_cast::{FromByteVec, IntoVecOf};
use hdf5::ByteOrder;
use lru::LruCache;

use crate::filters;
use crate::filters::byteorder::{Order, ToNative};
use crate::idx::Dataset;

pub struct DatasetReader<'a> {
    ds: &'a Dataset,
    fd: File,
    cache: LruCache<u64, Vec<u8>>,
    chunk_sz: u64,
}

impl<'a> DatasetReader<'a> {
    pub fn with_dataset<P>(ds: &'a Dataset, p: P) -> Result<DatasetReader, anyhow::Error>
    where
        P: AsRef<Path>,
    {
        let fd = File::open(p)?;

        const CACHE_SZ: u64 = 32 * 1024 * 1024;
        let chunk_sz = ds.chunk_shape.iter().product::<u64>() * ds.dsize as u64;
        let cache_sz = CACHE_SZ / chunk_sz;

        Ok(DatasetReader {
            ds,
            fd,
            cache: LruCache::new(cache_sz as usize),
            chunk_sz,
        })
    }

    pub fn read(
        &mut self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
    ) -> Result<Vec<u8>, anyhow::Error> {
        let counts: &[u64] = counts.unwrap_or_else(|| self.ds.shape.as_slice());

        let dsz = self.ds.dsize as u64;
        let vsz = counts.iter().product::<u64>() * dsz;
        let mut buf = Vec::with_capacity(vsz as usize);
        unsafe {
            buf.set_len(vsz as usize);
        }
        let mut buf_slice = &mut buf[..];

        for (c, start, end) in self.ds.chunk_slices(indices, Some(&counts)) {
            let start = (start * dsz) as usize;
            let end = (end * dsz) as usize;
            let slice_sz = end - start;

            if let Some(cache) = self.cache.get(&c.addr) {
                buf_slice[..slice_sz].copy_from_slice(&cache[start..end]);
            } else {
                let mut cache: Vec<u8> = Vec::with_capacity(c.size as usize);
                unsafe {
                    cache.set_len(c.size as usize);
                }

                self.fd.seek(SeekFrom::Start(c.addr))?;
                self.fd.read_exact(&mut cache)?;

                // we assume decompression comes before unshuffling
                let cache = if let Some(_) = self.ds.gzip {
                    let mut decache: Vec<u8> = Vec::with_capacity(self.chunk_sz as usize);
                    unsafe {
                        decache.set_len(self.chunk_sz as usize);
                    }

                    let mut dz = flate2::read::ZlibDecoder::new(&cache[..]);
                    dz.read_exact(&mut decache)?;

                    decache
                } else {
                    cache
                };

                let cache = if self.ds.shuffle {
                    filters::shuffle::unshuffle_sized(&cache, dsz as usize)
                } else {
                    cache
                };

                buf_slice[..slice_sz].copy_from_slice(&cache[start..end]);
                self.cache.put(c.addr, cache);
            }

            buf_slice = &mut buf_slice[slice_sz..];
        }

        Ok(buf)
    }

    pub fn values<T>(
        &mut self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
    ) -> Result<Vec<T>, anyhow::Error>
    where
        T: FromByteVec,
        [T]: ToNative,
    {
        // TODO: use as_slice_of() to avoid copy, or possible values_to(&mut buf) so that
        //       caller keeps ownership of slice too.

        let mut values = self.read(indices, counts)?.into_vec_of::<T>()?;

        let order: Order = match self.ds.order {
            ByteOrder::BigEndian => Order::BE,
            ByteOrder::LittleEndian => Order::LE,
            _ => unimplemented!(),
        };

        values.to_native(order);

        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idx::Index;

    #[test]
    fn read_t_float32() {
        let i = Index::index("tests/data/dmrpp/t_float.h5").unwrap();
        let mut r = DatasetReader::with_dataset(i.dataset("d32_1").unwrap(), i.path()).unwrap();

        let vs = r.values::<f32>(None, None).unwrap();

        let h = hdf5::File::open(i.path()).unwrap();
        let hvs = h.dataset("d32_1").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }

    #[test]
    fn read_chunked_1d() {
        let i = Index::index("tests/data/dmrpp/chunked_oneD.h5").unwrap();
        let mut r =
            DatasetReader::with_dataset(i.dataset("d_4_chunks").unwrap(), i.path()).unwrap();

        let vs = r.values::<f32>(None, None).unwrap();

        let h = hdf5::File::open(i.path()).unwrap();
        let hvs = h.dataset("d_4_chunks").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }

    #[test]
    fn read_chunked_2d() {
        let i = Index::index("tests/data/dmrpp/chunked_twoD.h5").unwrap();
        let mut r =
            DatasetReader::with_dataset(i.dataset("d_4_chunks").unwrap(), i.path()).unwrap();

        let vs = r.values::<f32>(None, None).unwrap();

        let h = hdf5::File::open(i.path()).unwrap();
        let hvs = h.dataset("d_4_chunks").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }

    #[test]
    fn read_chunked_shuffled_2d() {
        let i = Index::index("tests/data/dmrpp/chunked_shuffled_twoD.h5").unwrap();
        let mut r =
            DatasetReader::with_dataset(i.dataset("d_4_shuffled_chunks").unwrap(), i.path())
                .unwrap();

        let vs = r.values::<f32>(None, None).unwrap();

        let h = hdf5::File::open(i.path()).unwrap();
        let hvs = h
            .dataset("d_4_shuffled_chunks")
            .unwrap()
            .read_raw::<f32>()
            .unwrap();

        assert_eq!(vs, hvs);
    }

    #[test]
    fn read_chunked_gzipped_2d() {
        let i = Index::index("tests/data/dmrpp/chunked_gzipped_twoD.h5").unwrap();
        let mut r = DatasetReader::with_dataset(i.dataset("d_4_gzipped_chunks").unwrap(), i.path())
            .unwrap();

        let vs = r.values::<f32>(None, None).unwrap();

        // println!("{:?}", vs);

        // hdf5 having issues loading zlib
        let h = hdf5::File::open(i.path()).unwrap();
        let hvs = h
            .dataset("d_4_gzipped_chunks")
            .unwrap()
            .read_raw::<f32>()
            .unwrap();

        assert_eq!(vs, hvs);
    }
}
