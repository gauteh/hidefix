use std::convert::TryInto;
use std::io::{Read, Seek, SeekFrom};

use byte_slice_cast::{FromByteVec, IntoVecOf};
use lru::LruCache;
use strength_reduce::StrengthReducedU64;

use super::dataset::Reader;
use crate::filters;
use crate::filters::byteorder::ToNative;
use crate::idx::Dataset;

pub struct CacheReader<'a, R: Read + Seek, const D: usize>
where
    [u64; D]: std::array::LengthAtMost32,
    [StrengthReducedU64; D]: std::array::LengthAtMost32,
{
    ds: &'a Dataset<'a, D>,
    fd: R,
    cache: LruCache<u64, Vec<u8>>,
    chunk_sz: u64,
}

impl<'a, R: Read + Seek, const D: usize> CacheReader<'a, R, D>
where
    [u64; D]: std::array::LengthAtMost32,
    [StrengthReducedU64; D]: std::array::LengthAtMost32,
{
    pub fn with_dataset(ds: &'a Dataset<D>, fd: R) -> Result<CacheReader<'a, R, D>, anyhow::Error> {
        const CACHE_SZ: u64 = 32 * 1024 * 1024;
        let chunk_sz = ds.chunk_shape.iter().product::<u64>() * ds.dsize as u64;
        let cache_sz = std::cmp::max(CACHE_SZ / chunk_sz, 1);

        Ok(CacheReader {
            ds,
            fd,
            cache: LruCache::new(cache_sz as usize),
            chunk_sz,
        })
    }
}

impl<'a, R: Read + Seek, const D: usize> Reader for CacheReader<'a, R, D>
where
    [u64; D]: std::array::LengthAtMost32,
    [StrengthReducedU64; D]: std::array::LengthAtMost32,
{
    fn read(
        &mut self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
    ) -> Result<Vec<u8>, anyhow::Error> {
        let indices: Option<&[u64; D]> = indices
            .map(|i| i.try_into())
            .map_or(Ok(None), |v| v.map(Some))
            .unwrap();

        let counts: Option<&[u64; D]> = counts
            .map(|i| i.try_into())
            .map_or(Ok(None), |v| v.map(Some))
            .unwrap();
        let counts: &[u64; D] = counts.unwrap_or_else(|| &self.ds.shape);

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

                // Decompression comes before unshuffling
                let cache = if let Some(_) = self.ds.gzip {
                    let mut decache: Vec<u8> = Vec::with_capacity(self.chunk_sz as usize);
                    unsafe {
                        decache.set_len(self.chunk_sz as usize);
                    }

                    filters::gzip::decompress(&cache, &mut decache)?;

                    decache
                } else {
                    cache
                };

                // TODO: Keep buffers around to avoid allocations.
                // TODO: Write directly to buf_slice when on last filter.
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

    fn values<T>(
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
        values.to_native(self.ds.order);

        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use crate::idx::Index;
    use crate::reader::Reader;

    #[test]
    fn read_coads_sst() {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let mut r = i.reader("SST").unwrap();

        let vs = r.values::<f32>(None, None).unwrap();

        let h = hdf5::File::open(i.path().unwrap()).unwrap();
        let hvs = h.dataset("SST").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }

    #[test]
    fn read_t_float32() {
        let i = Index::index("tests/data/dmrpp/t_float.h5").unwrap();
        let mut r = i.reader("d32_1").unwrap();

        let vs = r.values::<f32>(None, None).unwrap();

        let h = hdf5::File::open(i.path().unwrap()).unwrap();
        let hvs = h.dataset("d32_1").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }

    #[test]
    fn read_chunked_1d() {
        let i = Index::index("tests/data/dmrpp/chunked_oneD.h5").unwrap();
        let mut r = i.reader("d_4_chunks").unwrap();

        let vs = r.values::<f32>(None, None).unwrap();

        let h = hdf5::File::open(i.path().unwrap()).unwrap();
        let hvs = h.dataset("d_4_chunks").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }

    #[test]
    fn read_chunked_2d() {
        let i = Index::index("tests/data/dmrpp/chunked_twoD.h5").unwrap();
        let mut r = i.reader("d_4_chunks").unwrap();

        let vs = r.values::<f32>(None, None).unwrap();

        let h = hdf5::File::open(i.path().unwrap()).unwrap();
        let hvs = h.dataset("d_4_chunks").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }

    #[test]
    fn read_chunked_shuffled_2d() {
        let i = Index::index("tests/data/dmrpp/chunked_shuffled_twoD.h5").unwrap();
        let mut r = i.reader("d_4_shuffled_chunks").unwrap();

        let vs = r.values::<f32>(None, None).unwrap();

        let h = hdf5::File::open(i.path().unwrap()).unwrap();
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
        let mut r = i.reader("d_4_gzipped_chunks").unwrap();

        let vs = r.values::<f32>(None, None).unwrap();

        // println!("{:?}", vs);

        // hdf5 having issues loading zlib
        let h = hdf5::File::open(i.path().unwrap()).unwrap();
        let hvs = h
            .dataset("d_4_gzipped_chunks")
            .unwrap()
            .read_raw::<f32>()
            .unwrap();

        assert_eq!(vs, hvs);
    }

    #[ignore]
    #[test]
    fn read_meps() {
        println!("meps");
        let i = Index::index("tests/data/meps_det_vc_2_5km_latest.nc").unwrap();
        println!("idx index: done");
        let mut r = i.reader("x_wind_ml").unwrap();

        // println!("ds size: {}", r.ds.size());
        // println!("cshape: {:?}", r.ds.chunk_shape);

        let vs = r
            .values::<i32>(Some(&[0, 0, 0, 0]), Some(&[2, 2, 1, 5]))
            .unwrap();
        println!("idx read: done: {}", vs.len());

        let h = hdf5::File::open("tests/data/meps_det_vc_2_5km_latest.nc").unwrap();
        let hvs = h.dataset("x_wind_ml").unwrap().read_dyn::<i32>().unwrap();
        println!("native: {}", hvs.len());
        println!("native: done");

        use ndarray::s;

        assert_eq!(
            vs,
            hvs.slice(s![0..2, 0..2, 0..1, 0..5])
                .iter()
                .map(|v| *v)
                .collect::<Vec<i32>>()
        );
    }
}
