use crate::filters::byteorder::{Order, ToNative};
use byte_slice_cast::{AsMutByteSlice, ToMutByteSlice};
use std::path::{Path, PathBuf};

use super::{
    chunk::{decode_chunk, read_chunk, read_chunk_to},
    dataset::Reader,
};
use crate::idx::{Chunk, Dataset};

pub struct UringReader<'a, const D: usize> {
    ds: &'a Dataset<'a, D>,
    path: PathBuf,
    chunk_sz: u64,
}

impl<'a, const D: usize> UringReader<'a, D> {
    pub fn with_dataset<P: AsRef<Path>>(
        ds: &'a Dataset<D>,
        path: P,
    ) -> Result<UringReader<'a, D>, anyhow::Error> {
        let chunk_sz = ds.chunk_shape.iter().product::<u64>() * ds.dsize as u64;

        Ok(UringReader {
            ds,
            path: path.as_ref().into(),
            chunk_sz,
        })
    }

    fn group_chunks(
        &self,
        indices: Option<&[u64; D]>,
        counts: &[u64; D],
    ) -> Vec<(&'a Chunk<D>, Vec<(u64, u64, u64)>)> {
        // Find chunks and calculate offset in destination vector.
        let mut chunks = self
            .ds
            .chunk_slices(indices, Some(counts))
            .scan(0u64, |offset, (c, start, end)| {
                let slice_sz = end - start;
                let current = *offset;
                *offset = *offset + slice_sz;

                Some((current, c, start, end))
            })
            .collect::<Vec<_>>();

        // Sort by chunk file address, not destination address.
        chunks.sort_unstable_by_key(|(_current, c, _start, _end)| c.addr.get());

        // Group by chunk
        let mut groups = Vec::<(&Chunk<D>, Vec<(u64, u64, u64)>)>::new();

        for (current, c, start, end) in chunks.iter() {
            match groups.last_mut() {
                Some((group_chunk, segments)) if *group_chunk == *c => {
                    segments.push((*current, *start, *end));
                }
                _ => {
                    groups.push((c, vec![(*current, *start, *end)]));
                }
            }
        }

        groups
    }

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

        let groups = self.group_chunks(indices, counts);

        let mut fd = std::fs::File::open(&self.path)?;

        // Read all chunks sequentially: maybe this is too big?
        let chunks = groups
            .iter()
            .map(|(c, segments)| {
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
        for (c, segments, chunk) in chunks {
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

    /// Must be called from within a `tokio_uring` runtime.
    pub async fn read_to_uring(
        &self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
        dst: &mut [u8],
    ) -> Result<u64, anyhow::Error> {
        use futures::stream::FuturesOrdered;
        use futures::{FutureExt, StreamExt};
        use std::assert_matches::assert_matches;
        use tokio_uring::fs::File;

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

        let groups = self.group_chunks(indices, counts);

        let fd = File::open(&self.path).await?;

        let chunks = groups
            .iter()
            .map(|(c, _)| {
                let buf = vec![0u8; c.size.get() as usize];
                let addr = c.addr.get();
                fd.read_at(buf, addr).map(|(res, v)| {
                    assert_matches!(res, Ok(i) if i == c.size.get() as usize);
                    // CPU intensive: should maybe be done in rayon pool?
                    decode_chunk(
                        v,
                        self.chunk_sz,
                        dsz,
                        self.ds.gzip.is_some(),
                        self.ds.shuffle,
                    )
                })
            })
            .collect::<FuturesOrdered<_>>()
            .collect::<Vec<_>>()
            .await;

        // All decoded chunks are now in `chunks`. Now copy slices to destination
        // vector.

        for (chunk, (_, segments)) in chunks.into_iter().zip(groups) {
            let chunk = chunk?;
            debug_assert_eq!(chunk.len(), self.chunk_sz as usize);
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

    pub async fn values_uring<T>(
        &self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
    ) -> Result<Vec<T>, anyhow::Error>
    where
        T: ToMutByteSlice,
        [T]: ToNative,
    {
        let dsz = self.dsize();
        ensure!(
            dsz % std::mem::size_of::<T>() == 0,
            "size of datatype ({}) not multiple of target {}",
            dsz,
            std::mem::size_of::<T>()
        );

        if dsz != std::mem::size_of::<T>() {
            error!("size of datatype ({}) not same as target {}, alignment may not match and result in unsoundness", dsz, std::mem::size_of::<T>());
        }

        let vsz = counts
            .unwrap_or_else(|| self.shape())
            .iter()
            .product::<u64>() as usize
            * dsz
            / std::mem::size_of::<T>();
        let mut values = Vec::<T>::with_capacity(vsz);
        unsafe {
            values.set_len(vsz);
        }

        {
            let dst = values.as_mut_byte_slice();
            self.read_to_uring(indices, counts, dst).await?;
            dst.to_native(self.order());
        }

        Ok(values)
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

        let groups = self.group_chunks(indices, counts);

        let mut fd = std::fs::File::open(&self.path)?;

        for (c, segments) in groups {
            // Read chunk
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

            for (current, start, end) in segments {
                let start = (start * dsz) as usize;
                let end = (end * dsz) as usize;
                let current = (current * dsz) as usize;

                debug_assert!(start <= cache.len());
                debug_assert!(end <= cache.len());

                let sz = end - start;
                dst[current..(current + sz)].copy_from_slice(&cache[start..end]);
            }
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
        let mut r = UringReader::with_dataset(ds, i.path().unwrap()).unwrap();

        let vs = r.values::<f32>(None, None).unwrap();

        let h = hdf5::File::open(i.path().unwrap()).unwrap();
        let hvs = h.dataset("SST").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }

    #[test]
    fn read_coads_sst_uring() {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let ds = if let DatasetD::D3(ds) = i.dataset("SST").unwrap() {
            ds
        } else {
            panic!()
        };
        let r = UringReader::with_dataset(ds, i.path().unwrap()).unwrap();

        let vs = tokio_uring::start(async { r.values_uring::<f32>(None, None).await.unwrap() });

        let h = hdf5::File::open(i.path().unwrap()).unwrap();
        let hvs = h.dataset("SST").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }
}
