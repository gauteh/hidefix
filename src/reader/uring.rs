use std::io::{Read, Seek};

// use super::{chunk::read_chunk, dataset::Reader};
use super::{chunk::read_chunk, dataset::Reader};
use crate::filters::byteorder::Order;
use crate::idx::{Chunk, Dataset};

pub struct UringReader<'a, R: Read + Seek, const D: usize> {
    ds: &'a Dataset<'a, D>,
    fd: R,
    chunk_sz: u64,
}

impl<'a, R: Read + Seek, const D: usize> UringReader<'a, R, D> {
    pub fn with_dataset(
        ds: &'a Dataset<D>,
        fd: R,
    ) -> Result<UringReader<'a, R, D>, anyhow::Error> {
        let chunk_sz = ds.chunk_shape.iter().product::<u64>() * ds.dsize as u64;

        Ok(UringReader { ds, fd, chunk_sz })
    }
}

impl<'a, R: Read + Seek, const D: usize> Reader for UringReader<'a, R, D> {
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

        // Sort chunk file address, not destination address.
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

        for (c, segments) in groups {
            // Read chunk
            let cache = read_chunk(
                &mut self.fd,
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
    use crate::prelude::*;
    use super::*;
    use std::fs;
    use crate::idx::DatasetD;

    #[test]
    fn read_coads_sst() {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let ds = if let DatasetD::D3(ds) = i.dataset("SST").unwrap() {
            ds
        } else {
            panic!()
        };
        let mut r = UringReader::with_dataset(ds, fs::File::open(i.path().unwrap()).unwrap()).unwrap();

        let vs = r.values::<f32>(None, None).unwrap();

        let h = hdf5::File::open(i.path().unwrap()).unwrap();
        let hvs = h.dataset("SST").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }
}
