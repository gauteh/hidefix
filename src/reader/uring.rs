use std::fs::File;
use std::path::Path;
use std::slice;

use byte_slice_cast::{FromByteVec, IntoVecOf};

use crate::idx::{Chunk, Dataset};

pub struct DatasetReader<'a> {
    ds: &'a Dataset,
    fd: File,
}

impl<'a> DatasetReader<'a> {
    pub fn with_dataset<P>(ds: &'a Dataset, p: P) -> Result<DatasetReader, anyhow::Error>
    where
        P: AsRef<Path>,
    {
        let fd = File::open(p)?;
        Ok(DatasetReader { ds, fd })
    }

    pub fn read(
        &self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
    ) -> Result<Vec<u8>, anyhow::Error> {
        let counts: &[u64] = counts.unwrap_or(self.ds.shape.as_slice());

        let dsz = self.ds.dtype.size() as u64;
        let vsz = counts.iter().product::<u64>() * dsz;

        // TODO: will only work on slices up to 512 bytes.
        let slices: Vec<(&Chunk, u64, u64)> =
            self.ds.chunk_slices(indices, Some(&counts)).collect();

        let buf = vec![0_u8; vsz as usize];
        let mut buffers: Vec<&mut [u8]> = Vec::with_capacity(slices.len());
        let (ptr, len, cap) = buf.into_raw_parts();

        // set up buffers
        let mut o: usize = 0;
        for (_, start, end) in slices.iter() {
            let slice_sz = ((end - start) * dsz) as usize;
            buffers.push(unsafe { slice::from_raw_parts_mut(ptr.offset(o as isize), slice_sz) });
            o += slice_sz;
        }

        let ring = rio::new()?;

        let mut i = 0;
        while i < slices.len() {
            let mut tasks = vec![];

            for _ in 0..std::cmp::min(6, slices.len() - i) {
                let (c, start, _) = slices[i];
                // println!("reading: {:?}", slices[i]);
                let addr = c.addr + start * dsz;
                // let slice_sz = ((end - start) * dsz) as usize;

                let task = ring.read_at(&self.fd, &buffers[i], addr);
                tasks.push(task);
                i += 1;
            }

            for task in tasks {
                task.wait()?;
            }
        }

        ring.submit_all();

        Ok(unsafe { Vec::from_raw_parts(ptr, len, cap) })
    }

    pub fn values<T>(
        &self,
        indices: Option<&[u64]>,
        counts: Option<&[u64]>,
    ) -> Result<Vec<T>, anyhow::Error>
    where
        T: FromByteVec,
    {
        // TODO: BE, LE conversion
        // TODO: use as_slice_of() to avoid copy, or possible values_to(&mut buf) so that
        //       caller keeps ownership of slice too.
        Ok(self.read(indices, counts)?.into_vec_of::<T>()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idx::Index;

    #[test]
    fn read_t_float64() {
        let i = Index::index("tests/data/t_float.h5").unwrap();
        let r = DatasetReader::with_dataset(i.dataset("d64_2").unwrap(), i.path()).unwrap();

        let vs = r.values::<f64>(None, None).unwrap();
        println!("vales: {:?}", vs);

        let h = hdf5::File::open(i.path()).unwrap();
        let hvs = h.dataset("d64_2").unwrap().read_raw::<f64>().unwrap();

        assert_eq!(vs, hvs);
    }

    #[test]
    fn read_chunked_1d() {
        let i = Index::index("tests/data/chunked_oneD.h5").unwrap();
        let r = DatasetReader::with_dataset(i.dataset("d_4_chunks").unwrap(), i.path()).unwrap();

        let vs = r.values::<f32>(None, None).unwrap();

        let h = hdf5::File::open(i.path()).unwrap();
        let hvs = h.dataset("d_4_chunks").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }

    #[test]
    fn read_chunked_2d() {
        let i = Index::index("tests/data/chunked_twoD.h5").unwrap();
        let r = DatasetReader::with_dataset(i.dataset("d_4_chunks").unwrap(), i.path()).unwrap();

        let vs = r.values::<f32>(None, None).unwrap();

        let h = hdf5::File::open(i.path()).unwrap();
        let hvs = h.dataset("d_4_chunks").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }
}
