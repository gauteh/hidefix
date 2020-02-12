use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use byte_slice_cast::{FromByteVec, IntoVecOf};

use super::idx::{Chunk, Dataset};

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
        &mut self,
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

        let ring = rio::new()?;

        // let mut tasks: Vec<_> = vec![];
        for i in 0..slices.len() {
            let (c, start, end) = slices[i];
            println!("reading: {:?}", slices[i]);
            let addr = c.addr + start * dsz;
            let slice_sz = ((end - start) * dsz) as usize;
            // self.fd.seek(SeekFrom::Start(addr));
            let task = ring.read_at(&self.fd, &buf, addr);
            task.wait()?;

            // tasks.push(task);

            // buf_slice = &mut buf_slice[slice_sz..];
        }

        // for task in tasks {
        //     task.wait()?;
        // }

        Ok(buf)
    }

    pub fn values<T>(
        &mut self,
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
        let mut r = DatasetReader::with_dataset(i.dataset("d64_2").unwrap(), i.path()).unwrap();

        let vs = r.values::<f64>(None, None).unwrap();
        println!("vales: {:?}", vs);

        let h = hdf5::File::open(i.path()).unwrap();
        let hvs = h.dataset("d64_2").unwrap().read_raw::<f64>().unwrap();

        assert_eq!(vs, hvs);
    }

    // #[test]
    // fn read_chunked_1d() {
    //     let i = Index::index("tests/data/chunked_oneD.h5").unwrap();
    //     let mut r =
    //         DatasetReader::with_dataset(i.dataset("d_4_chunks").unwrap(), i.path()).unwrap();

    //     let vs = r.values::<f32>(None, None).unwrap();

    //     let h = hdf5::File::open(i.path()).unwrap();
    //     let hvs = h.dataset("d_4_chunks").unwrap().read_raw::<f32>().unwrap();

    //     assert_eq!(vs, hvs);
    // }

    // #[test]
    // fn read_chunked_2d() {
    //     let i = Index::index("tests/data/chunked_twoD.h5").unwrap();
    //     let mut r =
    //         DatasetReader::with_dataset(i.dataset("d_4_chunks").unwrap(), i.path()).unwrap();

    //     let vs = r.values::<f32>(None, None).unwrap();

    //     let h = hdf5::File::open(i.path()).unwrap();
    //     let hvs = h.dataset("d_4_chunks").unwrap().read_raw::<f32>().unwrap();

    //     assert_eq!(vs, hvs);
    // }
}
