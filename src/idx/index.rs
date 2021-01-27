use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::path::{Path, PathBuf};

use hdf5::File;

use super::dataset::DatasetD;
use crate::reader::{Reader, Streamer};

#[derive(Debug, Serialize, Deserialize)]
pub struct Index<'a> {
    path: Option<PathBuf>,

    #[serde(borrow)]
    datasets: HashMap<String, DatasetD<'a>>,
}

impl TryFrom<&Path> for Index<'_> {
    type Error = anyhow::Error;

    fn try_from(p: &Path) -> Result<Index<'static>, anyhow::Error> {
        Index::index(p)
    }
}

impl TryFrom<&hdf5::File> for Index<'_> {
    type Error = anyhow::Error;

    fn try_from(f: &hdf5::File) -> Result<Index<'static>, anyhow::Error> {
        // Get file path
        //
        // XXX: Maybe this should be moved into Index::index, users would still be free to provide
        // a path to the readers.
        use hdf5_sys::h5f::H5Fget_name;
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let path = hdf5::sync::sync(|| {
            let sz: usize = unsafe { H5Fget_name(f.id(), std::ptr::null_mut(), 0) } as usize; // size without trailing 0

            ensure!(sz > 0, "No path available");

            let mut name = vec![0u8; sz + 1];
            ensure!(
                unsafe { H5Fget_name(f.id(), name.as_mut_ptr() as *mut _, sz + 1) } as usize == sz,
                "Mismatching length of path"
            );

            name.pop(); // remove trailing 0

            let name = OsString::from_vec(name);

            Ok(PathBuf::from(&name))
        })?;

        Index::index_file(f, Some(path))
    }
}

impl Index<'_> {
    /// Open an existing HDF5 file and index all variables.
    pub fn index<P>(path: P) -> Result<Index<'static>, anyhow::Error>
    where
        P: AsRef<Path>,
    {
        let path = path.as_ref();

        let hf = File::open(path)?;
        Index::index_file(&hf, Some(path))
    }

    /// Index an open HDF5 file.
    pub fn index_file<P>(hf: &hdf5::File, path: Option<P>) -> Result<Index<'static>, anyhow::Error>
    where
        P: Into<PathBuf>,
    {
        let datasets = hf
            .group("/")?
            .member_names()?
            .iter()
            .map(|m| hf.dataset(m).map(|d| (m, d)))
            .filter_map(Result::ok)
            .filter(|(_, d)| d.is_chunked() || d.offset().is_some()) // skipping un-allocated datasets.
            .map(|(m, d)| DatasetD::index(&d).map(|d| (m.clone(), d)))
            .collect::<Result<HashMap<String, DatasetD<'static>>, _>>()?;

        Ok(Index {
            path: path.map(|p| p.into()),
            datasets,
        })
    }

    #[must_use]
    pub fn dataset(&self, s: &str) -> Option<&DatasetD> {
        self.datasets.get(s)
    }

    pub fn datasets(&self) -> &HashMap<String, DatasetD> {
        &self.datasets
    }

    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_ref().map(|p| p.as_ref())
    }

    /// Create a cached reader for dataset.
    ///
    /// This is a convenience method to use a standard `std::fs::File` with a `cached` reader, you are
    /// free to create use anything else with `std::io::Read` and `std::io::Seek`.
    ///
    /// This method assumes the HDF5 file has the same location as at the time of
    /// indexing.
    pub fn reader(&self, ds: &str) -> Result<Box<dyn Reader + '_>, anyhow::Error> {
        let path = self.path().ok_or_else(|| anyhow!("missing path"))?;

        match self.dataset(ds) {
            Some(ds) => ds.as_reader(&path),
            None => Err(anyhow!("dataset does not exist")),
        }
    }

    /// Create a streaming reader for dataset.
    ///
    /// This is a convenience method to use a standard `std::fs::File` with a `stream` reader, you are
    /// free to create use anything else with `std::io::Read` and `std::io::Seek`.
    ///
    /// This method assumes the HDF5 file has the same location as at the time of
    /// indexing.
    pub fn streamer(&self, ds: &str) -> Result<Box<dyn Streamer + '_>, anyhow::Error> {
        let path = self.path().ok_or_else(|| anyhow!("missing path"))?;

        match self.dataset(ds) {
            Some(ds) => ds.as_streamer(&path),
            None => Err(anyhow!("dataset does not exist")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn index_t_float32() {
        let i = Index::index("tests/data/dmrpp/t_float.h5").unwrap();

        println!("index: {:#?}", i);
    }

    #[test]
    fn index_from_hdf5_file() {
        use std::convert::TryInto;

        let hf = hdf5::File::open("tests/data/coads_climatology.nc4").unwrap();
        let i: Index = (&hf).try_into().unwrap();
        let mut r = i.reader("SST").unwrap();
        r.values::<f32>(None, None).unwrap();
    }

    #[test]
    fn index_from_path() {
        use std::convert::TryInto;

        let p = PathBuf::from("tests/data/coads_climatology.nc4");
        let i: Index = p.as_path().try_into().unwrap();
        let mut r = i.reader("SST").unwrap();
        r.values::<f32>(None, None).unwrap();
    }

    #[test]
    fn chunked_1d() {
        let i = Index::index("tests/data/dmrpp/chunked_oneD.h5").unwrap();

        println!("index: {:#?}", i);
    }

    #[test]
    fn chunked_2d() {
        let i = Index::index("tests/data/dmrpp/chunked_twoD.h5").unwrap();

        println!("index: {:#?}", i);
    }

    #[ignore]
    #[test]
    fn index_meps() {
        println!("indexing meps");
        test::black_box(Index::index("tests/data/meps_det_vc_2_5km_latest.nc").unwrap());
    }

    #[test]
    fn serialize() {
        use flexbuffers::FlexbufferSerializer as ser;
        let i = Index::index("tests/data/dmrpp/chunked_oneD.h5").unwrap();
        println!("Original index: {:#?}", i);

        println!("serialize");
        let mut s = ser::new();
        i.serialize(&mut s).unwrap();

        println!("deserialize");
        let r = flexbuffers::Reader::get_root(s.view()).unwrap();
        let mi = Index::deserialize(r).unwrap();
        println!("Deserialized Index: {:#?}", mi);

        let s = bincode::serialize(&i).unwrap();
        bincode::deserialize::<Index>(&s).unwrap();
    }
}
