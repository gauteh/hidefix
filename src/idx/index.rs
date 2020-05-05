use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;
use serde::{Serialize, Deserialize};

use hdf5::File;

use super::dataset::Dataset;
use crate::reader::{cache, stream};

#[derive(Debug, Serialize, Deserialize)]
pub struct Index {
    path: PathBuf,
    datasets: HashMap<String, Dataset>,
}

impl Index {
    /// Open an existing HDF5 file and index all variables.
    pub fn index<P>(path: P) -> Result<Index, anyhow::Error>
    where
        P: AsRef<Path>,
    {
        let path = path.as_ref();

        let hf = File::open(path)?;

        let datasets = hf
            .group("/")?
            .member_names()?
            .iter()
            .map(|m| hf.dataset(m).map(|d| (m, d)))
            .filter_map(Result::ok)
            .filter(|(_, d)| d.is_chunked() || d.offset().is_some()) // skipping un-allocated datasets.
            .map(|(m, d)| Dataset::index(&d).map(|d| (m.clone(), d)))
            .collect::<Result<HashMap<String, Dataset>, _>>()?;

        Ok(Index {
            path: path.into(),
            datasets,
        })
    }

    #[must_use]
    pub fn dataset(&self, s: &str) -> Option<&Dataset> {
        self.datasets.get(s)
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        self.path.as_ref()
    }

    /// Create a cached reader for dataset.
    ///
    /// This is a convenience method to use a standard `std::fs::File` with a `cached` reader, you are
    /// free to create use anything else with `std::io::Read` and `std::io::Seek`.
    ///
    /// This method assumes the HDF5 file has the same location as at the time of
    /// indexing.
    pub fn reader(
        &self,
        ds: &str,
    ) -> Result<cache::DatasetReader<fs::File>, anyhow::Error> {
        match self.dataset(ds) {
            Some(ds) => cache::DatasetReader::with_dataset(&ds, fs::File::open(self.path())?),
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
    pub fn streamer(
        &self,
        ds: &str,
    ) -> Result<stream::DatasetReader, anyhow::Error> {
        match self.dataset(ds) {
            Some(ds) => stream::DatasetReader::with_dataset(&ds, self.path()),
            None => Err(anyhow!("dataset does not exist")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_t_float32() {
        let i = Index::index("tests/data/dmrpp/t_float.h5").unwrap();

        println!("index: {:#?}", i);
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
        let _i = Index::index("../data/meps_det_vc_2_5km_latest.nc").unwrap();
    }
}
