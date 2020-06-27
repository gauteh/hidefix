use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::fs;
use std::path::{Path, PathBuf};

use hdf5::File;

use super::dataset::Dataset;
use crate::reader::{cache, stream};

#[derive(Debug, Serialize, Deserialize)]
pub struct Index {
    path: Option<PathBuf>,
    datasets: HashMap<String, Dataset>,
}

impl TryFrom<&Path> for Index {
    type Error = anyhow::Error;

    fn try_from(p: &Path) -> Result<Index, anyhow::Error> {
        Index::index(p)
    }
}

impl TryFrom<&hdf5::File> for Index {
    type Error = anyhow::Error;

    fn try_from(f: &hdf5::File) -> Result<Index, anyhow::Error> {
        Index::index_file::<&str>(f, None)
    }
}

impl Index {
    /// Open an existing HDF5 file and index all variables.
    pub fn index<P>(path: P) -> Result<Index, anyhow::Error>
    where
        P: AsRef<Path>,
    {
        let path = path.as_ref();

        let hf = File::open(path)?;
        Index::index_file(&hf, Some(path))
    }

    /// Index an open HDF5 file.
    pub fn index_file<P>(hf: &hdf5::File, path: Option<P>) -> Result<Index, anyhow::Error>
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
            .map(|(m, d)| Dataset::index(&d).map(|d| (m.clone(), d)))
            .collect::<Result<HashMap<String, Dataset>, _>>()?;

        Ok(Index {
            path: path.map(|p| p.into()),
            datasets,
        })
    }

    #[must_use]
    pub fn dataset(&self, s: &str) -> Option<&Dataset> {
        self.datasets.get(s)
    }

    pub fn datasets(&self) -> &HashMap<String, Dataset> {
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
    pub fn reader(&self, ds: &str) -> Result<cache::DatasetReader<fs::File>, anyhow::Error> {
        let path = self.path().ok_or_else(|| anyhow!("missing path"))?;

        match self.dataset(ds) {
            Some(ds) => cache::DatasetReader::with_dataset(&ds, fs::File::open(path)?),
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
    pub fn streamer(&self, ds: &str) -> Result<stream::DatasetReader, anyhow::Error> {
        let path = self.path().ok_or_else(|| anyhow!("missing path"))?;

        match self.dataset(ds) {
            Some(ds) => stream::DatasetReader::with_dataset(&ds, path),
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
        let i = Index::index("tests/data/meps_det_vc_2_5km_latest.nc").unwrap();
        println!("{:#?}", i);
    }
}
