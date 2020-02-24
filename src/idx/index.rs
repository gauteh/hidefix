use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};

use hdf5::File;

use super::dataset::Dataset;

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
        let i = Index::index("../data/meps_det_vc_2_5km_latest.nc").unwrap();

        let s = serde_json::to_string(&i).unwrap();

        use std::io::prelude::*;
        let mut f = std::fs::File::create("/tmp/meps.idx").unwrap();
        f.write_all(s.as_bytes()).unwrap();
    }

    #[ignore]
    #[test]
    fn deserialize_meps() {
        use std::io::prelude::*;
        let f = std::fs::File::open("/tmp/meps.idx").unwrap();

        let i: Index = serde_json::from_reader(f).unwrap();

        println!("deserialized");

        loop {
            use std::{thread, time};

            let ten_millis = time::Duration::from_millis(10);
            let now = time::Instant::now();

            thread::sleep(ten_millis);
        }
    }
}
