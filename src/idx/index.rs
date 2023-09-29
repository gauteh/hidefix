use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::ops::Deref;
use std::path::{Path, PathBuf};

use hdf5::File;

use super::dataset::DatasetD;
use crate::reader::{Reader, Streamer};

#[derive(Debug, Serialize, Deserialize)]
pub struct Index<'a> {
    path: Option<PathBuf>,
    #[serde(borrow)]
    root: GroupIndex<'a>,
}

impl<'a> Deref for Index<'a> {
    type Target = GroupIndex<'a>;
    fn deref(&self) -> &Self::Target {
        &self.root
    }
}

impl Index<'_> {
    /// Open an existing HDF5 file and index all variables and groups.
    #[allow(clippy::self_named_constructors)]
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
        let path = path.map(|p| p.into());
        Ok(Index {
            path: path.clone(),
            root: GroupIndex::index_group(&hf.group("/")?, path)?,
        })
    }
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
        let path = PathBuf::from(&f.filename());

        Index::index_file(f, Some(path))
    }
}

#[cfg(feature = "netcdf")]
impl TryFrom<&netcdf::File> for Index<'_> {
    type Error = anyhow::Error;

    fn try_from(f: &netcdf::File) -> Result<Index<'static>, anyhow::Error> {
        let path = PathBuf::from(&f.path()?);

        Index::index(path)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GroupIndex<'a> {
    path: Option<PathBuf>,
    #[serde(borrow)]
    datasets: HashMap<String, DatasetD<'a>>,
    #[serde(borrow)]
    groups: HashMap<String, GroupIndex<'a>>,
}

impl GroupIndex<'_> {
    /// Index an HDF5 Group.
    pub fn index_group<P>(
        grp: &hdf5::Group,
        path: Option<P>,
    ) -> Result<GroupIndex<'static>, anyhow::Error>
    where
        P: Into<PathBuf>,
    {
        let path: Option<PathBuf> = path.map(|p| p.into());
        let datasets = grp
            .member_names()?
            .iter()
            .map(|m| grp.dataset(m).map(|d| (m, d)))
            .filter_map(Result::ok)
            .filter(|(_, d)| d.is_chunked() || d.offset().is_some()) // skipping un-allocated datasets.
            .map(|(m, d)| DatasetD::index(&d).map(|d| (m.clone(), d)))
            .collect::<Result<HashMap<String, DatasetD<'static>>, _>>()?;
        let groups = grp
            .groups()?
            .iter()
            .map(|grp| {
                GroupIndex::index_group(grp, path.clone())
                    .map(|idx| (grp.name().split('/').next_back().unwrap().to_owned(), idx))
            })
            .collect::<Result<HashMap<String, GroupIndex<'static>>, anyhow::Error>>()?;
        Ok(GroupIndex {
            path,
            datasets,
            groups,
        })
    }

    #[must_use]
    pub fn dataset(&self, s: &str) -> Option<&DatasetD> {
        let mut s = s.trim_start_matches('/').split('/');
        let ds_name = s.next_back()?;
        let grp = s.try_fold(self, |grp, grp_name| grp.groups.get(grp_name))?;
        grp.datasets.get(ds_name)
    }

    pub fn datasets(&self) -> &HashMap<String, DatasetD> {
        &self.datasets
    }

    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_ref().map(|p| p.as_ref())
    }

    #[must_use]
    pub fn group(&self, s: &str) -> Option<&GroupIndex<'_>> {
        s.trim_start_matches('/')
            .trim_end_matches('/')
            .split('/')
            .try_fold(self, |grp, grp_name| grp.groups.get(grp_name))
    }

    pub fn groups(&self) -> &HashMap<String, GroupIndex<'_>> {
        &self.groups
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
            Some(ds) => ds.as_reader(path),
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
            Some(ds) => ds.as_streamer(path),
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
        r.values::<f32, _>(..).unwrap();
    }

    #[test]
    #[cfg(feature = "netcdf")]
    fn index_from_netcdf() {
        use std::convert::TryInto;

        let f = netcdf::open("tests/data/coads_climatology.nc4").unwrap();
        let i: Index = (&f).try_into().unwrap();
        let mut r = i.reader("SST").unwrap();
        let iv = r.values::<f32, _>(..).unwrap();

        let nv = f.variable("SST").unwrap().values::<f32, _>(..).unwrap();

        assert_eq!(iv, nv);
    }

    #[test]
    fn index_from_path() {
        use std::convert::TryInto;

        let p = PathBuf::from("tests/data/coads_climatology.nc4");
        let i: Index = p.as_path().try_into().unwrap();
        let mut r = i.reader("SST").unwrap();
        r.values::<f32, _>(..).unwrap();
    }

    #[test]
    #[cfg(feature = "netcdf")]
    fn test_index_groups() {
        let path = std::env::temp_dir().join("test_index_groups.nc");
        {
            let mut ncfile = netcdf::create(path.clone()).unwrap();
            ncfile.add_dimension("x", 1).unwrap();
            ncfile
                .add_variable::<f64>("x", &["x"])
                .unwrap()
                .put_values(&[1.0], ..)
                .unwrap();
            let mut ab = ncfile.add_group("a/b").unwrap();
            ab.add_dimension("x", 1).unwrap();
            ab.add_variable::<f64>("x", &["x"])
                .unwrap()
                .put_values(&[1.0], ..)
                .unwrap();
            let mut abc = ab.add_group("c").unwrap();
            abc.add_dimension("x", 1).unwrap();
            abc.add_variable::<f64>("x", &["x"])
                .unwrap()
                .put_values(&[1.0], ..)
                .unwrap();
        }
        let idx = Index::index(path).unwrap();
        assert_eq!(idx.datasets().len(), 1);
        assert_eq!(
            idx.reader("x").unwrap().values::<f64>(None, None).unwrap(),
            vec![1.0]
        );
        assert_eq!(idx.groups().len(), 1);
        assert_eq!(idx.group("a").unwrap().groups().len(), 1);
        assert_eq!(idx.group("a/b").unwrap().groups().len(), 1);
        assert_eq!(
            idx.reader("a/b/x")
                .unwrap()
                .values::<f64>(None, None)
                .unwrap(),
            vec![1.0]
        );
        assert_eq!(
            idx.reader("a/b/c/x")
                .unwrap()
                .values::<f64>(None, None)
                .unwrap(),
            vec![1.0]
        );
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
        let mi = GroupIndex::deserialize(r).unwrap();
        println!("Deserialized Index: {:#?}", mi);

        let s = bincode::serialize(&i).unwrap();
        bincode::deserialize::<GroupIndex>(&s).unwrap();
    }
}
