use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::ops::Deref;
use std::path::{Path, PathBuf};

use hdf5::File;

use super::attributes::{dimension_names, read_attributes, Attributes, DatasetMeta};
use super::dataset::DatasetD;
use crate::reader::{Reader, Streamer};

/// HDF5 indexer holding reference to the root group indexer
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
    /// Open an existing HDF5 file and index datasets and groups.
    #[allow(clippy::self_named_constructors)]
    pub fn index<P>(path: P) -> Result<Index<'static>, anyhow::Error>
    where
        P: AsRef<Path>,
    {
        let path = path.as_ref();

        let hf = File::open(path)?;
        Index::index_file(&hf, Some(path))
    }

    /// Index an open HDF5 file and index datasets and groups.
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

    /// Take ownership of all borrowed chunk tables, untying the index from the
    /// buffer it was (zero-copy) deserialized from.
    pub fn into_owned(self) -> Index<'static> {
        Index {
            path: self.path,
            root: self.root.into_owned(),
        }
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

/// Indexer of HDF5 group holding references to both datasets and nested
/// groups within the HDF5 group
#[derive(Debug, Serialize, Deserialize)]
pub struct GroupIndex<'a> {
    path: Option<PathBuf>,
    #[serde(borrow)]
    datasets: HashMap<String, DatasetD<'a>>,
    #[serde(borrow)]
    groups: HashMap<String, GroupIndex<'a>>,
    /// Group attributes (root group: global attributes). `default` so indexes
    /// serialized before these fields existed still deserialize (from
    /// self-describing formats).
    #[serde(default)]
    attributes: Attributes,
    #[serde(default)]
    dataset_meta: HashMap<String, DatasetMeta>,
}

impl GroupIndex<'_> {
    /// Index datasets and nested groups of an HDF5 Group
    pub fn index_group<P>(
        grp: &hdf5::Group,
        path: Option<P>,
    ) -> Result<GroupIndex<'static>, anyhow::Error>
    where
        P: Into<PathBuf>,
    {
        let path: Option<PathBuf> = path.map(|p| p.into());
        let mut datasets = HashMap::new();
        let mut dataset_meta = HashMap::new();
        for m in grp.member_names()? {
            let Ok(d) = grp.dataset(&m) else { continue };
            if !(d.is_chunked() || d.offset().is_some()) {
                // skipping un-allocated datasets.
                continue;
            }
            dataset_meta.insert(
                m.clone(),
                DatasetMeta {
                    attributes: read_attributes(&d),
                    dim_names: dimension_names(&d),
                },
            );
            datasets.insert(m, DatasetD::index(&d)?);
        }
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
            attributes: read_attributes(grp),
            dataset_meta,
        })
    }

    /// Take ownership of all borrowed chunk tables, untying the group from the
    /// buffer it was (zero-copy) deserialized from.
    pub fn into_owned(self) -> GroupIndex<'static> {
        GroupIndex {
            path: self.path,
            datasets: self
                .datasets
                .into_iter()
                .map(|(k, v)| (k, v.into_owned()))
                .collect(),
            groups: self
                .groups
                .into_iter()
                .map(|(k, v)| (k, v.into_owned()))
                .collect(),
            attributes: self.attributes,
            dataset_meta: self.dataset_meta,
        }
    }

    /// Retrieves a reference to a dataset within a HDF5 group indexer hierarchy if it exists.
    ///
    /// This function allows you to access a dataset by providing its path, using the "path/to/dataset" naming structure.
    /// The function traverses nested groups based on the path until it finds the desired dataset.
    #[must_use]
    pub fn dataset(&self, s: &str) -> Option<&DatasetD<'_>> {
        let mut s = s.trim_start_matches('/').split('/');
        let ds_name = s.next_back()?;
        let grp = s.try_fold(self, |grp, grp_name| grp.groups.get(grp_name))?;
        grp.datasets.get(ds_name)
    }

    pub fn datasets(&self) -> &HashMap<String, DatasetD<'_>> {
        &self.datasets
    }

    /// Attributes of this group (for the root group: the global attributes).
    #[must_use]
    pub fn attributes(&self) -> &Attributes {
        &self.attributes
    }

    /// Attributes of a dataset, using the same "path/to/dataset" naming
    /// structure as [`GroupIndex::dataset`].
    #[must_use]
    pub fn dataset_attributes(&self, s: &str) -> Option<&Attributes> {
        self.dataset_meta(s).map(|m| &m.attributes)
    }

    /// netCDF dimension names of a dataset (in order), using the same
    /// "path/to/dataset" naming structure as [`GroupIndex::dataset`].
    #[must_use]
    pub fn dataset_dim_names(&self, s: &str) -> Option<&[String]> {
        self.dataset_meta(s).map(|m| m.dim_names.as_slice())
    }

    fn dataset_meta(&self, s: &str) -> Option<&DatasetMeta> {
        let mut s = s.trim_start_matches('/').split('/');
        let ds_name = s.next_back()?;
        let grp = s.try_fold(self, |grp, grp_name| grp.groups.get(grp_name))?;
        grp.dataset_meta.get(ds_name)
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

    /// Nested group getter
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
    ///
    /// This function allows you to access a dataset by providing its path, using the "path/to/dataset" naming structure.
    /// The function traverses nested groups based on the path until it finds the desired dataset.
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

        let nv = f.variable("SST").unwrap().get_values::<f32, _>(..).unwrap();

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
            idx.reader("x").unwrap().values::<f64, _>(..).unwrap(),
            vec![1.0]
        );
        assert_eq!(idx.groups().len(), 1);
        assert_eq!(idx.group("a").unwrap().groups().len(), 1);
        assert_eq!(idx.group("a/b").unwrap().groups().len(), 1);
        assert_eq!(
            idx.reader("a/b/x").unwrap().values::<f64, _>(..).unwrap(),
            vec![1.0]
        );
        assert_eq!(
            idx.reader("a/b/c/x").unwrap().values::<f64, _>(..).unwrap(),
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
        let mi = Index::deserialize(r).unwrap();
        println!("Deserialized Index: {:#?}", mi);

        let s = bincode::serialize(&i).unwrap();
        bincode::deserialize::<Index>(&s).unwrap();
    }

    #[test]
    fn serialize_metadata() {
        use crate::idx::AttributeValue;
        use flexbuffers::FlexbufferSerializer as ser;

        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();

        fn check(i: &Index) {
            assert!(i.attributes().contains_key("history"));
            assert_eq!(
                i.dataset_attributes("SST").unwrap().get("units"),
                Some(&AttributeValue::Str("Deg C".into()))
            );
            assert_eq!(
                i.dataset_dim_names("SST").unwrap(),
                ["TIME", "COADSY", "COADSX"]
            );
        }
        check(&i);

        let mut s = ser::new();
        i.serialize(&mut s).unwrap();
        let r = flexbuffers::Reader::get_root(s.view()).unwrap();
        let mi = Index::deserialize(r).unwrap();
        check(&mi);

        let b = bincode::serialize(&i).unwrap();
        let bi = bincode::deserialize::<Index>(&b).unwrap();
        check(&bi);
    }

    /// Indexes serialized before `attributes` and `dataset_meta` existed must
    /// still deserialize from self-describing formats (`serde(default)`).
    #[test]
    fn deserialize_index_without_metadata_fields() {
        use flexbuffers::FlexbufferSerializer as ser;

        #[derive(Serialize)]
        struct OldGroupIndex<'a> {
            path: Option<PathBuf>,
            datasets: HashMap<String, DatasetD<'a>>,
            groups: HashMap<String, OldGroupIndex<'a>>,
        }

        #[derive(Serialize)]
        struct OldIndex<'a> {
            path: Option<PathBuf>,
            root: OldGroupIndex<'a>,
        }

        let Index { path, root } = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let old = OldIndex {
            path,
            root: OldGroupIndex {
                path: root.path,
                datasets: root.datasets,
                groups: HashMap::new(), // no sub-groups in coads
            },
        };

        let mut s = ser::new();
        old.serialize(&mut s).unwrap();
        let r = flexbuffers::Reader::get_root(s.view()).unwrap();
        let mi = Index::deserialize(r).unwrap();

        assert!(mi.dataset("SST").is_some());
        assert!(mi.attributes().is_empty());
        assert!(mi.dataset_attributes("SST").is_none());
        assert!(mi.dataset_dim_names("SST").is_none());
    }
}
