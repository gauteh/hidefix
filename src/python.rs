//! Wrappers for using hidefix in Python.

use crate::filters::byteorder::{Order as ByteOrder, ToNative};
use byte_slice_cast::ToMutByteSlice;
use numpy::{PyArray, PyArray1, PyArrayDyn, PyArrayMethods};
use pyo3::{
    exceptions::{PyKeyError, PyTypeError},
    prelude::*,
    types::{PyBytes, PyDict, PyInt, PySlice, PyTuple},
};
use std::path::PathBuf;
use std::sync::Arc;

use crate::idx;
use crate::prelude::*;

#[pymodule]
fn hidefix(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Index>()?;
    #[cfg(feature = "s3")]
    m.add_class::<S3Source>()?;
    Ok(())
}

/// The location of, and connection configuration for, an object on S3 (or an
/// S3-compatible object store such as Minio).
///
/// Credentials come from exactly one source, and the modes are mutually
/// exclusive: pass `access_key`/`secret_key` (and optionally `session_token`)
/// for explicit signed requests, or `anonymous=True` for unsigned requests,
/// but not both (combining them raises `ValueError`); with neither the ambient
/// AWS configuration is used (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`
/// environment variables or the profile files).
///
/// `region` falls back to `AWS_REGION` / `AWS_DEFAULT_REGION` when not given.
/// A custom `endpoint` (e.g. `http://localhost:9000` for Minio) implies
/// path-style addressing unless overridden with `path_style`.
#[cfg(feature = "s3")]
#[pyclass(from_py_object)]
#[derive(Clone)]
struct S3Source {
    bucket: Box<s3::Bucket>,

    /// Key of the object in the bucket.
    #[pyo3(get)]
    key: String,
}

// Manual, credential-redacting Debug: the derived impl would print the boxed
// `s3::Bucket`, whose rust-s3 Debug includes the secret key. Only the bucket
// name, key and region are safe to format.
#[cfg(feature = "s3")]
impl std::fmt::Debug for S3Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3Source")
            .field("bucket", &self.bucket.name())
            .field("key", &self.key)
            .field("region", &self.bucket.region())
            .finish()
    }
}

#[cfg(feature = "s3")]
#[pymethods]
impl S3Source {
    #[new]
    #[pyo3(signature = (bucket, key, *, region=None, endpoint=None, anonymous=false, access_key=None, secret_key=None, session_token=None, path_style=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bucket: &str,
        key: &str,
        region: Option<String>,
        endpoint: Option<String>,
        anonymous: bool,
        access_key: Option<String>,
        secret_key: Option<String>,
        session_token: Option<String>,
        path_style: Option<bool>,
    ) -> PyResult<S3Source> {
        use pyo3::exceptions::PyValueError;
        use s3::creds::Credentials;

        if anonymous && (access_key.is_some() || secret_key.is_some() || session_token.is_some()) {
            return Err(PyValueError::new_err(
                "anonymous=True is mutually exclusive with \
                 access_key/secret_key/session_token: pass anonymous=True for \
                 unsigned requests, or the explicit credentials, not both",
            ));
        }

        let credentials = if anonymous {
            Credentials::anonymous()
        } else if access_key.is_some() || secret_key.is_some() {
            Credentials::new(
                access_key.as_deref(),
                secret_key.as_deref(),
                None,
                session_token.as_deref(),
                None,
            )
        } else {
            Credentials::default()
        }
        .map_err(|e| PyValueError::new_err(format!("S3 credentials: {e}")))?;

        let region = region
            .or_else(|| std::env::var("AWS_REGION").ok())
            .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok());

        let region = match (endpoint, region) {
            (Some(endpoint), region) => s3::Region::Custom {
                region: region.unwrap_or_else(|| "us-east-1".into()),
                endpoint,
            },
            (None, Some(region)) => region
                .parse()
                .map_err(|e| PyValueError::new_err(format!("S3 region: {e}")))?,
            (None, None) => {
                return Err(PyValueError::new_err(
                    "an S3 region (or a custom endpoint) is required: pass region= or \
                     endpoint=, or set AWS_REGION",
                ))
            }
        };

        let path_style = path_style.unwrap_or(matches!(region, s3::Region::Custom { .. }));

        let mut bucket = s3::Bucket::new(bucket, region, credentials)
            .map_err(|e| PyValueError::new_err(format!("S3 bucket: {e}")))?;
        if path_style {
            bucket = bucket.with_path_style();
        }

        Ok(S3Source {
            bucket,
            key: key.into(),
        })
    }

    /// Name of the bucket.
    #[getter]
    pub fn bucket(&self) -> String {
        self.bucket.name()
    }

    fn __repr__(&self) -> String {
        format!(
            "S3Source(bucket: {}, key: {})",
            self.bucket.name(),
            self.key
        )
    }
}

#[pyclass]
#[derive(Debug)]
struct Index {
    idx: Arc<idx::Index<'static>>,

    /// Size in bytes of the source file when it was indexed (0 if unknown).
    #[pyo3(get)]
    source_size: u64,

    /// Modification time (unix seconds) of the source file when it was
    /// indexed (0 if unknown).
    #[pyo3(get)]
    source_mtime: i64,
}

impl Index {
    fn group_index(&self, group: Option<&str>) -> PyResult<&idx::GroupIndex<'_>> {
        match group {
            Some(group) => self
                .idx
                .group(group)
                .ok_or_else(|| PyKeyError::new_err(format!("group not found: {group}"))),
            None => Ok(&self.idx),
        }
    }

    /// Serialize with the version and source-fingerprint header.
    fn serialized(&self) -> PyResult<Vec<u8>> {
        let s = idx::SerializedIndex::from_index(&self.idx, self.source_size, self.source_mtime)?;
        Ok(s.to_bytes()?)
    }
}

/// Build a numpy value of the concrete element type `T` from widened values.
///
/// Single-element attributes become a numpy scalar (e.g. `np.float32`), matching
/// netCDF4-python; multi-element attributes become a 1-D numpy array. Preserving
/// the width end-to-end keeps a packed variable's `scale_factor`/`add_offset` at
/// their original dtype so unpacking does not silently upcast (and drift) to
/// float64.
fn numeric_to_py<'py, T>(py: Python<'py>, vals: Vec<T>, scalar: bool) -> PyResult<Bound<'py, PyAny>>
where
    T: numpy::Element,
{
    let arr = PyArray1::<T>::from_vec(py, vals);
    if scalar {
        arr.as_any().get_item(0)
    } else {
        Ok(arr.into_any())
    }
}

fn attributes_to_py<'py>(py: Python<'py>, attrs: &idx::Attributes) -> PyResult<Bound<'py, PyAny>> {
    use idx::AttributeValue as A;

    // Narrow the widened storage value(s) back to the source byte-width and hand
    // the concrete Rust type to numpy so the dtype round-trips exactly.
    macro_rules! ints {
        ($vals:expr, $size:expr, $scalar:expr) => {
            match $size {
                1 => numeric_to_py(py, $vals.iter().map(|&x| x as i8).collect(), $scalar)?,
                2 => numeric_to_py(py, $vals.iter().map(|&x| x as i16).collect(), $scalar)?,
                4 => numeric_to_py(py, $vals.iter().map(|&x| x as i32).collect(), $scalar)?,
                _ => numeric_to_py(py, $vals.iter().map(|&x| x as i64).collect(), $scalar)?,
            }
        };
    }
    macro_rules! uints {
        ($vals:expr, $size:expr, $scalar:expr) => {
            match $size {
                1 => numeric_to_py(py, $vals.iter().map(|&x| x as u8).collect(), $scalar)?,
                2 => numeric_to_py(py, $vals.iter().map(|&x| x as u16).collect(), $scalar)?,
                4 => numeric_to_py(py, $vals.iter().map(|&x| x as u32).collect(), $scalar)?,
                _ => numeric_to_py(py, $vals.iter().map(|&x| x as u64).collect(), $scalar)?,
            }
        };
    }
    macro_rules! floats {
        ($vals:expr, $size:expr, $scalar:expr) => {
            match $size {
                4 => numeric_to_py(py, $vals.iter().map(|&x| x as f32).collect(), $scalar)?,
                _ => numeric_to_py(py, $vals.iter().map(|&x| x as f64).collect(), $scalar)?,
            }
        };
    }

    let dict = PyDict::new(py);
    for (k, v) in attrs {
        let v = match v {
            A::Str(v) => v.into_pyobject(py)?.into_any(),
            A::Strs(v) => v.into_pyobject(py)?.into_any(),
            A::Int(v, sz) => ints!(std::slice::from_ref(v), *sz, true),
            A::Ints(v, sz) => ints!(v.as_slice(), *sz, false),
            A::Uint(v, sz) => uints!(std::slice::from_ref(v), *sz, true),
            A::Uints(v, sz) => uints!(v.as_slice(), *sz, false),
            A::Float(v, sz) => floats!(std::slice::from_ref(v), *sz, true),
            A::Floats(v, sz) => floats!(v.as_slice(), *sz, false),
        };
        dict.set_item(k, v)?;
    }
    Ok(dict.into_any())
}

#[pymethods]
impl Index {
    #[new]
    pub fn new(p: PathBuf) -> PyResult<Index> {
        let idx = Arc::new(idx::Index::index(&p)?);
        let (source_size, source_mtime) = idx::serialized::file_fingerprint(&p);
        Ok(Index {
            idx,
            source_size,
            source_mtime,
        })
    }

    /// Path of the source file this index was built from.
    #[getter]
    pub fn source_path(&self) -> Option<String> {
        self.idx.path().map(|p| p.to_string_lossy().into_owned())
    }

    /// Serialize the index (with a format-version and source-fingerprint
    /// header) to bytes accepted by `load_index`.
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &self.serialized()?))
    }

    /// Serialize the index to a file, see `to_bytes`.
    pub fn save(&self, py: Python, p: PathBuf) -> PyResult<()> {
        let b = self.serialized()?;
        py.detach(|| std::fs::write(p, b))?;
        Ok(())
    }

    /// Load an index serialized with `save` or `to_bytes`, from a path or
    /// directly from bytes.
    #[staticmethod]
    pub fn load_index(py: Python, source: &Bound<'_, PyAny>) -> PyResult<Index> {
        let read;
        let bytes = if let Ok(b) = source.cast::<PyBytes>() {
            b.as_bytes()
        } else if let Ok(p) = source.extract::<PathBuf>() {
            read = py.detach(|| std::fs::read(&p))?;
            read.as_slice()
        } else {
            return Err(PyTypeError::new_err(
                "expected a path to, or the bytes of, a serialized index",
            ));
        };

        let s = idx::SerializedIndex::from_bytes(bytes)?;
        Ok(Index {
            idx: Arc::new(s.index()?),
            source_size: s.source_size,
            source_mtime: s.source_mtime,
        })
    }

    #[pyo3(signature = (s, group=None))]
    pub fn dataset(&self, s: &str, group: Option<&str>) -> Option<Dataset> {
        match group {
            Some(group) => self.idx.group(group).and_then(|g| g.dataset(s)),
            None => self.idx.dataset(s),
        }
        .map(|_| Dataset {
            idx: self.idx.clone(),
            group: group.map(String::from),
            ds: String::from(s),
            #[cfg(feature = "s3")]
            s3: None,
        })
    }

    fn __getitem__(&self, s: &str) -> Option<Dataset> {
        self.dataset(s, None)
    }

    /// Attributes of a group as a dict (`group=None`: the global attributes).
    pub fn attributes<'py>(
        &self,
        py: Python<'py>,
        group: Option<&str>,
    ) -> PyResult<Bound<'py, PyAny>> {
        attributes_to_py(py, self.group_index(group)?.attributes())
    }

    /// Attributes of a dataset as a dict.
    #[pyo3(signature = (s, group=None))]
    pub fn dataset_attributes<'a>(
        &self,
        py: Python<'a>,
        s: &str,
        group: Option<&str>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let attrs = self
            .group_index(group)?
            .dataset_attributes(s)
            .ok_or_else(|| PyKeyError::new_err(format!("dataset not found: {s}")))?;
        attributes_to_py(py, attrs)
    }

    /// netCDF dimension names of a dataset, in order.
    #[pyo3(signature = (s, group=None))]
    pub fn dataset_dims(&self, s: &str, group: Option<&str>) -> PyResult<Vec<String>> {
        Ok(self
            .group_index(group)?
            .dataset_dim_names(s)
            .ok_or_else(|| PyKeyError::new_err(format!("dataset not found: {s}")))?
            .to_vec())
    }

    #[pyo3(signature = (group=None))]
    pub fn datasets(&self, group: Option<&str>) -> Vec<String> {
        match group {
            Some(group) => match self.idx.group(group) {
                Some(group) => group.datasets().keys().cloned().collect::<Vec<_>>(),
                None => vec![],
            },
            None => self.idx.datasets().keys().cloned().collect::<Vec<_>>(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Index(file: {:?}, datasets: {}",
            self.idx.path(),
            self.idx.datasets().len()
        )
    }
}

#[pyclass]
#[derive(Debug)]
struct Dataset {
    idx: Arc<idx::Index<'static>>,
    group: Option<String>,
    ds: String,

    /// When set, reads fetch chunks from this object instead of the local
    /// indexed path.
    #[cfg(feature = "s3")]
    s3: Option<S3Source>,
}

impl Dataset {
    fn dataset(&self) -> &idx::DatasetD<'_> {
        match &self.group {
            Some(group) => self.idx.group(&group).unwrap().dataset(&self.ds).unwrap(),
            None => self.idx.dataset(&self.ds).unwrap(),
        }
    }

    fn read_py_array<'py, T>(
        &self,
        py: Python<'py>,
        ds: &idx::DatasetD<'_>,
        indices: &[u64],
        counts: &[u64],
    ) -> PyResult<Bound<'py, PyAny>>
    where
        T: numpy::Element + ToMutByteSlice + 'py,
        [T]: ToNative,
    {
        let mut dims = counts
            .iter()
            .cloned()
            .map(|d| d as usize)
            .filter(|d| *d > 1)
            .collect::<Vec<_>>();

        if dims.is_empty() {
            dims.push(1);
        }

        let a = PyArray::<T, _>::zeros(py, dims, false);

        {
            let mut rw = a.readwrite();
            let dst = rw.as_slice_mut()?;

            py.detach(|| -> Result<usize, anyhow::Error> {
                // Chunks are fetched with concurrent range requests (the S3 reader is
                // internally concurrent, not rayon-parallel like the local reader).
                #[cfg(feature = "s3")]
                if let Some(s3) = &self.s3 {
                    let mut r = ds.as_s3_reader(s3.bucket.clone(), &s3.key)?;
                    return r.values_to((indices, counts), dst);
                }

                let r = ds.as_par_reader(&self.idx.path().unwrap())?;
                r.values_to_par((indices, counts), dst)
            })?;
        }

        Ok(a.into_any())
    }

    #[cfg(any())]
    fn read_ndarray<'py, T>(
        &self,
        py: Python<'py>,
        ds: &idx::DatasetD<'_>,
        indices: &[u64],
        counts: &[u64],
    ) -> PyResult<Bound<'py, PyAny>>
    where
        T: Default + numpy::Element + ToMutByteSlice + 'py,
        [T]: ToNative,
    {
        let a = py.detach(|| {
            let r = ds.as_par_reader(&self.idx.path().unwrap())?;
            r.values_dyn_par((indices, counts))
        })?;

        let a = a.into_pyarray(py);

        Ok(a.into_any())
    }

    fn apply_fill_value_impl<'py, T>(
        &self,
        cond: &Bound<'py, PyAny>,
        fv: &Bound<'py, PyAny>,
        arr: &Bound<'py, PyAny>,
    ) where
        T: Clone + FromPyObjectOwned<'py> + numpy::Element + Sync + std::cmp::PartialEq + Copy,
        for<'a, 'b> <T as pyo3::FromPyObject<'a, 'b>>::Error: std::fmt::Debug,
    {
        let cond: T = cond.extract().unwrap();
        let fv: T = fv.extract().unwrap();
        let arr = arr.cast::<PyArrayDyn<T>>().unwrap();

        let mut rw = arr.readwrite();
        let mut v = rw.as_array_mut();
        ndarray::Zip::from(&mut v).par_for_each(|v| {
            if *v == cond {
                *v = fv
            }
        });
    }
}

#[pymethods]
impl Dataset {
    fn __repr__(&self) -> String {
        format!("Dataset (\"{}\")", self.ds)
    }

    /// A copy of this dataset handle that reads its chunks from `source`
    /// (which must hold the same file the index was built from) instead of
    /// the local indexed path.
    #[cfg(feature = "s3")]
    pub fn with_s3(&self, source: S3Source) -> Dataset {
        Dataset {
            idx: self.idx.clone(),
            group: self.group.clone(),
            ds: self.ds.clone(),
            s3: Some(source),
        }
    }

    fn __len__(&self) -> usize {
        self.dataset().size()
    }

    fn shape<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        PyArray::from_slice(py, self.dataset().shape())
    }

    fn chunk_shape<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        PyArray::from_slice(py, self.dataset().chunk_shape())
    }

    /// Numpy dtype string (e.g. "<f4").
    fn dtype(&self) -> String {
        let ds = self.dataset();
        let order = match ds.inner().order() {
            ByteOrder::BE => '>',
            ByteOrder::LE => '<',
            ByteOrder::Unknown => '=',
        };
        let (kind, size) = match ds.dtype() {
            Datatype::UInt(sz) => ('u', sz),
            Datatype::Int(sz) => ('i', sz),
            Datatype::Float(sz) => ('f', sz),
            Datatype::Custom(sz) => ('V', sz),
        };
        format!("{order}{kind}{size}")
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        slice: &Bound<'_, PyTuple>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let ds = self.dataset();
        let shape = ds.shape();

        // if there are fewer slices than dimensions they will be extended by the full dimension
        // when read.
        let (mut indices, (mut counts, mut strides)): (Vec<_>, (Vec<_>, Vec<_>)) = slice
            .iter()
            .map(|el| match el {
                el if el.is_instance_of::<PySlice>() => el.cast_into::<PySlice>().unwrap(),
                el if el.is_instance_of::<PyInt>() => {
                    let ind: isize = el.extract().unwrap();
                    PySlice::new(py, ind, ind + 1, 1)
                }
                _ => unimplemented!(),
            })
            .zip(shape)
            .map(|(slice, dim_sz)| {
                let i = slice
                    .indices((*dim_sz).try_into().unwrap())
                    .expect("slice could not be retrieved, too big for dimension?");
                (i.start as u64, ((i.stop - i.start) as u64, i.step as u64))
            })
            .unzip();

        indices.resize_with(shape.len(), || 0);
        strides.resize_with(shape.len(), || 1);
        counts.extend_from_slice(&shape[counts.len()..]);

        assert!(strides.iter().all(|i| *i == 1), "strides not yet supported");

        // read the data into correct datatype, convert to pyarray and cast as pyany.
        match ds.dtype() {
            Datatype::UInt(sz) if sz == 1 => self.read_py_array::<u8>(py, ds, &indices, &counts),
            Datatype::UInt(sz) if sz == 2 => self.read_py_array::<u16>(py, ds, &indices, &counts),
            Datatype::UInt(sz) if sz == 4 => self.read_py_array::<u32>(py, ds, &indices, &counts),
            Datatype::UInt(sz) if sz == 8 => self.read_py_array::<u64>(py, ds, &indices, &counts),
            Datatype::Int(sz) if sz == 1 => self.read_py_array::<i8>(py, ds, &indices, &counts),
            Datatype::Int(sz) if sz == 2 => self.read_py_array::<i16>(py, ds, &indices, &counts),
            Datatype::Int(sz) if sz == 4 => self.read_py_array::<i32>(py, ds, &indices, &counts),
            Datatype::Int(sz) if sz == 8 => self.read_py_array::<i64>(py, ds, &indices, &counts),
            Datatype::Float(sz) if sz == 4 => self.read_py_array::<f32>(py, ds, &indices, &counts),
            Datatype::Float(sz) if sz == 8 => self.read_py_array::<f64>(py, ds, &indices, &counts),
            _ => unimplemented!(),
        }
    }

    pub fn apply_fill_value<'py>(
        &self,
        _py: Python<'py>,
        cond: &Bound<'py, PyAny>,
        fv: &Bound<'py, PyAny>,
        arr: &Bound<'py, PyAny>,
    ) {
        let ds = self.dataset();
        match ds.dtype() {
            Datatype::UInt(sz) if sz == 1 => self.apply_fill_value_impl::<u8>(cond, fv, arr),
            Datatype::UInt(sz) if sz == 2 => self.apply_fill_value_impl::<u16>(cond, fv, arr),
            Datatype::UInt(sz) if sz == 4 => self.apply_fill_value_impl::<u32>(cond, fv, arr),
            Datatype::UInt(sz) if sz == 8 => self.apply_fill_value_impl::<u64>(cond, fv, arr),
            Datatype::Int(sz) if sz == 1 => self.apply_fill_value_impl::<i8>(cond, fv, arr),
            Datatype::Int(sz) if sz == 2 => self.apply_fill_value_impl::<i16>(cond, fv, arr),
            Datatype::Int(sz) if sz == 4 => self.apply_fill_value_impl::<i32>(cond, fv, arr),
            Datatype::Int(sz) if sz == 8 => self.apply_fill_value_impl::<i64>(cond, fv, arr),
            Datatype::Float(sz) if sz == 4 => self.apply_fill_value_impl::<f32>(cond, fv, arr),
            Datatype::Float(sz) if sz == 8 => self.apply_fill_value_impl::<f64>(cond, fv, arr),
            _ => unimplemented!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyFloat;

    /// The interpreter is initialized explicitly (rather than through pyo3's
    /// `auto-initialize` feature, which breaks wheel builds against statically linked
    /// pythons like the manylinux ones).
    fn with_gil<F: for<'py> FnOnce(Python<'py>) -> R, R>(f: F) -> R {
        Python::initialize();
        Python::attach(f)
    }

    #[test]
    fn coads_slice() {
        with_gil(|py| {
            let i = Index::new("tests/data/coads_climatology.nc4".into()).unwrap();
            let ds = i.dataset("SST", None).unwrap();

            let slice = PyTuple::new(py, vec![PySlice::new(py, 0, 10, 1)]).unwrap();
            let arr = ds.__getitem__(py, &slice);
            println!("{:?}", arr);
        });
    }

    #[test]
    fn coads_index_slice() {
        with_gil(|py| {
            let i = Index::new("tests/data/coads_climatology.nc4".into()).unwrap();
            let ds = i.dataset("SST", None).unwrap();

            let slice = PyTuple::new(py, vec![0, 10, 1]).unwrap();
            let arr = ds.__getitem__(py, &slice);
            println!("{:?}", arr);
        });
    }

    #[test]
    fn serialized_index_roundtrip() {
        with_gil(|py| {
            let i = Index::new("tests/data/coads_climatology.nc4".into()).unwrap();
            assert!(i.source_size > 0);
            assert!(i.source_mtime > 0);

            let b = i.to_bytes(py).unwrap();
            let li = Index::load_index(py, b.as_any()).unwrap();

            assert_eq!(li.source_size, i.source_size);
            assert_eq!(li.source_mtime, i.source_mtime);
            assert_eq!(
                li.source_path().as_deref(),
                Some("tests/data/coads_climatology.nc4")
            );

            let ds = li.dataset("SST", None).unwrap();
            let slice = PyTuple::new(py, vec![PySlice::new(py, 0, 10, 1)]).unwrap();
            ds.__getitem__(py, &slice).unwrap();
        });
    }

    #[test]
    fn fill_value() {
        with_gil(|py| {
            let i = Index::new("tests/data/coads_climatology.nc4".into()).unwrap();
            let ds = i.dataset("SST", None).unwrap();

            let arr = ds
                .__getitem__(py, &PyTuple::new(py, vec![0, 10, 1]).unwrap())
                .unwrap();
            println!("{:?}", arr);

            // apply fill value
            let cond = PyFloat::new(py, -1.0e+34);
            let fv = PyFloat::new(py, f64::NAN);
            ds.apply_fill_value(py, cond.as_any(), fv.as_any(), &arr);
        });
    }

    #[test]
    #[cfg(feature = "netcdf")]
    fn test_groups() {
        let path = std::env::temp_dir().join("test_index_groups2.nc");
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
        let idx = Index::new(path).unwrap();

        assert_eq!(idx.datasets(None), ["x"]);
        assert_eq!(idx.datasets(Some("a/b")), ["x"]);
        assert_eq!(idx.datasets(Some("a/b/c")), ["x"]);

        assert!(idx.dataset("x", None).is_some());
        assert!(idx.dataset("x", Some("a")).is_none());
        assert!(idx.dataset("x", Some("a/b")).is_some());
        // assert_eq!(idx.datasets(Some("a")), []);
    }
}
