//! Wrappers for using hidefix in Python.

use crate::filters::byteorder::ToNative;
use byte_slice_cast::ToMutByteSlice;
use ndarray::parallel::prelude::*;
use numpy::{PyArray, PyArray1, PyArrayDyn};
use pyo3::{
    prelude::*,
    types::{PyInt, PySlice, PyTuple},
};
use std::path::PathBuf;
use std::sync::Arc;

use crate::idx;
use crate::prelude::*;

#[pymodule]
fn hidefix(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Index>()?;
    Ok(())
}

#[pyclass]
#[derive(Debug)]
struct Index {
    idx: Arc<idx::Index<'static>>,
}

#[pymethods]
impl Index {
    #[new]
    pub fn new(p: PathBuf) -> PyResult<Index> {
        Ok(Index {
            idx: Arc::new(idx::Index::index(&p)?),
        })
    }

    pub fn dataset(&self, s: &str, group: Option<&str>) -> Option<Dataset> {
        match group {
            Some(group) => self.idx.group(group).and_then(|g| g.dataset(s)),
            None => self.idx.dataset(s),
        }
        .map(|_| Dataset {
            idx: self.idx.clone(),
            group: group.map(String::from),
            ds: String::from(s),
        })
    }

    fn __getitem__(&self, s: &str) -> Option<Dataset> {
        self.dataset(s, None)
    }

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
}

impl Dataset {
    fn dataset(&self) -> &idx::DatasetD {
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
    ) -> PyResult<&'py PyAny>
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

        let (a, dst) = unsafe {
            let a = PyArray::<T, _>::new(py, dims, false);
            let dst = a.as_slice_mut()?;

            (a, dst)
        };

        py.allow_threads(|| {
            let r = ds.as_par_reader(&self.idx.path().unwrap())?;
            r.values_to_par((indices, counts), dst)
        })?;

        Ok(a.as_ref())
    }

    #[cfg(off)]
    fn read_ndarray<'py, T>(
        &self,
        py: Python<'py>,
        ds: &idx::DatasetD<'_>,
        indices: &[u64],
        counts: &[u64],
    ) -> PyResult<&'py PyAny>
    where
        T: Default + numpy::Element + ToMutByteSlice + 'py,
        [T]: ToNative,
    {
        let a = py.allow_threads(|| {
            let r = ds.as_par_reader(&self.idx.path().unwrap())?;
            r.values_dyn_par((indices, counts))
        })?;

        let a = a.into_pyarray(py);

        Ok(a)
    }

    fn apply_fill_value_impl<'py, T>(
        &self,
        _py: Python<'py>,
        cond: &'py PyAny,
        fv: &'py PyAny,
        arr: &'py PyAny,
    ) where
        T: Clone
            + pyo3::conversion::FromPyObject<'py>
            + numpy::Element
            + Sync
            + std::cmp::PartialEq
            + Copy,
    {
        let cond: T = cond.extract().unwrap();
        let fv: T = fv.extract().unwrap();
        let arr = arr.downcast::<PyArrayDyn<T>>().unwrap();

        let mut v = unsafe { arr.as_array_mut() };
        v.mapv_inplace(|v| if v == cond { fv } else { v });
    }
}

#[pymethods]
impl Dataset {
    fn __repr__(&self) -> String {
        format!("Dataset (\"{}\")", self.ds)
    }

    fn __len__(&self) -> usize {
        self.dataset().size()
    }

    fn shape<'py>(&self, py: Python<'py>) -> &'py PyArray1<u64> {
        PyArray::from_slice(py, self.dataset().shape())
    }

    fn chunk_shape<'py>(&self, py: Python<'py>) -> &'py PyArray1<u64> {
        PyArray::from_slice(py, self.dataset().chunk_shape())
    }

    fn __getitem__<'py>(&self, py: Python<'py>, slice: &PyTuple) -> PyResult<&'py PyAny> {
        let ds = self.dataset();
        let shape = ds.shape();

        // if there are fewer slices than dimensions they will be extended by the full dimension
        // when read.
        let (mut indices, (mut counts, mut strides)): (Vec<_>, (Vec<_>, Vec<_>)) = slice
            .iter()
            .map(|el| match el {
                el if el.is_instance_of::<PySlice>() => el.downcast::<PySlice>().unwrap(),
                el if el.is_instance_of::<PyInt>() => {
                    let ind: isize = el.downcast::<PyInt>().unwrap().extract().unwrap();
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
        py: Python<'py>,
        cond: &PyAny,
        fv: &PyAny,
        arr: &'py PyAny,
    ) {
        let ds = self.dataset();
        match ds.dtype() {
            Datatype::UInt(sz) if sz == 1 => self.apply_fill_value_impl::<u8>(py, cond, fv, arr),
            Datatype::UInt(sz) if sz == 2 => self.apply_fill_value_impl::<u16>(py, cond, fv, arr),
            Datatype::UInt(sz) if sz == 4 => self.apply_fill_value_impl::<u32>(py, cond, fv, arr),
            Datatype::UInt(sz) if sz == 8 => self.apply_fill_value_impl::<u64>(py, cond, fv, arr),
            Datatype::Int(sz) if sz == 1 => self.apply_fill_value_impl::<i8>(py, cond, fv, arr),
            Datatype::Int(sz) if sz == 2 => self.apply_fill_value_impl::<i16>(py, cond, fv, arr),
            Datatype::Int(sz) if sz == 4 => self.apply_fill_value_impl::<i32>(py, cond, fv, arr),
            Datatype::Int(sz) if sz == 8 => self.apply_fill_value_impl::<i64>(py, cond, fv, arr),
            Datatype::Float(sz) if sz == 4 => self.apply_fill_value_impl::<f32>(py, cond, fv, arr),
            Datatype::Float(sz) if sz == 8 => self.apply_fill_value_impl::<f64>(py, cond, fv, arr),
            _ => unimplemented!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyFloat;

    #[test]
    fn coads_slice() {
        Python::with_gil(|py| {
            let i = Index::new("tests/data/coads_climatology.nc4".into()).unwrap();
            let ds = i.dataset("SST", None).unwrap();

            let arr = ds.__getitem__(py, PyTuple::new(py, vec![PySlice::new(py, 0, 10, 1)]));
            println!("{:?}", arr);
        });
    }

    #[test]
    fn coads_index_slice() {
        Python::with_gil(|py| {
            let i = Index::new("tests/data/coads_climatology.nc4".into()).unwrap();
            let ds = i.dataset("SST", None).unwrap();

            let arr = ds.__getitem__(py, PyTuple::new(py, vec![0, 10, 1]));
            println!("{:?}", arr);
        });
    }

    #[test]
    fn fill_value() {
        Python::with_gil(|py| {
            let i = Index::new("tests/data/coads_climatology.nc4".into()).unwrap();
            let ds = i.dataset("SST", None).unwrap();

            let arr = ds
                .__getitem__(py, PyTuple::new(py, vec![0, 10, 1]))
                .unwrap();
            println!("{:?}", arr);

            // apply fill value
            ds.apply_fill_value(
                py,
                PyFloat::new(py, -1.0e+34),
                PyFloat::new(py, f64::NAN),
                &arr,
            );
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
