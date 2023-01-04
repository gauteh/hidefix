//! Wrappers for using hidefix in Python.

use std::path::PathBuf;
use std::sync::Arc;
use pyo3::{prelude::*, types::PySlice};
use numpy::{PyArray, PyArray1, PyArrayDyn};

use crate::idx;
use crate::prelude::*;

#[pymodule]
fn hidefix(_py: Python, m: &PyModule) -> PyResult<()> {
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
            idx: Arc::new(idx::Index::index(&p)?)
        })
    }

    pub fn dataset(&self, s: &str) -> Option<Dataset> {
        self.idx.dataset(s).map(|_| Dataset { idx: self.idx.clone(), ds: String::from(s) })
    }

    pub fn datasets(&self) -> Vec<String> {
        self.idx.datasets().keys().cloned().collect::<Vec<_>>()
    }

    fn __repr__(&self) -> String {
        format!("Index(file: {:?}, datasets: {}", self.idx.path(), self.idx.datasets().len())
    }
}

#[pyclass]
#[derive(Debug)]
struct Dataset {
    idx: Arc<idx::Index<'static>>,
    ds: String,
}

#[pymethods]
impl Dataset {
    fn __repr__(&self) -> String {
        format!("Dataset (\"{}\")", self.ds)
    }

    fn __len__(&self) -> usize {
        self.idx.dataset(&self.ds).unwrap().size()
    }

    fn shape<'py>(&self, py: Python<'py>) -> &'py PyArray1<u64> {
        PyArray::from_slice(py,
            self.idx.dataset(&self.ds).unwrap().shape())
    }

    fn chunk_shape<'py>(&self, py: Python<'py>) -> &'py PyArray1<u64> {
        PyArray::from_slice(py,
            self.idx.dataset(&self.ds).unwrap().chunk_shape())
    }

    fn __getitem__<'py>(&self, py: Python<'py>, slice: &PySlice) -> &'py PyAny {
        let ds = self.idx.dataset(&self.ds).unwrap();
        println!("dtype: {:?}", ds.dtype());

        let arr = PyArray::arange(py, 0., 4., 1.);

        arr.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_coads() {
        Python::with_gil(|py| {
            let i = Index::new("tests/data/coads_climatology.nc4".into()).unwrap();
            let ds = i.dataset("SST").unwrap();

            let arr = ds.__getitem__(py, PySlice::new(py, 0, 10, 1));
            println!("{:?}", arr);
        });
    }
}

