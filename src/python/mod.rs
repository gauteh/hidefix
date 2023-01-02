//! Wrappers for using hidefix in Python.

use pyo3::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;

use crate::idx;

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
}

#[pyclass]
#[derive(Debug)]
struct Dataset {
    idx: Arc<idx::Index<'static>>,
    ds: String,
}
