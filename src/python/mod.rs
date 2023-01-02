//! Wrappers for using hidefix in Python.

use crate::idx;
use pyo3::prelude::*;

#[pymodule]
fn hidefix(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<Index>()?;
    Ok(())
}

#[pyclass]
struct Index {
    idx: idx::Index<'static>,
}
