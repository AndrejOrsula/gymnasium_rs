//! Python module of Gymnasium.
#![cfg(feature = "python")]
#![allow(unused)]

use pyo3::prelude::*;

#[pymodule]
fn gymnasium_rs(py: Python, module: &PyModule) -> PyResult<()> {
    Ok(())
}
