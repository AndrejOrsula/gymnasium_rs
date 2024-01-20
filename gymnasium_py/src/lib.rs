//! Python module for the Rust implementation of Gymnasium for interoperability with Rust environments.
#![cfg(feature = "python")]
#![allow(unused)]

use pyo3::prelude::*;

#[pymodule]
fn gymnasium_rs(py: Python, module: &PyModule) -> PyResult<()> {
    Ok(())
}
