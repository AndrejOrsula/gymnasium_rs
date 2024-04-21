//! Rust FFI bindings for Python implementation of Gymnasium.

// TODO: Remove once pyo3_bindgen properly updates to pyo3 0.21
#![allow(deprecated)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
pub use gymnasium::*;

mod manual_bindings;
