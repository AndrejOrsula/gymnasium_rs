//! Rust FFI bindings for Python implementation of Gymnasium.

#[allow(
    clippy::all,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals
)]
pub mod gym {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
