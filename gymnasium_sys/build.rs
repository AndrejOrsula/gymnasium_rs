fn main() {
    // Generate Rust bindings to the Python module
    pyo3_bindgen::build_bindings(
        "gymnasium",
        std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("bindings.rs"),
    )
    .unwrap();
}
