use pyo3_bindgen::Codegen;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate Rust bindings to the Python module
    Codegen::default()
        .module_name("gymnasium")?
        .build(std::path::Path::new(&std::env::var("OUT_DIR")?).join("bindings.rs"))?;
    Ok(())
}
