// //! More procedural macros for `PyO3`.
// use proc_macro::TokenStream;

// mod dict;

// /// Derive `pyo3::types::IntoPyDict` and `From<T> for pyo3::Py<pyo3::types::PyDict>`
// /// implementations for a struct with named fields.
// #[proc_macro_derive(IntoPyDict)]
// pub fn derive_into_pydict(input: TokenStream) -> TokenStream {
//     dict::impl_into_pydict(syn::parse_macro_input!(input))
// }
