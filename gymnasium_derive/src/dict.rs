// use syn::{Data, DataStruct, DeriveInput, Fields};

// pub fn impl_into_pydict(input: DeriveInput) -> proc_macro::TokenStream {
//     // The name of the struct for which `IntoPyDict` is being derived
//     let struct_ident = input.ident;

//     // The fields of the struct with named fields for which `IntoPyDict` is being derived
//     let named_fields = match &input.data {
//         Data::Struct(DataStruct {
//             fields: Fields::Named(fields),
//             ..
//         }) => &fields.named,
//         _ => panic!("Trait pyo3::types::IntoPyDict cannot be derived for {struct_ident} because it is not a struct with named fields."),
//     };

//     // The names of the fields of the struct for which `IntoPyDict` is being derived
//     let field_names = named_fields.iter().map(|field| &field.ident);

//     // The generated implementation
//     quote::quote! {
//         #[automatically_derived]
//         impl ::pyo3::types::IntoPyDict for #struct_ident {
//             fn into_py_dict(self, py: ::pyo3::Python) -> &::pyo3::types::PyDict {
//                 let kwargs = ::pyo3::types::PyDict::new(py);
//                 #(
//                     kwargs.set_item(::pyo3::intern!(py, stringify!(#field_names)), self.#field_names).expect(format!("Failed to convert field '{}' of struct '{}' into a pyo3::types::PyDict item.", stringify!(#field_names), stringify!(#struct_ident)).as_str());
//                 )*
//                 kwargs
//             }
//         }

//         #[automatically_derived]
//         impl ::std::convert::From<#struct_ident> for ::pyo3::Py<::pyo3::types::PyDict>
//         where
//             #struct_ident: ::pyo3::types::IntoPyDict,
//         {
//             fn from(value: #struct_ident) -> Self {
//                 ::pyo3::Python::with_gil(|py| {
//                     ::pyo3::types::IntoPyDict::into_py_dict(value, py).into()
//                 })
//             }
//         }
//     }.into()
// }
