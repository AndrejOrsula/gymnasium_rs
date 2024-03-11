use thiserror::Error;

#[derive(Error, Debug)]
pub enum GymnasiumError {
    // #[error(transparent)]
    // BuilderUninitializedFieldError(#[from] derive_builder::UninitializedFieldError),
    #[error("Dependency error: {0}")]
    DependencyError(String),
    #[error("Invalid space: {0}")]
    InvalidSpace(String),
    #[error(transparent)]
    IoError(#[from] std::io::Error),
    #[error(transparent)]
    PyError(#[from] pyo3::PyErr),
    #[error("Type error: {0}")]
    TypeError(String),
    #[error("Value error: {0}")]
    ValueError(String),
}

impl From<GymnasiumError> for pyo3::PyErr {
    fn from(e: GymnasiumError) -> Self {
        match e {
            // GymnasiumError::BuilderUninitializedFieldError(e) => {
            //     pyo3::exceptions::PyValueError::new_err(e.to_string())
            // }
            GymnasiumError::DependencyError(e) => pyo3::exceptions::PyImportError::new_err(e),
            GymnasiumError::InvalidSpace(e) => pyo3::exceptions::PyValueError::new_err(e),
            GymnasiumError::IoError(e) => pyo3::exceptions::PyIOError::new_err(e),
            GymnasiumError::PyError(e) => e,
            GymnasiumError::TypeError(e) => pyo3::exceptions::PyTypeError::new_err(e),
            GymnasiumError::ValueError(e) => pyo3::exceptions::PyValueError::new_err(e),
        }
    }
}
