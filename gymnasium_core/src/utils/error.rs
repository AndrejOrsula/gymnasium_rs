use thiserror::Error;

/// Error type for `gymnasium` operations.
#[derive(Error, Debug)]
pub enum GymnasiumError {
    #[error(transparent)]
    IoError(#[from] std::io::Error),

    #[cfg(feature = "python")]
    #[error(transparent)]
    PyError(#[from] pyo3::PyErr),

    #[cfg(feature = "python")]
    #[error(transparent)]
    PyDowncastError(#[from] pyo3::DowncastError<'static, 'static>),

    #[cfg(feature = "python")]
    #[error(transparent)]
    PyDowncastIntoError(#[from] pyo3::DowncastIntoError<'static>),

    #[cfg(feature = "ndarray")]
    #[error(transparent)]
    NdarrayShapeError(#[from] ndarray::ShapeError),

    #[cfg(feature = "candle")]
    #[error(transparent)]
    CandleCoreError(#[from] candle_core::Error),

    #[error("Dependency error: {0}")]
    DependencyError(String),
    #[error("Invalid space: {0}")]
    SpaceError(String),
    #[error("Type error: {0}")]
    TypeError(String),
    #[error("Value error: {0}")]
    ValueError(String),
    #[error("Invalid environment configuration: {0}")]
    InvalidConfigError(String),
    #[error("Failed to register environment: {0}")]
    RegistrationError(String),
}
