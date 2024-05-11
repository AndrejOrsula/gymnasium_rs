//! Gymnasium API for Reinforcement Learning.

pub mod backend;
pub mod env;
pub mod registration;
pub mod space;
pub mod utils;
pub mod wrappers;

// /// Prelude module for the gymnasium crate that re-exports the most commonly used items.
// pub mod prelude {
//     pub use crate::{Env, GymnasiumError, GymnasiumResult, Space};
// }

pub use backend::{DType, TensorLike};
pub use env::{Env, EnvConfig, RenderMode, RenderOutput, ResetReturn, StepReturn};
pub use space::{DynSpace, DynSpaceSampleUniform, Space};
pub use utils::{error::GymnasiumError, result::GymnasiumResult};
pub use wrappers::{WrappedEnv, Wrapper};

#[cfg(feature = "python")]
pub use backend::python::{PythonEnv, PythonEnvConfig};

pub(crate) use utils::result::Result;

// Re-export of the FFI bindings to the Python implementation of Gymnasium.
pub use gymnasium_sys as sys;
