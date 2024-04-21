//! Gymnasium API for Reinforcement Learning.

pub mod backend;
pub mod env;
pub mod space;
pub mod utils;
pub mod wrappers;

/// Prelude module for the gymnasium crate that re-exports the most commonly used items.
pub mod prelude {}

pub use env::Env;
pub use space::Space;
pub use utils::{error::GymnasiumError, result::GymnasiumResult};
pub use wrappers::{WrappedEnv, Wrapper};

pub(crate) use utils::result::Result;

// Re-export of the FFI bindings to the Python implementation of Gymnasium.
pub use gymnasium_sys as sys;
