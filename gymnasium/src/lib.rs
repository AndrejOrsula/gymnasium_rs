//! Gymnasium API for Reinforcement Learning.

/// Re-export of the FFI bindings to the Python implementation of Gymnasium.
pub use gymnasium_sys as sys;

/// Prelude module for the gymnasium crate that re-exports the most commonly used items.
pub mod prelude {
    pub use crate::utils::GymnasiumError;
}

pub mod utils;
