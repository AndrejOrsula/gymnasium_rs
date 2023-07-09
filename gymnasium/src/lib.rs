//! Gymnasium API for Reinforcement Learning.

/// Prelude module for the gymnasium crate that re-exports the most commonly used items.
pub mod prelude {
    pub use crate::space::Space;
    pub use crate::utils::GymnasiumError;
}

pub mod space;
pub mod utils;
