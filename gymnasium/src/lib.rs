//! Gymnasium API for Reinforcement Learning.

/// Prelude module for the gymnasium crate that re-exports the most commonly used items.
pub mod prelude {
    pub use crate::space::{
        BoxSpace, BoxSpaceIdentical, BoxSpaceIndependent, DiscreteSpace, MultiBinarySpace,
        MultiDiscreteSpace, Space, TextSpace,
    };
    pub use crate::utils::GymnasiumError;
}

pub mod space;
pub mod utils;
