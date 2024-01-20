//! Gymnasium API for Reinforcement Learning.

/// Re-export of the FFI bindings to the Python implementation of Gymnasium.
pub use gymnasium_sys as sys;

/// Prelude module for the gymnasium crate that re-exports the most commonly used items.
pub mod prelude {
    // pub use crate::env::Env;
    // pub use crate::registry::Registry;
    pub use crate::space::{
        BoxSpace, BoxSpaceIdentical, BoxSpaceIndependent, DiscreteSpace, MultiBinarySpace,
        MultiDiscreteSpace, Space, TextSpace,
    };
    pub use crate::utils::GymnasiumError;
}

// pub mod env;
// pub mod registry;
pub mod space;
pub mod utils;
