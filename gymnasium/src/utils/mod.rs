pub mod error;

pub use error::{GymnasiumError, Result};

pub(crate) type Rng = rand::rngs::SmallRng;
