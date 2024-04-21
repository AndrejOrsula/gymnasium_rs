use crate::{
    backend::{DType, TensorLike},
    Result, Space,
};

mod render;
mod return_types;

pub use render::{RenderMode, RenderOutput};
pub use return_types::{ResetReturn, StepReturn};

/// The main trait for implementing Reinforcement Learning environments.
pub trait Env {
    type DType: DType;
    type TensorLike: TensorLike<Self::DType>;

    /// Configuration for the constructor of the environment.
    type Config: EnvConfig;

    /// Action space of the environment.
    type ActionSpace: Space<Self::DType, Self::TensorLike>;

    /// Observation space of the environment.
    type ObservationSpace: Space<Self::DType, Self::TensorLike>;

    /// Type for the reward (floating point).
    type RewardType: num_traits::Float;

    /// Type containing additional information from the environment.
    type InfoType;

    /// Create a new [`Env`] instance with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `cfg` - The configuration specific to the environment.
    ///
    /// # Returns
    ///
    /// * The [`Env`] instance encapsulated in a [`Result`].
    ///
    /// # Errors
    ///
    /// * [`GymnasiumError::InvalidConfigError`] - If the provided configuration is invalid.
    fn new(cfg: Self::Config) -> Result<Self>
    where
        Self: Sized;

    /// Take a single time step in the environment using the given action.
    ///
    /// # Arguments
    ///
    /// * `action` - The action to take in the environment.
    ///
    /// # Returns
    ///
    /// * The [`StepReturn`] specific to the environment for taking the action.
    fn step(&mut self, action: Self::TensorLike) -> StepReturn<Self>
    where
        Self: Sized;

    /// Reset the environment to its initial state. After the initial reset, the environment
    /// should be reset whenever [`StepReturn`] returned by [`Env::step()`] indicates that
    /// the episode was either `terminated` or `truncated`.
    ///
    /// # Returns
    ///
    /// * The [`ResetReturn`] specific to the environment after the reset.
    fn reset(&mut self) -> ResetReturn<Self>
    where
        Self: Sized;

    /// Render the environment into various output formats.
    ///
    /// # Returns
    ///
    /// * [`Box`]-ed [`RenderOutput`] specific to the environment and render mode.
    ///
    /// # Notes
    ///
    /// The default implementation assumes [`RenderMode::None`] and therefore returns `()`.
    fn render(&mut self) -> Box<dyn RenderOutput> {
        Box::new(())
    }

    /// (Optional) Cleanly close external resources used by the environment.
    ///
    /// # Notes
    ///
    /// The default implementation does nothing.
    fn close(&mut self) {}
}

/// Trait that abstracts the configuration of an environment.
pub trait EnvConfig {
    /// The seed for the pseudo-random number generator of the environment.
    fn seed(&self) -> Option<u64> {
        None
    }

    /// The render mode of the environment.
    fn render_mode(&self) -> render::RenderMode {
        render::RenderMode::None
    }
}
