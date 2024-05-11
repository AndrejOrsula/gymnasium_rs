use crate::backend::{DType, TensorLike};

pub mod fundamental;
// pub mod composite;
mod r#dyn;

pub use fundamental::{
    AlphanumericSpace,
    BoxSpace,
    DiscreteSpace,
    // MultiBinarySpace,
    // MultiDiscreteSpace,
};
// pub use composite::{HashMapSpace, VecSpace};
pub use r#dyn::DynSpace;

/// Interface for all spaces that specify the valid values of actions and observations for each
/// environment.
///
/// There are two primary categories of spaces: [fundamental] and [composite]. Fundamental spaces
/// are the most basic spaces, and composite spaces combine together multiple fundamental spaces.
///
/// All spaces implement this trait and can be used interchangeably. For the full list of spaces,
/// see the [implementers](Space#implementers) of this trait.
pub trait Space<A: DType, T: TensorLike<A>> {
    /// Get the shape of the space.
    ///
    /// # Returns
    ///
    /// The shape of the space where each element represents the size of the corresponding dimension.
    fn shape(&self) -> &[usize];

    /// Check if a value is valid for the space.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to check.
    ///
    /// # Returns
    ///
    /// `true` if the value is valid for the space, `false` otherwise.
    fn contains(&self, value: &T) -> bool;
}

pub trait SpaceSampleUniform<A: DType, T: TensorLike<A>>: Space<A, T> {
    /// Uniformly sample a random value from the space.
    ///
    /// # Arguments
    ///
    /// * `rng` - The random number generator to use.
    ///
    /// # Returns
    ///
    /// A random value from the space.
    fn sample(&self, rng: &mut impl rand::Rng) -> T;
}
