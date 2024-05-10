use crate::backend::{DType, TensorLike};

pub mod fundamental;
// pub mod composite;

pub use fundamental::{
    AlphanumericSpace,
    BoxSpace,
    DiscreteSpace,
    // MultiBinarySpace,
    // MultiDiscreteSpace,
};
// pub use composite::{HashMapSpace, VecSpace};

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
    fn shape(&self) -> Vec<usize>;

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

pub trait SampleUniform<A: DType, T: TensorLike<A>>: Space<A, T> {
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

/// Dynamic space that can hold any space type.
pub struct DynSpace<A: DType, T: TensorLike<A>>(Box<dyn Space<A, T>>);

impl<A: DType, T: TensorLike<A>> Space<A, T> for DynSpace<A, T> {
    fn shape(&self) -> Vec<usize> {
        self.0.shape()
    }

    fn contains(&self, value: &T) -> bool {
        self.0.contains(value)
    }
}
