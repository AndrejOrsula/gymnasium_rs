//! Module containing the [Space] trait and all of its implementations.

use dfdx::shapes::{Shape, Unit};
use dfdx::tensor::{SampleTensor, Storage, Tensor};

/// Interface for all spaces that specify the valid values of actions and observations for each
/// environment.
///
/// There are two primary categories of spaces: [fundamental] and [composite]. Fundamental spaces
/// are the most basic spaces, and composite spaces combine together multiple fundamental spaces.
///
/// All spaces implement this trait and can be used interchangeably. For the full list of spaces,
/// see the [implementors](Space#implementors) of this trait.
pub trait Space {
    /// The shape of the space.
    type Shape: Shape;
    /// The data type of the space.
    type Dtype: Unit;
    /// The storage type of the space (e.g. [dfdx::tensor::Cpu] or [dfdx::tensor::Cuda].
    type Storage: Storage<Self::Dtype> + SampleTensor<Self::Dtype>;

    /// Check if a value is valid for the space.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to check.
    ///
    /// # Returns
    ///
    /// `true` if the value is valid, `false` otherwise.
    fn contains(&self, value: &Tensor<Self::Shape, Self::Dtype, Self::Storage>) -> bool;

    /// Uniformly sample a random value from the space.
    ///
    /// # Returns
    ///
    /// A random value from the space.
    fn sample(&self) -> Tensor<Self::Shape, Self::Dtype, Self::Storage>;
}
