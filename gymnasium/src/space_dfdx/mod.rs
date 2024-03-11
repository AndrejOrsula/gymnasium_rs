//! Module containing the [Space] trait and all of its implementations.

use dfdx::shapes::{Shape, Unit};
use dfdx::tensor::{Storage, Tensor};

// TODO: Consider moving more functions to the common
// TODO: Add new_with_seed constructors to all fundamental spaces
// TODO: try having new for all spaces

// pub mod composite;
pub mod fundamental;

// pub use composite::{HashMapSpace, VecSpace};
pub use fundamental::{
    BoxSpace, BoxSpaceIdentical, BoxSpaceIndependent, DiscreteSpace, MultiBinarySpace,
    MultiDiscreteSpace, TextSpace,
};

/// Interface for all spaces that specify the valid values of actions and observations for each
/// environment.
///
/// There are two primary categories of spaces: [fundamental] and [composite]. Fundamental spaces
/// are the most basic spaces, and composite spaces combine together multiple fundamental spaces.
///
/// All spaces implement this trait and can be used interchangeably. For the full list of spaces,
/// see the [implementers](Space#implementers) of this trait.
///
/// # Examples
///
/// The following example shows how to create each of the fundamental spaces.
///
/// ```
/// use gymnasium::prelude::*;
/// use dfdx::prelude::*;
///
/// let dev = AutoDevice::default();
///
/// // A discrete space with 3 valid values [6, 9)
/// let discrete_space = DiscreteSpace::new(3, 6, dev.clone(), None).unwrap();
///
/// // A box with a single dimension of size 3 and identical bounds [-1.0, 1.0]
/// let box_space = BoxSpaceIdentical::new(-1.0, 1.0, (Const::<3>,), dev.clone(), None).unwrap();
///
/// // A box with shape (2, 2), independent bounds and data type u8
/// let box_space = BoxSpaceIndependent::new(
///     dev.tensor([[0_u8, 1], [2, 3]]),
///     dev.tensor([[4, 5], [6, 7]]),
///     None,
/// );
///
/// // A multi-discrete space with shape (2,) and bounds [[-2, 2], [0, 1]]
/// let space = MultiDiscreteSpace::new(dev.tensor([5, 2]), dev.tensor([-2, 0]), None).unwrap();
///
/// // A multi-binary space with shape (3, 3, 3) and data type u8
/// let space = MultiBinarySpace::new((Const::<3>, Const::<3>, Const::<3>), dev.clone(), None);
///
/// // A text space with length bounds [2, 8]
/// let space = TextSpace::new(2, 8, dev.clone(), None).unwrap();
/// ```
pub trait Space {
    /// The shape of the space.
    type Shape: Shape;
    /// The data type of the space.
    type Dtype: Unit;
    /// The storage type of the space (e.g. [`dfdx::tensor::Cpu`] or [`dfdx::tensor::Cuda`].
    type Storage: Storage<Self::Dtype>;

    /// Check if a value is valid for the space.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to check.
    ///
    /// # Returns
    ///
    /// `true` if the value is valid, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use gymnasium::prelude::*;
    /// use dfdx::prelude::*;
    ///
    /// let dev = AutoDevice::default();
    ///
    /// // Valid values of this discrete space are 6, 7 and 8
    /// let discrete_space = DiscreteSpace::new(3, 6, dev.clone(), None).unwrap();
    /// assert_eq!(discrete_space.contains(&dev.tensor(5)), false);
    /// assert_eq!(discrete_space.contains(&dev.tensor(6)), true);
    /// assert_eq!(discrete_space.contains(&dev.tensor(7)), true);
    /// assert_eq!(discrete_space.contains(&dev.tensor(8)), true);
    /// assert_eq!(discrete_space.contains(&dev.tensor(9)), false);
    ///
    /// let box_space = BoxSpaceIdentical::new(-1.0, 1.0, (Const::<3>,), dev.clone(), None).unwrap();
    /// assert_eq!(box_space.contains(&dev.tensor([-1.0, 0.0, 1.0])), true);
    /// assert_eq!(box_space.contains(&dev.tensor([-2.0, 0.0, 1.0])), false);
    /// ```
    fn contains(&self, value: &Tensor<Self::Shape, Self::Dtype, Self::Storage>) -> bool;

    /// Uniformly sample a random value from the space.
    ///
    /// # Returns
    ///
    /// A random value from the space.
    ///
    /// # Examples
    ///
    /// ```
    /// use gymnasium::prelude::*;
    /// use dfdx::prelude::*;
    ///
    /// let dev = AutoDevice::default();
    ///
    /// // The PRNG can be seeded (e.g. with 42) to get reproducible results
    /// let discrete_space = DiscreteSpace::new(3, 6, dev.clone(), Some(42)).unwrap();
    /// let sample = discrete_space.sample();
    /// assert_eq!(discrete_space.contains(&sample), true);
    ///
    /// // The PRNG can also be left unseeded, in which case entropy is used
    /// let box_space = BoxSpaceIdentical::new(-1.0, 1.0, (Const::<3>,), dev.clone(), None).unwrap();
    /// let sample = box_space.sample();
    /// assert_eq!(box_space.contains(&sample), true);
    /// ```
    fn sample(&self) -> Tensor<Self::Shape, Self::Dtype, Self::Storage>;
}
