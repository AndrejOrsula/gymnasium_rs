use super::{BoxSpaceIdentical, BoxSpaceIndependent, Space};
use dfdx::{
    shapes::{Dtype, HasDtype, HasShape, HasUnitType, Shape},
    tensor::{HasErr, Storage, Tensor, TensorFromVec, TensorToArray},
};
use rand::distributions::uniform::SampleUniform;
use std::{
    convert::From,
    fmt::{self, Debug},
};

#[derive(Clone)]
pub enum BoxSpace<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    Identical(BoxSpaceIdentical<S, E, D>),
    Independent(BoxSpaceIndependent<S, E, D>),
}

#[derive(Clone, Debug)]
pub enum BoxBound<'a, S, E, D>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    Identical(E),
    Independent(&'a Tensor<S, E, D>),
}

impl<S, E, D> BoxSpace<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    pub fn device(&self) -> &D {
        match self {
            Self::Identical(space) => space.device(),
            Self::Independent(space) => space.device(),
        }
    }

    pub const fn low(&self) -> BoxBound<S, E, D> {
        match self {
            Self::Identical(space) => BoxBound::Identical(space.low()),
            Self::Independent(space) => BoxBound::Independent(space.low()),
        }
    }

    pub const fn high(&self) -> BoxBound<S, E, D> {
        match self {
            Self::Identical(space) => BoxBound::Identical(space.high()),
            Self::Independent(space) => BoxBound::Independent(space.high()),
        }
    }

    pub fn seed(&mut self, seed: u64) {
        match self {
            Self::Identical(space) => space.seed(seed),
            Self::Independent(space) => space.seed(seed),
        }
    }
}

impl<S, E, D> Space for BoxSpace<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E> + TensorFromVec<E>,
{
    type Shape = S;
    type Dtype = E;
    type Storage = D;

    fn contains(&self, value: &Tensor<Self::Shape, Self::Dtype, Self::Storage>) -> bool {
        match self {
            Self::Identical(space) => space.contains(value),
            Self::Independent(space) => space.contains(value),
        }
    }

    fn sample(&self) -> Tensor<Self::Shape, Self::Dtype, Self::Storage> {
        match self {
            Self::Identical(space) => space.sample(),
            Self::Independent(space) => space.sample(),
        }
    }
}

impl<S, E, D> From<BoxSpaceIdentical<S, E, D>> for BoxSpace<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    fn from(space: BoxSpaceIdentical<S, E, D>) -> Self {
        Self::Identical(space)
    }
}

impl<S, E, D> From<BoxSpaceIndependent<S, E, D>> for BoxSpace<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    fn from(space: BoxSpaceIndependent<S, E, D>) -> Self {
        Self::Independent(space)
    }
}

impl<S, E, D> HasShape for BoxSpace<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    type WithShape<New: Shape> = BoxSpace<New, E, D>;
    type Shape = S;
    fn shape(&self) -> &Self::Shape {
        match self {
            Self::Identical(space) => space.shape(),
            Self::Independent(space) => space.shape(),
        }
    }
}

impl<S, E, D> HasUnitType for BoxSpace<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    type Unit = E;
}

impl<S, E, D> HasDtype for BoxSpace<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    type Dtype = E;
}

impl<S, E, D> HasErr for BoxSpace<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    type Err = D::Err;
}

impl<S, E, D> Debug for BoxSpace<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E> + TensorToArray<S, E>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identical(space) => write!(f, "BoxSpace::Identical({space:?})"),
            Self::Independent(space) => write!(f, "BoxSpace::Independent({space:?})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::Result;
    use dfdx::{
        shapes::Const,
        tensor::{AutoDevice, TensorFrom},
    };

    const NUM_SAMPLES: usize = 10;

    #[test]
    fn trait_impls() {
        // Arrange
        fn trait_impls_noop<T: Sized + Send + Sync + Unpin + Clone + Debug>() {}

        // Act (successful compilation serves as the assertion)
        trait_impls_noop::<BoxSpace<(Const<3>, Const<3>, Const<3>), f32, AutoDevice>>();
    }

    #[test]
    fn contains_f64_identical() -> Result<()> {
        // Arrange
        let dev = AutoDevice::default();
        let space =
            BoxSpaceIdentical::new(-1.0_f64, 1.0, (Const::<3>,), dev.clone(), None)?.into_enum();

        // Act + Assert
        assert!(space.contains(&dev.tensor([-1.0, -1.0, -1.0])));
        assert!(space.contains(&dev.tensor([0.5, 0.0, -0.5])));
        assert!(space.contains(&dev.tensor([1.0, 1.0, 1.0])));
        assert!(!space.contains(&dev.tensor([-2.0, 0.0, 0.0])));
        assert!(!space.contains(&dev.tensor([0.0, 2.0, 0.0])));
        assert!(!space.contains(&dev.tensor([0.0, 0.0, f64::MAX])));

        Ok(())
    }

    #[test]
    fn contains_u8_independent() -> Result<()> {
        // Arrange
        let dev = AutoDevice::default();
        let space: BoxSpace<_, _, _> = BoxSpaceIndependent::new(
            dev.tensor([[0_u8, 1, 2], [3, 4, 5]]),
            dev.tensor([[10, 9, 8], [7, 6, 5]]),
            None,
        )?
        .into();

        // Act + Assert
        assert!(space.contains(&dev.tensor([[0, 1, 2], [3, 4, 5]])));
        assert!(space.contains(&dev.tensor([[5, 5, 5], [5, 5, 5]])));
        assert!(space.contains(&dev.tensor([[10, 9, 8], [7, 6, 5]])));
        assert!(!space.contains(&dev.tensor([[0, 0, 0], [0, 0, 0]])));
        assert!(!space.contains(&dev.tensor([[10, 10, 10], [10, 10, 10]])));
        assert!(!space.contains(&dev.tensor([[100, 5, 5], [5, 5, 5]])));
        assert!(!space.contains(&dev.tensor([[5, 5, 5], [5, 5, 0]])));
        assert!(!space.contains(&dev.tensor([[5, 5, 5], [5, 5, 10]])));

        Ok(())
    }

    #[test]
    fn sample_isize_identical() -> Result<()> {
        // Arrange
        let dev = AutoDevice::default();
        let space: BoxSpace<_, _, _> =
            BoxSpaceIdentical::new(-5000_isize, 10000, (Const::<3>,), dev, None)?.into();

        let mut all_samples = Vec::new();
        for _ in 0..NUM_SAMPLES {
            // Act
            let sample = space.sample();

            // Assert (sample must be in space)
            assert!(space.contains(&sample));

            // Act
            all_samples.push(sample);
        }

        // Assert (samples must differ)
        let mut is_different = false;
        'outer: for i in 0..NUM_SAMPLES {
            for j in 0..NUM_SAMPLES {
                if i != j {
                    is_different = true;
                    break 'outer;
                }
            }
        }
        assert!(
            is_different,
            "All samples are the same ({:?})",
            all_samples[0]
        );

        Ok(())
    }

    #[test]
    fn sample_f32_independent() -> Result<()> {
        // Arrange
        let dev = AutoDevice::default();
        let space = BoxSpaceIndependent::new(
            dev.tensor([[-1.0_f32, -2.0, -3.0], [-4.0, -5.0, -6.0]]),
            dev.tensor([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]),
            None,
        )?
        .into_enum();

        let mut all_samples = Vec::new();
        for _ in 0..NUM_SAMPLES {
            // Act
            let sample = space.sample();

            // Assert (sample must be in space)
            assert!(space.contains(&sample));

            // Act
            all_samples.push(sample);
        }

        // Assert (samples must differ)
        let mut is_different = false;
        'outer: for i in 0..NUM_SAMPLES {
            for j in 0..NUM_SAMPLES {
                if i != j {
                    is_different = true;
                    break 'outer;
                }
            }
        }
        assert!(
            is_different,
            "All samples are the same ({:?})",
            all_samples[0]
        );

        Ok(())
    }
}
