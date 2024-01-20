use super::{BoxSpace, Space};
use crate::utils::{GymnasiumError, Result, Rng};
use dfdx::{
    shapes::{Dtype, HasDtype, HasShape, HasUnitType, Shape},
    tensor::{AsArray, HasErr, Storage, Tensor, TensorFromVec, TensorToArray},
};
use rand::{
    distributions::{
        uniform::{SampleUniform, Uniform},
        Distribution,
    },
    SeedableRng,
};
use std::{
    fmt::{self, Debug},
    sync::{Arc, Mutex},
};

pub struct BoxSpaceIndependent<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    rng: Arc<Mutex<Rng>>,
    dist: Vec<Uniform<E>>,
    low: Tensor<S, E, D>,
    high: Tensor<S, E, D>,
}

impl<S, E, D> BoxSpaceIndependent<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    pub fn new(low: Tensor<S, E, D>, high: Tensor<S, E, D>, seed: Option<u64>) -> Result<Self> {
        Self::check_space(&low, &high)?;

        let rng = Arc::new(Mutex::new(Self::new_rng(seed)));
        let dist = Self::new_dist(&low, &high);

        Ok(Self {
            rng,
            dist,
            low,
            high,
        })
    }

    pub fn device(&self) -> &D {
        self.low.device()
    }

    pub const fn low(&self) -> &Tensor<S, E, D> {
        &self.low
    }

    pub const fn high(&self) -> &Tensor<S, E, D> {
        &self.high
    }

    pub fn seed(&mut self, seed: u64) {
        *self.rng.lock().unwrap() = Self::new_rng(Some(seed));
    }

    pub fn into_enum(self) -> BoxSpace<S, E, D> {
        self.into()
    }

    fn new_rng(seed: Option<u64>) -> Rng {
        seed.map_or_else(Rng::from_entropy, Rng::seed_from_u64)
    }

    fn new_dist(low: &Tensor<S, E, D>, high: &Tensor<S, E, D>) -> Vec<Uniform<E>> {
        low.as_vec()
            .iter()
            .zip(high.as_vec().iter())
            .map(|(l, h)| Uniform::new_inclusive(l, h))
            .collect()
    }

    fn check_space(low: &Tensor<S, E, D>, high: &Tensor<S, E, D>) -> Result<()> {
        if low
            .as_vec()
            .iter()
            .zip(high.as_vec().iter())
            .any(|(l, h)| l > h)
        {
            let index = low
                .as_vec()
                .iter()
                .zip(high.as_vec().iter())
                .position(|(l, h)| l > h)
                .unwrap();
            return Err(GymnasiumError::InvalidSpace(format!(
                "The lower bound cannot be greater than the upper bound \
                (low: {:?} > high: {:?} at index {})",
                low.as_vec()[index],
                high.as_vec()[index],
                index
            )));
        }

        Ok(())
    }
}

impl<S, E, D> Space for BoxSpaceIndependent<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E> + TensorFromVec<E>,
{
    type Shape = S;
    type Dtype = E;
    type Storage = D;

    fn contains(&self, value: &Tensor<Self::Shape, Self::Dtype, Self::Storage>) -> bool {
        self.low()
            .as_vec()
            .iter()
            .zip(self.high().as_vec().iter())
            .zip(value.as_vec().iter())
            .all(|((l, h), v)| l <= v && v <= h)
    }

    fn sample(&self) -> Tensor<Self::Shape, Self::Dtype, Self::Storage> {
        let mut rng = self.rng.lock().unwrap();
        self.device().tensor_from_vec(
            self.dist
                .iter()
                .map(|dist| dist.sample(&mut *rng))
                .collect(),
            *self.shape(),
        )
    }
}

impl<S, E, D> HasShape for BoxSpaceIndependent<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    type WithShape<New: Shape> = BoxSpaceIndependent<New, E, D>;
    type Shape = S;
    fn shape(&self) -> &Self::Shape {
        self.low.shape()
    }
}

impl<S, E, D> HasUnitType for BoxSpaceIndependent<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    type Unit = E;
}

impl<S, E, D> HasDtype for BoxSpaceIndependent<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    type Dtype = E;
}

impl<S, E, D> HasErr for BoxSpaceIndependent<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    type Err = D::Err;
}

impl<S, E, D> Clone for BoxSpaceIndependent<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    fn clone(&self) -> Self {
        Self {
            rng: self.rng.clone(),
            dist: Self::new_dist(&self.low, &self.high),
            low: self.low.clone(),
            high: self.high.clone(),
        }
    }
}

impl<S, E, D> Debug for BoxSpaceIndependent<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E> + TensorToArray<S, E>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BoxSpaceIndependent {{ shape: {:?}, low: {:?}, high: {:?}, dtype: {}, device: {} }}",
            self.shape().concrete(),
            self.low().array(),
            self.high().array(),
            std::any::type_name::<E>(),
            std::any::type_name::<D>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        trait_impls_noop::<BoxSpaceIndependent<(Const<2>, Const<2>), isize, AutoDevice>>();
    }

    #[test]
    fn invalid_bounds_usize() {
        // Arrange
        let dev = AutoDevice::default();

        // Act + Assert
        assert!(BoxSpaceIndependent::new(
            dev.tensor([0, usize::MAX]),
            dev.tensor([usize::MIN, 10]),
            None
        )
        .is_err());
    }

    #[test]
    fn invalid_bounds_f32() {
        // Arrange
        let dev = AutoDevice::default();

        // Act + Assert
        assert!(BoxSpaceIndependent::new(
            dev.tensor([[0.0], [0.0]]),
            dev.tensor([[f32::MIN], [1.0]]),
            None
        )
        .is_err());
    }

    #[test]
    fn contains_f32() -> Result<()> {
        // Arrange
        let dev = AutoDevice::default();
        let space: BoxSpaceIndependent<_, f32, _> = BoxSpaceIndependent::new(
            dev.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]]),
            dev.tensor([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]),
            None,
        )?;

        // Act + Assert
        assert!(space.contains(&dev.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])));
        assert!(space.contains(&dev.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])));
        assert!(space.contains(&dev.tensor([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])));
        assert!(!space.contains(&dev.tensor([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])));
        assert!(!space.contains(&dev.tensor([[0.0, -3.0, 0.0], [0.0, 0.0, 0.0]])));
        assert!(!space.contains(&dev.tensor([[0.0, 0.0, -4.0], [0.0, 0.0, 0.0]])));
        assert!(!space.contains(&dev.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])));
        assert!(!space.contains(&dev.tensor([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]])));
        assert!(!space.contains(&dev.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])));

        Ok(())
    }

    #[test]
    fn sample_u32() -> Result<()> {
        // Arrange
        let dev = AutoDevice::default();
        let space: BoxSpaceIndependent<_, u32, _> = BoxSpaceIndependent::new(
            dev.tensor([[0, 100, 200], [3, 40, 50]]),
            dev.tensor([[10000, 900, 8000], [7, 6000, 5000]]),
            None,
        )?;

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
