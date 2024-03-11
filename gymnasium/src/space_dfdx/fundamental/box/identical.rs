use super::{BoxSpace, Space};
use crate::{GymnasiumError, Result, Rng};
use dfdx::{
    shapes::{Dtype, HasDtype, HasShape, HasUnitType, Shape},
    tensor::{Storage, Tensor, TensorFromVec},
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

pub struct BoxSpaceIdentical<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    rng: Arc<Mutex<Rng>>,
    dist: Uniform<E>,
    device: D,
    shape: S,
    low: E,
    high: E,
}

impl<S, E, D> BoxSpaceIdentical<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    pub fn new(low: E, high: E, shape: S, device: D, seed: Option<u64>) -> Result<Self> {
        Self::check_space(low, high)?;

        let rng = Arc::new(Mutex::new(Self::new_rng(seed)));
        let dist = Self::new_dist(low, high);

        Ok(Self {
            rng,
            dist,
            device,
            shape,
            low,
            high,
        })
    }

    pub const fn device(&self) -> &D {
        &self.device
    }

    pub const fn low(&self) -> E {
        self.low
    }

    pub const fn high(&self) -> E {
        self.high
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

    fn new_dist(low: E, high: E) -> Uniform<E> {
        Uniform::new_inclusive(low, high)
    }

    fn check_space(low: E, high: E) -> Result<()> {
        if low > high {
            return Err(GymnasiumError::InvalidSpace(format!(
                "The lower bound cannot be greater than the upper bound \
                (low: {low:?} > high: {high:?})",
            )));
        }

        Ok(())
    }
}

impl<S, E, D> Space for BoxSpaceIdentical<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E> + TensorFromVec<E>,
{
    type Shape = S;
    type Dtype = E;
    type Storage = D;

    fn contains(&self, value: &Tensor<Self::Shape, Self::Dtype, Self::Storage>) -> bool {
        value
            .as_vec()
            .iter()
            .all(|v| self.low() <= *v && *v <= self.high())
    }

    fn sample(&self) -> Tensor<Self::Shape, Self::Dtype, Self::Storage> {
        let mut rng = self.rng.lock().unwrap();
        let vec: Vec<E> = (0..self.shape().num_elements())
            .map(|_| self.dist.sample(&mut *rng))
            .collect();
        self.device().tensor_from_vec(vec, *self.shape())
    }
}

impl<S, E, D> HasShape for BoxSpaceIdentical<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    type WithShape<New: Shape> = BoxSpaceIdentical<New, E, D>;
    type Shape = S;
    fn shape(&self) -> &Self::Shape {
        &self.shape
    }
}

impl<S, E, D> HasUnitType for BoxSpaceIdentical<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    type Unit = E;
}

impl<S, E, D> HasDtype for BoxSpaceIdentical<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    type Dtype = E;
}

impl<S, E, D> Clone for BoxSpaceIdentical<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    fn clone(&self) -> Self {
        Self {
            rng: self.rng.clone(),
            dist: Self::new_dist(self.low(), self.high()),
            device: self.device.clone(),
            shape: self.shape,
            low: self.low,
            high: self.high,
        }
    }
}

impl<S, E, D> Debug for BoxSpaceIdentical<S, E, D>
where
    S: Shape,
    E: Dtype + SampleUniform,
    D: Storage<E>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BoxSpaceIdentical {{ shape: {:?}, low: {:?}, high: {:?}, dtype: {}, device: {} }}",
            self.shape().concrete(),
            self.low(),
            self.high(),
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
        trait_impls_noop::<BoxSpaceIdentical<(Const<1>,), usize, AutoDevice>>();
    }

    #[test]
    fn invalid_bounds_u8() {
        // Arrange
        let dev = AutoDevice::default();

        // Act + Assert
        assert!(BoxSpaceIdentical::new(
            u8::MAX,
            0,
            (Const::<16>, Const::<16>, Const::<3>),
            dev,
            None
        )
        .is_err());
    }

    #[test]
    fn invalid_bounds_f64() {
        // Arrange
        let dev = AutoDevice::default();

        // Act + Assert
        assert!(BoxSpaceIdentical::new(f64::MIN_POSITIVE, 0.0, (Const::<8>,), dev, None).is_err());
    }

    #[test]
    fn contains_u16() -> Result<()> {
        // Arrange
        let dev = AutoDevice::default();
        let space: BoxSpaceIdentical<_, u16, _> =
            BoxSpaceIdentical::new(10, 100, (Const::<2>, Const::<2>), dev.clone(), None)?;

        // Act + Assert
        assert!(space.contains(&dev.tensor([[10, 10], [10, 10]])));
        assert!(space.contains(&dev.tensor([[50, 50], [50, 50]])));
        assert!(space.contains(&dev.tensor([[100, 100], [100, 100]])));
        assert!(!space.contains(&dev.tensor([[101, 50], [50, 50]])));
        assert!(!space.contains(&dev.tensor([[50, 50], [50, 9]])));
        assert!(!space.contains(&dev.tensor([[50, u16::MIN], [u16::MAX, 0]])));

        Ok(())
    }

    #[test]
    fn sample_f64() -> Result<()> {
        // Arrange
        let dev = AutoDevice::default();
        let space: BoxSpaceIdentical<(Const<3>,), f64, _> =
            BoxSpaceIdentical::new(-1.0, 1.0, Default::default(), dev, None)?;

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
