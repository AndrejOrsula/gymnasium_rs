use super::Space;
use crate::utils::{GymnasiumError, Result, Rng};
use dfdx::{
    shapes::{Dtype, HasDtype, HasUnitType, Rank0, Shape},
    tensor::{AsArray, HasErr, Storage, Tensor, TensorFrom, TensorToArray},
};
use num_traits::PrimInt;
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

pub struct DiscreteSpace<E, D>
where
    E: Dtype + PrimInt + SampleUniform,
    D: Storage<E>,
{
    rng: Arc<Mutex<Rng>>,
    dist: Uniform<E>,
    device: D,
    n: usize,
    start: E,
    end: E,
}

impl<E, D> DiscreteSpace<E, D>
where
    E: Dtype + PrimInt + SampleUniform,
    D: Storage<E>,
{
    pub fn new(n: usize, start: E, device: D, seed: Option<u64>) -> Result<Self> {
        Self::check_space(n, start)?;

        let end = start + E::from(n).unwrap();

        let rng = Arc::new(Mutex::new(Self::new_rng(seed)));
        let dist = Self::new_dist(start, end);

        Ok(Self {
            rng,
            dist,
            device,
            n,
            start,
            end,
        })
    }

    pub const fn device(&self) -> &D {
        &self.device
    }

    pub fn shape(&self) -> &impl Shape {
        &()
    }

    pub const fn n(&self) -> usize {
        self.n
    }

    pub const fn start(&self) -> E {
        self.start
    }

    pub const fn end(&self) -> E {
        self.end
    }

    pub fn seed(&mut self, seed: u64) {
        *self.rng.lock().unwrap() = Self::new_rng(Some(seed));
    }

    fn new_rng(seed: Option<u64>) -> Rng {
        seed.map_or_else(Rng::from_entropy, Rng::seed_from_u64)
    }

    fn new_dist(low: E, high: E) -> Uniform<E> {
        Uniform::new(low, high)
    }

    fn check_space(n: usize, start: E) -> Result<()> {
        if n == 0 {
            return Err(GymnasiumError::InvalidSpace(
                "The space must have at least one element".to_string(),
            ));
        }

        if start > E::max_value().saturating_sub(E::from(n).unwrap()) {
            return Err(GymnasiumError::InvalidSpace(format!(
                "The space overflows the maximum value of the data type \
                (start: {:?} + n: {:?} > MAX: {:?})",
                start,
                n,
                E::max_value()
            )));
        }

        Ok(())
    }
}

impl<E, D> Space for DiscreteSpace<E, D>
where
    E: Dtype + PrimInt + SampleUniform,
    D: Storage<E> + TensorFrom<E, Rank0, E> + TensorToArray<Rank0, E>,
    <D as TensorToArray<Rank0, E>>::Array: PartialOrd<E>,
{
    type Shape = Rank0;
    type Dtype = E;
    type Storage = D;

    fn contains(&self, value: &Tensor<Self::Shape, Self::Dtype, Self::Storage>) -> bool {
        let value = value.array();
        value >= self.start() && value < self.end()
    }

    fn sample(&self) -> Tensor<Self::Shape, Self::Dtype, Self::Storage> {
        let mut rng = self.rng.lock().unwrap();
        self.device().tensor(self.dist.sample(&mut *rng))
    }
}

impl<E, D> HasUnitType for DiscreteSpace<E, D>
where
    E: Dtype + PrimInt + SampleUniform,
    D: Storage<E>,
{
    type Unit = E;
}

impl<E, D> HasDtype for DiscreteSpace<E, D>
where
    E: Dtype + PrimInt + SampleUniform,
    D: Storage<E>,
{
    type Dtype = E;
}

impl<E, D> HasErr for DiscreteSpace<E, D>
where
    E: Dtype + PrimInt + SampleUniform,
    D: Storage<E>,
{
    type Err = D::Err;
}

impl<E, D> Clone for DiscreteSpace<E, D>
where
    E: Dtype + PrimInt + SampleUniform,
    D: Storage<E>,
{
    fn clone(&self) -> Self {
        Self {
            rng: self.rng.clone(),
            dist: Uniform::new(self.start, self.end()),
            device: self.device.clone(),
            n: self.n,
            start: self.start,
            end: self.end,
        }
    }
}

impl<E, D> Debug for DiscreteSpace<E, D>
where
    E: Dtype + PrimInt + SampleUniform,
    D: Storage<E>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DiscreteSpace {{ n: {}, start: {:?}, end: {:?}, dtype: {}, device: {} }}",
            self.n(),
            self.start(),
            self.end(),
            std::any::type_name::<E>(),
            std::any::type_name::<D>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dfdx::tensor::{AutoDevice, TensorFrom};

    const NUM_SAMPLES: usize = 10;

    #[test]
    fn trait_impls() {
        // Arrange
        fn trait_impls_noop<T: Sized + Send + Sync + Unpin + Clone + Debug>() {}

        // Act (successful compilation serves as the assertion)
        trait_impls_noop::<DiscreteSpace<usize, AutoDevice>>();
        trait_impls_noop::<DiscreteSpace<isize, AutoDevice>>();
    }

    #[test]
    fn invalid_bounds_u8() {
        // Arrange
        let dev = AutoDevice::default();

        // Act + Assert
        assert!(DiscreteSpace::new(1, u8::max_value(), dev, None).is_err());
    }

    #[test]
    fn invalid_bounds_isize() {
        // Arrange
        let dev = AutoDevice::default();

        // Act + Assert
        assert!(DiscreteSpace::new(101, isize::max_value() - 100, dev, None).is_err());
    }

    #[test]
    fn contains_u32() -> Result<()> {
        // Arrange
        let dev = AutoDevice::default();
        let space = DiscreteSpace::new(8, 2_u32, dev.clone(), None)?;

        // Act + Assert
        assert!(space.contains(&dev.tensor(2)));
        assert!(space.contains(&dev.tensor(3)));
        assert!(space.contains(&dev.tensor(5)));
        assert!(space.contains(&dev.tensor(6)));
        assert!(space.contains(&dev.tensor(8)));
        assert!(space.contains(&dev.tensor(9)));
        assert!(!space.contains(&dev.tensor(0)));
        assert!(!space.contains(&dev.tensor(1)));
        assert!(!space.contains(&dev.tensor(10)));
        assert!(!space.contains(&dev.tensor(11)));
        assert!(!space.contains(&dev.tensor(u32::MAX)));

        Ok(())
    }

    #[test]
    fn contains_i16() -> Result<()> {
        // Arrange
        let dev = AutoDevice::default();
        let space = DiscreteSpace::new(7, -3_i16, dev.clone(), None)?;

        // Act + Assert
        assert!(space.contains(&dev.tensor(-3)));
        assert!(space.contains(&dev.tensor(-2)));
        assert!(space.contains(&dev.tensor(-1)));
        assert!(space.contains(&dev.tensor(0)));
        assert!(space.contains(&dev.tensor(1)));
        assert!(space.contains(&dev.tensor(2)));
        assert!(space.contains(&dev.tensor(3)));
        assert!(!space.contains(&dev.tensor(-5)));
        assert!(!space.contains(&dev.tensor(-4)));
        assert!(!space.contains(&dev.tensor(4)));
        assert!(!space.contains(&dev.tensor(5)));
        assert!(!space.contains(&dev.tensor(i16::MIN)));
        assert!(!space.contains(&dev.tensor(i16::MAX)));

        Ok(())
    }

    #[test]
    fn sample_u16() -> Result<()> {
        // Arrange
        let dev = AutoDevice::default();
        let space = DiscreteSpace::new(256, 127_u16, dev, None)?;

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
    fn sample_i64() -> Result<()> {
        // Arrange
        let dev = AutoDevice::default();
        let space = DiscreteSpace::new(512, i64::MIN, dev, None)?;

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
