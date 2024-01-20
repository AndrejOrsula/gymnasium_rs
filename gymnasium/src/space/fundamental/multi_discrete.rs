use super::Space;
use crate::{GymnasiumError, Result, Rng};
use dfdx::{
    shapes::{Dtype, HasDtype, HasShape, HasUnitType, Shape},
    tensor::{AsArray, Storage, Tensor, TensorFromVec, TensorToArray},
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

pub struct MultiDiscreteSpace<S, E, D>
where
    S: Shape,
    E: Dtype + PrimInt + SampleUniform,
    D: Storage<E>,
{
    rng: Arc<Mutex<Rng>>,
    dist: Vec<Uniform<E>>,
    n: Tensor<S, E, D>,
    start: Tensor<S, E, D>,
    end: Tensor<S, E, D>,
}

impl<S, E, D> MultiDiscreteSpace<S, E, D>
where
    S: Shape,
    E: Dtype + PrimInt + SampleUniform,
    D: Storage<E> + TensorFromVec<E>,
{
    pub fn new(n: Tensor<S, E, D>, start: Tensor<S, E, D>, seed: Option<u64>) -> Result<Self> {
        Self::check_space(&n, &start)?;

        let end = n.device().tensor_from_vec(
            start
                .as_vec()
                .iter()
                .zip(n.as_vec().iter())
                .map(|(s, n)| *s + *n)
                .collect(),
            *n.shape(),
        );

        let rng = Arc::new(Mutex::new(Self::new_rng(seed)));
        let dist = Self::new_dist(&start, &end);
        Ok(Self {
            rng,
            dist,
            n,
            start,
            end,
        })
    }

    pub fn device(&self) -> &D {
        self.n.device()
    }

    pub const fn n(&self) -> &Tensor<S, E, D> {
        &self.n
    }

    pub const fn start(&self) -> &Tensor<S, E, D> {
        &self.start
    }

    pub const fn end(&self) -> &Tensor<S, E, D> {
        &self.end
    }

    pub fn seed(&mut self, seed: u64) {
        *self.rng.lock().unwrap() = Self::new_rng(Some(seed));
    }

    fn new_rng(seed: Option<u64>) -> Rng {
        seed.map_or_else(Rng::from_entropy, Rng::seed_from_u64)
    }

    fn new_dist(low: &Tensor<S, E, D>, high: &Tensor<S, E, D>) -> Vec<Uniform<E>> {
        low.as_vec()
            .iter()
            .zip(high.as_vec().iter())
            .map(|(l, h)| Uniform::new(l, h))
            .collect()
    }

    fn check_space(n: &Tensor<S, E, D>, start: &Tensor<S, E, D>) -> Result<()> {
        n.as_vec()
            .iter()
            .zip(start.as_vec().iter())
            .try_for_each(|(n, start)| {
                if start > &E::max_value().saturating_sub(*n) {
                    return Err(GymnasiumError::InvalidSpace(format!(
                        "The space overflows the maximum value of the data type \
                    (start: {:?} + n: {:?} > MAX: {:?})",
                        start,
                        n,
                        E::max_value()
                    )));
                }
                Ok(())
            })?;

        Ok(())
    }
}

impl<S, E, D> Space for MultiDiscreteSpace<S, E, D>
where
    S: Shape,
    E: Dtype + PrimInt + SampleUniform,
    D: Storage<E> + TensorFromVec<E>,
{
    type Shape = S;
    type Dtype = E;
    type Storage = D;

    fn contains(&self, value: &Tensor<Self::Shape, Self::Dtype, Self::Storage>) -> bool {
        self.start
            .as_vec()
            .iter()
            .zip(self.end.as_vec().iter())
            .zip(value.as_vec().iter())
            .all(|((start, end), value)| start <= value && value < end)
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

impl<S, E, D> HasShape for MultiDiscreteSpace<S, E, D>
where
    S: Shape,
    E: Dtype + PrimInt + SampleUniform,
    D: Storage<E>,
{
    type WithShape<New: Shape> = MultiDiscreteSpace<New, E, D>;
    type Shape = S;
    fn shape(&self) -> &Self::Shape {
        self.n.shape()
    }
}

impl<S, E, D> HasUnitType for MultiDiscreteSpace<S, E, D>
where
    S: Shape,
    E: Dtype + PrimInt + SampleUniform,
    D: Storage<E>,
{
    type Unit = E;
}

impl<S, E, D> HasDtype for MultiDiscreteSpace<S, E, D>
where
    S: Shape,
    E: Dtype + PrimInt + SampleUniform,
    D: Storage<E>,
{
    type Dtype = E;
}

impl<S, E, D> Clone for MultiDiscreteSpace<S, E, D>
where
    S: Shape,
    E: Dtype + PrimInt + SampleUniform,
    D: Storage<E> + TensorFromVec<E>,
{
    fn clone(&self) -> Self {
        Self {
            rng: self.rng.clone(),
            dist: Self::new_dist(&self.start, &self.end),
            n: self.n.clone(),
            start: self.start.clone(),
            end: self.end.clone(),
        }
    }
}

impl<S, E, D> Debug for MultiDiscreteSpace<S, E, D>
where
    S: Shape,
    E: Dtype + PrimInt + SampleUniform,
    D: Storage<E> + TensorToArray<S, E> + TensorFromVec<E>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MultiDiscreteSpace {{ shape: {:?}, start: {:?}, end: {:?}, dtype: {}, device: {} }}",
            self.shape().concrete(),
            self.start().array(),
            self.end().array(),
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
        trait_impls_noop::<MultiDiscreteSpace<(Const<1>,), usize, AutoDevice>>();
        trait_impls_noop::<MultiDiscreteSpace<(Const<2>, Const<2>), isize, AutoDevice>>();
    }

    #[test]
    fn invalid_bounds_u8() {
        // Arrange
        let dev = AutoDevice::default();

        // Act + Assert
        assert!(
            MultiDiscreteSpace::new(dev.tensor([1_u8, 3, 5]), dev.tensor([u8::MAX; 3]), None)
                .is_err()
        );
    }

    #[test]
    fn invalid_bounds_isize() {
        // Arrange
        let dev = AutoDevice::default();

        // Act + Assert
        assert!(MultiDiscreteSpace::new(
            dev.tensor([100, 101_isize]),
            dev.tensor([0, isize::max_value() - 100]),
            None
        )
        .is_err());
    }

    #[test]
    fn contains_u32() -> Result<()> {
        // Arrange
        let dev = AutoDevice::default();
        let space = MultiDiscreteSpace::new(dev.tensor([8_u32, 6]), dev.tensor([2, 4]), None)?;

        // Act + Assert
        assert!(space.contains(&dev.tensor([2, 4])));
        assert!(space.contains(&dev.tensor([3, 5])));
        assert!(space.contains(&dev.tensor([5, 6])));
        assert!(space.contains(&dev.tensor([6, 7])));
        assert!(space.contains(&dev.tensor([8, 8])));
        assert!(space.contains(&dev.tensor([9, 9])));
        assert!(!space.contains(&dev.tensor([0, 10])));
        assert!(!space.contains(&dev.tensor([1, 11])));
        assert!(!space.contains(&dev.tensor([10, 2])));
        assert!(!space.contains(&dev.tensor([11, 3])));
        assert!(!space.contains(&dev.tensor([u32::MAX, u32::MAX])));

        Ok(())
    }

    #[test]
    fn contains_i16() -> Result<()> {
        // Arrange
        let dev = AutoDevice::default();
        let space = MultiDiscreteSpace::new(
            dev.tensor([[3_i16, 5], [5, 3]]),
            dev.tensor([[-1, -2], [-2, -1]]),
            None,
        )?;

        // Act + Assert
        assert!(space.contains(&dev.tensor([[-1, -2], [-2, -1]])));
        assert!(space.contains(&dev.tensor([[0, -1], [-1, 0]])));
        assert!(space.contains(&dev.tensor([[0, 0], [0, 0]])));
        assert!(space.contains(&dev.tensor([[-1, 2], [2, -1]])));
        assert!(space.contains(&dev.tensor([[1, 2], [-2, -1]])));
        assert!(space.contains(&dev.tensor([[-1, -2], [2, 1]])));
        assert!(space.contains(&dev.tensor([[0, -2], [1, 0]])));
        assert!(!space.contains(&dev.tensor([[-2, -3], [-3, -2]])));
        assert!(!space.contains(&dev.tensor([[-1, 3], [3, -1]])));
        assert!(!space.contains(&dev.tensor([[2, -2], [-2, -1]])));
        assert!(!space.contains(&dev.tensor([[-1, -2], [-2, 2]])));
        assert!(!space.contains(&dev.tensor([[-1, 2], [-2, -2]])));
        assert!(!space.contains(&dev.tensor([[-1, -2], [i16::MAX, -1]])));

        Ok(())
    }

    #[test]
    fn sample_u16() -> Result<()> {
        // Arrange
        let dev = AutoDevice::default();
        let space = MultiDiscreteSpace::new(
            dev.tensor([16_u16, 8, 2, 4]),
            dev.tensor([512, 32, 2048, 1024]),
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

    #[test]
    fn sample_i64() -> Result<()> {
        // Arrange
        let dev = AutoDevice::default();
        let space =
            MultiDiscreteSpace::new(dev.tensor([[11; 3]; 3]), dev.tensor([[5_i64; 3]; 3]), None)?;

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
