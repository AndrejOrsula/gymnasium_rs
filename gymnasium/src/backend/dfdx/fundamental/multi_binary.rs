use super::Space;
use crate::Rng;
use dfdx::{
    shapes::{HasShape, HasUnitType, Shape},
    tensor::{Storage, Tensor, TensorFromVec},
};
use rand::{
    distributions::{Distribution, Standard},
    SeedableRng,
};
use std::{
    fmt::{self, Debug},
    sync::{Arc, Mutex},
};

#[derive(Clone)]
pub struct MultiBinarySpace<S, D>
where
    S: Shape,
    D: Storage<bool>,
{
    rng: Arc<Mutex<Rng>>,
    dist: Standard,
    device: D,
    shape: S,
}

impl<S, D> MultiBinarySpace<S, D>
where
    S: Shape,
    D: Storage<bool>,
{
    pub fn new(shape: S, device: D, seed: Option<u64>) -> Self {
        let rng = Arc::new(Mutex::new(Self::new_rng(seed)));
        let dist = Standard;

        Self {
            rng,
            dist,
            device,
            shape,
        }
    }

    pub const fn device(&self) -> &D {
        &self.device
    }

    pub fn seed(&mut self, seed: u64) {
        *self.rng.lock().unwrap() = Self::new_rng(Some(seed));
    }

    fn new_rng(seed: Option<u64>) -> Rng {
        seed.map_or_else(Rng::from_entropy, Rng::seed_from_u64)
    }
}

impl<S, D> Space for MultiBinarySpace<S, D>
where
    S: Shape,
    D: Storage<bool> + TensorFromVec<bool>,
{
    type Shape = S;
    type Dtype = bool;
    type Storage = D;

    fn contains(&self, _value: &Tensor<Self::Shape, Self::Dtype, Self::Storage>) -> bool {
        true
    }

    fn sample(&self) -> Tensor<Self::Shape, Self::Dtype, Self::Storage> {
        let mut rng = self.rng.lock().unwrap();
        let vec: Vec<bool> = (0..self.shape().num_elements())
            .map(|_| self.dist.sample(&mut *rng))
            .collect();
        self.device().tensor_from_vec(vec, *self.shape())
    }
}

impl<S, D> HasShape for MultiBinarySpace<S, D>
where
    S: Shape,
    D: Storage<bool>,
{
    type WithShape<New: Shape> = MultiBinarySpace<New, D>;
    type Shape = S;
    fn shape(&self) -> &Self::Shape {
        &self.shape
    }
}

impl<S, D> HasUnitType for MultiBinarySpace<S, D>
where
    S: Shape,
    D: Storage<bool>,
{
    type Unit = bool;
}

impl<S, D> Debug for MultiBinarySpace<S, D>
where
    S: Shape,
    D: Storage<bool>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MultiBinarySpace {{ shape: {:?}, device: {} }}",
            self.shape().concrete(),
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
        trait_impls_noop::<MultiBinarySpace<(Const<1>,), AutoDevice>>();
        trait_impls_noop::<MultiBinarySpace<(Const<2>, Const<2>), AutoDevice>>();
    }

    #[test]
    fn contains() {
        // Arrange
        let dev = AutoDevice::default();
        let space: MultiBinarySpace<(Const<2>,), _> =
            MultiBinarySpace::new(Default::default(), dev.clone(), None);

        // Act + Assert
        assert!(space.contains(&dev.tensor([true; 2])));
        assert!(space.contains(&dev.tensor([true, false])));
        assert!(space.contains(&dev.tensor([false, true])));
        assert!(space.contains(&dev.tensor([false; 2])));
    }

    #[test]
    fn sample() {
        // Arrange
        let dev = AutoDevice::default();
        let space = MultiBinarySpace::new((Const::<3>, Const::<3>, Const::<3>), dev, None);

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
    }
}
