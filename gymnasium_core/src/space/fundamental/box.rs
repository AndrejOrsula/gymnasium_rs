use super::{Space, SpaceSampleUniform};
use crate::{
    backend::{DType, TensorLike},
    GymnasiumError, Result,
};

#[derive(Clone)]
pub struct BoxSpace<V>
where
    V: DType + rand::distributions::uniform::SampleUniform,
    <V as rand::distributions::uniform::SampleUniform>::Sampler: Clone,
{
    shape: Vec<usize>,
    bounds: BoxBounds<V>,
    dist: BoxDistribution<V>,
}

impl<T, V> Space<V, T> for BoxSpace<V>
where
    T: TensorLike<V>,
    V: DType + rand::distributions::uniform::SampleUniform + num_traits::PrimInt + std::fmt::Debug,
    <V as rand::distributions::uniform::SampleUniform>::Sampler: Clone,
{
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn contains(&self, value: &T) -> bool {
        match &self.bounds {
            BoxBounds::Identical((low, high)) => value
                .as_slice()
                .unwrap()
                .iter()
                .all(|v| low <= v && v <= high),
            BoxBounds::Independent(bounds) => value
                .as_slice()
                .unwrap()
                .iter()
                .zip(bounds.iter())
                .all(|(v, (low, high))| low <= v && v <= high),
        }
    }
}

impl<T, V> SpaceSampleUniform<V, T> for BoxSpace<V>
where
    T: TensorLike<V>,
    V: DType + rand::distributions::uniform::SampleUniform + num_traits::PrimInt + std::fmt::Debug,
    <V as rand::distributions::uniform::SampleUniform>::Sampler: Clone,
{
    fn sample(&self, rng: &mut impl rand::Rng) -> T {
        match &self.dist {
            BoxDistribution::Identical(dist) => {
                let data = rand::distributions::Distribution::sample_iter(dist, rng)
                    .take(self.shape.iter().sum())
                    .collect::<Vec<V>>();
                T::from_shape_vec(self.shape.as_slice(), data).unwrap_or_else(|_| {
                    unreachable!(
                        "The data is generated from the same shape, so it should be valid."
                    )
                })
            }
            BoxDistribution::Independent(dists) => {
                let data = dists
                    .iter()
                    .map(|dist| rand::distributions::Distribution::sample(dist, rng))
                    .collect::<Vec<V>>();
                T::from_shape_vec(self.shape.as_slice(), data).unwrap_or_else(|_| {
                    unreachable!(
                        "The data is generated from the same shape, so it should be valid."
                    )
                })
            }
        }
    }
}

impl<V> BoxSpace<V>
where
    V: DType + rand::distributions::uniform::SampleUniform + num_traits::PrimInt + std::fmt::Debug,
    <V as rand::distributions::uniform::SampleUniform>::Sampler: Clone,
{
    pub fn new(shape: Vec<usize>, bounds: BoxBounds<V>) -> Result<Self> {
        bounds.validate()?;

        let dist = match &bounds {
            BoxBounds::Identical((low, high)) => {
                BoxDistribution::Identical(rand::distributions::Uniform::new(*low, *high))
            }
            BoxBounds::Independent(bounds) => {
                if bounds.len() != shape.iter().sum::<usize>() {
                    return Err(GymnasiumError::SpaceError(format!(
                        "Box space must have the same number of bounds as the number of elements \
                            ({:?} [bounds] != {:?} [elements])",
                        bounds.len(),
                        shape.iter().sum::<usize>(),
                    )));
                }
                BoxDistribution::Independent(
                    bounds
                        .iter()
                        .map(|(low, high)| rand::distributions::Uniform::new(*low, *high))
                        .collect(),
                )
            }
        };

        Ok(Self {
            shape,
            bounds,
            dist,
        })
    }

    pub fn new_identical_bounds(shape: Vec<usize>, low: V, high: V) -> Result<Self> {
        Self::new(shape, BoxBounds::Identical((low, high)))
    }

    pub fn new_independent_bounds(shape: Vec<usize>, bounds: Vec<(V, V)>) -> Result<Self> {
        Self::new(shape, BoxBounds::Independent(bounds))
    }

    pub const fn bounds(&self) -> &BoxBounds<V> {
        &self.bounds
    }
}

#[derive(Clone, Debug)]
pub enum BoxBounds<V> {
    Identical((V, V)),
    Independent(Vec<(V, V)>),
}

impl<V> BoxBounds<V> {
    fn validate(&self) -> Result<()>
    where
        V: std::fmt::Debug + PartialOrd,
    {
        match self {
            Self::Identical((low, high)) => {
                if low >= high {
                    return Err(GymnasiumError::SpaceError(format!(
                        "Box space must have valid bounds \
                            ({low:?} [low] >= {high:?} [high])",
                    )));
                }
            }
            Self::Independent(bounds) => {
                bounds
                    .iter()
                    .enumerate()
                    .try_for_each(|(i, (low, high))| -> Result<()> {
                        if low >= high {
                            return Err(GymnasiumError::SpaceError(format!(
                                "Box space must have valid bounds \
                                    ({low:?} [low] >= {high:?} [high] @ index {i})",
                            )));
                        }
                        Ok(())
                    })?;
            }
        }

        Ok(())
    }
}

#[derive(Clone)]
enum BoxDistribution<V>
where
    V: DType + rand::distributions::uniform::SampleUniform,
    <V as rand::distributions::uniform::SampleUniform>::Sampler: Clone,
{
    Identical(rand::distributions::Uniform<V>),
    Independent(Vec<rand::distributions::Uniform<V>>),
}

impl<V> std::fmt::Debug for BoxSpace<V>
where
    V: DType
        + rand::distributions::uniform::SampleUniform
        + num_traits::PrimInt
        + num_traits::ConstZero
        + std::fmt::Debug,
    <V as rand::distributions::uniform::SampleUniform>::Sampler: Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BoxSpace {{ shape: {:?}, bounds: {:?} dtype: {} }}",
            self.shape,
            self.bounds,
            std::any::type_name::<V>(),
        )
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use dfdx::{
//         shapes::Const,
//         tensor::{AutoDevice, TensorFrom},
//     };

//     const NUM_SAMPLES: usize = 10;

//     #[test]
//     fn trait_impls() {
//         // Arrange
//         fn trait_impls_noop<T: Sized + Send + Sync + Unpin + Clone + Debug>() {}

//         // Act (successful compilation serves as the assertion)
//         trait_impls_noop::<BoxSpace<(Const<1>,), usize, AutoDevice>>();
//     }

//     #[test]
//     fn invalid_bounds_u8() {
//         // Arrange
//         let dev = AutoDevice::default();

//         // Act + Assert
//         assert!(BoxSpace::new(
//             u8::MAX,
//             0,
//             (Const::<16>, Const::<16>, Const::<3>),
//             dev,
//             None
//         )
//         .is_err());
//     }

//     #[test]
//     fn invalid_bounds_f64() {
//         // Arrange
//         let dev = AutoDevice::default();

//         // Act + Assert
//         assert!(BoxSpace::new(f64::MIN_POSITIVE, 0.0, (Const::<8>,), dev, None).is_err());
//     }

//     #[test]
//     fn contains_u16() -> Result<()> {
//         // Arrange
//         let dev = AutoDevice::default();
//         let space: BoxSpace<_, u16, _> =
//             BoxSpace::new(10, 100, (Const::<2>, Const::<2>), dev.clone(), None)?;

//         // Act + Assert
//         assert!(space.contains(&dev.tensor([[10, 10], [10, 10]])));
//         assert!(space.contains(&dev.tensor([[50, 50], [50, 50]])));
//         assert!(space.contains(&dev.tensor([[100, 100], [100, 100]])));
//         assert!(!space.contains(&dev.tensor([[101, 50], [50, 50]])));
//         assert!(!space.contains(&dev.tensor([[50, 50], [50, 9]])));
//         assert!(!space.contains(&dev.tensor([[50, u16::MIN], [u16::MAX, 0]])));

//         Ok(())
//     }

//     #[test]
//     fn sample_f64() -> Result<()> {
//         // Arrange
//         let dev = AutoDevice::default();
//         let space: BoxSpace<(Const<3>,), f64, _> =
//             BoxSpace::new(-1.0, 1.0, Default::default(), dev, None)?;

//         let mut all_samples = Vec::new();
//         for _ in 0..NUM_SAMPLES {
//             // Act
//             let sample = space.sample();

//             // Assert (sample must be in space)
//             assert!(space.contains(&sample));

//             // Act
//             all_samples.push(sample);
//         }

//         // Assert (samples must differ)
//         let mut is_different = false;
//         'outer: for i in 0..NUM_SAMPLES {
//             for j in 0..NUM_SAMPLES {
//                 if i != j {
//                     is_different = true;
//                     break 'outer;
//                 }
//             }
//         }
//         assert!(
//             is_different,
//             "All samples are the same ({:?})",
//             all_samples[0]
//         );

//         Ok(())
//     }
// }
