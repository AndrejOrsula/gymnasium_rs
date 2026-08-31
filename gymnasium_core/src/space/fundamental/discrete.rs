use super::{Space, SpaceSampleUniform};
use crate::{backend::TensorLike, GymnasiumError, Result};

#[derive(Clone)]
pub struct DiscreteSpace {
    range: std::ops::Range<usize>,
    dist: rand::distributions::Uniform<usize>,
}

impl<T> Space<usize, T> for DiscreteSpace
where
    T: TensorLike<usize>,
{
    fn shape(&self) -> &[usize] {
        &[1]
    }

    fn contains(&self, value: &T) -> bool {
        let value = value.as_slice().unwrap()[0];
        self.range.contains(&value)
    }
}

impl<T> SpaceSampleUniform<usize, T> for DiscreteSpace
where
    T: TensorLike<usize>,
{
    fn sample(&self, rng: &mut rand::rngs::SmallRng) -> T {
        let value = rand::distributions::Distribution::sample(&self.dist, rng);
        T::from_vec(vec![value])
    }
}

impl DiscreteSpace {
    pub fn new(n: usize) -> Result<Self> {
        Self::new_at(0, n)
    }

    pub fn new_at(start: usize, n: usize) -> Result<Self> {
        if n == 0 {
            return Err(GymnasiumError::SpaceError(format!(
                "Discrete space must have at least one element \
                    ({n:?} [n] <= {:?} [ZERO])",
                0
            )));
        }

        if start.checked_add(n).is_none() {
            return Err(GymnasiumError::SpaceError(format!(
                "Discrete space overflows the maximum value of the data type \
                    ({start:?} [start] + {n:?} [n] > {:?} [MAX]!",
                usize::max_value()
            )));
        }

        let end = start + n;
        Ok(Self {
            range: std::ops::Range { start, end },
            dist: rand::distributions::Uniform::new(start, end),
        })
    }

    pub fn from_bounds(start: usize, end: usize) -> Result<Self> {
        Self::from_range(std::ops::Range { start, end })
    }

    pub fn from_range(range: std::ops::Range<usize>) -> Result<Self> {
        if range.start > range.end {
            return Err(GymnasiumError::SpaceError(format!(
                "Discrete space must have a valid range \
                    ({:?} [start] > {:?} [end])",
                range.start, range.end,
            )));
        }

        let dist = rand::distributions::Uniform::new(range.start, range.end);
        Ok(Self { range, dist })
    }

    pub const fn start(&self) -> usize {
        self.range.start
    }

    pub const fn end(&self) -> usize {
        self.range.end
    }
}

impl TryFrom<std::ops::Range<usize>> for DiscreteSpace {
    type Error = GymnasiumError;
    fn try_from(range: std::ops::Range<usize>) -> Result<Self> {
        Self::from_range(range)
    }
}

impl std::fmt::Debug for DiscreteSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DiscreteSpace {{ start: {:?}, end: {:?}, dtype: {} }}",
            self.range.start,
            self.range.end,
            std::any::type_name::<usize>(),
        )
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use dfdx::tensor::{AutoDevice, TensorFrom};

//     const NUM_SAMPLES: usize = 10;

//     #[test]
//     fn trait_impls() {
//         // Arrange
//         fn trait_impls_noop<T: Sized + Send + Sync + Unpin + Clone + Debug>() {}

//         // Act (successful compilation serves as the assertion)
//         trait_impls_noop::<DiscreteSpace<usize, AutoDevice>>();
//         trait_impls_noop::<DiscreteSpace<isize, AutoDevice>>();
//     }

//     #[test]
//     fn invalid_bounds_u8() {
//         // Arrange
//         let dev = AutoDevice::default();

//         // Act + Assert
//         assert!(DiscreteSpace::new(1, u8::max_value(), dev, None).is_err());
//     }

//     #[test]
//     fn invalid_bounds_isize() {
//         // Arrange
//         let dev = AutoDevice::default();

//         // Act + Assert
//         assert!(DiscreteSpace::new(101, isize::max_value() - 100, dev, None).is_err());
//     }

//     #[test]
//     fn contains_u32() -> Result<()> {
//         // Arrange
//         let dev = AutoDevice::default();
//         let space = DiscreteSpace::new(8, 2_u32, dev.clone(), None)?;

//         // Act + Assert
//         assert!(space.contains(&dev.tensor(2)));
//         assert!(space.contains(&dev.tensor(3)));
//         assert!(space.contains(&dev.tensor(5)));
//         assert!(space.contains(&dev.tensor(6)));
//         assert!(space.contains(&dev.tensor(8)));
//         assert!(space.contains(&dev.tensor(9)));
//         assert!(!space.contains(&dev.tensor(0)));
//         assert!(!space.contains(&dev.tensor(1)));
//         assert!(!space.contains(&dev.tensor(10)));
//         assert!(!space.contains(&dev.tensor(11)));
//         assert!(!space.contains(&dev.tensor(u32::MAX)));

//         Ok(())
//     }

//     #[test]
//     fn contains_i16() -> Result<()> {
//         // Arrange
//         let dev = AutoDevice::default();
//         let space = DiscreteSpace::new(7, -3_i16, dev.clone(), None)?;

//         // Act + Assert
//         assert!(space.contains(&dev.tensor(-3)));
//         assert!(space.contains(&dev.tensor(-2)));
//         assert!(space.contains(&dev.tensor(-1)));
//         assert!(space.contains(&dev.tensor(0)));
//         assert!(space.contains(&dev.tensor(1)));
//         assert!(space.contains(&dev.tensor(2)));
//         assert!(space.contains(&dev.tensor(3)));
//         assert!(!space.contains(&dev.tensor(-5)));
//         assert!(!space.contains(&dev.tensor(-4)));
//         assert!(!space.contains(&dev.tensor(4)));
//         assert!(!space.contains(&dev.tensor(5)));
//         assert!(!space.contains(&dev.tensor(i16::MIN)));
//         assert!(!space.contains(&dev.tensor(i16::MAX)));

//         Ok(())
//     }

//     #[test]
//     fn sample_u16() -> Result<()> {
//         // Arrange
//         let dev = AutoDevice::default();
//         let space = DiscreteSpace::new(256, 127_u16, dev, None)?;

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

//     #[test]
//     fn sample_i64() -> Result<()> {
//         // Arrange
//         let dev = AutoDevice::default();
//         let space = DiscreteSpace::new(512, i64::MIN, dev, None)?;

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
