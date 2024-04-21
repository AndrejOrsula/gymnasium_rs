use super::{SampleUniform, Space};
use crate::{
    backend::{DType, TensorLike},
    GymnasiumError, Result,
};

#[derive(Clone)]
pub struct AlphanumericSpace<V>
where
    V: DType + rand::distributions::uniform::SampleUniform,
{
    _type: std::marker::PhantomData<V>,
    len_range: std::ops::Range<usize>,
    dist_len: rand::distributions::Uniform<usize>,
    dist_char: rand::distributions::Alphanumeric,
}

impl<T, V> Space<T> for AlphanumericSpace<V>
where
    T: TensorLike,
    V: DType + rand::distributions::uniform::SampleUniform + Into<u8> + Copy,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.len_range.end]
    }

    fn contains(&self, value: &T) -> bool {
        let value: &[V] = value.as_slice();
        let len = value.len();
        (len >= self.len_range.start && len < self.len_range.end)
            && value.iter().all(|v| {
                let v: u8 = (*v).into();
                v.is_ascii_alphanumeric()
            })
    }
}

impl<T, V> SampleUniform<T> for AlphanumericSpace<V>
where
    T: TensorLike,
    V: DType + rand::distributions::uniform::SampleUniform + Into<u8> + From<u8> + Copy,
{
    fn sample(&self, rng: &mut impl rand::Rng) -> T {
        let len = rand::distributions::Distribution::sample(&self.dist_len, rng);
        let data = rand::distributions::Distribution::sample_iter(&self.dist_char, rng)
            .take(len)
            .map(|c| c.into())
            .collect::<Vec<V>>();
        T::from_vec(data)
    }
}

impl<V> AlphanumericSpace<V>
where
    V: DType + rand::distributions::uniform::SampleUniform,
{
    pub fn new(max_len: usize) -> Result<Self> {
        Ok(Self {
            _type: std::marker::PhantomData,
            len_range: 1..max_len,
            dist_len: rand::distributions::Uniform::new_inclusive(1, max_len),
            dist_char: rand::distributions::Alphanumeric,
        })
    }

    pub fn new_at(min_len: usize, n: usize) -> Result<Self> {
        if n == 0 {
            return Err(GymnasiumError::SpaceError(format!(
                "Alphanumeric space must have a minimum length of at least 1 char \
                    ({:?} [n] < 1 [MIN])",
                n,
            )));
        }

        if min_len.checked_add(n).is_none() {
            return Err(GymnasiumError::SpaceError(format!(
                "Alphanumeric space overflows the maximum value of the data type \
                    ({min_len:?} [min_len] + {n:?} [n] > {:?} [MAX]!",
                usize::MAX
            )));
        }

        let max_len = min_len + n;
        Ok(Self {
            _type: std::marker::PhantomData,
            len_range: min_len..max_len,
            dist_len: rand::distributions::Uniform::new(min_len, max_len),
            dist_char: rand::distributions::Alphanumeric,
        })
    }

    pub fn from_bounds(start: usize, end: usize) -> Result<Self> {
        Self::from_range(start..end)
    }

    pub fn from_range(range: std::ops::Range<usize>) -> Result<Self> {
        if range.start == 0 {
            return Err(GymnasiumError::SpaceError(format!(
                "Alphanumeric space must have a minimum length of at least 1 char \
                    ({:?} [min_len] < 1 [MIN])",
                range.start,
            )));
        }

        if range.start > range.end {
            return Err(GymnasiumError::SpaceError(format!(
                "Alphanumeric space minimum length cannot be greater than the maximum length \
                    ({:?} [min_len] > {:?} [max_len])",
                range.start, range.end,
            )));
        }

        let dist_len = rand::distributions::Uniform::new(range.start, range.end);
        Ok(Self {
            _type: std::marker::PhantomData,
            len_range: range,
            dist_len,
            dist_char: rand::distributions::Alphanumeric,
        })
    }

    pub const fn min_len(&self) -> usize {
        self.len_range.start
    }

    pub const fn max_len(&self) -> usize {
        self.len_range.end
    }
}

impl<V> std::fmt::Debug for AlphanumericSpace<V>
where
    V: DType
        + rand::distributions::uniform::SampleUniform
        + num_traits::PrimInt
        + num_traits::ConstZero
        + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AlphanumericSpace {{ min_len: {:?}, max_len: {:?}, dtype: {} }}",
            self.len_range.start,
            self.len_range.end,
            std::any::type_name::<V>(),
        )
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use dfdx::tensor::AutoDevice;

//     const NUM_SAMPLES: usize = 10;

//     #[test]
//     fn trait_impls() {
//         // Arrange
//         fn trait_impls_noop<T: Sized + Send + Sync + Unpin + Clone + Debug>() {}

//         // Act (successful compilation serves as the assertion)
//         trait_impls_noop::<AlphanumericSpace<AutoDevice>>();
//     }

//     #[test]
//     fn invalid_bounds() {
//         // Arrange
//         let dev = AutoDevice::default();

//         // Act + Assert
//         assert!(AlphanumericSpace::new(2, 1, dev.clone(), None).is_err());
//         assert!(AlphanumericSpace::new(0, 10, dev, None).is_err());
//     }

//     #[test]
//     fn contains() -> Result<()> {
//         // Arrange
//         let dev = AutoDevice::default();
//         let space = AlphanumericSpace::new(3, 5, dev.clone(), None)?;

//         // Act + Assert
//         assert!(space.contains(&dev.tensor_from_vec(vec![b'a', b'b', b'c'], (3,))));
//         assert!(space.contains(&dev.tensor_from_vec(String::from("1234").into_bytes(), (4,))));
//         assert!(space.contains(&dev.tensor_from_vec(String::from(".\t|\n<").into_bytes(), (5,))));
//         assert!(!space.contains(&dev.tensor_from_vec(vec![b'1'], (1,))));
//         assert!(!space.contains(&dev.tensor_from_vec(vec![b'b', b'a'], (2,))));
//         assert!(!space.contains(&dev.tensor_from_vec(vec![0_u8, 1_u8, 254_u8, 255_u8], (4,))));
//         assert!(!space.contains(&dev.tensor_from_vec(String::from("abcdef").into_bytes(), (6,))));

//         Ok(())
//     }

//     #[test]
//     fn contains_str() -> Result<()> {
//         // Arrange
//         let dev = AutoDevice::default();
//         let space = AlphanumericSpace::new(3, 5, dev, None)?;

//         // Act + Assert
//         assert!(space.contains_str("abc"));
//         assert!(space.contains_str("1234"));
//         assert!(space.contains_str(".\t|\n<"));
//         assert!(!space.contains_str("1"));
//         assert!(!space.contains_str("ba"));
//         assert!(!space.contains_str("abcdef"));

//         Ok(())
//     }

//     #[test]
//     fn sample() -> Result<()> {
//         // Arrange
//         let dev = AutoDevice::default();
//         let space = AlphanumericSpace::new(1, 64, dev, None)?;

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
//     fn sample_string() -> Result<()> {
//         // Arrange
//         let dev = AutoDevice::default();
//         let space = AlphanumericSpace::new(1, 64, dev, None)?;

//         let mut all_samples = Vec::new();
//         for _ in 0..NUM_SAMPLES {
//             // Act
//             let sample = space.sample_string();

//             // Assert (sample must be in space)
//             assert!(space.contains_str(&sample));

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
