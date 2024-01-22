use super::Space;
use crate::{GymnasiumError, Result, Rng};
use dfdx::{
    shapes::{HasDtype, HasUnitType},
    tensor::{Storage, Tensor, TensorFromVec},
};
use rand::{
    distributions::{uniform::Uniform, Alphanumeric, DistString, Distribution},
    SeedableRng,
};
use std::{
    fmt::{self, Debug},
    sync::{Arc, Mutex},
};

#[derive(Clone)]
pub struct TextSpace<D>
where
    D: Storage<u8>,
{
    rng: Arc<Mutex<Rng>>,
    dist_len: Uniform<usize>,
    dist_text: Alphanumeric,
    device: D,
    min_len: usize,
    max_len: usize,
}

impl<D> TextSpace<D>
where
    D: Storage<u8>,
{
    pub fn new(min_len: usize, max_len: usize, device: D, seed: Option<u64>) -> Result<Self> {
        Self::check_space(min_len, max_len)?;

        let rng = Arc::new(Mutex::new(Self::new_rng(seed)));
        let (dist_len, dist_text) = Self::new_dists(min_len, max_len);

        Ok(Self {
            rng,
            dist_len,
            dist_text,
            device,
            min_len,
            max_len,
        })
    }

    pub const fn contains_str(&self, value: &str) -> bool {
        let len = value.len();
        len >= self.min_len && len <= self.max_len
    }

    pub fn sample_string(&self) -> String {
        let mut rng = self.rng.lock().unwrap();
        let len = self.dist_len.sample(&mut *rng);
        self.dist_text.sample_string(&mut *rng, len)
    }

    pub const fn device(&self) -> &D {
        &self.device
    }

    pub const fn min_len(&self) -> usize {
        self.min_len
    }

    pub const fn max_len(&self) -> usize {
        self.max_len
    }

    pub fn seed(&mut self, seed: u64) {
        *self.rng.lock().unwrap() = Self::new_rng(Some(seed));
    }

    fn new_rng(seed: Option<u64>) -> Rng {
        seed.map_or_else(Rng::from_entropy, Rng::seed_from_u64)
    }

    fn new_dists(low: usize, high: usize) -> (Uniform<usize>, Alphanumeric) {
        (Uniform::new_inclusive(low, high), Alphanumeric)
    }

    fn check_space(min_len: usize, max_len: usize) -> Result<()> {
        if min_len == 0 {
            return Err(GymnasiumError::InvalidSpace(
                "The minimum length of the text space must be greater than 0".to_string(),
            ));
        }

        if min_len > max_len {
            return Err(GymnasiumError::InvalidSpace(format!(
                "The minimum length of the text space cannot be greater than the maximum length \
                 (min_len: {min_len}, max_len: {max_len})",
            )));
        }

        Ok(())
    }
}

impl<D> Space for TextSpace<D>
where
    D: Storage<u8> + TensorFromVec<u8>,
{
    type Shape = (usize,);
    type Dtype = u8;
    type Storage = D;

    fn contains(&self, value: &Tensor<Self::Shape, Self::Dtype, Self::Storage>) -> bool {
        String::from_utf8(value.as_vec()).map_or(false, |value| self.contains_str(&value))
    }

    fn sample(&self) -> Tensor<Self::Shape, Self::Dtype, Self::Storage> {
        let text = self.sample_string();
        let bytes = text.into_bytes();
        let len = bytes.len();
        self.device().tensor_from_vec(bytes, (len,))
    }
}

impl<D> HasUnitType for TextSpace<D>
where
    D: Storage<u8>,
{
    type Unit = u8;
}

impl<D> HasDtype for TextSpace<D>
where
    D: Storage<u8>,
{
    type Dtype = u8;
}

impl<D> Debug for TextSpace<D>
where
    D: Storage<u8>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TextSpace {{ min_len: {}, max_len: {}, device: {} }}",
            self.min_len(),
            self.max_len(),
            std::any::type_name::<D>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dfdx::tensor::AutoDevice;

    const NUM_SAMPLES: usize = 10;

    #[test]
    fn trait_impls() {
        // Arrange
        fn trait_impls_noop<T: Sized + Send + Sync + Unpin + Clone + Debug>() {}

        // Act (successful compilation serves as the assertion)
        trait_impls_noop::<TextSpace<AutoDevice>>();
    }

    #[test]
    fn invalid_bounds() {
        // Arrange
        let dev = AutoDevice::default();

        // Act + Assert
        assert!(TextSpace::new(2, 1, dev.clone(), None).is_err());
        assert!(TextSpace::new(0, 10, dev, None).is_err());
    }

    #[test]
    fn contains() -> Result<()> {
        // Arrange
        let dev = AutoDevice::default();
        let space = TextSpace::new(3, 5, dev.clone(), None)?;

        // Act + Assert
        assert!(space.contains(&dev.tensor_from_vec(vec![b'a', b'b', b'c'], (3,))));
        assert!(space.contains(&dev.tensor_from_vec(String::from("1234").into_bytes(), (4,))));
        assert!(space.contains(&dev.tensor_from_vec(String::from(".\t|\n<").into_bytes(), (5,))));
        assert!(!space.contains(&dev.tensor_from_vec(vec![b'1'], (1,))));
        assert!(!space.contains(&dev.tensor_from_vec(vec![b'b', b'a'], (2,))));
        assert!(!space.contains(&dev.tensor_from_vec(vec![0_u8, 1_u8, 254_u8, 255_u8], (4,))));
        assert!(!space.contains(&dev.tensor_from_vec(String::from("abcdef").into_bytes(), (6,))));

        Ok(())
    }

    #[test]
    fn contains_str() -> Result<()> {
        // Arrange
        let dev = AutoDevice::default();
        let space = TextSpace::new(3, 5, dev, None)?;

        // Act + Assert
        assert!(space.contains_str("abc"));
        assert!(space.contains_str("1234"));
        assert!(space.contains_str(".\t|\n<"));
        assert!(!space.contains_str("1"));
        assert!(!space.contains_str("ba"));
        assert!(!space.contains_str("abcdef"));

        Ok(())
    }

    #[test]
    fn sample() -> Result<()> {
        // Arrange
        let dev = AutoDevice::default();
        let space = TextSpace::new(1, 64, dev, None)?;

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
    fn sample_string() -> Result<()> {
        // Arrange
        let dev = AutoDevice::default();
        let space = TextSpace::new(1, 64, dev, None)?;

        let mut all_samples = Vec::new();
        for _ in 0..NUM_SAMPLES {
            // Act
            let sample = space.sample_string();

            // Assert (sample must be in space)
            assert!(space.contains_str(&sample));

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
