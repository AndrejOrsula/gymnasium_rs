use crate::{
    backend::{DType, TensorLike},
    Env, Wrapper,
};

/// Wrapper that clips actions to a fixed range.
pub struct ActionClipping<T: PartialOrd + Copy> {
    /// Minimum value for each action.
    min: T,
    /// Maximum value for each action.
    max: T,
}

impl<T: PartialOrd + Copy> ActionClipping<T> {
    /// Creates a new [`ActionClipping`] instance.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value for each action.
    /// * `max` - The maximum value for each action.
    ///
    /// # Returns
    ///
    /// * The newly created [`ActionClipping`] instance.
    pub fn new(min: T, max: T) -> Self {
        Self { min, max }
    }
}

impl<E: Env<DType = T>, T: PartialOrd + Copy + DType> Wrapper<E> for ActionClipping<T> {
    fn wrap_action(&self, _env: &E, action: &mut <E as Env>::TensorLike) {
        action.as_slice_mut().unwrap().iter_mut().for_each(|a| {
            if *a < self.min {
                *a = self.min;
            } else if *a > self.max {
                *a = self.max;
            }
        });
    }
}
