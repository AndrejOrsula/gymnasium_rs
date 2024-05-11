use crate::{
    backend::{DType, TensorLike},
    Space,
};

/// Dynamic wrapper around [`Space`] trait object to allow for runtime polymorphism.
pub struct DynSpace<A: DType, T: TensorLike<A>>(Box<dyn Space<A, T>>);

impl<A: DType, T: TensorLike<A>> Space<A, T> for DynSpace<A, T> {
    fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    fn contains(&self, value: &T) -> bool {
        self.0.contains(value)
    }
}
