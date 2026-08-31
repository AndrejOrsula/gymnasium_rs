use crate::{
    backend::{DType, TensorLike},
    space::SpaceSampleUniform,
    Space,
};

/// Dynamic wrapper around [`Space`] trait object to allow for runtime polymorphism.
pub struct DynSpace<A: DType, T: TensorLike<A>>(pub Box<dyn Space<A, T>>);

impl<A: DType, T: TensorLike<A>> Space<A, T> for DynSpace<A, T> {
    fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    fn contains(&self, value: &T) -> bool {
        self.0.contains(value)
    }
}

pub struct DynSpaceSampleUniform<A: DType, T: TensorLike<A>>(pub Box<dyn SpaceSampleUniform<A, T>>);

impl<A: DType, T: TensorLike<A>> Space<A, T> for DynSpaceSampleUniform<A, T> {
    fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    fn contains(&self, value: &T) -> bool {
        self.0.contains(value)
    }
}

impl<A: DType, T: TensorLike<A>> SpaceSampleUniform<A, T> for DynSpaceSampleUniform<A, T> {
    fn sample(&self, rng: &mut rand::rngs::SmallRng) -> T {
        self.0.sample(rng)
    }
}
