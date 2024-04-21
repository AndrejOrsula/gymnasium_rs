use crate::Result;

mod common;
mod std;

mod ndarray;
// mod dfdx;

pub use common::DummyDevice;

pub trait TensorLike {
    fn shape(&self) -> Vec<usize>;

    fn from_vec<T>(data: Vec<T>) -> Self
    where
        T: DType;

    fn from_shape_vec<S, T>(shape: S, data: Vec<T>) -> Result<Self>
    where
        Self: Sized,
        S: Shape,
        T: DType;

    fn to_vec<T>(&self) -> Vec<T>;

    fn as_slice<T>(&self) -> &[T];

    fn as_slice_mut<T>(&mut self) -> &mut [T];
}

pub trait DType {}

pub trait Device {}

pub trait Shape {}

pub trait Rng: rand::Rng {}
impl<R: rand::Rng> Rng for R {}

pub trait Distribution<T>: rand::distributions::Distribution<T> {}
impl<T, D: rand::distributions::Distribution<T>> Distribution<T> for D {}
