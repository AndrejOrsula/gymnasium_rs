use crate::Result;

mod common;
mod std;

#[cfg(feature = "candle")]
mod candle;
#[cfg(feature = "ndarray")]
mod ndarray;
// #[cfg(feature = "dfdx")]
// mod dfdx;

pub use common::DummyDevice;

pub trait TensorLike<T>
where
    T: DType,
{
    fn shape(&self) -> Vec<usize>;

    fn from_vec(data: Vec<T>) -> Self;

    fn from_shape_vec(shape: &[usize], data: Vec<T>) -> Result<Self>
    where
        Self: Sized;

    fn into_vec(self) -> Vec<T>;

    fn as_slice(&self) -> Option<&[T]>;

    fn as_slice_mut(&mut self) -> Option<&mut [T]>;
}

pub trait DType {}

pub trait Device {}

pub trait Shape {}

pub trait Rng: rand::Rng {}
impl<R: rand::Rng> Rng for R {}

pub trait Distribution<T>: rand::distributions::Distribution<T> {}
impl<T, D: rand::distributions::Distribution<T>> Distribution<T> for D {}
