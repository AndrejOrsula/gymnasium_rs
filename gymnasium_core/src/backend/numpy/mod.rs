use super::{DType, Result, TensorLike};
use crate::RenderOutput;

impl<T> TensorLike<T> for pyo3::Bound<'_, numpy::PyArray<T, numpy::IxDyn>>
where
    T: DType,
{
    fn shape(&self) -> Vec<usize> {
        todo!()
    }

    fn from_vec(data: Vec<T>) -> Self {
        let shape = vec![data.len()];
        Self::from_shape_vec(&shape, data).unwrap()
    }

    fn from_shape_vec(shape: &[usize], data: Vec<T>) -> Result<Self>
    where
        Self: Sized,
    {
        let _shape = shape.to_vec();
        let _data = data.into_iter().collect::<Vec<_>>();
        todo!()
    }

    fn into_vec(self) -> Vec<T> {
        todo!()
    }

    fn as_slice(&self) -> Option<&[T]> {
        todo!()
    }

    fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        todo!()
    }
}

impl<T> TensorLike<T> for pyo3::Py<numpy::PyArray<T, numpy::IxDyn>>
where
    T: DType,
{
    fn shape(&self) -> Vec<usize> {
        todo!()
    }

    fn from_vec(data: Vec<T>) -> Self {
        let shape = vec![data.len()];
        Self::from_shape_vec(&shape, data).unwrap()
    }

    fn from_shape_vec(shape: &[usize], data: Vec<T>) -> Result<Self>
    where
        Self: Sized,
    {
        let _shape = shape.to_vec();
        let _data = data.into_iter().collect::<Vec<_>>();
        todo!()
    }

    fn into_vec(self) -> Vec<T> {
        todo!()
    }

    fn as_slice(&self) -> Option<&[T]> {
        todo!()
    }

    fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        todo!()
    }
}

impl<T> RenderOutput for pyo3::Py<numpy::PyArray<T, numpy::IxDyn>> where T: DType {}
