use super::{DType, Result, TensorLike};

impl<T> TensorLike<T> for ndarray::Array<T, ndarray::IxDyn>
where
    T: DType,
{
    fn shape(&self) -> Vec<usize> {
        self.shape().to_vec()
    }

    fn from_vec(data: Vec<T>) -> Self {
        let shape = vec![data.len()];
        Self::from_shape_vec(shape, data).unwrap()
    }

    fn from_shape_vec(shape: &[usize], data: Vec<T>) -> Result<Self>
    where
        Self: Sized,
    {
        let shape = shape.to_vec();
        let data = data.into_iter().collect::<Vec<_>>();
        Ok(ndarray::Array::from_shape_vec(shape, data).unwrap())
    }

    fn into_vec(self) -> Vec<T> {
        self.into_raw_vec()
    }

    fn as_slice(&self) -> Option<&[T]> {
        self.as_slice()
    }

    fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        self.as_slice_mut()
    }
}
