use super::{DType, Result, TensorLike};

impl<T> TensorLike<T> for candle_core::Tensor
where
    T: DType + candle_core::WithDType,
{
    fn shape(&self) -> Vec<usize> {
        self.shape().dims().to_vec()
    }

    fn from_vec(data: Vec<T>) -> Self {
        let shape = vec![data.len()];
        candle_core::Tensor::from_vec(
            data,
            shape,
            &candle_core::Device::cuda_if_available(0).unwrap(),
        )
        .unwrap()
    }

    fn from_shape_vec(shape: &[usize], data: Vec<T>) -> Result<Self>
    where
        Self: Sized,
    {
        let shape = shape.to_vec();
        let data = data.into_iter().collect::<Vec<_>>();
        Ok(candle_core::Tensor::from_vec(
            data,
            shape,
            &candle_core::Device::cuda_if_available(0).unwrap(),
        )
        .unwrap())
    }

    fn into_vec(self) -> Vec<T> {
        self.reshape(((),)).unwrap().to_vec1().unwrap()
    }

    fn as_slice(&self) -> Option<&[T]> {
        todo!("Unavailable, might need to change the trait API")
    }

    fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        todo!("Unavailable, might need to change the trait API")
    }
}
