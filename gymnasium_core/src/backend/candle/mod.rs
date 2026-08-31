use super::{DType, Result, TensorLike};
use ::std::borrow::Cow;

impl<T> TensorLike<T> for candle_core::Tensor
where
    T: DType + candle_core::WithDType,
{
    fn shape(&self) -> Cow<[usize]> {
        Cow::Borrowed(self.shape().dims())
    }

    fn from_shape_vec(shape: &[usize], data: Vec<T>) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(candle_core::Tensor::from_vec(
            data,
            shape,
            &candle_core::Device::cuda_if_available(0).unwrap(),
        )?)
    }

    fn is_contiguous(&self) -> bool {
        candle_core::Tensor::is_contiguous(self)
    }

    fn ensure_contiguous(&mut self) {
        if !candle_core::Tensor::is_contiguous(self) {
            *self = candle_core::Tensor::contiguous(self).unwrap();
        }
    }

    fn as_slice(&self) -> Option<&[T]> {
        // TODO: Fix
        unimplemented!("Unavailable, might need to change the trait API")
    }

    fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        // TODO: Fix
        unimplemented!("Unavailable, might need to change the trait API")
    }
}
