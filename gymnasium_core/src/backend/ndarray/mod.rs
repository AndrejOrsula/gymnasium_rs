use super::{DType, Result, TensorLike};
use ::std::borrow::Cow;

impl<T> TensorLike<T> for ndarray::Array<T, ndarray::IxDyn>
where
    T: DType + Clone,
{
    fn shape(&self) -> Cow<[usize]> {
        Cow::Borrowed(ndarray::Array::shape(self))
    }

    fn from_shape_vec(shape: &[usize], data: Vec<T>) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(ndarray::Array::from_shape_vec(shape, data)?)
    }

    fn from_vec(data: Vec<T>) -> Self
    where
        Self: Sized,
    {
        ndarray::Array::from_vec(data).into_dyn()
    }

    fn is_contiguous(&self) -> bool {
        ndarray::Array::is_standard_layout(self)
    }

    fn ensure_contiguous(&mut self) {
        let standard_layout = ndarray::Array::as_standard_layout(self);
        if standard_layout.is_owned() {
            *self = standard_layout.to_owned();
        }
    }

    fn as_slice(&self) -> Option<&[T]> {
        ndarray::Array::as_slice(self)
    }

    fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        ndarray::Array::as_slice_mut(self)
    }
}
