use crate::{DType, RenderOutput, Result, TensorLike};
use ::std::borrow::Cow;
use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use pyo3::types::PyAnyMethods;

impl<T> TensorLike<T> for pyo3::Py<numpy::PyArray<T, numpy::IxDyn>>
where
    T: DType + numpy::Element,
{
    fn shape(&self) -> Cow<[usize]> {
        pyo3::Python::with_gil(|py| {
            Cow::Owned(PyUntypedArrayMethods::shape(self.bind(py)).to_owned())
        })
    }

    fn from_shape_vec(shape: &[usize], data: Vec<T>) -> Result<Self>
    where
        Self: Sized,
        T: Clone,
    {
        pyo3::Python::with_gil(|py| {
            let array = numpy::PyArray::from_vec_bound(py, data);
            let array = PyArrayMethods::reshape(&array, shape).unwrap();
            Ok(array.unbind())
        })
    }

    fn from_vec(data: Vec<T>) -> Self
    where
        Self: Sized,
    {
        pyo3::Python::with_gil(|py| {
            let array = numpy::PyArray::from_vec_bound(py, data);
            let array = PyArrayMethods::to_dyn(&array).to_owned();
            array.unbind()
        })
    }

    fn is_contiguous(&self) -> bool {
        pyo3::Python::with_gil(|py| PyUntypedArrayMethods::is_contiguous(self.bind(py)))
    }

    fn ensure_contiguous(&mut self) {
        pyo3::Python::with_gil(|py| {
            let slf = self.bind(py);
            if !PyUntypedArrayMethods::is_contiguous(slf) {
                let contiguous_slf: pyo3::Bound<numpy::PyArray<T, numpy::IxDyn>> =
                    numpy::get_array_module(py)
                        .unwrap()
                        .call_method1("ascontiguousarray", (slf,))
                        .unwrap()
                        .downcast_into()
                        .unwrap();
                *self = contiguous_slf.unbind();
            }
        });
    }

    /// Returns an immutable view of the internal data as a slice.
    ///
    /// # Safety
    ///
    /// Calling this method is undefined behaviour if the underlying array
    /// is aliased mutably by other instances of `PyArray`
    /// or concurrently modified by Python or other native code.
    ///
    /// Please consider the safe alternative [`PyReadonlyArray::as_slice`].
    fn as_slice(&self) -> Option<&[T]> {
        pyo3::Python::with_gil(|py| {
            let slf = self.bind(py);
            if PyUntypedArrayMethods::is_contiguous(slf) {
                Some(unsafe {
                    std::slice::from_raw_parts(
                        PyArrayMethods::data(slf),
                        PyUntypedArrayMethods::len(slf),
                    )
                })
            } else {
                None
            }
        })
    }

    /// Returns a mutable view of the internal data as a slice.
    ///
    /// # Safety
    ///
    /// Calling this method is undefined behaviour if the underlying array
    /// is aliased immutably or mutably by other instances of [`PyArray`]
    /// or concurrently modified by Python or other native code.
    ///
    /// Please consider the safe alternative [`PyReadwriteArray::as_slice_mut`].
    fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        pyo3::Python::with_gil(|py| {
            let slf = self.bind(py);
            if PyUntypedArrayMethods::is_contiguous(slf) {
                Some(unsafe {
                    std::slice::from_raw_parts_mut(
                        PyArrayMethods::data(slf),
                        PyUntypedArrayMethods::len(slf),
                    )
                })
            } else {
                None
            }
        })
    }
}

impl<T> RenderOutput for pyo3::Py<numpy::PyArray<T, numpy::IxDyn>> where T: DType {}
