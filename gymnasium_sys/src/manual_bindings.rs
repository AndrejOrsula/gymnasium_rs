pub trait EnvMethodsManual {
    fn action_space(&self) -> ::pyo3::PyResult<::pyo3::Bound<::pyo3::types::PyAny>>;
    fn set_action_space(
        &self,
        p_value: impl ::pyo3::IntoPy<::pyo3::Py<::pyo3::types::PyAny>>,
    ) -> ::pyo3::PyResult<()>;
    fn observation_space(&self) -> ::pyo3::PyResult<::pyo3::Bound<::pyo3::types::PyAny>>;
    fn set_observation_space(
        &self,
        p_value: impl ::pyo3::IntoPy<::pyo3::Py<::pyo3::types::PyAny>>,
    ) -> ::pyo3::PyResult<()>;
}

impl EnvMethodsManual for pyo3::Bound<'_, crate::Env> {
    fn action_space(&self) -> ::pyo3::PyResult<::pyo3::Bound<::pyo3::types::PyAny>> {
        ::pyo3::types::PyAnyMethods::extract(&::pyo3::types::PyAnyMethods::getattr(
            self.as_any(),
            ::pyo3::intern!(self.py(), "action_space"),
        )?)
    }
    fn set_action_space(
        &self,
        p_value: impl ::pyo3::IntoPy<::pyo3::Py<::pyo3::types::PyAny>>,
    ) -> ::pyo3::PyResult<()> {
        let py = self.py();
        let p_value = ::pyo3::IntoPy::<::pyo3::Py<::pyo3::types::PyAny>>::into_py(p_value, py);
        let p_value = p_value.bind(py);
        ::pyo3::types::PyAnyMethods::setattr(
            self.as_any(),
            ::pyo3::intern!(py, "action_space"),
            p_value,
        )
    }
    fn observation_space(&self) -> ::pyo3::PyResult<::pyo3::Bound<::pyo3::types::PyAny>> {
        ::pyo3::types::PyAnyMethods::extract(&::pyo3::types::PyAnyMethods::getattr(
            self.as_any(),
            ::pyo3::intern!(self.py(), "observation_space"),
        )?)
    }
    fn set_observation_space(
        &self,
        p_value: impl ::pyo3::IntoPy<::pyo3::Py<::pyo3::types::PyAny>>,
    ) -> ::pyo3::PyResult<()> {
        let py = self.py();
        let p_value = ::pyo3::IntoPy::<::pyo3::Py<::pyo3::types::PyAny>>::into_py(p_value, py);
        let p_value = p_value.bind(py);
        ::pyo3::types::PyAnyMethods::setattr(
            self.as_any(),
            ::pyo3::intern!(py, "observation_space"),
            p_value,
        )
    }
}
