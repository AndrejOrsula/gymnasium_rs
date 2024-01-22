use crate::{Env, Space};

impl Env {
    ///Return the :attr:`Env` :attr:`action_space` unless overwritten then the wrapper :attr:`action_space` is used.
    pub fn action_space<'py>(
        &'py self,
        py: ::pyo3::marker::Python<'py>,
    ) -> ::pyo3::PyResult<&'py Space> {
        self.getattr(::pyo3::intern!(py, "action_space"))?.extract()
    }
    ///Setter for the `action_space` attribute
    pub fn set_action_space<'py>(
        &'py self,
        py: ::pyo3::marker::Python<'py>,
        value: &'py Space,
    ) -> ::pyo3::PyResult<()> {
        self.setattr(::pyo3::intern!(py, "action_space"), value)?;
        Ok(())
    }
    ///Return the :attr:`Env` :attr:`observation_space` unless overwritten then the wrapper :attr:`observation_space` is used.
    pub fn observation_space<'py>(
        &'py self,
        py: ::pyo3::marker::Python<'py>,
    ) -> ::pyo3::PyResult<&'py Space> {
        self.getattr(::pyo3::intern!(py, "observation_space"))?
            .extract()
    }
    ///Setter for the `observation_space` attribute
    pub fn set_observation_space<'py>(
        &'py self,
        py: ::pyo3::marker::Python<'py>,
        value: &'py Space,
    ) -> ::pyo3::PyResult<()> {
        self.setattr(::pyo3::intern!(py, "observation_space"), value)?;
        Ok(())
    }
}
