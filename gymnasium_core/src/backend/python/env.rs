use pyo3::types::{PyAnyMethods, PyDictMethods};
use sys::core::EnvMethods;

use crate::{
    sys, DynSpace, Env, EnvConfig, RenderMode, RenderOutput, ResetReturn, Result, StepReturn,
};

pub struct PythonEnv<T: ::numpy::Element> {
    cfg: PythonEnvConfig,
    env: pyo3::Py<sys::Env>,
    dtype: std::marker::PhantomData<T>,
}

impl<T: ::numpy::Element + 'static> Env for PythonEnv<T> {
    type DType = T;
    type TensorLike = pyo3::Py<numpy::PyArray<T, numpy::IxDyn>>;
    type Config = PythonEnvConfig;
    type ActionSpace = DynSpace<Self::DType, Self::TensorLike>;
    type ObservationSpace = DynSpace<Self::DType, Self::TensorLike>;
    type RewardType = f32;
    type InfoType = pyo3::Py<pyo3::types::PyDict>;

    fn new(cfg: Self::Config) -> Result<Self>
    where
        Self: Sized,
    {
        pyo3::prepare_freethreaded_python();
        let env = pyo3::Python::with_gil(|py| {
            let kwargs = pyo3::types::PyDict::new_bound(py);
            kwargs.set_item("render_mode", cfg.render_mode.to_string())?;

            let env = sys::make(py, &cfg.id, None, None, None, None, Some(kwargs))?;
            let env = env.unbind();

            Result::Ok(env)
        })?;

        Ok(PythonEnv {
            cfg,
            env,
            dtype: std::marker::PhantomData,
        })
    }

    fn step(&mut self, action: Self::TensorLike) -> StepReturn<Self>
    where
        Self: Sized,
    {
        pyo3::Python::with_gil(|py| {
            let env = self.env.bind(py);
            let (observation, reward, terminated, truncated, info) = env.step(action).unwrap();

            let observation: pyo3::Bound<numpy::PyArray<T, numpy::IxDyn>> =
                observation.downcast_into().unwrap();
            let observation = observation.unbind();

            let reward = reward.extract().unwrap();

            let info = info.unbind();

            StepReturn {
                observation,
                reward,
                terminated,
                truncated,
                info,
            }
        })
    }

    fn reset(&mut self) -> ResetReturn<Self>
    where
        Self: Sized,
    {
        pyo3::Python::with_gil(|py| {
            let env = self.env.bind(py);
            let (observation, info) = env.reset(None, None).unwrap();

            let observation: pyo3::Bound<numpy::PyArray<T, numpy::IxDyn>> =
                observation.downcast_into().unwrap();
            let observation = observation.unbind();

            let info = info.unbind();

            ResetReturn { observation, info }
        })
    }

    fn render(&mut self) -> Box<dyn RenderOutput> {
        match self.cfg.render_mode {
            RenderMode::Image => pyo3::Python::with_gil(|py| {
                let env = self.env.bind(py);
                let render_output = env.render().unwrap();

                let render_output: pyo3::Bound<numpy::PyArray<T, numpy::IxDyn>> =
                    render_output.downcast_into().unwrap();
                let render_output = render_output.unbind();

                Box::new(render_output)
            }),
            RenderMode::Text => pyo3::Python::with_gil(|py| {
                let env = self.env.bind(py);
                let render_output = env.render().unwrap();

                let render_output: String = render_output.extract().unwrap();

                Box::new(render_output)
            }),
            _ => pyo3::Python::with_gil(|py| {
                let env = self.env.bind(py);
                let _render_output = env.render().unwrap();

                Box::new(())
            }),
        }
    }

    fn close(&mut self) {
        pyo3::Python::with_gil(|py| {
            let env = self.env.bind(py);
            env.close().unwrap();
        });
    }
}

pub struct PythonEnvConfig {
    pub id: String,
    pub seed: Option<u64>,
    pub render_mode: RenderMode,
}

impl EnvConfig for PythonEnvConfig {
    /// The seed for the pseudo-random number generator of the environment.
    fn seed(&self) -> Option<u64> {
        self.seed
    }

    /// The render mode of the environment.
    fn render_mode(&self) -> RenderMode {
        self.render_mode.clone()
    }
}
