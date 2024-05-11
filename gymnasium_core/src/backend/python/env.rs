use crate::{
    space::BoxSpace, sys, DynSpaceSampleUniform, Env, EnvConfig, RenderMode, RenderOutput,
    ResetReturn, Result, StepReturn,
};
use pyo3::types::{IntoPyDict, PyAnyMethods};
use sys::{core::EnvMethods, spaces::r#box::BoxMethods, EnvMethodsManual};

pub struct PythonEnv<T: ::numpy::Element> {
    pub cfg: PythonEnvConfig,
    pub env: pyo3::Py<sys::Env>,
    pub action_space: DynSpaceSampleUniform<T, pyo3::Py<numpy::PyArray<T, numpy::IxDyn>>>,
    pub observation_space: DynSpaceSampleUniform<T, pyo3::Py<numpy::PyArray<T, numpy::IxDyn>>>,
    pub dtype: std::marker::PhantomData<T>,
}

impl<
        T: Copy
            + num_traits::FromPrimitive
            + numpy::Element
            + rand::distributions::uniform::SampleUniform
            + std::cmp::PartialOrd
            + std::fmt::Debug
            + 'static,
    > Env for PythonEnv<T>
where
    <T as rand::distributions::uniform::SampleUniform>::Sampler: Clone,
{
    type DType = T;
    type TensorLike = pyo3::Py<numpy::PyArray<T, numpy::IxDyn>>;
    type Config = PythonEnvConfig;
    type ActionSpace = DynSpaceSampleUniform<Self::DType, Self::TensorLike>;
    type ObservationSpace = DynSpaceSampleUniform<Self::DType, Self::TensorLike>;
    type RewardType = f32;
    type InfoType = pyo3::Py<pyo3::types::PyDict>;

    fn action_space(&self) -> &Self::ActionSpace {
        &self.action_space
    }

    fn observation_space(&self) -> &Self::ObservationSpace {
        &self.observation_space
    }

    fn new(cfg: Self::Config) -> Result<Self>
    where
        Self: Sized,
    {
        pyo3::prepare_freethreaded_python();
        pyo3::Python::with_gil(|py| {
            let env = sys::make(
                py,
                &cfg.env_id,
                None,
                None,
                None,
                None,
                Some([("render_mode", cfg.render_mode.to_string_py())].into_py_dict_bound(py)),
            )?;

            let action_space = Self::extract_space(&env.action_space()?)?;
            let observation_space = Self::extract_space(&env.observation_space()?)?;

            Result::Ok(PythonEnv {
                cfg,
                env: env.unbind(),
                action_space,
                observation_space,
                dtype: std::marker::PhantomData,
            })
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

impl<T: numpy::Element> PythonEnv<T> {
    fn extract_space<'py>(
        space: &'py pyo3::Bound<'py, pyo3::PyAny>,
    ) -> Result<DynSpaceSampleUniform<T, pyo3::Py<numpy::PyArray<T, numpy::IxDyn>>>>
    where
        T: Copy
            + num_traits::FromPrimitive
            + rand::distributions::uniform::SampleUniform
            + std::cmp::PartialOrd
            + std::fmt::Debug
            + 'static,
        <T as rand::distributions::uniform::SampleUniform>::Sampler: Clone,
    {
        // Parse the name of the space
        // TODO: Make more robust | support other spaces
        let space_name = space.to_string();
        let space_name = space_name
            .split_once('(')
            .map(|x| x.0)
            .unwrap_or(&space_name);

        Result::Ok(match space_name {
            "Box" => {
                let space: pyo3::Bound<sys::spaces::Box> = space.extract()?;

                let shape = space
                    .shape()?
                    .into_iter()
                    .map(|x| x as usize)
                    .collect::<Vec<_>>();

                // TODO: Get the actual lower & upper bounds from the box space
                // let low = space.getattr("low")?;
                let low = T::from_f32(-1.0).unwrap();
                let high = T::from_f32(1.0).unwrap();

                DynSpaceSampleUniform(Box::new(BoxSpace::new_identical_bounds(shape, low, high)?))
            }
            // "Discrete" => {
            //     let space: pyo3::Bound<sys::spaces::Discrete> = space.extract()?;

            //     let n: i64 = space.getattr("n")?.extract()?;
            //     let start: i64 = space.getattr("start")?.extract()?;

            //     DynSpaceSampleUniform(Box::new(DiscreteSpace::new_at(start as usize, n as usize)?))
            // }
            _ => {
                unimplemented!("Unsupported Python space: {space_name}")
            }
        })
    }
}

pub struct PythonEnvConfig {
    pub env_id: String,
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
