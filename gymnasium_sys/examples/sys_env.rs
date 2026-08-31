use gymnasium_sys::{core::EnvMethods, spaces::space::SpaceMethods, EnvMethodsManual, Space};
use pyo3::types::{IntoPyDict, PyAnyMethods};

pub fn main() -> pyo3::PyResult<()> {
    pyo3::prepare_freethreaded_python();
    pyo3::Python::with_gil(|py| {
        let env = gymnasium_sys::make(
            py,
            "LunarLanderContinuous-v2",
            None,
            None,
            None,
            None,
            Some([("render_mode", "human")].into_py_dict_bound(py)),
        )?;

        let (_observation, _info) = env.reset(None, None)?;
        for _ in 0..1000 {
            let action_space = env.action_space()?.downcast_into::<Space>()?;
            let action = action_space.sample(None)?;
            let (_observation, _reward, terminated, truncated, _info) = env.step(action)?;
            if terminated || truncated {
                let (_observation, _info) = env.reset(None, None)?;
            }
        }
        env.close()?;
        Ok(())
    })
}
