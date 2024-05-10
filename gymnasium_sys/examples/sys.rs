use gymnasium_sys::{core::EnvMethods, spaces::space::SpaceMethods, EnvMethodsManual, Space};
use pyo3::prelude::*;

pub fn main() -> pyo3::PyResult<()> {
    pyo3::prepare_freethreaded_python();
    pyo3::Python::with_gil(|py| {
        let kwargs = pyo3::types::PyDict::new_bound(py);
        kwargs.set_item("render_mode", "human").unwrap();
        let env_id = pyo3::types::PyString::new_bound(py, "LunarLander-v2");
        let env = gymnasium_sys::make(py, env_id, None, None, None, None, Some(kwargs)).unwrap();
        let reset_ret = env.reset(None, None)?;
        let (_observation, _info) = reset_ret;
        for _ in 0..1000 {
            let action_space: Bound<Space> = env.action_space()?.downcast_into()?;
            let action = action_space.sample(None)?;
            let step_ret = env.step(action)?;
            let (_observation, _reward, terminated, truncated, _info) = step_ret;
            if terminated || truncated {
                let reset_ret = env.reset(None, None)?;
                let (_observation, _info) = reset_ret;
            }
        }
        env.close()?;
        Ok(())
    })
}
