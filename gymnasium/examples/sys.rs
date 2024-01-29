use gymnasium::sys as gymnasium;

pub fn main() -> pyo3::PyResult<()> {
    pyo3::Python::with_gil(|py| {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("render_mode", "human").unwrap();
        let env_id = pyo3::types::PyString::new(py, "LunarLander-v2");
        let env = gymnasium::make(py, env_id, None, None, None, None, kwargs).unwrap();
        let reset_ret = env.reset(py, None, None)?;
        let (_observation, _info) = reset_ret;
        for _ in 0..1000 {
            let action_space = env.action_space(py)?;
            let action = action_space.sample(py, None)?;
            let step_ret = env.step(py, action)?;
            let (_observation, _reward, terminated, truncated, _info) = step_ret;
            if terminated || truncated {
                let reset_ret = env.reset(py, None, None)?;
                let (_observation, _info) = reset_ret;
            }
        }
        env.close(py)?;
        Ok(())
    })
}
