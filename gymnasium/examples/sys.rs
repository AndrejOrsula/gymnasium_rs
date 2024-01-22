pub use gymnasium::sys;

use itertools::Itertools;
use pyo3::{prelude::*, types::PyString};

pub fn main() -> pyo3::PyResult<()> {
    let env: pyo3::Py<sys::Env> = pyo3::Python::with_gil(|py| {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("render_mode", "human").unwrap();
        let env_id = PyString::new(py, "LunarLander-v2");
        let env: &sys::Env = sys::make(py, env_id, None, None, None, None, kwargs)
            .unwrap()
            .extract()
            .unwrap();

        env.into()
    });

    pyo3::Python::with_gil(|py| {
        let env = env.as_ref(py);
        let reset_ret = env.reset(py, None, None)?;
        let (_observation, _info) = reset_ret.iter().collect_tuple().unwrap();

        for _ in 0..1000 {
            let action_space = env.action_space(py)?;
            let action = action_space.sample(py, None)?;
            let step_ret = env.step(py, action)?;
            let (_observation, _reward, terminated, truncated, _info) =
                step_ret.iter().collect_tuple().unwrap();
            if terminated.is_true()? || truncated.is_true()? {
                let reset_ret = env.reset(py, None, None)?;
                let (_observation, _info) = reset_ret.iter().collect_tuple().unwrap();
            }
        }

        env.close(py)?;

        Ok(())
    })

    // pyo3::Python::with_gil(|py| {
    //     let kwargs = pyo3::types::PyDict::new(py);
    //     kwargs.set_item("render_mode", "human")?;
    //     let env_id = PyString::new(py, "LunarLander-v2");
    //     let env: &sys::Wrapper =
    //         sys::make(py, env_id, None, None, None, None, kwargs)?.extract()?;

    //     let reset_ret = env.reset(py, None, None)?;
    //     let (observation, info) = reset_ret.iter().collect_tuple().unwrap();

    //     for _ in 0..1000 {
    //         let action_space: &sys::Space = env.action_space(py)?.extract()?;
    //         let action = action_space.sample(py, None)?;
    //         let step_ret = env.step(py, action)?;
    //         let (observation, reward, terminated, truncated, info) =
    //             step_ret.iter().collect_tuple().unwrap();
    //         if terminated.is_true()? || truncated.is_true()? {
    //             let reset_ret = env.reset(py, None, None)?;
    //             let (observation, info) = reset_ret.iter().collect_tuple().unwrap();
    //         }
    //     }

    //     env.close(py)?;

    //     Ok(())
    // })
}
