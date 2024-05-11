use gymnasium::{
    space::SpaceSampleUniform, Env, GymnasiumResult, PythonEnv, PythonEnvConfig, RenderMode,
};

pub fn main() -> GymnasiumResult<()> {
    let mut env = PythonEnv::<f32>::new(PythonEnvConfig {
        env_id: "LunarLanderContinuous-v2".to_string(),
        render_mode: RenderMode::Human,
        seed: None,
    })?;

    let mut rng = rand::SeedableRng::from_entropy();

    let _reset_return = env.reset();
    for _ in 0..1000 {
        let action = env.action_space().sample(&mut rng);
        let step_return = env.step(action);
        if step_return.terminated || step_return.truncated {
            let _reset_return = env.reset();
        }
    }
    env.close();

    Ok(())
}
