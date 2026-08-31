use crate::env::{Env, RenderOutput, ResetReturn, StepReturn};

pub mod clipping;
pub mod time_limit;

/// Abstraction for wrapping [`Env`] instances with additional functionality.
pub trait Wrapper<E: Env> {
    /// In-place modification of the input [`Env::ActionSpace`] used by [`Env::step()`].
    /// This wrapper is applied before [`Wrapper::pre_step()`].
    fn wrap_action(&self, _env: &E, _action: &mut E::TensorLike) {}

    /// Custom call before [`Env::step()`].
    fn pre_step(&mut self, _env: &E) {}

    /// Custom call after [`Env::step()`].
    fn post_step(&mut self, _env: &E) {}

    /// In-place modification of the entire [`StepReturn`] returned by [`Env::step()`].
    /// This wrapper is applied after [`Wrapper::post_step()`].
    fn wrap_step_return(&self, _env: &E, _step_return: &mut StepReturn<E>) {}

    /// Custom call before [`Env::reset()`].
    fn pre_reset(&mut self, _env: &E) {}

    /// Custom call after [`Env::reset()`].
    fn post_reset(&mut self, _env: &E) {}

    /// In-place modification of the entire [`ResetReturn`] returned by [`Env::reset()`].
    /// This wrapper is applied after [`Wrapper::post_reset()`].
    fn wrap_reset_return(&self, _env: &E, _reset_return: &mut ResetReturn<E>) {}

    /// In-place modification of the output observations contained in [`StepReturn`] and [`ResetReturn`].
    /// This wrapper is applied both after [`Wrapper::wrap_step_return()`] and [`Wrapper::wrap_reset_return()`].
    fn wrap_observation(&self, _env: &E, _observation: &mut E::TensorLike) {}

    /// In-place modification of the output rewards contained in [`StepReturn`].
    /// This wrapper is applied after [`Wrapper::wrap_step_return()`].
    fn wrap_reward(&self, _env: &E, _reward: &mut E::RewardType) {}

    /// In-place modification of the output information contained in [`StepReturn`] and [`ResetReturn`].
    /// This wrapper is applied both after [`Wrapper::wrap_step_return()`] and [`Wrapper::wrap_reset_return()`].
    fn wrap_info(&self, _env: &E, _info: &mut E::InfoType) {}

    /// Custom call before [`Env::render()`].
    fn pre_render(&mut self, _env: &E) {}

    /// Custom call after [`Env::render()`].
    fn post_render(&mut self, _env: &E) {}

    /// In-place modification of the output render data.
    fn wrap_render_output(&self, _env: &E, _render_output: &mut Box<dyn RenderOutput>) {}

    /// Custom call before [`Env::close()`].
    fn pre_close(&mut self, _env: &E) {}

    /// Custom call after [`Env::close()`].
    fn post_close(&mut self, _env: &E) {}
}

/// Implementation of a wrapped [`Env`] with ordered sequence of [`Wrapper`] instances.
pub struct WrappedEnv<E: Env> {
    /// Inner [`Env`] instance.
    pub env: E,
    /// Ordered collection of [`Wrapper`] instances that are applied in sequence.
    pub wrappers: Vec<Box<dyn Wrapper<E>>>,
}

impl<E: Env> WrappedEnv<E> {
    /// Create a new [`WrappedEnv`] around the given [`Env`] instance. The [`WrappedEnv`] is
    /// initialized with no [`Wrapper`] instances.
    ///
    /// # Arguments
    ///
    /// * `env` - The [`Env`] instance to wrap.
    ///
    /// # Returns
    ///
    /// * The newly created [`WrappedEnv`] instance.
    pub fn new(env: E) -> Self {
        Self {
            env,
            wrappers: Vec::new(),
        }
    }

    /// Add a single [`Wrapper`] instance to the end of the sequence of wrappers.
    pub fn add_wrapper(&mut self, wrapper: impl Wrapper<E> + 'static) {
        self.wrappers.push(Box::new(wrapper));
    }

    /// Add multiple [`Wrapper`] instances to the end of the sequence of wrappers.
    pub fn add_wrappers(&mut self, wrappers: Vec<impl Wrapper<E> + 'static>) {
        self.wrappers.extend(
            wrappers
                .into_iter()
                .map(|wrapper| Box::new(wrapper) as Box<dyn Wrapper<E>>),
        );
    }

    /// Add a single [`Wrapper`] instance to the end of the sequence of wrappers. Chainable.
    pub fn with_wrapper(mut self, wrapper: impl Wrapper<E> + 'static) -> Self {
        self.add_wrapper(wrapper);
        self
    }

    /// Add multiple [`Wrapper`] instances to the end of the sequence of wrappers. Chainable.
    pub fn with_wrappers(mut self, wrappers: Vec<impl Wrapper<E> + 'static>) -> Self {
        self.add_wrappers(wrappers);
        self
    }

    /// Wrapper for [`Env::step()`] that applies all [`Wrapper`] instances in sequence.
    pub fn step(&mut self, mut action: E::TensorLike) -> StepReturn<E> {
        for wrapper in &mut self.wrappers {
            wrapper.wrap_action(&self.env, &mut action);
            wrapper.pre_step(&self.env);
        }

        let mut step_return = self.env.step(action);

        for wrapper in &mut self.wrappers {
            wrapper.post_step(&self.env);
            wrapper.wrap_step_return(&self.env, &mut step_return);
            wrapper.wrap_observation(&self.env, &mut step_return.observation);
            wrapper.wrap_reward(&self.env, &mut step_return.reward);
            wrapper.wrap_info(&self.env, &mut step_return.info);
        }

        step_return
    }

    /// Wrapper for [`Env::reset()`] that applies all [`Wrapper`] instances in sequence.
    pub fn reset(&mut self) -> ResetReturn<E> {
        for wrapper in &mut self.wrappers {
            wrapper.pre_reset(&self.env);
        }

        let mut reset_return = self.env.reset();

        for wrapper in &mut self.wrappers {
            wrapper.post_reset(&self.env);
            wrapper.wrap_reset_return(&self.env, &mut reset_return);
            wrapper.wrap_observation(&self.env, &mut reset_return.observation);
            wrapper.wrap_info(&self.env, &mut reset_return.info);
        }

        reset_return
    }

    /// Wrapper for [`Env::render()`] that applies all [`Wrapper`] instances in sequence.
    pub fn render(&mut self) -> Box<dyn RenderOutput> {
        for wrapper in &mut self.wrappers {
            wrapper.pre_render(&self.env);
        }

        let mut render_output = self.env.render();

        for wrapper in &mut self.wrappers {
            wrapper.post_render(&self.env);
            wrapper.wrap_render_output(&self.env, &mut render_output);
        }

        render_output
    }

    /// Wrapper for [`Env::close()`] that applies all [`Wrapper`] instances in sequence.
    pub fn close(&mut self) {
        for wrapper in &mut self.wrappers {
            wrapper.pre_close(&self.env);
        }

        self.env.close();

        for wrapper in &mut self.wrappers {
            wrapper.post_close(&self.env);
        }
    }
}
