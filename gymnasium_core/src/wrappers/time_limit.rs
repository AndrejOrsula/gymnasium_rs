use crate::{Env, Wrapper};

/// Wrapper that always truncates the episode after a fixed number of [`Env::step`] calls.
pub struct TimeLimit {
    /// Maximum number of steps before truncating the episode.
    max_steps: u64,
    /// Number of steps elapsed in the current episode.
    current_step: u64,
}

impl TimeLimit {
    /// Creates a new [`TimeLimit`].
    ///
    /// # Arguments
    ///
    /// * `max_steps` - The maximum number of steps before truncating the episode.
    ///
    /// # Returns
    ///
    /// * The newly created [`TimeLimit`] instance.
    #[must_use]
    pub fn new(max_steps: u64) -> Self {
        Self {
            max_steps,
            current_step: 0,
        }
    }
}

impl<E: Env> Wrapper<E> for TimeLimit {
    fn post_step(&mut self, _env: &E) {
        self.current_step += 1;
    }

    fn wrap_step_return(&self, _env: &E, step_return: &mut crate::env::StepReturn<E>) {
        if self.current_step >= self.max_steps {
            step_return.truncated = true;
        }
    }

    fn post_reset(&mut self, _env: &E) {
        self.current_step = 0;
    }
}
