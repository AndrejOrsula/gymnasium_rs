use super::Env;

/// The return type of [`Env::step()`].
pub struct StepReturn<E: Env> {
    /// The observation of the environment after taking an action.
    pub observation: E::TensorLike,
    /// The reward after taking an action.
    pub reward: E::RewardType,
    /// Indication that the agent has reached a terminal state after taking an action, as defined by the task formulation.
    /// If true, [`Env::reset()`] must be called to restart the environment.
    pub terminated: bool,
    /// Indication that the agent has reached a truncation condition after taking an action, which is outside the scope of the task formulation.
    /// If true, [`Env::reset()`] must be called to restart the environment.
    pub truncated: bool,
    /// Additional information from the environment after taking an action.
    pub info: E::InfoType,
}

/// [StepReturn] represented as a tuple.
type StepReturnTuple<E> = (
    <E as Env>::TensorLike,
    <E as Env>::RewardType,
    bool,
    bool,
    <E as Env>::InfoType,
);

impl<E: Env> From<StepReturn<E>> for StepReturnTuple<E> {
    fn from(value: StepReturn<E>) -> Self {
        (
            value.observation,
            value.reward,
            value.terminated,
            value.truncated,
            value.info,
        )
    }
}

impl<E: Env> From<StepReturnTuple<E>> for StepReturn<E> {
    fn from(value: StepReturnTuple<E>) -> Self {
        Self {
            observation: value.0,
            reward: value.1,
            terminated: value.2,
            truncated: value.3,
            info: value.4,
        }
    }
}

/// The return type of [`Env::reset()`].
pub struct ResetReturn<E: Env> {
    /// The initial observation of the environment after the reset.
    pub observation: E::TensorLike,
    /// Additional initial information from the environment after the reset.
    pub info: E::InfoType,
}

/// [ResetReturn] represented as a tuple.
type ResetReturnTuple<E> = (<E as Env>::TensorLike, <E as Env>::InfoType);

impl<E: Env> From<ResetReturn<E>> for ResetReturnTuple<E> {
    fn from(value: ResetReturn<E>) -> Self {
        (value.observation, value.info)
    }
}

impl<E: Env> From<ResetReturnTuple<E>> for ResetReturn<E> {
    fn from(value: ResetReturnTuple<E>) -> Self {
        Self {
            observation: value.0,
            info: value.1,
        }
    }
}
