use crate::space::Space;
use std::collections::HashMap;
use std::convert::From;

type Info = HashMap<String, ()>;

pub enum RenderMode {
    None,
    Human,
    Image,
    Text,
}

pub struct StepResult<ObservationType: Space, RewardType> {
    observation: ObservationType,
    reward: RewardType,
    terminated: bool,
    truncated: bool,
    info: Info,
}

impl<ObservationType: Space, RewardType> StepResult<ObservationType, RewardType> {
    pub fn new(
        observation: ObservationType,
        reward: RewardType,
        terminated: bool,
        truncated: bool,
        info: Info,
    ) -> Self {
        Self {
            observation,
            reward,
            terminated,
            truncated,
            info,
        }
    }

    pub fn observation(&self) -> &ObservationType {
        &self.observation
    }

    // pub fn reward(&self) -> RewardType {
    //     self.reward
    // }

    pub fn terminated(&self) -> bool {
        self.terminated
    }

    pub fn truncated(&self) -> bool {
        self.truncated
    }

    pub fn info(&self) -> &Info {
        &self.info
    }

    pub fn done(&self) -> bool {
        self.terminated || self.truncated
    }
}

impl<ObservationType: Space, RewardType> From<StepResult<ObservationType, RewardType>>
    for (ObservationType, RewardType, bool, bool, Info)
{
    fn from(step_result: StepResult<ObservationType, RewardType>) -> Self {
        (
            step_result.observation,
            step_result.reward,
            step_result.terminated,
            step_result.truncated,
            step_result.info,
        )
    }
}

pub trait Env<ActionType: Space, ObservationType: Space, RewardType> {
    fn step(&mut self, action: ActionType) -> StepResult<ObservationType, RewardType>;
    fn reset(
        &mut self,
        seed: Option<usize>,
        options: Option<HashMap<String, String>>,
    ) -> (ObservationType, Info);
    // fn render(&self) -> Option<Image>;
    // fn close(&mut self);

    // fn action_space(&self) -> ObservationType;
    // fn observation_space(&self) -> ActionType;
    // fn reward_range(&self) -> Range<RewardType> {
    //     use std::ops::Range;
    //     Range {
    //         start: RewardType::neg_infinity(),
    //         end: RewardType::infinity(),
    //     }
    // }
    // fn metadata(&self) -> Info;
    // fn render_mode(&self) -> RenderMode;

    fn build(config: HashMap<String, String>, seed: Option<usize>) -> Self;
}
