use thiserror::Error;

pub type Result<T> = std::result::Result<T, GymnasiumError>;

#[derive(Error, Debug)]
pub enum GymnasiumError {
    #[error("Invalid space: {0}")]
    InvalidSpace(String),
}
