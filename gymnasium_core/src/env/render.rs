use std::{fmt::Display, str::FromStr};

/// Enum that represents the possible rendering modes.
#[derive(Debug, Clone, PartialEq)]
pub enum RenderMode {
    /// No rendering. Nothing is expected to be rendered and the output is ignored.
    None,
    /// Render to a human viewer. The environment should generally be rendered in
    /// an interactive window and the output is ignored.
    Human,
    /// Render to image. The environment should render to an image and return it as
    /// the output.
    Image,
    /// Render to text. The environment should produce a string and return it as the
    /// output.
    Text,
    /// Render to any other custom environment-specific format.
    Other(String),
}

impl Default for RenderMode {
    fn default() -> Self {
        Self::None
    }
}

impl FromStr for RenderMode {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "" | "none" => Self::None,
            "human" => Self::Human,
            "image" => Self::Image,
            "text" => Self::Text,
            _ => Self::Other(s.to_string()),
        })
    }
}

impl Display for RenderMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Other(s) => write!(f, "{s}"),
            _ => write!(f, "{:?}", self),
        }
    }
}

/// Trait that abstracts the rendering of the environment output.
pub trait RenderOutput {}

/// Implementation for [`RenderMode::None`] and [`RenderMode::Human`] rendering modes with the `()` output.
impl RenderOutput for () {}

/// Implementation for [`RenderMode::Text`] rendering mode with the `String` output.
impl RenderOutput for String {}
