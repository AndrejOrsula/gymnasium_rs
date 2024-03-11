use super::Space;

pub mod r#box;
pub mod discrete;
pub mod multi_binary;
pub mod multi_discrete;
pub mod text;

pub use discrete::DiscreteSpace;
pub use multi_binary::MultiBinarySpace;
pub use multi_discrete::MultiDiscreteSpace;
pub use r#box::{BoxSpace, BoxSpaceIdentical, BoxSpaceIndependent};
pub use text::TextSpace;
