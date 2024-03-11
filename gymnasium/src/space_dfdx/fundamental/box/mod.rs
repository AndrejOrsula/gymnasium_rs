use super::Space;

pub mod box_enum;
pub mod identical;
pub mod independent;

pub use box_enum::BoxSpace;
pub use identical::BoxSpaceIdentical;
pub use independent::BoxSpaceIndependent;
