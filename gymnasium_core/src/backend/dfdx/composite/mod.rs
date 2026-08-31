use super::Space;

pub mod hashmap;
pub mod vec;

pub use hashmap::HashMapSpace;
pub use vec::VecSpace;

// type BoxedSpace = Box<dyn Space>;
