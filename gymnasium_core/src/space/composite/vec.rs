use super::{BoxedSpace, Space};
use std::ops::{Deref, DerefMut};
use std::vec::Vec;

#[repr(transparent)]
pub struct VecSpace(Vec<BoxedSpace>);

impl Deref for VecSpace {
    type Target = Vec<BoxedSpace>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for VecSpace {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl VecSpace {
    pub fn new() -> Self {
        todo!()
    }
}

impl Space for VecSpace {}
