use super::{BoxedSpace, Space};
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

#[repr(transparent)]
pub struct HashMapSpace(HashMap<String, BoxedSpace>);

impl Deref for HashMapSpace {
    type Target = HashMap<String, BoxedSpace>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for HashMapSpace {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl HashMapSpace {
    pub fn new() -> Self {
        todo!()
    }
}

impl Space for HashMapSpace {}
