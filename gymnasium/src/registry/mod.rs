// use crate::env::Env;
// use std::collections::HashMap;
// use std::ops::{Deref, DerefMut};

#[repr(transparent)]
pub struct Registry {
    // (HashMap<String, Box<dyn Fn() -> Box<dyn Env>>>)
}

// impl Deref for Registry {
//     type Target = HashMap<String, Box<dyn Fn() -> Box<dyn Env>>>;

//     fn deref(&self) -> &Self::Target {
//         &self.inner
//     }
// }

// impl DerefMut for Registry {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.inner
//     }
// }

// impl Default for Registry {
//     fn default() -> Self {
//         Self {
//             inner: HashMap::new(),
//         }
//     }
// }

// impl Registry   {
//     pub fn register<T: Into<String>>(&mut self, name: T, env: Box<dyn Fn() -> Box<dyn Env>>) {
//         self.inner.insert(name.into(), env);
//     }
// }
