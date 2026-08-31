use super::Device;

/// Dummy CPU device.
pub type DummyDevice = ();

impl Device for DummyDevice {}
