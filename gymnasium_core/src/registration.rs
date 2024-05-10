use crate::{Env, GymnasiumError, Result};
use std::any::Any;

/// Register a new [`Env`] with the given unique identifier `id`.
///
/// # Generic
///
/// * `E` - The type of [`Env`] to register.
///
/// # Arguments
///
/// * `id` - The unique identifier for the environment.
///
/// # Errors
///
/// * [`GymnasiumError::RegistrationError`] - If an environment is already registered with the same `id`.
///
/// # Panics
///
/// * If the lock holding the registry is poisoned and cannot be acquired.
pub fn register<E: Env + 'static>(id: &str) -> Result<()> {
    let mut registry = registry()
        .write()
        .expect("Failed to acquire write lock holding the environment registry");
    use std::collections::hash_map::Entry;
    match registry.entry(id.to_string()) {
        Entry::Vacant(entry) => {
            // Create a new constructor for the environment
            let constructor = Box::new(|cfg: Box<dyn Any>| {
                // Downcast the configuration into the concrete type for the environment
                let cfg = *cfg
                .downcast()
                .expect("Failed to downcast `Any` into the concrete `Env::Config` type for the environment with ID '{id}'");

                // Create a new environment instance
                let env = E::new(cfg);

                // Box the result and cast it into `Any`
                Box::new(env) as Box<dyn Any>
            });

            // Insert the constructor into the registry
            entry.insert(Box::new(constructor));
            Ok(())
        }
        Entry::Occupied(_) => Err(GymnasiumError::RegistrationError(format!(
            "Environment with ID '{id}' is already registered"
        ))),
    }
}

/// Create a new [`Env`] instance with the given unique identifier `id` and configuration `config`.
///
/// # Generic
///
/// * `E` - The type of [`Env`] to create.
///
/// # Arguments
///
/// * `id` - The unique identifier for the environment that was registered for `E` via [`register()`].
/// * `cfg` - The configuration specific to the environment `E`.
///
/// # Returns
///
/// * The [`Env`] instance encapsulated in a [`Result`].
///
/// # Errors
///
/// * [`GymnasiumError::ValueError`] - If `id` does not match any environment in the registry.
/// * [`GymnasiumError::TypeError`] - If the registered environment does not match the expected type `E`.
///
/// # Panics
///
/// * If the lock holding the registry is poisoned and cannot be acquired.
pub fn make<E: Env + 'static>(id: &str, cfg: E::Config) -> Result<E> {
    // Box the configuration and cast it into `Any`
    let cfg = Box::new(cfg) as Box<dyn Any>;

    // Retrieve the constructor for the environment
    let registry = registry()
        .read()
        .expect("Failed to acquire read lock holding the environment registry");
    let constructor = if let Some(constructor) = registry.get(id) {
        constructor
    } else {
        return Err(GymnasiumError::ValueError(format!(
            "Environment ID '{id}' not found in registry",
        )));
    };

    // Call the constructor with the configuration
    let output = constructor(cfg);

    // Downcast and dereference the output of the constructor from `Box<dyn Any>` to `Result<E>`
    if let Ok(env) = output.downcast::<Result<E>>() {
        *env
    } else {
        Err(GymnasiumError::TypeError(format!(
            "Failed to downcast `Any` into the concrete `Result<Env>` type for the environment with ID '{id}'"
        )))
    }
}

/// Alias for a dynamically-typed registry of multiple environments ([`Env`]). The registry maps
/// unique string identifiers to constructors of the corresponding environments. Dynamic typing
/// is used to support various environments to have different configuration types.
type Registry =
    std::collections::HashMap<String, Box<dyn Fn(Box<dyn Any>) -> Box<dyn Any> + Send + Sync>>;

/// Accessor for the static mutable registry of environments.
///
/// # Returns
///
/// * Static reference to [`std::sync::RwLock`] holding the [`Registry`].
///
/// # Notes
///
/// * [`std::sync::RwLock`] is used instead here of [`std::sync::Mutex`] to allow multiple threads
/// to concurrently create new [`Env`] instances via [`make()`].
fn registry() -> &'static std::sync::RwLock<Registry> {
    static REGISTRY: std::sync::OnceLock<std::sync::RwLock<Registry>> = std::sync::OnceLock::new();
    REGISTRY.get_or_init(Default::default)
}
