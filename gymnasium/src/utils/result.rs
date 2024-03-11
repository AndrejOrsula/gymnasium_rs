/// Result wrapper for `GymnasiumError`.
pub type GymnasiumResult<T> = std::result::Result<T, crate::GymnasiumError>;

/// Crate-local alias for `GymnasiumResult`.
pub(crate) type Result<T> = GymnasiumResult<T>;
