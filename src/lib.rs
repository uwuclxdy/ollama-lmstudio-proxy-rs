// src/lib.rs - Updated module organization with new consolidated systems

// Core modules
pub mod constants;
pub mod model;
pub mod server;
pub mod utils;
pub mod handlers;
pub mod common;

// Public re-exports for easy access
pub use server::{Config, ProxyServer};
pub use utils::{ProxyError, Logger, validate_config};
pub use model::{ModelInfo, ModelResolver, clean_model_name};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Re-export commonly used constants for external use
pub use constants::{
    // Buffer and streaming limits
    MAX_BUFFER_SIZE,
    MAX_CHUNK_COUNT,
    MAX_PARTIAL_CONTENT_SIZE,

    // Default values
    DEFAULT_MODEL_SIZE_BYTES,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,

    // Error messages
    ERROR_MISSING_MODEL,
    ERROR_MISSING_MESSAGES,
    ERROR_MISSING_PROMPT,
    ERROR_MISSING_INPUT,
    ERROR_CANCELLED,
    ERROR_BUFFER_OVERFLOW,
    ERROR_TIMEOUT,

    // Headers and content types
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_SSE,

    // Logging prefixes
    LOG_PREFIX_REQUEST,
    LOG_PREFIX_SUCCESS,
    LOG_PREFIX_ERROR,
    LOG_PREFIX_WARNING,
    LOG_PREFIX_CANCEL,
};
