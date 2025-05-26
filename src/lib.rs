/// src/lib.rs - Main library file, re-exporting core modules and constants

// Core modules
pub mod constants;
pub mod model;
pub mod server;
pub mod utils;
pub mod handlers;
pub mod common;

pub use common::RequestContext;
pub use model::{clean_model_name, ModelInfo, ModelResolver};
// Public re-exports for easy access
pub use server::{Config, ProxyServer};
pub use utils::{validate_config, ProxyError};

/// Version information for the application
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Name of the application
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Re-export runtime configuration functions
pub use constants::{
    get_runtime_config,
    init_runtime_config,
    RuntimeConfig,
};


/// Re-export optimized constants for external use
pub use constants::{
    // Timing and performance constants
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_SSE,
    DEFAULT_CONTEXT,
    DEFAULT_LOAD_DURATION_NS,

    // Default values
    DEFAULT_MODEL_SIZE_BYTES,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,

    // Error messages
    ERROR_BUFFER_OVERFLOW,
    ERROR_CANCELLED,
    ERROR_CHUNK_LIMIT,
    ERROR_LM_STUDIO_UNAVAILABLE,
    ERROR_MISSING_INPUT,
    ERROR_MISSING_MESSAGES,
    ERROR_MISSING_MODEL,
    ERROR_MISSING_PROMPT,
    ERROR_REQUEST_TOO_LARGE,
    ERROR_TIMEOUT,

    // Headers and content types
    HEADER_ACCESS_CONTROL_ALLOW_HEADERS,
    HEADER_ACCESS_CONTROL_ALLOW_METHODS,
    HEADER_ACCESS_CONTROL_ALLOW_ORIGIN,
    HEADER_CACHE_CONTROL,
    HEADER_CONNECTION,
    SSE_DATA_PREFIX,
    SSE_DONE_MESSAGE,

    // SSE parsing constants
    SSE_MESSAGE_BOUNDARY,
    TIMING_EVAL_RATIO,
    TIMING_PROMPT_RATIO,

    // Default context for responses
    TOKEN_TO_CHAR_RATIO,
};


/// Centralized logging - re-export the global logger
pub use utils::{init_global_logger, log_error, log_info, log_request, log_timed, log_warning};
