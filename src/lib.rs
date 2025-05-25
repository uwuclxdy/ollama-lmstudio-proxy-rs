// src/lib.rs - Enhanced module organization with metrics and runtime configuration

// Core modules
pub mod constants;
pub mod model;
pub mod server;
pub mod utils;
pub mod handlers;
pub mod common;
pub mod metrics;

// Public re-exports for easy access
pub use server::{Config, ProxyServer};
pub use utils::{ProxyError, Logger, validate_config};
pub use model::{ModelInfo, ModelResolver, clean_model_name};
pub use common::RequestContext;
pub use metrics::{MetricsCollector, get_global_metrics, init_global_metrics};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Re-export runtime configuration functions
pub use constants::{
    RuntimeConfig,
    init_runtime_config,
    get_runtime_config,
};

/// Re-export optimized constants for external use
pub use constants::{
    // Timing and performance constants
    TOKEN_TO_CHAR_RATIO,
    DEFAULT_LOAD_DURATION_NS,
    TIMING_EVAL_RATIO,
    TIMING_PROMPT_RATIO,

    // Default values
    DEFAULT_MODEL_SIZE_BYTES,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_REPEAT_PENALTY,

    // Error messages (static str for efficiency)
    ERROR_MISSING_MODEL,
    ERROR_MISSING_MESSAGES,
    ERROR_MISSING_PROMPT,
    ERROR_MISSING_INPUT,
    ERROR_CANCELLED,
    ERROR_BUFFER_OVERFLOW,
    ERROR_CHUNK_LIMIT,
    ERROR_TIMEOUT,
    ERROR_LM_STUDIO_UNAVAILABLE,
    ERROR_REQUEST_TOO_LARGE,

    // Headers and content types
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_SSE,
    HEADER_CACHE_CONTROL,
    HEADER_CONNECTION,
    HEADER_ACCESS_CONTROL_ALLOW_ORIGIN,
    HEADER_ACCESS_CONTROL_ALLOW_METHODS,
    HEADER_ACCESS_CONTROL_ALLOW_HEADERS,

    // SSE parsing constants
    SSE_DATA_PREFIX,
    SSE_DONE_MESSAGE,
    SSE_MESSAGE_BOUNDARY,

    // Logging prefixes
    LOG_PREFIX_REQUEST,
    LOG_PREFIX_SUCCESS,
    LOG_PREFIX_ERROR,
    LOG_PREFIX_WARNING,
    LOG_PREFIX_CANCEL,
    LOG_PREFIX_METRICS,

    // Default context for responses
    DEFAULT_CONTEXT,
};
