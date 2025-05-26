/// src/lib.rs - Main library file with native and legacy API support

// Core modules
pub mod constants;
pub mod model;           // Native LM Studio API support
pub mod model_legacy;    // Legacy OpenAI-compatible API support
pub mod server;
pub mod utils;
pub mod handlers;
pub mod common;

// Public re-exports for easy access
pub use common::RequestContext;

// Native API exports (default)
pub use model::{clean_model_name, ModelInfo, ModelResolver};

// Legacy API exports
pub use model_legacy::{
    clean_model_name_legacy, ModelInfoLegacy, ModelResolverLegacy
};

// Server exports
pub use server::{Config, ModelResolverType, ProxyServer};

// Utility exports
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
    // API endpoint constants
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_SSE,
    DEFAULT_CONTEXT,
    DEFAULT_LOAD_DURATION_NS,
    DEFAULT_MODEL_SIZE_BYTES,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,

    // Timing and performance constants
    DEFAULT_TOP_P,
    ERROR_BUFFER_OVERFLOW,
    ERROR_CANCELLED,
    ERROR_CHUNK_LIMIT,

    // Default values
    ERROR_LM_STUDIO_UNAVAILABLE,
    ERROR_MISSING_INPUT,
    ERROR_MISSING_MESSAGES,
    ERROR_MISSING_MODEL,
    ERROR_MISSING_PROMPT,

    // Error messages
    ERROR_NATIVE_API_UNAVAILABLE,
    ERROR_REQUEST_TOO_LARGE,
    ERROR_TIMEOUT,
    HEADER_ACCESS_CONTROL_ALLOW_HEADERS,
    HEADER_ACCESS_CONTROL_ALLOW_METHODS,
    HEADER_ACCESS_CONTROL_ALLOW_ORIGIN,
    HEADER_CACHE_CONTROL,
    HEADER_CONNECTION,
    LM_STUDIO_LEGACY_CHAT,
    LM_STUDIO_LEGACY_COMPLETIONS,
    LM_STUDIO_LEGACY_EMBEDDINGS,

    // Headers and content types
    LM_STUDIO_LEGACY_MODELS,
    LM_STUDIO_NATIVE_CHAT,
    LM_STUDIO_NATIVE_COMPLETIONS,
    LM_STUDIO_NATIVE_EMBEDDINGS,
    LM_STUDIO_NATIVE_MODELS,
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

/// Helper function to determine which API mode is being used
pub fn get_api_mode_info(use_legacy: bool) -> (&'static str, &'static str) {
    if use_legacy {
        ("Legacy", "OpenAI-compatible API (/v1/ endpoints)")
    } else {
        ("Native", "LM Studio native API (/api/v0/ endpoints)")
    }
}

/// Helper function to get recommended LM Studio version for each mode
pub fn get_lm_studio_requirements(use_legacy: bool) -> &'static str {
    if use_legacy {
        "LM Studio 0.2.0+ (any version with OpenAI compatibility)"
    } else {
        "LM Studio 0.3.6+ (required for native API)"
    }
}

/// Helper to create appropriate model resolver based on configuration
pub fn create_model_resolver(
    lmstudio_url: String,
    cache: moka::future::Cache<String, String>,
    use_legacy: bool,
) -> ModelResolverType {
    if use_legacy {
        ModelResolverType::Legacy(std::sync::Arc::new(
            ModelResolverLegacy::new_legacy(lmstudio_url, cache)
        ))
    } else {
        ModelResolverType::Native(std::sync::Arc::new(
            ModelResolver::new(lmstudio_url, cache)
        ))
    }
}

/// Enhanced error handling for API compatibility issues
pub fn handle_api_compatibility_error(error: &ProxyError, use_legacy: bool) -> String {
    if error.status_code == 404 && !use_legacy {
        format!(
            "Native API endpoint not found: {}. \
            This may indicate LM Studio version < 0.3.6. \
            Try using --legacy flag for older versions.",
            error.message
        )
    } else if error.status_code == 404 && use_legacy {
        format!(
            "Legacy API endpoint not found: {}. \
            This may indicate an unsupported LM Studio version.",
            error.message
        )
    } else {
        error.message.clone()
    }
}

/// Feature comparison between API modes
pub struct ApiFeatureComparison {
    pub native_features: Vec<&'static str>,
    pub legacy_features: Vec<&'static str>,
    pub native_limitations: Vec<&'static str>,
    pub legacy_limitations: Vec<&'static str>,
}

impl ApiFeatureComparison {
    pub fn new() -> Self {
        Self {
            native_features: vec![
                "Real model loading state detection",
                "Accurate context length limits",
                "Performance metrics (tokens/sec, TTFT)",
                "Model architecture information",
                "Publisher and quantization details",
                "Loaded vs available model distinction",
                "Enhanced error messages",
            ],
            legacy_features: vec![
                "Broad LM Studio version compatibility",
                "Estimated model information",
                "Basic functionality coverage",
                "Fallback compatibility",
            ],
            native_limitations: vec![
                "Requires LM Studio 0.3.6+",
                "May have API changes in beta",
            ],
            legacy_limitations: vec![
                "No real model state information",
                "Estimated timing and metrics",
                "Limited model metadata",
                "No context length awareness",
            ],
        }
    }
}

impl Default for ApiFeatureComparison {
    fn default() -> Self {
        Self::new()
    }
}
