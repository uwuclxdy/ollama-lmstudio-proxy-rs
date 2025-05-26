/// src/constants.rs - Runtime configurable constants and static values

use std::sync::OnceLock;


/// Global configuration that can be set at runtime
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub max_buffer_size: usize,
    pub max_partial_content_size: usize,
    pub string_buffer_size: usize,
    pub enable_chunk_recovery: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 256 * 1024,
            max_partial_content_size: 50_000,
            string_buffer_size: 2048,
            enable_chunk_recovery: true,
        }
    }
}

static RUNTIME_CONFIG: OnceLock<RuntimeConfig> = OnceLock::new();

/// Initialize runtime configuration
pub fn init_runtime_config(config: RuntimeConfig) {
    RUNTIME_CONFIG.set(config).ok();
}

/// Get current runtime configuration
pub fn get_runtime_config() -> &'static RuntimeConfig {
    RUNTIME_CONFIG.get().unwrap_or_else(|| {
        static DEFAULT: OnceLock<RuntimeConfig> = OnceLock::new();
        DEFAULT.get_or_init(RuntimeConfig::default)
    })
}

/// Timing and performance constants
pub const TOKEN_TO_CHAR_RATIO: f64 = 0.25;
pub const DEFAULT_LOAD_DURATION_NS: u64 = 1_000_000;
pub const TIMING_EVAL_RATIO: u64 = 2;
pub const TIMING_PROMPT_RATIO: u64 = 4;

/// Default model size estimate
pub const DEFAULT_MODEL_SIZE_BYTES: u64 = 4_000_000_000;

/// Response headers
pub const CONTENT_TYPE_JSON: &str = "application/json; charset=utf-8";
pub const CONTENT_TYPE_SSE: &str = "text/event-stream";
pub const HEADER_CACHE_CONTROL: &str = "no-cache";
pub const HEADER_CONNECTION: &str = "keep-alive";
pub const HEADER_ACCESS_CONTROL_ALLOW_ORIGIN: &str = "*";
pub const HEADER_ACCESS_CONTROL_ALLOW_METHODS: &str = "GET, POST, PUT, DELETE, OPTIONS";
pub const HEADER_ACCESS_CONTROL_ALLOW_HEADERS: &str = "Content-Type, Authorization";

/// Default parameter values
pub const DEFAULT_TEMPERATURE: f64 = 0.7;
pub const DEFAULT_TOP_P: f64 = 0.9;
pub const DEFAULT_TOP_K: u32 = 40;
pub const DEFAULT_REPEAT_PENALTY: f64 = 1.1;
pub const DEFAULT_KEEP_ALIVE_MINUTES: i64 = 5;
pub const EMBEDDING_TRUNCATE_CHAR_LIMIT: usize = 131072;

/// Error messages
pub const ERROR_MISSING_MODEL: &str = "Missing 'model' field";
pub const ERROR_MISSING_MESSAGES: &str = "Missing 'messages' field";
pub const ERROR_MISSING_PROMPT: &str = "Missing 'prompt' field";
pub const ERROR_MISSING_INPUT: &str = "Missing 'input' or 'prompt' field";
pub const ERROR_BUFFER_OVERFLOW: &str = "Stream buffer overflow";
pub const ERROR_CHUNK_LIMIT: &str = "Stream exceeded maximum chunk limit";
pub const ERROR_TIMEOUT: &str = "Stream timeout";
pub const ERROR_CANCELLED: &str = "Request cancelled by client";
pub const ERROR_LM_STUDIO_UNAVAILABLE: &str = "LM Studio not available";
pub const ERROR_REQUEST_TOO_LARGE: &str = "Request body too large";

/// SSE parsing constants
pub const SSE_DATA_PREFIX: &str = "data: ";
pub const SSE_DONE_MESSAGE: &str = "[DONE]";
pub const SSE_MESSAGE_BOUNDARY: &str = "\n\n";

/// Logging prefixes
pub const LOG_PREFIX_REQUEST: &str = "🔄";
pub const LOG_PREFIX_SUCCESS: &str = "✅";
pub const LOG_PREFIX_ERROR: &str = "❌";
pub const LOG_PREFIX_WARNING: &str = "⚠️";
pub const LOG_PREFIX_CANCEL: &str = "🚫";

/// Default context array for generate responses
pub const DEFAULT_CONTEXT: [u32; 3] = [1, 2, 3];
