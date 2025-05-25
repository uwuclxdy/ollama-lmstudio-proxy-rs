// src/utils.rs - Optimized utilities with macros and efficient string handling

use std::cell::RefCell;
use std::error::Error;
use std::fmt::{self, Write};
use std::time::{Duration, Instant};
use warp::reject::Reject;

// Thread-local string buffer for reuse
thread_local! {
    static STRING_BUFFER: RefCell<String> = RefCell::new(String::with_capacity(crate::constants::STRING_BUFFER_SIZE));
}

/// Macro for efficient error handling in handlers
#[macro_export]
macro_rules! handle_lm_error {
    ($response:expr) => {
        if !$response.status().is_success() {
            let status = $response.status();
            return Err(ProxyError::new(
                format!("LM Studio error: {}", status),
                status.as_u16()
            ));
        }
    };
}

/// Macro for cancellation checking
#[macro_export]
macro_rules! check_cancelled {
    ($token:expr) => {
        if $token.is_cancelled() {
            return Err(ProxyError::request_cancelled());
        }
    };
}

/// Macro for efficient logging with timing
#[macro_export]
macro_rules! log_with_timing {
    ($logger:expr, $prefix:expr, $operation:expr, $start:expr) => {
        if $logger.enabled {
            let duration = $start.elapsed();
            STRING_BUFFER.with(|buf| {
                let mut buffer = buf.borrow_mut();
                buffer.clear();
                write!(buffer, "{} {} ({})", $prefix, $operation, format_duration(duration)).unwrap();
                println!("[{}] {}", chrono::Local::now().format("%H:%M:%S"), buffer);
            });
        }
    };
}

/// Custom error type for the proxy server
#[derive(Debug, Clone)]
pub struct ProxyError {
    pub message: String,
    pub status_code: u16,
    kind: ProxyErrorKind,
}

#[derive(Debug, Clone)]
enum ProxyErrorKind {
    RequestCancelled,
    InternalServerError,
    BadRequest,
    NotFound,
    NotImplemented,
    Custom,
}

impl ProxyError {
    pub fn new(message: String, status_code: u16) -> Self {
        Self {
            message,
            status_code,
            kind: ProxyErrorKind::Custom,
        }
    }

    pub fn internal_server_error(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 500,
            kind: ProxyErrorKind::InternalServerError,
        }
    }

    pub fn bad_request(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 400,
            kind: ProxyErrorKind::BadRequest,
        }
    }

    pub fn not_found(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 404,
            kind: ProxyErrorKind::NotFound,
        }
    }

    pub fn not_implemented(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 501,
            kind: ProxyErrorKind::NotImplemented,
        }
    }

    pub fn request_cancelled() -> Self {
        Self {
            message: "Request was cancelled".to_string(),
            status_code: 499,
            kind: ProxyErrorKind::RequestCancelled,
        }
    }

    pub fn is_cancelled(&self) -> bool {
        matches!(self.kind, ProxyErrorKind::RequestCancelled)
    }
}

impl fmt::Display for ProxyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ProxyError {}: {}", self.status_code, self.message)
    }
}

impl Error for ProxyError {}
impl Reject for ProxyError {}

/// Simplified logger for single-client use
#[derive(Debug, Clone)]
pub struct Logger {
    pub enabled: bool,
}

impl Logger {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Log with timing information (efficient implementation)
    pub fn log_timed(&self, prefix: &str, operation: &str, start: Instant) {
        if self.enabled {
            let duration = start.elapsed();
            STRING_BUFFER.with(|buf| {
                let mut buffer = buf.borrow_mut();
                buffer.clear();
                write!(buffer, "{} {} ({})", prefix, operation, format_duration(duration)).unwrap();
                println!("[{}] {}", chrono::Local::now().format("%H:%M:%S"), buffer);
            });
        }
    }

    /// Simple log without timing
    pub fn log(&self, message: &str) {
        if self.enabled {
            println!("[{}] {}", chrono::Local::now().format("%H:%M:%S"), message);
        }
    }

    /// Log request with optional model
    pub fn log_request(&self, method: &str, path: &str, model: Option<&str>) {
        if self.enabled {
            STRING_BUFFER.with(|buf| {
                let mut buffer = buf.borrow_mut();
                buffer.clear();
                match model {
                    Some(m) => write!(buffer, "🔄 {} {} (model: {})", method, path, m).unwrap(),
                    None => write!(buffer, "🔄 {} {}", method, path).unwrap(),
                }
                println!("[{}] {}", chrono::Local::now().format("%H:%M:%S"), buffer);
            });
        }
    }

    /// Log error with operation context
    pub fn log_error(&self, operation: &str, error: &str) {
        if self.enabled {
            STRING_BUFFER.with(|buf| {
                let mut buffer = buf.borrow_mut();
                buffer.clear();
                write!(buffer, "❌ {} failed: {}", operation, error).unwrap();
                println!("[{}] {}", chrono::Local::now().format("%H:%M:%S"), buffer);
            });
        }
    }
}

/// Simplified model loading error detection (no regex for performance)
pub fn is_model_loading_error(message: &str) -> bool {
    let lower = message.to_lowercase();
    (lower.contains("no") || lower.contains("not") || lower.contains("missing") ||
        lower.contains("invalid") || lower.contains("unknown") || lower.contains("failed")) &&
        (lower.contains("model") || lower.contains("load") || lower.contains("available"))
}

/// Fast duration formatting
pub fn format_duration(duration: Duration) -> String {
    let total_micros = duration.as_micros();

    if total_micros < 1_000 { // Less than 1ms
        format!("{}µs", total_micros)
    } else if total_micros < 1_000_000 { // Less than 1s
        format!("{:.3}ms", total_micros as f64 / 1_000.0)
    } else { // 1s or more
        format!("{:.3}s", total_micros as f64 / 1_000_000.0)
    }
}

/// Validate model name and return warnings for potentially malformed names
pub fn validate_model_name(name: &str) -> (bool, Option<String>) {
    if name.is_empty() {
        return (false, Some("Model name cannot be empty".to_string()));
    }

    let mut warnings = Vec::new();

    // Check for common issues
    if name.contains("::") {
        warnings.push("Multiple consecutive colons".to_string());
    }

    if name.starts_with(':') && name.len() > 1 {
        warnings.push("Starts with colon".to_string());
    }

    if name.ends_with(':') {
        warnings.push("Ends with colon".to_string());
    }

    if name.len() > 200 {
        warnings.push("Unusually long name".to_string());
    }

    if name.contains(char::is_whitespace) {
        warnings.push("Contains whitespace".to_string());
    }

    if name.chars().any(|c| c.is_control() && c != '\t' && c != '\n' && c != '\r') {
        warnings.push("Contains control characters".to_string());
    }

    let colon_count = name.matches(':').count();
    if colon_count > 4 {
        warnings.push(format!("Too many colons ({})", colon_count));
    }

    let is_valid = warnings.is_empty();
    let warning_message = if warnings.is_empty() {
        None
    } else {
        Some(warnings.join("; "))
    };

    (is_valid, warning_message)
}

/// Optimized config validation for single client
pub fn validate_config(config: &crate::server::Config) -> Result<(), String> {
    if config.request_timeout_seconds == 0 {
        return Err("Request timeout must be greater than 0".to_string());
    }

    if config.stream_timeout_seconds == 0 {
        return Err("Stream timeout must be greater than 0".to_string());
    }

    if config.load_timeout_seconds == 0 {
        return Err("Load timeout must be greater than 0".to_string());
    }

    // Parse listen address
    if config.listen.parse::<std::net::SocketAddr>().is_err() {
        return Err(format!("Invalid listen address: {}", config.listen));
    }

    // Basic URL validation
    if !config.lmstudio_url.starts_with("http://") && !config.lmstudio_url.starts_with("https://") {
        return Err(format!("Invalid LM Studio URL: {}", config.lmstudio_url));
    }

    Ok(())
}

/// Simple model identifier validation
pub fn is_valid_model_identifier(name: &str) -> bool {
    !name.is_empty() &&
        name.len() <= 200 &&
        name.chars().any(|c| c.is_alphanumeric()) &&
        !name.chars().any(|c| c.is_control())
}
