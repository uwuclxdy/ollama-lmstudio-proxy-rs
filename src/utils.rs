// src/utils.rs - Consolidated utilities with enhanced error handling

use std::cell::RefCell;
use std::error::Error;
use std::fmt::{self, Write};
use std::time::{Duration, Instant};
use warp::reject::Reject;

use crate::constants::*;

// Thread-local string buffer for reuse
thread_local! {
    static STRING_BUFFER: RefCell<String> = RefCell::new(String::with_capacity(get_runtime_config().string_buffer_size));
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

/// Enhanced error type for the proxy server
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
    LMStudioUnavailable,
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
            message: ERROR_CANCELLED.to_string(),
            status_code: 499,
            kind: ProxyErrorKind::RequestCancelled,
        }
    }

    pub fn lm_studio_unavailable(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 503,
            kind: ProxyErrorKind::LMStudioUnavailable,
        }
    }

    pub fn is_cancelled(&self) -> bool {
        matches!(self.kind, ProxyErrorKind::RequestCancelled)
    }

    pub fn is_lm_studio_unavailable(&self) -> bool {
        matches!(self.kind, ProxyErrorKind::LMStudioUnavailable)
    }
}

impl fmt::Display for ProxyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ProxyError {}: {}", self.status_code, self.message)
    }
}

impl Error for ProxyError {}
impl Reject for ProxyError {}

/// Enhanced logger with metrics integration
#[derive(Debug, Clone)]
pub struct Logger {
    pub enabled: bool,
}

impl Logger {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Log with timing information using efficient string buffer
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
                    Some(m) => write!(buffer, "{} {} {} (model: {})", LOG_PREFIX_REQUEST, method, path, m).unwrap(),
                    None => write!(buffer, "{} {} {}", LOG_PREFIX_REQUEST, method, path).unwrap(),
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
                write!(buffer, "{} {} failed: {}", LOG_PREFIX_ERROR, operation, error).unwrap();
                println!("[{}] {}", chrono::Local::now().format("%H:%M:%S"), buffer);
            });
        }
    }

    /// Log warning message
    pub fn log_warning(&self, operation: &str, warning: &str) {
        if self.enabled {
            STRING_BUFFER.with(|buf| {
                let mut buffer = buf.borrow_mut();
                buffer.clear();
                write!(buffer, "{} {} warning: {}", LOG_PREFIX_WARNING, operation, warning).unwrap();
                println!("[{}] {}", chrono::Local::now().format("%H:%M:%S"), buffer);
            });
        }
    }
}

/// Enhanced model loading error detection
pub fn is_model_loading_error(message: &str) -> bool {
    let lower = message.to_lowercase();

    // More comprehensive error pattern matching
    let error_indicators = [
        "no model", "not loaded", "model not found", "model unavailable",
        "model not available", "invalid model", "unknown model",
        "failed to load", "loading failed", "model error"
    ];

    error_indicators.iter().any(|&pattern| lower.contains(pattern)) ||
        ((lower.contains("no") || lower.contains("not") || lower.contains("missing") ||
            lower.contains("invalid") || lower.contains("unknown") || lower.contains("failed")) &&
            (lower.contains("model") || lower.contains("load") || lower.contains("available")))
}

/// Fast duration formatting with better precision
pub fn format_duration(duration: Duration) -> String {
    let total_nanos = duration.as_nanos();

    if total_nanos < 1_000 { // Less than 1µs
        format!("{}ns", total_nanos)
    } else if total_nanos < 1_000_000 { // Less than 1ms
        format!("{:.1}µs", total_nanos as f64 / 1_000.0)
    } else if total_nanos < 1_000_000_000 { // Less than 1s
        format!("{:.2}ms", total_nanos as f64 / 1_000_000.0)
    } else { // 1s or more
        format!("{:.3}s", total_nanos as f64 / 1_000_000_000.0)
    }
}

/// Consolidated model name validation
pub fn validate_model_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("Model name cannot be empty".to_string());
    }

    if name.len() > 200 {
        return Err("Model name too long (max: 200 characters)".to_string());
    }

    // Check for control characters (except common whitespace)
    if name.chars().any(|c| c.is_control() && !matches!(c, '\t' | '\n' | '\r')) {
        return Err("Model name contains invalid control characters".to_string());
    }

    // Check for suspicious patterns
    if name.contains("::") {
        return Err("Model name contains multiple consecutive colons".to_string());
    }

    if name.matches(':').count() > 4 {
        return Err("Model name contains too many colons".to_string());
    }

    Ok(())
}

/// Enhanced config validation
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

    // Validate listen address
    if config.listen.parse::<std::net::SocketAddr>().is_err() {
        return Err(format!("Invalid listen address: {}", config.listen));
    }

    // Enhanced URL validation
    if !config.lmstudio_url.starts_with("http://") && !config.lmstudio_url.starts_with("https://") {
        return Err(format!("Invalid LM Studio URL (must start with http:// or https://): {}", config.lmstudio_url));
    }

    // Validate URL format more thoroughly
    if let Err(e) = url::Url::parse(&config.lmstudio_url) {
        return Err(format!("Invalid LM Studio URL format: {}", e));
    }

    // Validate buffer sizes
    if config.max_buffer_size == 0 {
        return Err("Max buffer size must be greater than 0".to_string());
    }

    if config.max_chunk_count == 0 {
        return Err("Max chunk count must be greater than 0".to_string());
    }

    if config.max_request_size == 0 {
        return Err("Max request size must be greater than 0".to_string());
    }

    // Reasonable limits validation
    if config.max_buffer_size > 100 * 1024 * 1024 { // 100MB
        return Err("Max buffer size too large (max: 100MB)".to_string());
    }

    if config.max_request_size > 1024 * 1024 * 1024 { // 1GB
        return Err("Max request size too large (max: 1GB)".to_string());
    }

    Ok(())
}

/// Check if endpoint requires authentication (for future use)
pub fn is_protected_endpoint(path: &str) -> bool {
    // Currently no protected endpoints, but framework for future use
    matches!(path, "/admin/*" | "/config/*")
}

/// Sanitize log message to prevent log injection
pub fn sanitize_log_message(message: &str) -> String {
    message
        .chars()
        .map(|c| if c.is_control() && !matches!(c, '\t' | '\n' | '\r') { '?' } else { c })
        .collect()
}

/// Extract client IP from request headers (for logging/metrics)
pub fn extract_client_ip(headers: &warp::http::HeaderMap) -> Option<String> {
    // Check common headers for client IP
    let ip_headers = [
        "x-forwarded-for",
        "x-real-ip",
        "cf-connecting-ip",
        "x-client-ip"
    ];

    for header_name in &ip_headers {
        if let Some(header_value) = headers.get(*header_name) {
            if let Ok(ip_str) = header_value.to_str() {
                // Take first IP if comma-separated list
                let ip = ip_str.split(',').next().unwrap_or(ip_str).trim();
                if !ip.is_empty() {
                    return Some(ip.to_string());
                }
            }
        }
    }

    None
}
