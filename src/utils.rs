/// src/utils.rs - Centralized logging and utilities

use std::cell::RefCell;
use std::error::Error;
use std::fmt::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use warp::reject::Reject;


use crate::constants::*;

// Global logging state
static LOGGING_ENABLED: AtomicBool = AtomicBool::new(true);

// Thread-local string buffer for reuse
thread_local! {
    pub static STRING_BUFFER: RefCell<String> = RefCell::new(String::with_capacity(get_runtime_config().string_buffer_size));
}

/// Initialize global logger
pub fn init_global_logger(enabled: bool) {
    LOGGING_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Check if logging is enabled
#[inline]
pub fn is_logging_enabled() -> bool {
    LOGGING_ENABLED.load(Ordering::Relaxed)
}

/// Centralized logging functions - use these throughout the application

/// Log informational message
pub fn log_info(message: &str) {
    if is_logging_enabled() {
        println!("[{}] ℹ️ {}", chrono::Local::now().format("%H:%M:%S"), sanitize_log_message(message));
    }
}

/// Log warning message
pub fn log_warning(operation: &str, warning: &str) {
    if is_logging_enabled() {
        STRING_BUFFER.with(|buf| {
            let mut buffer = buf.borrow_mut();
            buffer.clear();
            write!(buffer, "{} {} warning: {}", LOG_PREFIX_WARNING, sanitize_log_message(operation), sanitize_log_message(warning)).unwrap();
            println!("[{}] {}", chrono::Local::now().format("%H:%M:%S"), buffer);
        });
    }
}

/// Log error message
pub fn log_error(operation: &str, error: &str) {
    if is_logging_enabled() {
        STRING_BUFFER.with(|buf| {
            let mut buffer = buf.borrow_mut();
            buffer.clear();
            write!(buffer, "{} {} failed: {}", LOG_PREFIX_ERROR, sanitize_log_message(operation), sanitize_log_message(error)).unwrap();
            println!("[{}] {}", chrono::Local::now().format("%H:%M:%S"), buffer);
        });
    }
}

/// Log request with optional model
pub fn log_request(method: &str, path: &str, model: Option<&str>) {
    if is_logging_enabled() {
        STRING_BUFFER.with(|buf| {
            let mut buffer = buf.borrow_mut();
            buffer.clear();
            match model {
                Some(m) => write!(buffer, "{} {} {} (model: {})", LOG_PREFIX_REQUEST, method, sanitize_log_message(path), sanitize_log_message(m)).unwrap(),
                None => write!(buffer, "{} {} {}", LOG_PREFIX_REQUEST, method, sanitize_log_message(path)).unwrap(),
            }
            println!("[{}] {}", chrono::Local::now().format("%H:%M:%S"), buffer);
        });
    }
}

/// Log with timing information
pub fn log_timed(prefix: &str, operation: &str, start: Instant) {
    if is_logging_enabled() {
        let duration = start.elapsed();
        STRING_BUFFER.with(|buf| {
            let mut buffer = buf.borrow_mut();
            buffer.clear();
            write!(buffer, "{} {} ({})", prefix, operation, format_duration(duration)).unwrap();
            println!("[{}] {}", chrono::Local::now().format("%H:%M:%S"), buffer);
        });
    }
}

/// Macro for efficient error handling in handlers
#[macro_export]
macro_rules! handle_lm_error {
    ($response:expr) => {
        if !$response.status().is_success() {
            let status = $response.status();
            let error_body = $response.text().await.unwrap_or_else(|_| "Unknown error body".to_string());
            return Err(ProxyError::new(
                format!("LM Studio error: {} - {}", status, error_body),
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
    /// Create new proxy error
    pub fn new(message: String, status_code: u16) -> Self {
        Self {
            message,
            status_code,
            kind: ProxyErrorKind::Custom,
        }
    }

    /// Create internal server error
    pub fn internal_server_error(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 500,
            kind: ProxyErrorKind::InternalServerError,
        }
    }

    /// Create bad request error
    pub fn bad_request(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 400,
            kind: ProxyErrorKind::BadRequest,
        }
    }

    /// Create not found error
    pub fn not_found(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 404,
            kind: ProxyErrorKind::NotFound,
        }
    }

    /// Create not implemented error
    pub fn not_implemented(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 501,
            kind: ProxyErrorKind::NotImplemented,
        }
    }

    /// Create request cancelled error
    pub fn request_cancelled() -> Self {
        Self {
            message: ERROR_CANCELLED.to_string(),
            status_code: 499,
            kind: ProxyErrorKind::RequestCancelled,
        }
    }

    /// Create LM Studio unavailable error
    pub fn lm_studio_unavailable(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 503,
            kind: ProxyErrorKind::LMStudioUnavailable,
        }
    }

    /// Check if request is canceled
    pub fn is_cancelled(&self) -> bool {
        matches!(self.kind, ProxyErrorKind::RequestCancelled)
    }

    /// Check if LM Studio is unavailable
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

/// Enhanced model loading error detection
pub fn is_model_loading_error(message: &str) -> bool {
    let lower = message.to_lowercase();

    let error_indicators = [
        "no model", "not loaded", "model not found", "model unavailable",
        "model not available", "invalid model", "unknown model",
        "failed to load", "loading failed", "model error", "is not embedding"
    ];

    error_indicators.iter().any(|&pattern| lower.contains(pattern)) ||
        ((lower.contains("no") || lower.contains("not") || lower.contains("missing") ||
            lower.contains("invalid") || lower.contains("unknown") || lower.contains("failed")) &&
            (lower.contains("model") || lower.contains("load") || lower.contains("available")))
}

/// Fast duration formatting with better precision
pub fn format_duration(duration: Duration) -> String {
    let total_nanos = duration.as_nanos();

    if total_nanos < 1_000 {
        format!("{}ns", total_nanos)
    } else if total_nanos < 1_000_000 {
        format!("{:.1}µs", total_nanos as f64 / 1_000.0)
    } else if total_nanos < 1_000_000_000 {
        format!("{:.2}ms", total_nanos as f64 / 1_000_000.0)
    } else {
        format!("{:.3}s", total_nanos as f64 / 1_000_000_000.0)
    }
}

/// Enhanced config validation
pub fn validate_config(config: &crate::server::Config) -> Result<(), String> {
    if config.listen.parse::<std::net::SocketAddr>().is_err() {
        return Err(format!("Invalid listen address: {}", config.listen));
    }
    if !config.lmstudio_url.starts_with("http://") && !config.lmstudio_url.starts_with("https://") {
        return Err(format!("Invalid LM Studio URL (must start with http:// or https://): {}", config.lmstudio_url));
    }
    if let Err(e) = url::Url::parse(&config.lmstudio_url) {
        return Err(format!("Invalid LM Studio URL format: {}", e));
    }

    Ok(())
}

/// Check if endpoint requires authentication
pub fn is_protected_endpoint(path: &str) -> bool {
    matches!(path, "/admin/*" | "/config/*")
}

/// Sanitize log message to prevent log injection
pub fn sanitize_log_message(message: &str) -> String {
    message
        .chars()
        .map(|c| if c.is_control() && !matches!(c, '\t' | '\n' | '\r') { '?' } else { c })
        .collect()
}

/// Extract client IP from request headers
pub fn extract_client_ip(headers: &warp::http::HeaderMap) -> Option<String> {
    let ip_headers = [
        "x-forwarded-for",
        "x-real-ip",
        "cf-connecting-ip",
        "x-client-ip"
    ];
    for header_name in &ip_headers {
        if let Some(header_value) = headers.get(*header_name) {
            if let Ok(ip_str) = header_value.to_str() {
                let ip = ip_str.split(',').next().unwrap_or(ip_str).trim();
                if !ip.is_empty() {
                    return Some(ip.to_string());
                }
            }
        }
    }
    None
}
