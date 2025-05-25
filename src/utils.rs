// src/utils.rs - Optimized utilities with centralized error handling

use std::error::Error;
use std::fmt;
use warp::reject::Reject;
use regex::Regex;
use once_cell::sync::Lazy;

/// Custom error type for the proxy server
#[derive(Debug, Clone)]
pub struct ProxyError {
    pub message: String,
    pub status_code: u16,
    kind: ProxyErrorKind,
}

#[derive(Debug, Clone)]
pub enum ProxyErrorKind {
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
            kind: ProxyErrorKind::Custom
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

/// Logger utility with consistent formatting
#[derive(Debug, Clone)]
pub struct Logger {
    pub enabled: bool,
}

impl Logger {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn log(&self, message: &str) {
        if self.enabled {
            println!("[PROXY] {}", message);
        }
    }

    /// Log with specific prefix
    pub fn log_with_prefix(&self, prefix: &str, message: &str) {
        if self.enabled {
            println!("[PROXY] {} {}", prefix, message);
        }
    }

    /// Log request start
    pub fn log_request(&self, method: &str, path: &str, model: Option<&str>) {
        if self.enabled {
            match model {
                Some(m) => println!("[PROXY] 🔄 {} {} (model: {})", method, path, m),
                None => println!("[PROXY] 🔄 {} {}", method, path),
            }
        }
    }

    /// Log successful completion
    pub fn log_success(&self, operation: &str, duration: std::time::Duration) {
        if self.enabled {
            println!("[PROXY] ✅ {} completed (took {})", operation, format_duration(duration));
        }
    }

    /// Log error
    pub fn log_error(&self, operation: &str, error: &str) {
        if self.enabled {
            println!("[PROXY] ❌ {} failed: {}", operation, error);
        }
    }

    /// Log warning
    pub fn log_warning(&self, message: &str) {
        if self.enabled {
            println!("[PROXY] ⚠️ {}", message);
        }
    }

    /// Log cancellation
    pub fn log_cancellation(&self, operation: &str) {
        if self.enabled {
            println!("[PROXY] 🚫 {} cancelled by client disconnection", operation);
        }
    }
}

/// Optimized error pattern detection using compiled regex
static MODEL_ERROR_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(no|not|missing|invalid|unknown|failed|cannot|unable).*(model|load|available|found)")
        .expect("Failed to compile model error regex")
});

static HTTP_ERROR_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(404|503|400|422).*(model|service)")
        .expect("Failed to compile HTTP error regex")
});

/// Check if error message indicates model loading issues
pub fn is_model_loading_error(message: &str) -> bool {
    MODEL_ERROR_REGEX.is_match(message) || HTTP_ERROR_REGEX.is_match(message)
}

/// Format a duration into human-readable string
pub fn format_duration(duration: std::time::Duration) -> String {
    let total_ms = duration.as_millis();

    if total_ms >= 1000 {
        let seconds = total_ms as f64 / 1000.0;
        format!("{:.2}s", seconds)
    } else {
        format!("{}ms", total_ms)
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

/// Validate server configuration
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

    // Warnings for potentially problematic values
    if config.load_timeout_seconds > 300 {
        eprintln!("Warning: Load timeout is very high ({}s)", config.load_timeout_seconds);
    }

    if config.request_timeout_seconds > 3600 {
        eprintln!("Warning: Request timeout is very high ({}s)", config.request_timeout_seconds);
    }

    if config.stream_timeout_seconds < 5 {
        eprintln!("Warning: Stream timeout is very low ({}s)", config.stream_timeout_seconds);
    }

    // Parse listen address
    if config.listen.parse::<std::net::SocketAddr>().is_err() {
        return Err(format!("Invalid listen address: {}", config.listen));
    }

    // Basic URL validation
    if !config.lmstudio_url.starts_with("http://") && !config.lmstudio_url.starts_with("https://") {
        return Err(format!("Invalid LM Studio URL (must start with http:// or https://): {}", config.lmstudio_url));
    }

    Ok(())
}

/// Check if a string is likely a valid model identifier
pub fn is_valid_model_identifier(name: &str) -> bool {
    if name.is_empty() || name.len() > 200 {
        return false;
    }

    // Must contain at least one alphanumeric character
    if !name.chars().any(|c| c.is_alphanumeric()) {
        return false;
    }

    // Should not contain control characters (except tab, newline, carriage return)
    if name.chars().any(|c| c.is_control() && c != '\t' && c != '\n' && c != '\r') {
        return false;
    }

    // Should not start or end with special characters
    let first_char = name.chars().next().unwrap();
    let last_char = name.chars().last().unwrap();

    if !first_char.is_alphanumeric() && first_char != '_' {
        return false;
    }

    if !last_char.is_alphanumeric() && last_char != '_' {
        return false;
    }

    true
}

/// Extract HTTP status code from error message
pub fn extract_status_code(message: &str) -> Option<u16> {
    static STATUS_REGEX: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"\b([1-5][0-9]{2})\b").expect("Failed to compile status code regex")
    });

    STATUS_REGEX
        .find(message)
        .and_then(|m| m.as_str().parse().ok())
}

/// Check if an error message indicates a timeout
pub fn is_timeout_error(message: &str) -> bool {
    let lower = message.to_lowercase();
    lower.contains("timeout") ||
        lower.contains("timed out") ||
        lower.contains("deadline exceeded") ||
        lower.contains("connection timeout")
}

/// Check if an error message indicates a connection issue
pub fn is_connection_error(message: &str) -> bool {
    let lower = message.to_lowercase();
    lower.contains("connection") ||
        lower.contains("network") ||
        lower.contains("unreachable") ||
        lower.contains("refused") ||
        lower.contains("reset") ||
        lower.contains("broken pipe")
}

/// Sanitize error message for safe logging (remove sensitive information)
pub fn sanitize_error_message(message: &str) -> String {
    static SENSITIVE_REGEX: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?i)(password|token|key|secret|auth)[=:]\s*\S+")
            .expect("Failed to compile sensitive data regex")
    });

    SENSITIVE_REGEX.replace_all(message, "$1=***").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_loading_error_detection() {
        assert!(is_model_loading_error("No model loaded"));
        assert!(is_model_loading_error("Model not found"));
        assert!(is_model_loading_error("404 model not available"));
        assert!(is_model_loading_error("503 service unavailable"));
        assert!(!is_model_loading_error("Generic error message"));
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(std::time::Duration::from_millis(500)), "500ms");
        assert_eq!(format_duration(std::time::Duration::from_millis(1500)), "1.50s");
    }

    #[test]
    fn test_model_name_validation() {
        assert!(is_valid_model_identifier("llama3.2"));
        assert!(is_valid_model_identifier("model_name"));
        assert!(!is_valid_model_identifier(""));
        assert!(!is_valid_model_identifier("   "));
        assert!(!is_valid_model_identifier("\x00invalid"));
    }

    #[test]
    fn test_status_code_extraction() {
        assert_eq!(extract_status_code("HTTP 404 Not Found"), Some(404));
        assert_eq!(extract_status_code("Error 500 occurred"), Some(500));
        assert_eq!(extract_status_code("No status here"), None);
    }
}
