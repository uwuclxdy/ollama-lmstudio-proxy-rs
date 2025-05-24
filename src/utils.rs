use std::fmt;
use std::error::Error;
use warp::reject::Reject;

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
    // ...existing variants...
}

impl ProxyError {
    /// Create a new ProxyError with custom message and status code
    pub fn new(message: String, status_code: u16) -> Self {
        Self { message, status_code, kind: ProxyErrorKind::RequestCancelled }
    }

    /// Create an internal server error (500)
    pub fn internal_server_error(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 500,
            kind: ProxyErrorKind::RequestCancelled,
        }
    }

    /// Create a bad request error (400)
    pub fn bad_request(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 400,
            kind: ProxyErrorKind::RequestCancelled,
        }
    }

    /// Create a not found error (404)
    pub fn not_found(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 404,
            kind: ProxyErrorKind::RequestCancelled,
        }
    }

    /// Create a not implemented error (501)
    pub fn not_implemented(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 501,
            kind: ProxyErrorKind::RequestCancelled,
        }
    }

    /// NEW: Create a request cancelled error (499 - Client Closed Request)
    pub fn request_cancelled() -> Self {
        Self {
            message: "Request was cancelled".to_string(),
            status_code: 499,
            kind: ProxyErrorKind::RequestCancelled,
        }
    }

    /// NEW: Check if this error represents a cancellation
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

/// Logger utility for the proxy server
#[derive(Debug, Clone)]
pub struct Logger {
    pub enabled: bool,
}

impl Logger {
    /// Create a new Logger instance
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Log a message with [PROXY] prefix if logging is enabled
    pub fn log(&self, message: &str) {
        if self.enabled {
            println!("[PROXY] {}", message);
        }
    }
}

/// Clean model name by removing :latest and numeric suffixes
///
/// This function handles various model name formats:
/// - Removes `:latest` suffix if present
/// - Removes pure numeric suffixes (e.g., `:2`, `:123`) but only if there's content before the colon
/// - Preserves non-numeric suffixes (e.g., `:custom`, `:alpha`, `:v1.2`)
/// - Handles multiple colons correctly - only processes the last segment for numeric removal
///
/// Examples:
/// - "deepseek-r1-distill-qwen-14b:2" → "deepseek-r1-distill-qwen-14b"
/// - "llama3.2:latest" → "llama3.2"
/// - "model-name:3" → "model-name"
/// - "namespace:model:tag:version" → "namespace:model:tag:version" (non-numeric preserved)
/// - "namespace:model:tag:2" → "namespace:model:tag" (only last numeric removed)
/// - ":123" → ":123" (preserved when no model name before colon)
/// - "model:custom" → "model:custom" (non-numeric suffix preserved)
pub fn clean_model_name(name: &str) -> String {
    if name.is_empty() {
        return name.to_string();
    }

    // First remove :latest suffix if present
    let after_latest = if name.ends_with(":latest") {
        &name[..name.len() - 7]  // ":latest" is 7 characters, not 8
    } else {
        name
    };

    // Then check if we should remove a numeric suffix
    if let Some(colon_pos) = after_latest.rfind(':') {
        let suffix = &after_latest[colon_pos + 1..];

        // Only remove suffix if:
        // 1. It's purely numeric (not empty, all digits)
        // 2. There's actually content before the colon (not just removing everything)
        if !suffix.is_empty()
            && suffix.chars().all(|c| c.is_ascii_digit())
            && colon_pos > 0 {  // Don't remove if it would result in empty string
            return after_latest[..colon_pos].to_string();
        }
    }

    after_latest.to_string()
}

/// Check if an error message indicates that no models are loaded
/// This is used to detect when LM Studio needs to have a model loaded
pub fn is_no_models_loaded_error(message: &str) -> bool {
    let lower_msg = message.to_lowercase();
    lower_msg.contains("no model")
        || lower_msg.contains("model not loaded")
        || lower_msg.contains("no models loaded")
        || lower_msg.contains("model loading")
        || lower_msg.contains("load a model")
        || lower_msg.contains("model is not loaded")
}

/// Format a duration into a human-readable string
/// Examples:
/// - 1500ms → "1.50s"
/// - 500ms → "500ms"
/// - 2500ms → "2.50s"
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
///
/// This function checks for common issues in model names that might indicate
/// user error or unexpected input patterns.
///
/// Returns (is_valid, warning_message)
pub fn validate_model_name(name: &str) -> (bool, Option<String>) {
    if name.is_empty() {
        return (false, Some("Model name cannot be empty".to_string()));
    }

    // Check for suspicious patterns
    let mut warnings = Vec::new();

    // Check for multiple consecutive colons
    if name.contains("::") {
        warnings.push("Multiple consecutive colons detected".to_string());
    }

    // Check for colons at start/end (might be intentional but often indicates error)
    if name.starts_with(':') && name.len() > 1 {
        warnings.push("Model name starts with colon".to_string());
    }

    if name.ends_with(':') {
        warnings.push("Model name ends with colon".to_string());
    }

    // Check for extremely long names (might indicate pasted content)
    if name.len() > 200 {
        warnings.push("Model name is unusually long".to_string());
    }

    // Check for whitespace (spaces/tabs) which are often copy-paste errors
    if name.contains(char::is_whitespace) {
        warnings.push("Model name contains whitespace characters".to_string());
    }

    // Check for unusual characters that might indicate encoding issues
    if name.chars().any(|c| c.is_control() && c != '\t' && c != '\n' && c != '\r') {
        warnings.push("Model name contains control characters".to_string());
    }

    // Too many colons might indicate confusion about format
    let colon_count = name.matches(':').count();
    if colon_count > 4 {
        warnings.push(format!("Model name has {} colons, which seems excessive", colon_count));
    }

    let is_valid = warnings.is_empty();
    let warning_message = if warnings.is_empty() {
        None
    } else {
        Some(warnings.join("; "))
    };

    (is_valid, warning_message)
}
