use std::fmt;
use std::error::Error;
use warp::reject::Reject;

/// Custom error type for the proxy server
#[derive(Debug, Clone)]
pub struct ProxyError {
    pub message: String,
    pub status_code: u16,
}

impl ProxyError {
    /// Create a new ProxyError with custom message and status code
    pub fn new(message: String, status_code: u16) -> Self {
        Self { message, status_code }
    }

    /// Create an internal server error (500)
    pub fn internal_server_error(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 500,
        }
    }

    /// Create a bad request error (400)
    pub fn bad_request(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 400,
        }
    }

    /// Create a not found error (404)
    pub fn not_found(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 404,
        }
    }

    /// Create a not implemented error (501)
    pub fn not_implemented(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 501,
        }
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
/// Examples:
/// - "deepseek-r1-distill-qwen-14b:2" → "deepseek-r1-distill-qwen-14b"
/// - "llama3.2:latest" → "llama3.2"
/// - "model-name:3" → "model-name"
pub fn clean_model_name(name: &str) -> String {
    // First remove :latest suffix
    let name = if name.ends_with(":latest") {
        &name[..name.len() - 8]
    } else {
        name
    };

    // Then remove numeric suffixes like :2, :3, etc.
    if let Some(colon_pos) = name.rfind(':') {
        let suffix = &name[colon_pos + 1..];
        if suffix.chars().all(|c| c.is_ascii_digit()) {
            return name[..colon_pos].to_string();
        }
    }

    name.to_string()
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_clean_model_name() {
        assert_eq!(clean_model_name("deepseek-r1-distill-qwen-14b:2"), "deepseek-r1-distill-qwen-14b");
        assert_eq!(clean_model_name("llama3.2:latest"), "llama3.2");
        assert_eq!(clean_model_name("model-name:3"), "model-name");
        assert_eq!(clean_model_name("simple-model"), "simple-model");
        assert_eq!(clean_model_name("model:tag:version"), "model:tag");
    }

    #[test]
    fn test_is_no_models_loaded_error() {
        assert!(is_no_models_loaded_error("No model loaded"));
        assert!(is_no_models_loaded_error("Model not loaded"));
        assert!(is_no_models_loaded_error("Please load a model first"));
        assert!(is_no_models_loaded_error("Model loading required"));
        assert!(!is_no_models_loaded_error("Invalid request"));
        assert!(!is_no_models_loaded_error("Connection failed"));
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_millis(500)), "500ms");
        assert_eq!(format_duration(Duration::from_millis(1500)), "1.50s");
        assert_eq!(format_duration(Duration::from_millis(2500)), "2.50s");
        assert_eq!(format_duration(Duration::from_millis(999)), "999ms");
        assert_eq!(format_duration(Duration::from_millis(1000)), "1.00s");
    }

    #[test]
    fn test_proxy_error_constructors() {
        let err = ProxyError::internal_server_error("Test error");
        assert_eq!(err.status_code, 500);
        assert_eq!(err.message, "Test error");

        let err = ProxyError::bad_request("Bad request");
        assert_eq!(err.status_code, 400);
        assert_eq!(err.message, "Bad request");

        let err = ProxyError::not_found("Not found");
        assert_eq!(err.status_code, 404);
        assert_eq!(err.message, "Not found");

        let err = ProxyError::not_implemented("Not implemented");
        assert_eq!(err.status_code, 501);
        assert_eq!(err.message, "Not implemented");
    }

    #[test]
    fn test_logger() {
        let logger = Logger::new(true);
        assert!(logger.enabled);

        let logger = Logger::new(false);
        assert!(!logger.enabled);
    }
}
