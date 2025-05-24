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
/// 
/// This function handles various model name formats:
/// - Removes `:latest` suffix if present
/// - Removes pure numeric suffixes (e.g., `:2`, `:123`) but only if there's content before the colon
/// - Preserves non-numeric suffixes (e.g., `:custom`, `:alpha`, `:v1.2`) 
/// - Handles multiple colons correctly - only processes the last segment
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_clean_model_name() {
        // Basic cases that should work
        assert_eq!(clean_model_name("simple-model"), "simple-model");
        assert_eq!(clean_model_name("model-name"), "model-name");
        
        // :latest suffix removal
        assert_eq!(clean_model_name("llama3.2:latest"), "llama3.2");
        assert_eq!(clean_model_name("deepseek-r1:latest"), "deepseek-r1");
        assert_eq!(clean_model_name("model:latest"), "model");
        
        // Numeric suffix removal (version numbers)
        assert_eq!(clean_model_name("deepseek-r1-distill-qwen-14b:2"), "deepseek-r1-distill-qwen-14b");
        assert_eq!(clean_model_name("model-name:3"), "model-name");
        assert_eq!(clean_model_name("llama:1"), "llama");
        assert_eq!(clean_model_name("model:123"), "model");
        
        // Non-numeric suffixes should be preserved
        assert_eq!(clean_model_name("model:custom"), "model:custom");
        assert_eq!(clean_model_name("model:alpha"), "model:alpha");
        assert_eq!(clean_model_name("model:beta1"), "model:beta1");
        assert_eq!(clean_model_name("model:v2.1"), "model:v2.1");
        
        // Multiple colons - only remove last numeric suffix
        assert_eq!(clean_model_name("namespace:model:tag:version"), "namespace:model:tag:version");
        assert_eq!(clean_model_name("namespace:model:tag:2"), "namespace:model:tag");
        assert_eq!(clean_model_name("org:model:custom:latest"), "org:model:custom");
        assert_eq!(clean_model_name("a:b:c:d:123"), "a:b:c:d");
        
        // Edge cases
        assert_eq!(clean_model_name("model:"), "model:");
        assert_eq!(clean_model_name(":123"), ":123");
        assert_eq!(clean_model_name(""), "");
        assert_eq!(clean_model_name("model::123"), "model:");
        
        // Complex real-world examples
        assert_eq!(clean_model_name("huggingface:microsoft/DialoGPT-medium:latest"), "huggingface:microsoft/DialoGPT-medium");
        assert_eq!(clean_model_name("registry.com:5000/org/model:v1.2"), "registry.com:5000/org/model:v1.2");
        assert_eq!(clean_model_name("registry.com:5000/org/model:1"), "registry.com:5000/org/model");
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

    #[test]
    fn test_validate_model_name() {
        // Valid names
        assert_eq!(validate_model_name("llama3.2"), (true, None));
        assert_eq!(validate_model_name("model:latest"), (true, None));
        assert_eq!(validate_model_name("namespace:model:tag"), (true, None));
        assert_eq!(validate_model_name("simple-model"), (true, None));
        
        // Invalid names
        let (valid, warning) = validate_model_name("");
        assert!(!valid);
        assert!(warning.unwrap().contains("empty"));
        
        let (valid, warning) = validate_model_name("model::tag");
        assert!(!valid);
        assert!(warning.unwrap().contains("consecutive colons"));
        
        let (valid, warning) = validate_model_name("model:");
        assert!(!valid);
        assert!(warning.unwrap().contains("ends with colon"));
        
        let (valid, warning) = validate_model_name(":model");
        assert!(!valid);
        assert!(warning.unwrap().contains("starts with colon"));
        
        let (valid, warning) = validate_model_name("model name with spaces");
        assert!(!valid);
        assert!(warning.unwrap().contains("whitespace"));
        
        let long_name = "a".repeat(250);
        let (valid, warning) = validate_model_name(&long_name);
        assert!(!valid);
        assert!(warning.unwrap().contains("unusually long"));
        
        let (valid, warning) = validate_model_name("a:b:c:d:e:f");
        assert!(!valid);
        assert!(warning.unwrap().contains("5 colons"));
    }
}
