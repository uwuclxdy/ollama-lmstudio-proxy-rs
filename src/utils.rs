use std::error::Error;
use std::fmt;
use warp::reject::Reject;
use crate::common::CancellableRequest;
use crate::server::ProxyServer;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

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
    /// Create a new ProxyError with custom message and status code
    pub fn new(message: String, status_code: u16) -> Self {
        Self {
            message,
            status_code,
            kind: ProxyErrorKind::Custom
        }
    }

    /// Create an internal server error (500)
    pub fn internal_server_error(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 500,
            kind: ProxyErrorKind::InternalServerError,
        }
    }

    /// Create a bad request error (400)
    pub fn bad_request(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 400,
            kind: ProxyErrorKind::BadRequest,
        }
    }

    /// Create a not found error (404)
    pub fn not_found(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 404,
            kind: ProxyErrorKind::NotFound,
        }
    }

    /// Create a not implemented error (501)
    pub fn not_implemented(message: &str) -> Self {
        Self {
            message: message.to_string(),
            status_code: 501,
            kind: ProxyErrorKind::NotImplemented,
        }
    }

    /// Create a request cancelled error (499 - Client Closed Request)
    pub fn request_cancelled() -> Self {
        Self {
            message: "Request was cancelled".to_string(),
            status_code: 499,
            kind: ProxyErrorKind::RequestCancelled,
        }
    }

    /// Check if this error represents a cancellation
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

/// Logger utility
#[derive(Debug, Clone)]
pub struct Logger {
    pub enabled: bool,
}

impl Logger {
    /// Create new Logger instance
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Log messages
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
pub fn clean_model_name(name: &str) -> String {
    if name.is_empty() {
        return name.to_string();
    }

    let after_latest = if name.ends_with(":latest") {
        &name[..name.len() - 7]
    } else {
        name
    };

    if let Some(colon_pos) = after_latest.rfind(':') {
        let suffix = &after_latest[colon_pos + 1..];

        if !suffix.is_empty()
            && suffix.chars().all(|c| c.is_ascii_digit())
            && colon_pos > 0 {
            return after_latest[..colon_pos].to_string();
        }
    }

    after_latest.to_string()
}

/// Enhanced model name mapping from Ollama format to LM Studio format
/// This tries to intelligently map common Ollama model names to their LM Studio equivalents
pub fn map_ollama_to_lmstudio_model(ollama_name: &str) -> Vec<String> {
    let cleaned = clean_model_name(ollama_name);
    let lower_name = cleaned.to_lowercase();

    let mut candidates = Vec::new();

    // First, always try the cleaned name as-is (for direct matches)
    candidates.push(cleaned.clone());

    // Common model name mappings based on popular models in LM Studio
    if lower_name.contains("llama") {
        if lower_name.contains("3.1") && lower_name.contains("8b") {
            candidates.push("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF".to_string());
            candidates.push("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF".to_string());
        } else if lower_name.contains("3.1") && lower_name.contains("70b") {
            candidates.push("lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF".to_string());
        } else if lower_name.contains("3") && lower_name.contains("8b") {
            candidates.push("lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF".to_string());
            candidates.push("bartowski/Meta-Llama-3-8B-Instruct-GGUF".to_string());
        } else if lower_name.contains("2") && lower_name.contains("7b") {
            candidates.push("lmstudio-community/Llama-2-7b-Chat-GGUF".to_string());
            candidates.push("TheBloke/Llama-2-7B-Chat-GGUF".to_string());
        }
    } else if lower_name.contains("qwen") {
        if lower_name.contains("2.5") && lower_name.contains("7b") {
            candidates.push("lmstudio-community/Qwen2.5-7B-Instruct-GGUF".to_string());
            candidates.push("bartowski/Qwen2.5-7B-Instruct-GGUF".to_string());
        } else if lower_name.contains("2.5") && lower_name.contains("14b") {
            candidates.push("lmstudio-community/Qwen2.5-14B-Instruct-GGUF".to_string());
        } else if lower_name.contains("2") && lower_name.contains("7b") {
            candidates.push("lmstudio-community/Qwen2-7B-Instruct-GGUF".to_string());
        }
    } else if lower_name.contains("mistral") {
        if lower_name.contains("7b") {
            candidates.push("lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF".to_string());
            candidates.push("bartowski/Mistral-7B-Instruct-v0.3-GGUF".to_string());
        }
    } else if lower_name.contains("gemma") {
        if lower_name.contains("2b") {
            candidates.push("lmstudio-community/gemma-2b-it-GGUF".to_string());
        } else if lower_name.contains("7b") {
            candidates.push("lmstudio-community/gemma-7b-it-GGUF".to_string());
        }
    } else if lower_name.contains("phi") {
        if lower_name.contains("3") {
            candidates.push("lmstudio-community/Phi-3-mini-4k-instruct-GGUF".to_string());
            candidates.push("microsoft/Phi-3-mini-4k-instruct-gguf".to_string());
        }
    } else if lower_name.contains("deepseek") {
        if lower_name.contains("coder") {
            candidates.push("lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF".to_string());
        }
    }

    // Add some generic patterns that might work
    if candidates.len() == 1 { // Only the cleaned name was added
        // Try with common prefixes
        candidates.push(format!("lmstudio-community/{}", cleaned));
        candidates.push(format!("bartowski/{}", cleaned));
        candidates.push(format!("TheBloke/{}", cleaned));

        // Try with common suffixes if not present
        if !lower_name.contains("instruct") && !lower_name.contains("chat") {
            candidates.push(format!("{}-Instruct", cleaned));
            candidates.push(format!("{}-Chat", cleaned));
            candidates.push(format!("lmstudio-community/{}-Instruct-GGUF", cleaned));
        }
    }

    candidates
}

/// Enhanced function to check if an error message indicates that a model needs to be loaded or switched
/// Now includes more LM Studio specific patterns and common model switching scenarios
pub fn is_model_loading_error(message: &str) -> bool {
    let lower_msg = message.to_lowercase();

    // Original "no models loaded" patterns
    if lower_msg.contains("no model")
        || lower_msg.contains("model not loaded")
        || lower_msg.contains("no models loaded")
        || lower_msg.contains("model loading")
        || lower_msg.contains("load a model")
        || lower_msg.contains("model is not loaded") {
        return true;
    }

    // Model mismatch/unavailable patterns
    if lower_msg.contains("model not found")
        || lower_msg.contains("model not available")
        || lower_msg.contains("model does not exist")
        || lower_msg.contains("unknown model")
        || lower_msg.contains("invalid model")
        || lower_msg.contains("model not supported")
        || lower_msg.contains("model mismatch")
        || lower_msg.contains("requested model")
        || lower_msg.contains("model is not available") {
        return true;
    }

    // LM Studio specific patterns (common error messages)
    if lower_msg.contains("failed to load model")
        || lower_msg.contains("model failed to load")
        || lower_msg.contains("cannot find model")
        || lower_msg.contains("model path not found")
        || lower_msg.contains("model file not found") {
        return true;
    }

    // Enhanced LM Studio specific error patterns
    if lower_msg.contains("model switching")
        || lower_msg.contains("switching model")
        || lower_msg.contains("model switch")
        || lower_msg.contains("loading model")
        || lower_msg.contains("model initialization")
        || lower_msg.contains("model not ready")
        || lower_msg.contains("model busy")
        || lower_msg.contains("model unavailable")
        || lower_msg.contains("please load a model")
        || lower_msg.contains("no active model")
        || lower_msg.contains("model required")
        || lower_msg.contains("specify a model")
        || lower_msg.contains("missing model")
        || lower_msg.contains("model parameter") {
        return true;
    }

    // HTTP status patterns that might indicate model issues
    if (lower_msg.contains("404") || lower_msg.contains("not found")) && lower_msg.contains("model") {
        return true;
    }

    if (lower_msg.contains("400") || lower_msg.contains("bad request")) &&
        (lower_msg.contains("model") || lower_msg.contains("invalid")) {
        return true;
    }

    // Additional HTTP patterns for model switching
    if lower_msg.contains("422") && lower_msg.contains("model") {
        return true;
    }

    if lower_msg.contains("503") && (lower_msg.contains("model") || lower_msg.contains("service")) {
        return true;
    }

    // Pattern for when LM Studio returns errors about model compatibility
    if lower_msg.contains("incompatible")
        || lower_msg.contains("unsupported format")
        || lower_msg.contains("wrong model type") {
        return true;
    }

    false
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

    // Check for suspicious patterns
    let mut warnings = Vec::new();

    if name.contains("::") {
        warnings.push("Multiple consecutive colons detected".to_string());
    }

    if name.starts_with(':') && name.len() > 1 {
        warnings.push("Model name starts with colon".to_string());
    }

    if name.ends_with(':') {
        warnings.push("Model name ends with colon".to_string());
    }

    if name.len() > 200 {
        warnings.push("Model name is unusually long".to_string());
    }

    if name.contains(char::is_whitespace) {
        warnings.push("Model name contains whitespace characters".to_string());
    }

    if name.chars().any(|c| c.is_control() && c != '\t' && c != '\n' && c != '\r') {
        warnings.push("Model name contains control characters".to_string());
    }

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

/// Resolve an Ollama model name to the actual LM Studio model name
/// This fetches available models from LM Studio and finds the best match
pub async fn resolve_model_name(
    server: &ProxyServer,
    ollama_model: &str,
    cancellation_token: CancellationToken,
) -> Result<String, ProxyError> {
    let cleaned_ollama = clean_model_name(ollama_model);

    // First, get the list of available models from LM Studio
    let available_models = get_available_lmstudio_models(server, cancellation_token.clone()).await?;

    // Try to find the best match
    if let Some(matched_model) = find_best_model_match(&cleaned_ollama, &available_models) {
        server.logger.log(&format!("✅ Resolved '{}' -> '{}'", ollama_model, matched_model));
        Ok(matched_model)
    } else {
        // Fallback to cleaned name if no match found
        server.logger.log(&format!("⚠️  No match found for '{}', using cleaned name '{}'", ollama_model, cleaned_ollama));
        Ok(cleaned_ollama)
    }
}

/// Get available models from LM Studio
async fn get_available_lmstudio_models(
    server: &ProxyServer,
    cancellation_token: CancellationToken,
) -> Result<Vec<String>, ProxyError> {
    let url = format!("{}/v1/models", server.config.lmstudio_url);

    let request = CancellableRequest::new(
        server.client.clone(),
        cancellation_token,
        server.logger.clone(),
        server.config.request_timeout_seconds
    );

    let response = request.make_request(reqwest::Method::GET, &url, None).await?;

    if !response.status().is_success() {
        return Ok(vec![]);  // Return empty list if unable to fetch models
    }

    let models_response: Value = response.json().await
        .map_err(|e| ProxyError::internal_server_error(&format!("Failed to parse models response: {}", e)))?;

    let mut model_names = Vec::new();

    if let Some(data) = models_response.get("data").and_then(|d| d.as_array()) {
        for model in data {
            if let Some(model_id) = model.get("id").and_then(|id| id.as_str()) {
                model_names.push(model_id.to_string());
            }
        }
    }

    Ok(model_names)
}

/// Find the best matching LM Studio model for an Ollama model name
fn find_best_model_match(ollama_name: &str, available_models: &[String]) -> Option<String> {
    let lower_ollama = ollama_name.to_lowercase();

    // Direct match first
    for model in available_models {
        if model.to_lowercase() == lower_ollama {
            return Some(model.clone());
        }
    }

    // Pattern matching for common model families
    for model in available_models {
        let lower_model = model.to_lowercase();

        // Llama family matching
        if lower_ollama.contains("llama") {
            if lower_model.contains("llama") && models_match_size(&lower_ollama, &lower_model) {
                return Some(model.clone());
            }
        }

        // Qwen family matching  
        else if lower_ollama.contains("qwen") {
            if lower_model.contains("qwen") && models_match_size(&lower_ollama, &lower_model) {
                return Some(model.clone());
            }
        }

        // Mistral family matching
        else if lower_ollama.contains("mistral") {
            if lower_model.contains("mistral") && models_match_size(&lower_ollama, &lower_model) {
                return Some(model.clone());
            }
        }

        // Gemma family matching
        else if lower_ollama.contains("gemma") {
            if lower_model.contains("gemma") && models_match_size(&lower_ollama, &lower_model) {
                return Some(model.clone());
            }
        }

        // Phi family matching
        else if lower_ollama.contains("phi") {
            if lower_model.contains("phi") && models_match_size(&lower_ollama, &lower_model) {
                return Some(model.clone());
            }
        }

        // DeepSeek family matching
        else if lower_ollama.contains("deepseek") {
            if lower_model.contains("deepseek") && models_match_size(&lower_ollama, &lower_model) {
                return Some(model.clone());
            }
        }
    }

    // Fallback: partial name matching
    for model in available_models {
        let lower_model = model.to_lowercase();

        // Check if any significant part of the Ollama name appears in the LM Studio model name
        let ollama_parts: Vec<&str> = lower_ollama.split(&['-', '_', ':', '.']).collect();
        let mut matches = 0;

        for part in &ollama_parts {
            if part.len() > 2 && lower_model.contains(part) {
                matches += 1;
            }
        }

        // If at least 2 parts match, consider it a candidate
        if matches >= 2 {
            return Some(model.clone());
        }
    }

    None
}

/// Check if two model names refer to the same model size
fn models_match_size(name1: &str, name2: &str) -> bool {
    let sizes = ["0.5b", "1.5b", "2b", "3b", "7b", "8b", "9b", "11b", "13b", "14b", "27b", "30b", "32b", "70b"];

    for size in &sizes {
        if name1.contains(size) && name2.contains(size) {
            return true;
        }
    }

    // If no size found in either, consider them matching (size unknown)
    true
}
