// src/utils.rs - Consolidated utilities with optimized lookups and centralized model resolution

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use warp::reject::Reject;
use tokio_util::sync::CancellationToken;
use serde_json::Value;
use once_cell::sync::Lazy;

use crate::common::CancellableRequest;
use crate::server::ProxyServer;

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

/// Logger utility
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
}

// ===== OPTIMIZED LOOKUP TABLES =====

/// Optimized parameter size lookup using HashMap for O(1) access
static PARAMETER_SIZES: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert("0.5b", "0.5B");
    map.insert("1.5b", "1.5B");
    map.insert("2b", "2B");
    map.insert("3b", "3B");
    map.insert("7b", "7B");
    map.insert("8b", "8B");
    map.insert("9b", "9B");
    map.insert("13b", "13B");
    map.insert("14b", "14B");
    map.insert("27b", "27B");
    map.insert("30b", "30B");
    map.insert("32b", "32B");
    map.insert("70b", "70B");
    map
});

/// Optimized error pattern detection using static patterns
static ERROR_PATTERNS: Lazy<Vec<&'static str>> = Lazy::new(|| {
    vec![
        "no model", "model not loaded", "no models loaded", "model loading",
        "load a model", "model is not loaded", "model not found", "model not available",
        "model does not exist", "unknown model", "invalid model", "model not supported",
        "model mismatch", "requested model", "model is not available", "failed to load model",
        "model failed to load", "cannot find model", "model path not found", "model file not found",
        "model switching", "switching model", "model switch", "loading model",
        "model initialization", "model not ready", "model busy", "model unavailable",
        "please load a model", "no active model", "model required", "specify a model",
        "missing model", "model parameter"
    ]
});

// ===== CORE UTILITIES =====

/// Clean model name by removing :latest and numeric suffixes
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

/// Determine parameter size based on model name using optimized lookup
pub fn determine_parameter_size(model_name: &str) -> &'static str {
    let lower_name = model_name.to_lowercase();

    for (pattern, size) in PARAMETER_SIZES.iter() {
        if lower_name.contains(pattern) {
            return size;
        }
    }

    "7B" // Fallback
}

/// Check if error indicates model loading issues using optimized pattern matching
pub fn is_model_loading_error(message: &str) -> bool {
    let lower_msg = message.to_lowercase();

    ERROR_PATTERNS.iter().any(|pattern| lower_msg.contains(pattern))
        || (lower_msg.contains("404") && lower_msg.contains("model"))
        || (lower_msg.contains("400") && (lower_msg.contains("model") || lower_msg.contains("invalid")))
        || (lower_msg.contains("422") && lower_msg.contains("model"))
        || (lower_msg.contains("503") && (lower_msg.contains("model") || lower_msg.contains("service")))
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

// ===== CENTRALIZED MODEL RESOLVER =====

/// Centralized model resolver - handles all model name resolution logic
pub struct ModelResolver {
    server: Arc<ProxyServer>,
}

impl ModelResolver {
    pub fn new(server: Arc<ProxyServer>) -> Self {
        Self { server }
    }

    /// Resolve Ollama model name to actual LM Studio model name
    /// This is the SINGLE source of truth for model resolution
    pub async fn resolve_model_name(
        &self,
        ollama_model: &str,
        cancellation_token: CancellationToken,
    ) -> Result<String, ProxyError> {
        let cleaned_ollama = clean_model_name(ollama_model);

        // Get available models from LM Studio
        let available_models = self.get_available_models(cancellation_token).await?;

        // Find the best match using simplified logic
        if let Some(matched_model) = self.find_best_match(&cleaned_ollama, &available_models) {
            self.server.logger.log(&format!("✅ Resolved '{}' -> '{}'", ollama_model, matched_model));
            Ok(matched_model)
        } else {
            // Fallback to cleaned name - let LM Studio handle it
            self.server.logger.log(&format!("⚠️  No match found for '{}', using cleaned name '{}'", ollama_model, cleaned_ollama));
            Ok(cleaned_ollama)
        }
    }

    /// Get available models from LM Studio
    async fn get_available_models(&self, cancellation_token: CancellationToken) -> Result<Vec<String>, ProxyError> {
        let url = format!("{}/v1/models", self.server.config.lmstudio_url);

        let request = CancellableRequest::new(
            self.server.client.clone(),
            cancellation_token,
            self.server.logger.clone(),
            self.server.config.request_timeout_seconds
        );

        let response = match request.make_request(reqwest::Method::GET, &url, None).await {
            Ok(response) => response,
            Err(_) => {
                // Return empty list
                return Ok(vec![]);
            }
        };

        if !response.status().is_success() {
            return Ok(vec![]);
        }

        let models_response: Value = match response.json().await {
            Ok(json) => json,
            Err(_) => return Ok(vec![]),
        };

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
    /// Simplified logic - no hardcoded mappings, just pattern matching
    fn find_best_match(&self, ollama_name: &str, available_models: &[String]) -> Option<String> {
        let lower_ollama = ollama_name.to_lowercase();

        // Direct match first
        for model in available_models {
            if model.to_lowercase() == lower_ollama {
                return Some(model.clone());
            }
        }

        // Family and size matching
        for model in available_models {
            let lower_model = model.to_lowercase();

            // Check for family match + size match
            if self.models_match_family_and_size(&lower_ollama, &lower_model) {
                return Some(model.clone());
            }
        }

        // Fallback: partial name matching
        for model in available_models {
            let lower_model = model.to_lowercase();
            let ollama_parts: Vec<&str> = lower_ollama.split(&['-', '_', ':', '.']).filter(|s| s.len() > 2).collect();
            let mut matches = 0;

            for part in &ollama_parts {
                if lower_model.contains(part) {
                    matches += 1;
                }
            }

            // If at least 2 significant parts match, consider it a candidate
            if matches >= 2 {
                return Some(model.clone());
            }
        }

        None
    }

    /// Check if two model names match in family and size
    fn models_match_family_and_size(&self, ollama_name: &str, lm_name: &str) -> bool {
        let families = ["llama", "qwen", "mistral", "gemma", "phi", "deepseek"];
        let sizes = ["0.5b", "1.5b", "2b", "3b", "7b", "8b", "9b", "13b", "14b", "27b", "30b", "32b", "70b"];

        // Check family match
        let family_match = families.iter().any(|family| {
            ollama_name.contains(family) && lm_name.contains(family)
        });

        if !family_match {
            return false;
        }

        // Check size match
        for size in &sizes {
            if ollama_name.contains(size) && lm_name.contains(size) {
                return true;
            }
        }

        // If no size found in either, consider them matching (size unknown)
        true
    }
}
