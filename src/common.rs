// src/common.rs - Enhanced infrastructure with runtime configuration support

use serde_json::Value;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::constants::*;
use crate::utils::{ProxyError, Logger};
use crate::{check_cancelled, handle_lm_error};

/// Lightweight request context for concurrent request handling
#[derive(Clone)]
pub struct RequestContext<'a> {
    pub client: &'a reqwest::Client,
    pub logger: &'a Logger,
    pub lmstudio_url: &'a str,
    pub timeout_seconds: u64,
}

/// Optimized cancellable request handler
pub struct CancellableRequest<'a> {
    context: RequestContext<'a>,
    token: CancellationToken,
}

impl<'a> CancellableRequest<'a> {
    pub fn new(context: RequestContext<'a>, token: CancellationToken) -> Self {
        Self { context, token }
    }

    /// Make a cancellable HTTP request with proper error handling
    pub async fn make_request(
        &self,
        method: reqwest::Method,
        url: &str,
        body: Option<Value>,
    ) -> Result<reqwest::Response, ProxyError> {
        check_cancelled!(self.token);

        let mut request_builder = self.context.client.request(method, url);

        if let Some(body) = body {
            request_builder = request_builder
                .header("Content-Type", CONTENT_TYPE_JSON)
                .json(&body);
        }

        request_builder = request_builder.timeout(Duration::from_secs(self.context.timeout_seconds));

        // Race between request and cancellation with proper error handling
        tokio::select! {
            result = request_builder.send() => {
                match result {
                    Ok(response) => Ok(response),
                    Err(err) => {
                        let error_msg = if err.is_timeout() {
                            "Request timeout"
                        } else if err.is_connect() {
                            ERROR_LM_STUDIO_UNAVAILABLE
                        } else if err.is_request() {
                            "Invalid request"
                        } else {
                            "Request failed"
                        };
                        Err(ProxyError::internal_server_error(error_msg))
                    }
                }
            }
            _ = self.token.cancelled() => {
                Err(ProxyError::request_cancelled())
            }
        }
    }
}

/// Enhanced JSON response handling with cancellation support
pub async fn handle_json_response(
    response: reqwest::Response,
    cancellation_token: CancellationToken
) -> Result<Value, ProxyError> {
    check_cancelled!(cancellation_token);
    handle_lm_error!(response);

    tokio::select! {
        result = response.json::<Value>() => {
            result.map_err(|e| {
                ProxyError::internal_server_error(&format!("Invalid JSON from LM Studio: {}", e))
            })
        }
        _ = cancellation_token.cancelled() => {
            Err(ProxyError::request_cancelled())
        }
    }
}

/// Enhanced request size validation using runtime configuration
pub fn validate_request_size(body: &Value) -> Result<(), ProxyError> {
    let config = get_runtime_config();

    // Fast size estimation without full serialization
    let estimated_size = estimate_json_size_optimized(body);

    if estimated_size > config.max_request_size_bytes {
        return Err(ProxyError::bad_request(&format!(
            "{} (size: {} bytes, max: {} bytes)",
            ERROR_REQUEST_TOO_LARGE,
            estimated_size,
            config.max_request_size_bytes
        )));
    }

    Ok(())
}

/// Optimized JSON size estimation with better accuracy
fn estimate_json_size_optimized(value: &Value) -> usize {
    match value {
        Value::Null => 4, // "null"
        Value::Bool(true) => 4, // "true"
        Value::Bool(false) => 5, // "false"
        Value::Number(n) => {
            // More accurate number size estimation
            if n.is_i64() {
                n.as_i64().unwrap().to_string().len()
            } else if n.is_u64() {
                n.as_u64().unwrap().to_string().len()
            } else {
                n.as_f64().unwrap().to_string().len()
            }
        },
        Value::String(s) => {
            // Account for escaped characters and quotes
            s.len() + 2 + count_escape_chars(s)
        },
        Value::Array(arr) => {
            // More accurate array size calculation
            if arr.is_empty() {
                2 // "[]"
            } else {
                2 + arr.iter().map(estimate_json_size_optimized).sum::<usize>() + (arr.len() - 1) // [] + commas
            }
        },
        Value::Object(obj) => {
            // More accurate object size calculation
            if obj.is_empty() {
                2 // "{}"
            } else {
                2 + obj.iter().map(|(k, v)| {
                    k.len() + 3 + count_escape_chars(k) + estimate_json_size_optimized(v) // key + ": + quotes
                }).sum::<usize>() + (obj.len() - 1) // {} + commas
            }
        }
    }
}

/// Count characters that need escaping in JSON strings
fn count_escape_chars(s: &str) -> usize {
    s.chars().filter(|&c| {
        matches!(c, '"' | '\\' | '\n' | '\r' | '\t' | '\u{08}' | '\u{0C}')
    }).count()
}

/// Enhanced model name extraction with validation
pub fn extract_model_name<'a>(body: &'a Value, field_name: &str) -> Result<&'a str, ProxyError> {
    let model = body.get(field_name)
        .and_then(|m| m.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| match field_name {
            "model" => ProxyError::bad_request(ERROR_MISSING_MODEL),
            _ => ProxyError::bad_request("Missing required field"),
        })?;

    // Enhanced validation
    if model.len() > 200 {
        return Err(ProxyError::bad_request("Model name too long (max: 200 characters)"));
    }

    // Check for potentially problematic characters
    if model.chars().any(|c| c.is_control() && c != '\t' && c != '\n' && c != '\r') {
        return Err(ProxyError::bad_request("Model name contains invalid characters"));
    }

    Ok(model)
}

/// Enhanced request builder with common parameters
pub struct RequestBuilder {
    body: serde_json::Map<String, Value>,
}

impl RequestBuilder {
    pub fn new() -> Self {
        Self {
            body: serde_json::Map::new(),
        }
    }

    /// Add required field
    pub fn add_required<T: Into<Value>>(mut self, key: &str, value: T) -> Self {
        self.body.insert(key.to_string(), value.into());
        self
    }

    /// Add optional field
    pub fn add_optional<T: Into<Value>>(mut self, key: &str, value: Option<T>) -> Self {
        if let Some(v) = value {
            self.body.insert(key.to_string(), v.into());
        }
        self
    }

    /// Add field from another JSON object if it exists
    pub fn add_from_source(mut self, key: &str, source: &Value) -> Self {
        if let Some(value) = source.get(key) {
            self.body.insert(key.to_string(), value.clone());
        }
        self
    }

    /// Build the final JSON value
    pub fn build(self) -> Value {
        Value::Object(self.body)
    }
}

impl Default for RequestBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Common parameter mapping for LM Studio requests
pub fn map_ollama_to_lmstudio_params(ollama_options: Option<&Value>) -> serde_json::Map<String, Value> {
    let mut params = serde_json::Map::new();

    if let Some(options) = ollama_options {
        // Direct parameter mappings
        const DIRECT_MAPPINGS: &[&str] = &[
            "temperature", "top_p", "top_k", "presence_penalty",
            "frequency_penalty", "seed", "stop"
        ];

        for &param in DIRECT_MAPPINGS {
            if let Some(value) = options.get(param) {
                params.insert(param.to_string(), value.clone());
            }
        }

        // Special mappings
        if let Some(max_tokens) = options.get("num_predict") {
            params.insert("max_tokens".to_string(), max_tokens.clone());
        }

        if let Some(repeat_penalty) = options.get("repeat_penalty") {
            if !params.contains_key("frequency_penalty") {
                params.insert("frequency_penalty".to_string(), repeat_penalty.clone());
            }
        }

        // Handle system message if present
        if let Some(system) = options.get("system") {
            params.insert("system".to_string(), system.clone());
        }
    }

    params
}

/// Utility function to merge JSON objects efficiently
pub fn merge_json_objects(base: &mut serde_json::Map<String, Value>, overlay: serde_json::Map<String, Value>) {
    for (key, value) in overlay {
        base.insert(key, value);
    }
}
