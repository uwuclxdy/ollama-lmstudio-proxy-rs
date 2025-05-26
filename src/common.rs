/// src/common.rs - Enhanced infrastructure with centralized logging

use serde_json::Value;
use std::time::Duration;
use tokio_util::sync::CancellationToken;


use crate::constants::*;
use crate::utils::{log_error, ProxyError};
use crate::{check_cancelled, handle_lm_error};

/// Lightweight request context for concurrent request handling
#[derive(Clone)]
pub struct RequestContext<'a> {
    pub client: &'a reqwest::Client,
    pub lmstudio_url: &'a str,
    pub timeout_seconds: u64,
}

/// Optimized cancellable request handler
pub struct CancellableRequest<'a> {
    context: RequestContext<'a>,
    token: CancellationToken,
}

impl<'a> CancellableRequest<'a> {
    /// Create new cancellable request handler
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

        if let Some(body_content) = body {
            request_builder = request_builder
                .header("Content-Type", CONTENT_TYPE_JSON)
                .json(&body_content);
        }

        request_builder = request_builder.timeout(Duration::from_secs(self.context.timeout_seconds));

        // Race request against cancellation
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
                        log_error("CancellableRequest send", &format!("{}: {:?}", error_msg, err));
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

/// Enhanced model name extraction
pub fn extract_model_name<'a>(body: &'a Value, field_name: &str) -> Result<&'a str, ProxyError> {
    body.get(field_name)
        .and_then(|m| m.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| match field_name {
            "model" => ProxyError::bad_request(ERROR_MISSING_MODEL),
            _ => ProxyError::bad_request("Missing required field"),
        })
}

/// Enhanced request builder with common parameters
pub struct RequestBuilder {
    body: serde_json::Map<String, Value>,
}

impl RequestBuilder {
    /// Create new request builder
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

    /// Build the JSON value
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

        if let Some(repeat_penalty_val) = options.get("repeat_penalty") {
            // Map to frequency_penalty
            if !params.contains_key("frequency_penalty") && !params.contains_key("presence_penalty") {
                params.insert("repeat_penalty".to_string(), repeat_penalty_val.clone());
            } else if !params.contains_key("frequency_penalty") {
                params.insert("frequency_penalty".to_string(), repeat_penalty_val.clone());
            }
        }

        // Handle system message
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
