// src/common.rs - Optimized infrastructure for single-client use

use serde_json::Value;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::constants::*;
use crate::utils::{ProxyError, Logger};
use crate::{check_cancelled, handle_lm_error};

/// Lightweight request context for single client (no Arc needed)
#[derive(Clone)]
pub struct RequestContext<'a> {
    pub client: &'a reqwest::Client,
    pub logger: &'a Logger,
    pub lmstudio_url: &'a str,
    pub timeout_seconds: u64,
}

/// Simplified cancellable request without request ID overhead
pub struct CancellableRequest<'a> {
    context: RequestContext<'a>,
    token: CancellationToken,
}

impl<'a> CancellableRequest<'a> {
    pub fn new(context: RequestContext<'a>, token: CancellationToken) -> Self {
        Self { context, token }
    }

    /// Make a cancellable HTTP request (optimized for single client)
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

        // Race between request and cancellation
        tokio::select! {
            result = request_builder.send() => {
                match result {
                    Ok(response) => Ok(response),
                    Err(err) => {
                        let error_msg = if err.is_timeout() {
                            "Request timeout"
                        } else if err.is_connect() {
                            "Connection failed"
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

/// Fast JSON response handling with cancellation
pub async fn handle_json_response(
    response: reqwest::Response,
    cancellation_token: CancellationToken
) -> Result<Value, ProxyError> {
    check_cancelled!(cancellation_token);
    handle_lm_error!(response);

    tokio::select! {
        result = response.json::<Value>() => {
            result.map_err(|_| ProxyError::internal_server_error("Invalid JSON from LM Studio"))
        }
        _ = cancellation_token.cancelled() => {
            Err(ProxyError::request_cancelled())
        }
    }
}

/// Optimized request size validation
pub fn validate_request_size(body: &Value) -> Result<(), ProxyError> {
    // Fast size estimation without full serialization
    let estimated_size = estimate_json_size(body);

    if estimated_size > MAX_REQUEST_SIZE_BYTES {
        return Err(ProxyError::bad_request("Request body too large"));
    }

    Ok(())
}

/// Fast JSON size estimation (avoids full serialization)
fn estimate_json_size(value: &Value) -> usize {
    match value {
        Value::Null => 4, // "null"
        Value::Bool(_) => 5, // "false" (max)
        Value::Number(n) => n.to_string().len(),
        Value::String(s) => s.len() + 2, // quotes
        Value::Array(arr) => {
            2 + arr.iter().map(estimate_json_size).sum::<usize>() + arr.len().saturating_sub(1) // [] + commas
        }
        Value::Object(obj) => {
            2 + obj.iter().map(|(k, v)| k.len() + 3 + estimate_json_size(v)).sum::<usize>() + obj.len().saturating_sub(1) // {} + quotes + colons + commas
        }
    }
}

/// Fast model name extraction
pub fn extract_model_name<'a>(body: &'a Value, field_name: &str) -> Result<&'a str, ProxyError> {
    body.get(field_name)
        .and_then(|m| m.as_str())
        .filter(|s| !s.is_empty() && s.len() <= 200)
        .ok_or_else(|| match field_name {
            "model" => ProxyError::bad_request(ERROR_MISSING_MODEL),
            _ => ProxyError::bad_request("Missing required field"),
        })
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_json_size_estimation() {
        let small = json!({"model": "test"});
        assert!(estimate_json_size(&small) < 50);

        let large = json!({"model": "test", "prompt": "x".repeat(1000)});
        assert!(estimate_json_size(&large) > 1000);
    }

    #[test]
    fn test_validate_request_size() {
        let small_body = json!({"model": "test", "prompt": "hello"});
        assert!(validate_request_size(&small_body).is_ok());
    }

    #[test]
    fn test_extract_model_name() {
        let valid_body = json!({"model": "llama3.2"});
        assert_eq!(extract_model_name(&valid_body, "model").unwrap(), "llama3.2");

        let invalid_body = json!({"prompt": "hello"});
        assert!(extract_model_name(&invalid_body, "model").is_err());
    }
}
