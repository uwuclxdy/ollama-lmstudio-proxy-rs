// src/common.rs - Updated core infrastructure for cancellable operations using constants

use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::constants::*;
use crate::utils::{ProxyError, Logger};

/// Global counter for unique request IDs
static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Unified wrapper for cancellable HTTP requests
pub struct CancellableRequest {
    client: reqwest::Client,
    token: CancellationToken,
    request_id: String,
    logger: Logger,
    timeout_seconds: u64,
}

impl CancellableRequest {
    pub fn new(client: reqwest::Client, token: CancellationToken, logger: Logger, timeout_seconds: u64) -> Self {
        // Generate truly unique request ID using process ID and atomic counter
        let request_id = format!("{}_{}_{}",
                                 REQUEST_ID_PREFIX,
                                 std::process::id(),
                                 REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)
        );

        Self {
            client,
            token,
            request_id,
            logger,
            timeout_seconds
        }
    }

    /// Make a cancellable HTTP request that can be aborted mid-flight
    pub async fn make_request(
        &self,
        method: reqwest::Method,
        url: &str,
        body: Option<Value>,
    ) -> Result<reqwest::Response, ProxyError> {
        let mut request_builder = self.client.request(method, url);

        if let Some(body) = body {
            request_builder = request_builder
                .header("Content-Type", CONTENT_TYPE_JSON)
                .json(&body);
        }

        request_builder = request_builder.timeout(Duration::from_secs(self.timeout_seconds));

        self.logger.log_with_prefix(LOG_PREFIX_REQUEST, &format!("[{}] Starting request to LM Studio: {}", self.request_id, url));

        let request_future = request_builder.send();

        // Use tokio::select to race between the request and cancellation
        tokio::select! {
            result = request_future => {
                match result {
                    Ok(response) => {
                        self.logger.log_with_prefix(LOG_PREFIX_SUCCESS, &format!("[{}] Request completed successfully", self.request_id));
                        Ok(response)
                    },
                    Err(err) => {
                        let error_msg = if err.is_timeout() {
                            format!("Request timeout after {}s", self.timeout_seconds)
                        } else if err.is_connect() {
                            format!("Connection failed: {}", err)
                        } else {
                            format!("Request failed: {}", err)
                        };

                        self.logger.log_error(&format!("[{}] Request", self.request_id), &error_msg);
                        Err(ProxyError::internal_server_error(&format!("Failed to reach LM Studio: {}", error_msg)))
                    }
                }
            }

            // Request was cancelled
            _ = self.token.cancelled() => {
                self.logger.log_with_prefix(LOG_PREFIX_CANCEL, &format!("[{}] HTTP request to LM Studio cancelled by client disconnection", self.request_id));
                Err(ProxyError::request_cancelled())
            }
        }
    }
}

/// Generic helper for handling cancellable JSON responses
pub async fn handle_json_response(response: reqwest::Response, cancellation_token: CancellationToken) -> Result<Value, ProxyError> {
    if cancellation_token.is_cancelled() {
        return Err(ProxyError::request_cancelled());
    }

    let response_future = response.json::<Value>();

    tokio::select! {
        result = response_future => {
            result.map_err(|e| {
                let error_msg = if e.is_decode() {
                    "Invalid JSON response from LM Studio"
                } else {
                    "Failed to parse LM Studio response"
                };
                ProxyError::internal_server_error(&format!("{}: {}", error_msg, e))
            })
        }
        _ = cancellation_token.cancelled() => {
            Err(ProxyError::request_cancelled())
        }
    }
}

/// Generic helper for handling cancellable text responses
pub async fn handle_text_response(response: reqwest::Response, cancellation_token: CancellationToken) -> Result<String, ProxyError> {
    if cancellation_token.is_cancelled() {
        return Err(ProxyError::request_cancelled());
    }

    let response_future = response.text();

    tokio::select! {
        result = response_future => {
            result.map_err(|e| ProxyError::internal_server_error(&format!("Failed to read LM Studio response: {}", e)))
        }
        _ = cancellation_token.cancelled() => {
            Err(ProxyError::request_cancelled())
        }
    }
}

/// Validate request body size to prevent resource exhaustion
pub fn validate_request_size(body: &Value) -> Result<(), ProxyError> {
    let body_str = serde_json::to_string(body)
        .map_err(|e| ProxyError::bad_request(&format!("Invalid JSON: {}", e)))?;

    if body_str.len() > MAX_REQUEST_SIZE_BYTES {
        return Err(ProxyError::bad_request(&format!(
            "Request body too large: {} bytes (max: {} bytes)",
            body_str.len(),
            MAX_REQUEST_SIZE_BYTES
        )));
    }

    Ok(())
}

/// Extract and validate model name from request body
pub fn extract_model_name(body: &Value, field_name: &str) -> Result<String, ProxyError> {
    let model = body.get(field_name)
        .and_then(|m| m.as_str())
        .ok_or_else(|| match field_name {
            "model" => ProxyError::bad_request(ERROR_MISSING_MODEL),
            _ => ProxyError::bad_request(&format!("Missing '{}' field", field_name)),
        })?;

    if model.is_empty() {
        return Err(ProxyError::bad_request("Model name cannot be empty"));
    }

    if model.len() > 200 {
        return Err(ProxyError::bad_request("Model name too long"));
    }

    Ok(model.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_request_id_generation() {
        let logger = Logger::new(false);
        let token = CancellationToken::new();
        let client = reqwest::Client::new();

        let req1 = CancellableRequest::new(client.clone(), token.clone(), logger.clone(), 30);
        let req2 = CancellableRequest::new(client, token, logger, 30);

        assert_ne!(req1.request_id, req2.request_id);
        assert!(req1.request_id.starts_with(REQUEST_ID_PREFIX));
    }

    #[test]
    fn test_validate_request_size() {
        let small_body = json!({"model": "test", "prompt": "hello"});
        assert!(validate_request_size(&small_body).is_ok());

        let large_string = "x".repeat(MAX_REQUEST_SIZE_BYTES + 1);
        let large_body = json!({"model": "test", "prompt": large_string});
        assert!(validate_request_size(&large_body).is_err());
    }

    #[test]
    fn test_extract_model_name() {
        let valid_body = json!({"model": "llama3.2"});
        assert_eq!(extract_model_name(&valid_body, "model").unwrap(), "llama3.2");

        let invalid_body = json!({"prompt": "hello"});
        assert!(extract_model_name(&invalid_body, "model").is_err());

        let empty_model_body = json!({"model": ""});
        assert!(extract_model_name(&empty_model_body, "model").is_err());
    }
}
