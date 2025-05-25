// src/handlers/retry.rs - Updated retry logic using constants and improved error handling

use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;
use serde_json::json;

use crate::constants::*;
use crate::server::ProxyServer;
use crate::model::clean_model_name;
use crate::utils::{is_model_loading_error, ProxyError};
use crate::common::CancellableRequest;

/// Trigger model loading with minimal request using constants
pub async fn trigger_model_loading(
    server: &ProxyServer,
    model_name: &str,
    cancellation_token: CancellationToken,
) -> Result<bool, ProxyError> {
    let cleaned_model = clean_model_name(model_name);
    server.logger.log_with_prefix(LOG_PREFIX_REQUEST, &format!("Triggering model loading for: {}", cleaned_model));

    let url = format!("{}/v1/chat/completions", server.config.lmstudio_url);

    // Truly minimal request - let LM Studio use GUI defaults
    let minimal_request = json!({
        "model": cleaned_model,
        "messages": [{"role": "user", "content": "hi"}]
        // No other parameters - LM Studio will use its GUI-configured defaults
    });

    let request = CancellableRequest::new(
        server.client.clone(),
        cancellation_token.clone(),
        server.logger.clone(),
        server.config.request_timeout_seconds
    );

    let response = match request.make_request(
        reqwest::Method::POST,
        &url,
        Some(minimal_request)
    ).await {
        Ok(response) => response,
        Err(e) => {
            if e.is_cancelled() {
                server.logger.log_cancellation("Model loading trigger");
                return Err(ProxyError::request_cancelled());
            }
            server.logger.log_error("Model loading trigger", &e.message);
            return Err(ProxyError::internal_server_error(&format!("Failed to communicate with LM Studio: {}", e)));
        }
    };

    let status = response.status();

    if status.is_success() {
        server.logger.log_with_prefix(LOG_PREFIX_SUCCESS, &format!("Model loading triggered for: {}", cleaned_model));
        Ok(true)
    } else if status.as_u16() == 400 || status.as_u16() == 404 {
        // These status codes might still trigger model loading in LM Studio
        server.logger.log_with_prefix(LOG_PREFIX_WARNING, &format!("Model loading request returned {} for {}, but may have triggered loading", status, cleaned_model));
        Ok(true)
    } else {
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        server.logger.log_error("Model loading", &format!("Status {}: {}", status, error_text));
        Ok(false)
    }
}

/// Simplified retry wrapper with standardized error handling
pub async fn with_retry_and_cancellation<F, Fut, T>(
    server: &Arc<ProxyServer>,
    model_name: &str,
    operation: F,
    cancellation_token: CancellationToken,
) -> Result<T, ProxyError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output=Result<T, ProxyError>>,
{
    // Check if already cancelled
    if cancellation_token.is_cancelled() {
        return Err(ProxyError::request_cancelled());
    }

    let cleaned_model = clean_model_name(model_name);
    server.logger.log_with_prefix(LOG_PREFIX_REQUEST, &format!("Attempting operation with model: {}", cleaned_model));

    // First attempt
    match operation().await {
        Ok(result) => {
            server.logger.log_with_prefix(LOG_PREFIX_SUCCESS, &format!("Operation succeeded on first attempt for: {}", cleaned_model));
            Ok(result)
        },
        Err(e) if e.is_cancelled() => {
            server.logger.log_cancellation("Operation");
            Err(ProxyError::request_cancelled())
        }
        Err(e) => {
            let error_msg = &e.message;
            server.logger.log_with_prefix(LOG_PREFIX_WARNING, &format!("First attempt failed for '{}': {}", cleaned_model, error_msg));

            // Only retry if it's clearly a model loading issue
            if is_model_loading_error(error_msg) {
                server.logger.log_with_prefix(LOG_PREFIX_REQUEST, &format!("Detected model loading error for '{}', attempting retry...", cleaned_model));

                // Check cancellation before retry
                if cancellation_token.is_cancelled() {
                    return Err(ProxyError::request_cancelled());
                }

                // Try to trigger model loading
                match trigger_model_loading(server, &cleaned_model, cancellation_token.clone()).await {
                    Ok(true) => {
                        server.logger.log_with_prefix(LOG_PREFIX_REQUEST, &format!("Model loading triggered for '{}', waiting briefly...", cleaned_model));

                        // Brief wait for model loading
                        let sleep_future = sleep(Duration::from_secs(server.config.load_timeout_seconds));

                        tokio::select! {
                            _ = sleep_future => {
                                server.logger.log_with_prefix(LOG_PREFIX_REQUEST, &format!("Wait period completed, retrying operation for: {}", cleaned_model));
                            }
                            _ = cancellation_token.cancelled() => {
                                server.logger.log_cancellation("Model loading wait");
                                return Err(ProxyError::request_cancelled());
                            }
                        }

                        // Final cancellation check before retry
                        if cancellation_token.is_cancelled() {
                            return Err(ProxyError::request_cancelled());
                        }

                        server.logger.log_with_prefix(LOG_PREFIX_REQUEST, &format!("Retrying operation after wait for: {}", cleaned_model));

                        // Retry the operation
                        match operation().await {
                            Ok(result) => {
                                server.logger.log_with_prefix(LOG_PREFIX_SUCCESS, &format!("Operation succeeded after retry for: {}", cleaned_model));
                                Ok(result)
                            },
                            Err(retry_error) => {
                                server.logger.log_error("Operation retry", &format!("Failed for '{}': {}", cleaned_model, retry_error.message));
                                Err(retry_error)
                            }
                        }
                    },
                    Ok(false) => {
                        server.logger.log_error("Model loading trigger", &format!("Failed for: {}", cleaned_model));
                        Err(e)
                    },
                    Err(trigger_error) => {
                        server.logger.log_error("Model loading trigger", &format!("Error for '{}': {}", cleaned_model, trigger_error.message));
                        Err(e) // Return original error, not trigger error
                    }
                }
            } else {
                // Not a model loading error, don't retry
                server.logger.log_with_prefix(LOG_PREFIX_WARNING, &format!("Error is not model-related for '{}', not retrying: {}", cleaned_model, error_msg));
                Err(e)
            }
        }
    }
}

/// Simple retry wrapper for endpoints without specific models (like /api/tags)
pub async fn with_simple_retry<F, Fut, T>(
    operation: F,
    cancellation_token: CancellationToken,
) -> Result<T, ProxyError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output=Result<T, ProxyError>>,
{
    // Check if already cancelled
    if cancellation_token.is_cancelled() {
        return Err(ProxyError::request_cancelled());
    }

    // For endpoints like /api/tags that don't have a specific model,
    // just make one attempt without model-specific retry logic
    operation().await
}
