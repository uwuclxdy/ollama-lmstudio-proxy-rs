// src/handlers/retry.rs - Unified auto-retry infrastructure with cancellation support

use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;
use serde_json::json;

use crate::server::ProxyServer;
use crate::utils::{is_model_loading_error, ProxyError, clean_model_name};
use crate::common::CancellableRequest;

/// Model loading trigger that works for both initial loading and model switching
pub async fn trigger_model_loading(
    server: &ProxyServer,
    model_name: &str,
    cancellation_token: CancellationToken,
) -> Result<bool, ProxyError> {
    let cleaned_model = clean_model_name(model_name);
    server.logger.log(&format!("Attempting to trigger loading/switching for model: {}", cleaned_model));

    // Make a minimal chat completion request to trigger model loading/switching
    let url = format!("{}/v1/chat/completions", server.config.lmstudio_url);

    let minimal_request = json!({
        "model": cleaned_model,
        "messages": [
            {
                "role": "user",
                "content": "test"
            }
        ],
        "max_tokens": 1,
        "stream": false,
        "temperature": 0.7
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
                server.logger.log("Model loading trigger was cancelled");
                return Err(ProxyError::request_cancelled());
            }
            server.logger.log(&format!("Error triggering model loading: {}", e));
            return Err(ProxyError::internal_server_error(&format!("Failed to communicate with LM Studio: {}", e)));
        }
    };

    let status = response.status();

    if status.is_success() {
        server.logger.log(&format!("Model loading/switching triggered successfully for: {}", cleaned_model));
        Ok(true)
    } else if status.as_u16() == 400 || status.as_u16() == 404 {
        // These status codes might still trigger model loading in LM Studio
        server.logger.log(&format!("Model loading request returned status {} for {}, but may have triggered loading", status, cleaned_model));
        Ok(true)
    } else {
        // Log error response
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        server.logger.log(&format!("Model loading failed with status {}: {}", status, error_text));

        // Still return success as the request might have triggered something
        Ok(true)
    }
}

/// Generic retry wrapper with cancellation support and model-specific loading
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
    server.logger.log(&format!("Attempting operation with model: {}", cleaned_model));

    // First attempt
    match operation().await {
        Ok(result) => {
            server.logger.log(&format!("Operation succeeded on first attempt for model: {}", cleaned_model));
            Ok(result)
        },
        Err(e) if e.is_cancelled() => {
            server.logger.log("Operation was cancelled");
            Err(ProxyError::request_cancelled())
        }
        Err(e) => {
            let error_msg = &e.message;
            server.logger.log(&format!("First attempt failed for model '{}': {}", cleaned_model, error_msg));

            if is_model_loading_error(error_msg) {
                server.logger.log(&format!("Detected model loading error for '{}', attempting retry with model loading/switching...", cleaned_model));

                // Check if cancelled before attempting retry
                if cancellation_token.is_cancelled() {
                    return Err(ProxyError::request_cancelled());
                }

                // Trigger model loading/switching with cancellation support
                match trigger_model_loading(server, &cleaned_model, cancellation_token.clone()).await {
                    Ok(true) => {
                        server.logger.log(&format!("Model loading/switching triggered for '{}', waiting for model to be ready...", cleaned_model));

                        // Wait for model loading with cancellation support
                        let sleep_future = sleep(Duration::from_secs(server.config.load_timeout_seconds));

                        tokio::select! {
                            _ = sleep_future => {
                                server.logger.log(&format!("Wait period completed, retrying operation for: {}", cleaned_model));
                            }
                            _ = cancellation_token.cancelled() => {
                                server.logger.log("Retry cancelled during model loading wait");
                                return Err(ProxyError::request_cancelled());
                            }
                        }

                        // Check cancellation one more time before retry
                        if cancellation_token.is_cancelled() {
                            return Err(ProxyError::request_cancelled());
                        }

                        server.logger.log(&format!("Retrying operation after model loading/switching for: {}", cleaned_model));

                        // Retry the operation
                        match operation().await {
                            Ok(result) => {
                                server.logger.log(&format!("Operation succeeded after retry for model: {}", cleaned_model));
                                Ok(result)
                            },
                            Err(retry_error) => {
                                server.logger.log(&format!("Operation failed even after retry for model '{}': {}", cleaned_model, retry_error.message));
                                Err(retry_error)
                            }
                        }
                    },
                    Ok(false) => {
                        server.logger.log(&format!("Model loading trigger returned false for: {}", cleaned_model));
                        Err(e)
                    },
                    Err(trigger_error) => {
                        server.logger.log(&format!("Failed to trigger model loading for '{}': {}", cleaned_model, trigger_error.message));
                        Err(e) // Return original error, not trigger error
                    }
                }
            } else {
                // If not a model loading error, return original error
                server.logger.log(&format!("Error is not model-related for '{}', not retrying: {}", cleaned_model, error_msg));
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
