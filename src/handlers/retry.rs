// src/handlers/retry.rs - Unified auto-retry infrastructure with cancellation support

use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;
use serde_json::json;

use crate::server::ProxyServer;
use crate::utils::{is_no_models_loaded_error, ProxyError, clean_model_name};
use crate::common::CancellableRequest;

/// Auto-retry infrastructure: trigger model loading by making a minimal chat completion request
pub async fn trigger_model_loading(
    server: &ProxyServer,
    model_name: &str,
    cancellation_token: CancellationToken,
) -> Result<bool, ProxyError> {
    let cleaned_model = clean_model_name(model_name);
    server.logger.log(&format!("Attempting to trigger loading for model: {}", cleaned_model));

    // Make a minimal chat completion request to trigger model loading
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
        "stream": false
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

    if response.status().is_success() {
        server.logger.log(&format!("Model loading triggered successfully for: {}", cleaned_model));
        Ok(true)
    } else {
        // Log status but return success anyway - failed request may still trigger model loading
        server.logger.log(&format!("Model loading request returned status {}, but may have triggered loading", response.status()));
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

    // First attempt
    match operation().await {
        Ok(result) => Ok(result),
        Err(e) if e.is_cancelled() => {
            Err(ProxyError::request_cancelled())
        }
        Err(e) => {
            if is_no_models_loaded_error(&e.message) {
                server.logger.log(&format!("Detected 'no models loaded' error for model '{}', attempting retry with model loading...", model_name));

                // Check if cancelled before attempting retry
                if cancellation_token.is_cancelled() {
                    return Err(ProxyError::request_cancelled());
                }

                // Trigger model loading with cancellation support
                if trigger_model_loading(server, model_name, cancellation_token.clone()).await.unwrap_or(false) {
                    server.logger.log("Waiting for model to load...");

                    // Wait for model loading with cancellation support
                    let sleep_future = sleep(Duration::from_secs(server.config.load_timeout_seconds));

                    tokio::select! {
                        _ = sleep_future => { }
                        _ = cancellation_token.cancelled() => {
                            server.logger.log("Retry cancelled during model loading wait");
                            return Err(ProxyError::request_cancelled());
                        }
                    }

                    server.logger.log(&format!("Retrying operation after model loading for: {}", model_name));

                    // Check cancellation one more time before retry
                    if cancellation_token.is_cancelled() {
                        return Err(ProxyError::request_cancelled());
                    }

                    return operation().await;
                }
            }

            // If not a model loading error or retry failed, return original error
            Err(e)
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
