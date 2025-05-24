// src/handlers/retry.rs - Unified auto-retry infrastructure with cancellation support

use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use crate::server::ProxyServer;
use crate::utils::{is_no_models_loaded_error, ProxyError};

/// Auto-retry infrastructure: trigger model loading by calling /v1/models with cancellation support
pub async fn trigger_model_loading(
    server: &ProxyServer,
    cancellation_token: CancellationToken,
) -> Result<bool, ProxyError> {
    server.logger.log("Attempting to trigger model loading...");

    let url = format!("{}/v1/models", server.config.lmstudio_url);

    let request_future = server.client.get(&url).send();

    let response = tokio::select! {
        result = request_future => {
            match result {
                Ok(response) => response,
                Err(e) => {
                    server.logger.log(&format!("Error triggering model loading: {}", e));
                    return Err(ProxyError::internal_server_error(&format!("Failed to communicate with LM Studio: {}", e)));
                }
            }
        }
        _ = cancellation_token.cancelled() => {
            server.logger.log("Model loading trigger was cancelled");
            return Err(ProxyError::request_cancelled());
        }
    };

    if response.status().is_success() {
        server.logger.log("Model loading triggered successfully");
        Ok(true)
    } else {
        server.logger.log(&format!("Failed to trigger model loading: {}", response.status()));
        Ok(false)
    }
}

/// Generic retry wrapper with cancellation support
pub async fn with_retry_and_cancellation<F, Fut, T>(
    server: &Arc<ProxyServer>,
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
                server.logger.log("Detected 'no models loaded' error, attempting retry with model loading...");

                // Check if cancelled before attempting retry
                if cancellation_token.is_cancelled() {
                    return Err(ProxyError::request_cancelled());
                }

                // Trigger model loading with cancellation support
                if trigger_model_loading(server, cancellation_token.clone()).await.unwrap_or(false) {
                    server.logger.log("Waiting for model to load...");

                    // Wait for model loading with cancellation support
                    let sleep_future = sleep(Duration::from_secs(server.config.load_timeout_seconds));

                    tokio::select! {
                        _ = sleep_future => {
                            // Wait completed, proceed with retry
                        }
                        _ = cancellation_token.cancelled() => {
                            server.logger.log("Retry cancelled during model loading wait");
                            return Err(ProxyError::request_cancelled());
                        }
                    }

                    server.logger.log("Retrying operation after model loading...");

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
