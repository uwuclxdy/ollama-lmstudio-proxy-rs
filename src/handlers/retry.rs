use std::time::Duration;
use tokio::time::sleep;

use crate::server::ProxyServer;
use crate::utils::{is_no_models_loaded_error, ProxyError};

/// Auto-retry infrastructure: trigger model loading by calling /v1/models
pub async fn trigger_model_loading(server: &ProxyServer) -> Result<bool, ProxyError> {
    server.logger.log("Attempting to trigger model loading...");

    let url = format!("{}/v1/models", server.config.lmstudio_url);

    match server.client.get(&url).send().await {
        Ok(response) => {
            if response.status().is_success() {
                server.logger.log("Model loading triggered successfully");
                Ok(true)
            } else {
                server.logger.log(&format!("Failed to trigger model loading: {}", response.status()));
                Ok(false)
            }
        }
        Err(e) => {
            server.logger.log(&format!("Error triggering model loading: {}", e));
            Err(ProxyError::internal_server_error(&format!("Failed to communicate with LM Studio: {}", e)))
        }
    }
}

/// Generic retry wrapper that attempts model loading and retries operation
pub async fn with_retry<F, T, Fut>(server: &ProxyServer, operation: F) -> Result<T, ProxyError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output=Result<T, ProxyError>>,
{
    // First attempt
    match operation().await {
        Ok(result) => Ok(result),
        Err(e) => {
            if is_no_models_loaded_error(&e.message) {
                server.logger.log("Detected 'no models loaded' error, attempting retry with model loading...");

                // Trigger model loading
                if trigger_model_loading(server).await.unwrap_or(false) {
                    server.logger.log("Waiting for model to load...");
                    sleep(Duration::from_secs(server.config.load_timeout_seconds)).await;

                    server.logger.log("Retrying operation after model loading...");
                    return operation().await;
                }
            }

            // If not a model loading error or retry failed, return original error
            Err(e)
        }
    }
}