// src/handlers/retry.rs - Simplified retry logic for single-client use

use serde_json::json;
use std::time::Duration;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use crate::common::{CancellableRequest, RequestContext};
use crate::model::clean_model_name;
use crate::utils::{is_model_loading_error, ProxyError};
use crate::check_cancelled;

/// Simplified model loading trigger
pub async fn trigger_model_loading(
    context: &RequestContext<'_>,
    model_name: &str,
    cancellation_token: CancellationToken,
) -> Result<bool, ProxyError> {
    let cleaned_model = clean_model_name(model_name);

    let url = format!("{}/v1/chat/completions", context.lmstudio_url);

    // Minimal request to trigger loading
    let minimal_request = json!({
        "model": cleaned_model,
        "messages": [{"role": "user", "content": "hi"}]
    });

    let request = CancellableRequest::new(context.clone(), cancellation_token);

    match request.make_request(reqwest::Method::POST, &url, Some(minimal_request)).await {
        Ok(response) => {
            let status = response.status();
            // Even 400/404 might trigger model loading in LM Studio
            Ok(status.is_success() || status.as_u16() == 400 || status.as_u16() == 404)
        },
        Err(e) if e.is_cancelled() => Err(ProxyError::request_cancelled()),
        Err(_) => Ok(false),
    }
}

/// Simplified retry wrapper for model-specific operations
pub async fn with_retry_and_cancellation<F, Fut, T>(
    context: &RequestContext<'_>,
    model_name: &str,
    load_timeout_seconds: u64,
    operation: F,
    cancellation_token: CancellationToken,
) -> Result<T, ProxyError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output=Result<T, ProxyError>>,
{
    check_cancelled!(cancellation_token);

    // First attempt
    match operation().await {
        Ok(result) => Ok(result),
        Err(e) if e.is_cancelled() => Err(ProxyError::request_cancelled()),
        Err(e) => {
            // Only retry if it's a model loading error
            if is_model_loading_error(&e.message) {
                // Try to trigger model loading
                if trigger_model_loading(context, model_name, cancellation_token.clone()).await.unwrap_or(false) {
                    // Brief wait for model loading
                    tokio::select! {
                        _ = sleep(Duration::from_secs(load_timeout_seconds)) => {},
                        _ = cancellation_token.cancelled() => {
                            return Err(ProxyError::request_cancelled());
                        }
                    }

                    check_cancelled!(cancellation_token);

                    // Retry the operation
                    operation().await
                } else {
                    Err(e)
                }
            } else {
                Err(e)
            }
        }
    }
}

/// Simple retry wrapper for endpoints without specific models
pub async fn with_simple_retry<F, Fut, T>(
    operation: F,
    cancellation_token: CancellationToken,
) -> Result<T, ProxyError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output=Result<T, ProxyError>>,
{
    check_cancelled!(cancellation_token);
    operation().await
}
