// src/handlers/retry.rs - Fail-fast retry logic with metrics integration

use serde_json::json;
use std::time::Duration;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use crate::common::{CancellableRequest, RequestContext};
use crate::constants::ERROR_LM_STUDIO_UNAVAILABLE;
use crate::metrics::get_global_metrics;
use crate::model::clean_model_name;
use crate::utils::{is_model_loading_error, ProxyError};
use crate::check_cancelled;

/// Fail-fast model loading trigger
pub async fn trigger_model_loading(
    context: &RequestContext<'_>,
    model_name: &str,
    cancellation_token: CancellationToken,
) -> Result<bool, ProxyError> {
    let cleaned_model = clean_model_name(model_name);

    // Record model load attempt in metrics
    if let Some(metrics) = get_global_metrics() {
        metrics.record_model_load(cleaned_model, false).await; // Start as failed, update if successful
    }

    let url = format!("{}/v1/chat/completions", context.lmstudio_url);

    // Minimal request to trigger loading
    let minimal_request = json!({
        "model": cleaned_model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1
    });

    let request = CancellableRequest::new(context.clone(), cancellation_token);

    match request.make_request(reqwest::Method::POST, &url, Some(minimal_request)).await {
        Ok(response) => {
            let status = response.status();
            let success = status.is_success() || status.as_u16() == 400 || status.as_u16() == 404;

            // Update metrics with actual result
            if let Some(metrics) = get_global_metrics() {
                metrics.record_model_load(cleaned_model, success).await;
            }

            if !success {
                context.logger.log_warning("Model loading", &format!("Trigger returned status: {}", status));
            }

            Ok(success)
        }
        Err(e) if e.is_cancelled() => Err(ProxyError::request_cancelled()),
        Err(e) if e.is_lm_studio_unavailable() => {
            // Fail fast if LM Studio is unavailable
            Err(ProxyError::lm_studio_unavailable(ERROR_LM_STUDIO_UNAVAILABLE))
        }
        Err(e) => {
            context.logger.log_error("Model loading trigger", &e.message);
            Ok(false)
        }
    }
}

/// Fail-fast retry wrapper for model-specific operations
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
        Err(e) if e.is_lm_studio_unavailable() => {
            // Fail fast for LM Studio unavailability - don't retry
            context.logger.log_error("Request failed", "LM Studio unavailable - failing fast");
            Err(e)
        }
        Err(e) => {
            // Only retry if it's a model loading error
            if is_model_loading_error(&e.message) {
                context.logger.log_warning("Model loading", &format!("Attempting to load model: {}", model_name));

                // Try to trigger model loading
                match trigger_model_loading(context, model_name, cancellation_token.clone()).await {
                    Ok(true) => {
                        // Brief wait for model loading with cancellation support
                        tokio::select! {
                            _ = sleep(Duration::from_secs(load_timeout_seconds)) => {},
                            _ = cancellation_token.cancelled() => {
                                return Err(ProxyError::request_cancelled());
                            }
                        }

                        check_cancelled!(cancellation_token);

                        // Retry the operation once
                        context.logger.log("Retrying operation after model loading");
                        match operation().await {
                            Ok(result) => {
                                context.logger.log("Retry successful after model loading");
                                Ok(result)
                            }
                            Err(retry_error) => {
                                // If retry fails, return the original error for context
                                context.logger.log_error("Retry failed", &retry_error.message);
                                Err(e) // Return original error, not retry error
                            }
                        }
                    }
                    Ok(false) => {
                        context.logger.log_error("Model loading", "Failed to trigger model loading");
                        Err(e)
                    }
                    Err(loading_error) => {
                        // If we can't even trigger loading, fail fast
                        context.logger.log_error("Model loading trigger", &loading_error.message);
                        Err(loading_error)
                    }
                }
            } else {
                // Not a model loading error, don't retry
                Err(e)
            }
        }
    }
}

/// Simple retry wrapper for endpoints without specific models (fail-fast approach)
pub async fn with_simple_retry<F, Fut, T>(
    operation: F,
    cancellation_token: CancellationToken,
) -> Result<T, ProxyError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output=Result<T, ProxyError>>,
{
    check_cancelled!(cancellation_token);

    // No retries for simple operations - fail fast
    operation().await
}

/// Check LM Studio availability before operations (fail-fast health check)
pub async fn check_lm_studio_availability(
    context: &RequestContext<'_>,
    cancellation_token: CancellationToken,
) -> Result<(), ProxyError> {
    let url = format!("{}/v1/models", context.lmstudio_url);
    let request = CancellableRequest::new(context.clone(), cancellation_token);

    match request.make_request(reqwest::Method::GET, &url, None).await {
        Ok(response) => {
            if response.status().is_success() {
                Ok(())
            } else {
                Err(ProxyError::lm_studio_unavailable(
                    &format!("LM Studio health check failed: {}", response.status())
                ))
            }
        }
        Err(e) if e.is_cancelled() => Err(ProxyError::request_cancelled()),
        Err(_) => Err(ProxyError::lm_studio_unavailable(ERROR_LM_STUDIO_UNAVAILABLE)),
    }
}

/// Enhanced operation wrapper that includes health checking
pub async fn with_health_check_and_retry<F, Fut, T>(
    context: &RequestContext<'_>,
    model_name: Option<&str>,
    load_timeout_seconds: u64,
    operation: F,
    cancellation_token: CancellationToken,
) -> Result<T, ProxyError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output=Result<T, ProxyError>>,
{
    // Optional: Quick health check first (can be disabled for performance)
    // check_lm_studio_availability(context, cancellation_token.clone()).await?;

    match model_name {
        Some(model) => {
            with_retry_and_cancellation(
                context,
                model,
                load_timeout_seconds,
                operation,
                cancellation_token,
            ).await
        }
        None => {
            with_simple_retry(operation, cancellation_token).await
        }
    }
}

/// Utility to determine if an error is worth retrying
pub fn should_retry_error(error: &ProxyError) -> bool {
    // Only retry model loading errors
    if is_model_loading_error(&error.message) {
        return true;
    }

    // Don't retry these specific cases
    if error.is_cancelled() || error.is_lm_studio_unavailable() {
        return false;
    }

    // Don't retry client errors (4xx) except 404 which might be model not found
    if error.status_code >= 400 && error.status_code < 500 && error.status_code != 404 {
        return false;
    }

    // Don't retry by default for fail-fast approach
    false
}

/// Exponential backoff calculation (unused in fail-fast mode, but available)
pub fn calculate_backoff_delay(attempt: u32, base_delay_ms: u64) -> Duration {
    let delay_ms = base_delay_ms * 2_u64.pow(attempt.min(5)); // Cap at 32x base delay
    Duration::from_millis(delay_ms.min(30_000)) // Cap at 30 seconds
}
