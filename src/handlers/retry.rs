/// src/handlers/retry.rs - Fail-fast retry logic with model loading triggers.

use serde_json::json;
use std::time::Duration;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use crate::common::{CancellableRequest, RequestContext};
use crate::constants::ERROR_LM_STUDIO_UNAVAILABLE;
use crate::metrics::get_global_metrics;
use crate::model::clean_model_name;
use crate::check_cancelled;
use crate::utils::{is_model_loading_error, ProxyError};

/// Trigger model loading via minimal request
pub async fn trigger_model_loading(
    context: &RequestContext<'_>,
    ollama_model_name: &str,
    cancellation_token: CancellationToken,
) -> Result<bool, ProxyError> {
    let cleaned_ollama_model_for_logging = clean_model_name(ollama_model_name);

    // Record metrics attempt
    if let Some(metrics) = get_global_metrics() {
        let model_name_metrics = cleaned_ollama_model_for_logging.to_string();
        tokio::spawn(async move {
            metrics.record_model_load(&model_name_metrics, false).await;
        });
    }

    let model_for_lm_studio_trigger = cleaned_ollama_model_for_logging;

    let url = format!("{}/v1/chat/completions", context.lmstudio_url);
    let minimal_request_body = json!({
        "model": model_for_lm_studio_trigger,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": false
    });

    let request = CancellableRequest::new(context.clone(), cancellation_token.clone());
    context.logger.log(&format!("Attempting to trigger load for model '{}' (as '{}') via minimal chat request.", ollama_model_name, model_for_lm_studio_trigger));

    match request.make_request(reqwest::Method::POST, &url, Some(minimal_request_body)).await {
        Ok(response) => {
            let status = response.status();
            // Success or client error
            let trigger_considered_successful = status.is_success() || status.is_client_error();

            if let Some(metrics) = get_global_metrics() {
                let model_name_metrics = cleaned_ollama_model_for_logging.to_string();
                tokio::spawn(async move {
                    metrics.record_model_load(&model_name_metrics, trigger_considered_successful).await;
                });
            }

            if !trigger_considered_successful {
                context.logger.log_warning("Model loading trigger", &format!("Trigger returned non-successful/non-client-error status: {}", status));
            } else {
                context.logger.log(&format!("Model loading trigger for '{}' completed with status: {}. Assuming model is loading/loaded.", ollama_model_name, status));
            }
            Ok(trigger_considered_successful)
        }
        Err(e) if e.is_cancelled() => Err(ProxyError::request_cancelled()),
        Err(e) if e.is_lm_studio_unavailable() => {
            Err(ProxyError::lm_studio_unavailable(ERROR_LM_STUDIO_UNAVAILABLE))
        }
        Err(e) => {
            context.logger.log_error("Model loading trigger request failed", &e.message);
            if let Some(metrics) = get_global_metrics() {
                let model_name_metrics = cleaned_ollama_model_for_logging.to_string();
                tokio::spawn(async move {
                    metrics.record_model_load(&model_name_metrics, false).await;
                });
            }
            Ok(false)
        }
    }
}

/// Trigger model loading for Ollama load hints
pub async fn trigger_model_loading_for_ollama(
    context: &RequestContext<'_>,
    ollama_model_name: &str,
    cancellation_token: CancellationToken,
) -> Result<(), ProxyError> {
    match trigger_model_loading(context, ollama_model_name, cancellation_token).await {
        Ok(true) => Ok(()),
        Ok(false) => {
            context.logger.log_warning("Model load hint", &format!("Trigger for '{}' returned false, proceeding with main request.", ollama_model_name));
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Retry wrapper with model loading on failure
pub async fn with_retry_and_cancellation<F, Fut, T>(
    context: &RequestContext<'_>,
    ollama_model_name: &str,
    load_timeout_seconds: u64,
    operation: F,
    cancellation_token: CancellationToken,
) -> Result<T, ProxyError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output=Result<T, ProxyError>>,
{
    check_cancelled!(cancellation_token);

    match operation().await {
        Ok(result) => Ok(result),
        Err(e) if e.is_cancelled() => Err(ProxyError::request_cancelled()),
        Err(e) if e.is_lm_studio_unavailable() => {
            context.logger.log_error("Request failed", "LM Studio unavailable - failing fast");
            Err(e)
        }
        Err(e) => {
            if is_model_loading_error(&e.message) {
                context.logger.log_warning("Model operation failed (likely model not loaded)", &format!("Attempting to trigger load for model: {}", ollama_model_name));

                match trigger_model_loading(context, ollama_model_name, cancellation_token.clone()).await {
                    Ok(true) => {
                        context.logger.log(&format!("Model load triggered for {}. Waiting {}s before retry.", ollama_model_name, load_timeout_seconds));
                        tokio::select! {
                            _ = sleep(Duration::from_secs(load_timeout_seconds)) => {},
                            _ = cancellation_token.cancelled() => {
                                return Err(ProxyError::request_cancelled());
                            }
                        }
                        check_cancelled!(cancellation_token);
                        context.logger.log(&format!("Retrying operation for model: {}", ollama_model_name));
                        match operation().await {
                            Ok(result) => {
                                context.logger.log(&format!("Retry successful for model: {}", ollama_model_name));
                                Ok(result)
                            }
                            Err(retry_error) => {
                                context.logger.log_error(&format!("Retry failed for model: {}", ollama_model_name), &retry_error.message);
                                Err(e) // Return original error
                            }
                        }
                    }
                    Ok(false) => {
                        context.logger.log_error("Model loading trigger", &format!("Failed to confirm model loading for {} or model may not exist. Original error: {}", ollama_model_name, e.message));
                        Err(e)
                    }
                    Err(loading_trigger_error) => {
                        context.logger.log_error("Model loading trigger failed", &loading_trigger_error.message);
                        Err(loading_trigger_error)
                    }
                }
            } else {
                Err(e)
            }
        }
    }
}

/// Simple retry without model-specific logic
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

/// Check LM Studio availability
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
        Err(e) => Err(ProxyError::lm_studio_unavailable(&format!("{}: {}", ERROR_LM_STUDIO_UNAVAILABLE, e.message))),
    }
}

/// Enhanced operation wrapper with optional health checking
pub async fn with_health_check_and_retry<F, Fut, T>(
    context: &RequestContext<'_>,
    ollama_model_name: Option<&str>,
    load_timeout_seconds: u64,
    operation: F,
    cancellation_token: CancellationToken,
) -> Result<T, ProxyError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output=Result<T, ProxyError>>,
{
    match ollama_model_name {
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

/// Determine if error is worth retrying
pub fn should_retry_error(error: &ProxyError) -> bool {
    if is_model_loading_error(&error.message) {
        return true;
    }
    if error.is_cancelled() || error.is_lm_studio_unavailable() {
        return false;
    }
    // Don't retry 4xx except 404
    if error.status_code >= 400 && error.status_code < 500 && error.status_code != 404 {
        return false;
    }
    false
}

/// Calculate exponential backoff delay
pub fn calculate_backoff_delay(attempt: u32, base_delay_ms: u64) -> Duration {
    let delay_ms = base_delay_ms * 2_u64.pow(attempt.min(5)); // Cap at 32x
    Duration::from_millis(delay_ms.min(30_000)) // Cap at 30s
}
