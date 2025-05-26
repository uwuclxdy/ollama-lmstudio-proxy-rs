/// src/handlers/retry.rs - Enhanced retry logic with model loading detection and timing
use serde::Serialize;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use crate::check_cancelled;
use crate::common::{CancellableRequest, RequestContext};
use crate::constants::ERROR_LM_STUDIO_UNAVAILABLE;
use crate::model_legacy::clean_model_name_legacy;
use crate::utils::{is_model_loading_error, log_error, log_timed, log_warning, ProxyError};

#[derive(Serialize)]
struct MinimalChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct MinimalChatRequestPayload<'a> {
    model: &'a str,
    messages: Vec<MinimalChatMessage<'a>>,
    max_tokens: u32,
    stream: bool,
}

/// Trigger model loading via minimal request
pub async fn trigger_model_loading(
    context: &RequestContext<'_>,
    ollama_model_name: &str,
    cancellation_token: CancellationToken,
) -> Result<bool, ProxyError> {
    let cleaned_ollama_model_for_logging = clean_model_name_legacy(ollama_model_name);
    let model_for_lm_studio_trigger = cleaned_ollama_model_for_logging;

    let url = format!("{}/v1/chat/completions", context.lmstudio_url);
    let minimal_request_body = MinimalChatRequestPayload {
        model: model_for_lm_studio_trigger,
        messages: vec![MinimalChatMessage {
            role: "user",
            content: "ping",
        }],
        max_tokens: 1,
        stream: false,
    };

    let request = CancellableRequest::new(context.clone(), cancellation_token.clone());

    match request
        .make_request(
            reqwest::Method::POST,
            &url,
            Some(minimal_request_body),
        )
        .await
    {
        Ok(response) => {
            let status = response.status();
            let trigger_considered_successful = status.is_success() || status.is_client_error();

            if !trigger_considered_successful {
                log_warning("Model trigger", &format!("Status: {}", status));
            }
            Ok(trigger_considered_successful)
        }
        Err(e) if e.is_cancelled() => Err(ProxyError::request_cancelled()),
        Err(e) if e.is_lm_studio_unavailable() => {
            Err(ProxyError::lm_studio_unavailable(
                ERROR_LM_STUDIO_UNAVAILABLE,
            ))
        }
        Err(e) => {
            log_error("Model trigger", &e.message);
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
            log_warning("Load hint", &format!("Trigger for '{}' failed, proceeding", ollama_model_name));
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Enhanced retry wrapper with model loading detection and timing
pub async fn with_retry_and_cancellation<F, Fut, T>(
    context: &RequestContext<'_>,
    ollama_model_name: &str,
    load_timeout_seconds: u64,
    operation: F,
    cancellation_token: CancellationToken,
) -> Result<T, ProxyError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, ProxyError>>,
{
    check_cancelled!(cancellation_token);

    match operation().await {
        Ok(result) => Ok(result),
        Err(e) if e.is_cancelled() => Err(ProxyError::request_cancelled()),
        Err(e) if e.is_lm_studio_unavailable() => {
            log_error("Request failed", "LM Studio unavailable - failing fast");
            Err(e)
        }
        Err(e) => {
            if is_model_loading_error(&e.message) {
                let model_loading_start = Instant::now();
                log_timed(crate::constants::LOG_PREFIX_INFO, &format!("{} not loaded, triggering", ollama_model_name), model_loading_start);

                match trigger_model_loading(context, ollama_model_name, cancellation_token.clone())
                    .await
                {
                    Ok(true) => {
                        tokio::select! {
                            _ = sleep(Duration::from_secs(load_timeout_seconds)) => {},
                            _ = cancellation_token.cancelled() => {
                                return Err(ProxyError::request_cancelled());
                            }
                        }
                        check_cancelled!(cancellation_token);

                        match operation().await {
                            Ok(result) => {
                                log_timed(crate::constants::LOG_PREFIX_SUCCESS, &format!("{} loaded", ollama_model_name), model_loading_start);
                                Ok(result)
                            }
                            Err(retry_error) => {
                                log_error(&format!("Retry failed for {}", ollama_model_name), &retry_error.message);
                                Err(e) // Return original error
                            }
                        }
                    }
                    Ok(false) => {
                        log_error("Model trigger", &format!("Failed for {} - model may not exist. Original: {}", ollama_model_name, e.message));
                        Err(e)
                    }
                    Err(loading_trigger_error) => {
                        log_error("Model trigger error", &loading_trigger_error.message);
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
    Fut: std::future::Future<Output = Result<T, ProxyError>>,
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
    let start_time = Instant::now();

    match request
        .make_request(reqwest::Method::GET, &url, None::<serde_json::Value>)
        .await
    {
        Ok(response) => {
            if response.status().is_success() {
                log_timed(crate::constants::LOG_PREFIX_SUCCESS, "LM Studio availability check", start_time);
                Ok(())
            } else {
                log_timed(crate::constants::LOG_PREFIX_ERROR, &format!("LM Studio health check failed: {}", response.status()), start_time);
                Err(ProxyError::lm_studio_unavailable(&format!(
                    "LM Studio health check failed: {}",
                    response.status()
                )))
            }
        }
        Err(e) if e.is_cancelled() => Err(ProxyError::request_cancelled()),
        Err(e) => {
            log_timed(crate::constants::LOG_PREFIX_ERROR, &format!("{}: {}", ERROR_LM_STUDIO_UNAVAILABLE, e.message), start_time);
            Err(ProxyError::lm_studio_unavailable(&format!(
                "{}: {}",
                ERROR_LM_STUDIO_UNAVAILABLE, e.message
            )))
        }
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
    Fut: std::future::Future<Output = Result<T, ProxyError>>,
{
    match ollama_model_name {
        Some(model) => {
            with_retry_and_cancellation(
                context,
                model,
                load_timeout_seconds,
                operation,
                cancellation_token,
            )
                .await
        }
        None => with_simple_retry(operation, cancellation_token).await,
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
