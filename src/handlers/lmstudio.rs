// src/handlers/lmstudio.rs - Updated LM Studio passthrough with model resolution and constants

use serde_json::Value;
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::constants::*;
use crate::server::ProxyServer;
use crate::model::ModelResolver;
use crate::utils::ProxyError;
use crate::common::{CancellableRequest, handle_json_response};
use crate::handlers::helpers::json_response;
use crate::handlers::retry::{with_simple_retry, with_retry_and_cancellation};
use crate::handlers::streaming::{is_streaming_request, handle_passthrough_streaming_response};

/// Handle direct LM Studio API passthrough with model resolution and improved error handling
pub async fn handle_lmstudio_passthrough(
    server: Arc<ProxyServer>,
    method: &str,
    endpoint: &str,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    // Extract model name from request body for proper resolution
    let original_model_name = body.get("model")
        .and_then(|m| m.as_str())
        .map(|m| m.to_string());

    let operation = {
        let server = server.clone();
        let method = method.to_string();
        let endpoint = endpoint.to_string();
        let body = body.clone();
        let cancellation_token = cancellation_token.clone();
        let original_model_name = original_model_name.clone();

        move || {
            let server = server.clone();
            let method = method.clone();
            let endpoint = endpoint.clone();
            let mut body = body.clone();
            let cancellation_token = cancellation_token.clone();
            let original_model_name = original_model_name.clone();

            async move {
                // Resolve model name if present in passthrough request
                if let Some(ref model_name) = original_model_name {
                    let resolver = ModelResolver::new(server.clone());
                    let resolved_model = resolver.resolve_model_name(model_name, cancellation_token.clone()).await?;

                    // Update the request body with the resolved model name
                    if let Some(body_obj) = body.as_object_mut() {
                        body_obj.insert("model".to_string(), Value::String(resolved_model.clone()));
                        server.logger.log_with_prefix(LOG_PREFIX_SUCCESS, &format!("Updated passthrough model: '{}' -> '{}'", model_name, resolved_model));
                    }
                }

                let url = format!("{}{}", server.config.lmstudio_url, endpoint);
                let is_streaming = is_streaming_request(&body);

                server.logger.log_request(&method, &url, original_model_name.as_deref());

                let request_method = match method.as_str() {
                    "GET" => reqwest::Method::GET,
                    "POST" => reqwest::Method::POST,
                    "PUT" => reqwest::Method::PUT,
                    "DELETE" => reqwest::Method::DELETE,
                    _ => return Err(ProxyError::bad_request(&format!("Unsupported method: {}", method))),
                };

                let request = CancellableRequest::new(
                    server.client.clone(),
                    cancellation_token.clone(),
                    server.logger.clone(),
                    server.config.request_timeout_seconds
                );

                let request_body = if method == "GET" || method == "DELETE" {
                    None
                } else {
                    Some(body.clone())
                };

                let response = request.make_request(request_method, &url, request_body).await?;
                let status = response.status();

                if !status.is_success() {
                    let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());

                    // Provide more specific error messages based on status codes
                    let error_message = match status.as_u16() {
                        404 => format!("LM Studio endpoint not found: {}", endpoint),
                        503 => format!("{}: {}", ERROR_LM_STUDIO_UNAVAILABLE, error_text),
                        400 => format!("Bad request to LM Studio: {}", error_text),
                        401 | 403 => format!("Authentication/Authorization error: {}", error_text),
                        500 => format!("LM Studio internal error: {}", error_text),
                        _ => format!("LM Studio error ({}): {}", status, error_text),
                    };

                    return Err(ProxyError::new(error_message, status.as_u16()));
                }

                if is_streaming {
                    handle_passthrough_streaming_response(
                        response,
                        cancellation_token.clone(),
                        server.logger.clone(),
                        server.config.stream_timeout_seconds
                    ).await
                } else {
                    let json_data = handle_json_response(response, cancellation_token).await?;
                    Ok(json_response(&json_data))
                }
            }
        }
    };

    // Use model-specific retry if we have a model name, otherwise use simple retry
    let result = if let Some(model) = original_model_name {
        server.logger.log_with_prefix(LOG_PREFIX_REQUEST, &format!("Using model-specific retry for passthrough with model: {}", model));
        with_retry_and_cancellation(&server, &model, operation, cancellation_token).await?
    } else {
        server.logger.log_with_prefix(LOG_PREFIX_REQUEST, "Using simple retry for passthrough (no model specified)");
        with_simple_retry(operation, cancellation_token).await?
    };

    server.logger.log_success("LM Studio passthrough", start_time.elapsed());
    Ok(result)
}

/// Validate LM Studio endpoint for security
pub fn validate_lmstudio_endpoint(endpoint: &str) -> Result<(), ProxyError> {
    // Only allow specific LM Studio API endpoints
    let allowed_endpoints = [
        "/v1/models",
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
    ];

    // Check if endpoint starts with an allowed pattern
    let is_allowed = allowed_endpoints.iter().any(|allowed| {
        endpoint == *allowed || endpoint.starts_with(&format!("{}/", allowed))
    });

    if !is_allowed {
        return Err(ProxyError::bad_request(&format!("Endpoint not allowed: {}", endpoint)));
    }

    // Additional security checks
    if endpoint.contains("..") {
        return Err(ProxyError::bad_request("Path traversal not allowed"));
    }

    if endpoint.len() > 200 {
        return Err(ProxyError::bad_request("Endpoint path too long"));
    }

    Ok(())
}

/// Get LM Studio server status (for health checks)
pub async fn get_lmstudio_status(
    server: Arc<ProxyServer>,
    cancellation_token: CancellationToken,
) -> Result<Value, ProxyError> {
    let url = format!("{}/v1/models", server.config.lmstudio_url);

    let request = CancellableRequest::new(
        server.client.clone(),
        cancellation_token.clone(),
        server.logger.clone(),
        5 // Short timeout for health check
    );

    match request.make_request(reqwest::Method::GET, &url, None).await {
        Ok(response) => {
            let status = response.status();
            let is_healthy = status.is_success();

            Ok(serde_json::json!({
                "status": if is_healthy { "healthy" } else { "unhealthy" },
                "lmstudio_url": server.config.lmstudio_url,
                "http_status": status.as_u16(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        },
        Err(_) => {
            Ok(serde_json::json!({
                "status": "unreachable",
                "lmstudio_url": server.config.lmstudio_url,
                "error": ERROR_LM_STUDIO_UNAVAILABLE,
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}
