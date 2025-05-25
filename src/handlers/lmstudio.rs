// src/handlers/lmstudio.rs - Optimized LM Studio handlers with lightweight context

use serde_json::Value;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::common::{handle_json_response, CancellableRequest, RequestContext};
use crate::constants::*;
use crate::handlers::helpers::json_response;
use crate::handlers::retry::{with_retry_and_cancellation, with_simple_retry};
use crate::handlers::streaming::{handle_passthrough_streaming_response, is_streaming_request};
use crate::model::ModelResolver;
use crate::utils::ProxyError;

/// Handle direct LM Studio API passthrough with model resolution
pub async fn handle_lmstudio_passthrough(
    context: RequestContext<'_>,
    method: &str,
    endpoint: &str,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    // Extract model name for proper resolution
    let original_model_name = body.get("model")
        .and_then(|m| m.as_str());

    let operation = {
        let context = context.clone();
        let method = method.to_string();
        let endpoint = endpoint.to_string();
        let body = body.clone();
        let cancellation_token = cancellation_token.clone();
        let original_model_name = original_model_name.map(|s| s.to_string());

        move || {
            let context = context.clone();
            let method = method.clone();
            let endpoint = endpoint.clone();
            let mut body = body.clone();
            let cancellation_token = cancellation_token.clone();
            let original_model_name = original_model_name.clone();

            async move {
                // Resolve model name if present
                if let Some(ref model_name) = original_model_name {
                    let resolver = ModelResolver::new(context.clone());
                    let resolved_model = resolver.resolve_model_name(model_name, cancellation_token.clone()).await?;

                    // Update request body with resolved model name
                    if let Some(body_obj) = body.as_object_mut() {
                        body_obj.insert("model".to_string(), Value::String(resolved_model.clone()));
                    }
                }

                let url = format!("{}{}", context.lmstudio_url, endpoint);
                let is_streaming = is_streaming_request(&body);

                context.logger.log_request(&method, &url, original_model_name.as_deref());

                let request_method = match method.as_str() {
                    "GET" => reqwest::Method::GET,
                    "POST" => reqwest::Method::POST,
                    "PUT" => reqwest::Method::PUT,
                    "DELETE" => reqwest::Method::DELETE,
                    _ => return Err(ProxyError::bad_request(&format!("Unsupported method: {}", method))),
                };

                let request = CancellableRequest::new(context.clone(), cancellation_token.clone());

                let request_body = if method == "GET" || method == "DELETE" {
                    None
                } else {
                    Some(body.clone())
                };

                let response = request.make_request(request_method, &url, request_body).await?;

                // Enhanced error handling
                if !response.status().is_success() {
                    let status = response.status();
                    let error_message = match status.as_u16() {
                        404 => format!("LM Studio endpoint not found: {}", endpoint),
                        503 => ERROR_LM_STUDIO_UNAVAILABLE.to_string(),
                        400 => "Bad request to LM Studio".to_string(),
                        401 | 403 => "Authentication/Authorization error".to_string(),
                        500 => "LM Studio internal error".to_string(),
                        _ => format!("LM Studio error ({})", status),
                    };
                    return Err(ProxyError::new(error_message, status.as_u16()));
                }

                if is_streaming {
                    handle_passthrough_streaming_response(
                        response,
                        cancellation_token.clone(),
                        context.logger,
                        context.timeout_seconds
                    ).await
                } else {
                    let json_data = handle_json_response(response, cancellation_token).await?;
                    Ok(json_response(&json_data))
                }
            }
        }
    };

    // Use model-specific retry if we have a model name, otherwise simple retry
    let result = if let Some(model) = original_model_name {
        with_retry_and_cancellation(&context, &model, 3, operation, cancellation_token).await?
    } else {
        with_simple_retry(operation, cancellation_token).await?
    };

    context.logger.log_timed(LOG_PREFIX_SUCCESS, "LM Studio passthrough", start_time);
    Ok(result)
}

/// Validate LM Studio endpoint for security
pub fn validate_lmstudio_endpoint(endpoint: &str) -> Result<(), ProxyError> {
    // Allowed endpoints
    const ALLOWED_ENDPOINTS: &[&str] = &[
        "/v1/models",
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
    ];

    // Check if endpoint is allowed
    let is_allowed = ALLOWED_ENDPOINTS.iter().any(|allowed| {
        endpoint == *allowed || endpoint.starts_with(&format!("{}/", allowed))
    });

    if !is_allowed {
        return Err(ProxyError::bad_request(&format!("Endpoint not allowed: {}", endpoint)));
    }

    // Security checks
    if endpoint.contains("..") {
        return Err(ProxyError::bad_request("Path traversal not allowed"));
    }

    if endpoint.len() > 200 {
        return Err(ProxyError::bad_request("Endpoint path too long"));
    }

    Ok(())
}

/// Get LM Studio server status for health checks
pub async fn get_lmstudio_status(
    context: RequestContext<'_>,
    cancellation_token: CancellationToken,
) -> Result<Value, ProxyError> {
    let url = format!("{}/v1/models", context.lmstudio_url);

    let request = CancellableRequest::new(context.clone(), cancellation_token.clone());

    match request.make_request(reqwest::Method::GET, &url, None).await {
        Ok(response) => {
            let status = response.status();
            let is_healthy = status.is_success();

            Ok(serde_json::json!({
                "status": if is_healthy { "healthy" } else { "unhealthy" },
                "lmstudio_url": context.lmstudio_url,
                "http_status": status.as_u16(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        },
        Err(_) => {
            Ok(serde_json::json!({
                "status": "unreachable",
                "lmstudio_url": context.lmstudio_url,
                "error": ERROR_LM_STUDIO_UNAVAILABLE,
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}
