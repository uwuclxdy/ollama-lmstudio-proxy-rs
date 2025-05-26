/// src/handlers/lmstudio.rs - LM Studio API passthrough handlers with native and legacy support
use serde_json::Value;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::common::{handle_json_response, CancellableRequest, RequestContext};
use crate::constants::*;
use crate::handlers::helpers::json_response;
use crate::handlers::retry::{with_retry_and_cancellation, with_simple_retry};
use crate::handlers::streaming::{handle_passthrough_streaming_response, is_streaming_request};
use crate::server::ModelResolverType;
use crate::utils::{log_request, log_timed, ProxyError};

/// Handle direct LM Studio API passthrough with dual API support
pub async fn handle_lmstudio_passthrough(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    method: &str,
    endpoint: &str,
    body: Value,
    cancellation_token: CancellationToken,
    load_timeout_seconds: u64,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    let original_model_name = body.get("model").and_then(|m| m.as_str());

    let operation = {
        let context = context.clone();
        let model_resolver = model_resolver.clone();
        let method_str = method.to_string();
        let endpoint_str = endpoint.to_string();
        let body_clone = body.clone();
        let cancellation_token_clone = cancellation_token.clone();
        let original_model_name_clone = original_model_name.map(|s| s.to_string());

        move || {
            let context = context.clone();
            let model_resolver = model_resolver.clone();
            let current_method = method_str.clone();
            let current_endpoint = endpoint_str.clone();
            let mut current_body = body_clone.clone();
            let current_cancellation_token = cancellation_token_clone.clone();
            let current_original_model_name = original_model_name_clone.clone();

            async move {
                // Resolve model name based on API type
                if let Some(ref model_name) = current_original_model_name {
                    let resolved_model = match &model_resolver {
                        ModelResolverType::Native(resolver) => {
                            resolver
                                .resolve_model_name(
                                    model_name,
                                    context.client,
                                    current_cancellation_token.clone(),
                                )
                                .await?
                        }
                        ModelResolverType::Legacy(resolver) => {
                            resolver
                                .resolve_model_name_legacy(
                                    model_name,
                                    context.client,
                                    current_cancellation_token.clone(),
                                )
                                .await?
                        }
                    };

                    if let Some(body_obj) = current_body.as_object_mut() {
                        body_obj.insert("model".to_string(), Value::String(resolved_model.clone()));
                    }
                }

                // Determine the correct endpoint URL based on API type and requested endpoint
                let final_endpoint_url = determine_passthrough_endpoint_url(
                    &context.lmstudio_url,
                    &current_endpoint,
                    &model_resolver,
                );

                let is_streaming = is_streaming_request(&current_body);

                log_request(
                    &current_method,
                    &final_endpoint_url,
                    current_original_model_name.as_deref(),
                );

                let request_method = match current_method.as_str() {
                    "GET" => reqwest::Method::GET,
                    "POST" => reqwest::Method::POST,
                    "PUT" => reqwest::Method::PUT,
                    "DELETE" => reqwest::Method::DELETE,
                    _ => {
                        return Err(ProxyError::bad_request(&format!(
                            "Unsupported method: {}",
                            current_method
                        )))
                    }
                };

                let request = CancellableRequest::new(context.clone(), current_cancellation_token.clone());

                let request_body_opt = if current_method == "GET" || current_method == "DELETE" {
                    None
                } else {
                    Some(current_body.clone())
                };

                let response = request
                    .make_request(request_method, &final_endpoint_url, request_body_opt)
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let error_message = match status.as_u16() {
                        404 => {
                            // Provide helpful message for native API 404s
                            if current_endpoint.starts_with("/v1/") && matches!(model_resolver, ModelResolverType::Native(_)) {
                                format!(
                                    "LM Studio endpoint not found: {}. Note: Using native API mode, which targets /api/v0/ endpoints. Error from: {}",
                                    current_endpoint, final_endpoint_url
                                )
                            } else if current_endpoint.starts_with("/api/v0/") && matches!(model_resolver, ModelResolverType::Legacy(_)) {
                                format!(
                                    "LM Studio native API endpoint not available: {}. Try removing --legacy flag or update to LM Studio 0.3.6+. Error from: {}",
                                    current_endpoint, final_endpoint_url
                                )
                            } else {
                                format!("LM Studio endpoint not found: {}", final_endpoint_url)
                            }
                        }
                        503 => ERROR_LM_STUDIO_UNAVAILABLE.to_string(),
                        400 => "Bad request to LM Studio".to_string(),
                        401 | 403 => "Authentication/Authorization error with LM Studio".to_string(),
                        500 => "LM Studio internal error".to_string(),
                        _ => format!("LM Studio error ({})", status),
                    };
                    return Err(ProxyError::new(error_message, status.as_u16()));
                }

                if is_streaming {
                    handle_passthrough_streaming_response(
                        response,
                        current_cancellation_token.clone(),
                        60,
                    )
                        .await
                } else {
                    let json_data = handle_json_response(response, current_cancellation_token).await?;
                    Ok(json_response(&json_data))
                }
            }
        }
    };

    let result = if let Some(model) = original_model_name {
        with_retry_and_cancellation(
            &context,
            model,
            load_timeout_seconds,
            operation,
            cancellation_token,
        )
            .await?
    } else {
        with_simple_retry(operation, cancellation_token).await?
    };

    log_timed(LOG_PREFIX_SUCCESS, "LM Studio passthrough", start_time);
    Ok(result)
}

/// Determine the correct endpoint URL based on API type and requested path
fn determine_passthrough_endpoint_url(
    lmstudio_base_url: &str,
    requested_endpoint: &str,
    model_resolver: &ModelResolverType,
) -> String {
    match model_resolver {
        ModelResolverType::Native(_) => {
            // For native mode, convert v1 endpoints to v0 endpoints
            let converted_endpoint = if requested_endpoint.starts_with("/v1/") {
                requested_endpoint.replace("/v1/", "/api/v0/")
            } else {
                requested_endpoint.to_string()
            };
            format!("{}{}", lmstudio_base_url, converted_endpoint)
        }
        ModelResolverType::Legacy(_) => {
            // For legacy mode, keep v1 endpoints as-is, convert v0 to v1
            let converted_endpoint = if requested_endpoint.starts_with("/api/v0/") {
                requested_endpoint.replace("/api/v0/", "/v1/")
            } else {
                requested_endpoint.to_string()
            };
            format!("{}{}", lmstudio_base_url, converted_endpoint)
        }
    }
}

/// Get LM Studio server status for health checks with dual API support
pub async fn get_lmstudio_status(
    context: RequestContext<'_>,
    model_resolver: Option<&ModelResolverType>,
    cancellation_token: CancellationToken,
) -> Result<Value, ProxyError> {
    let endpoint = match model_resolver {
        Some(ModelResolverType::Native(_)) => "/api/v0/models",
        _ => "/v1/models", // Default to legacy for health checks
    };

    let url = format!("{}{}", context.lmstudio_url, endpoint);
    let request = CancellableRequest::new(context.clone(), cancellation_token.clone());

    match request
        .make_request(reqwest::Method::GET, &url, None::<Value>)
        .await
    {
        Ok(response) => {
            let status = response.status();
            let is_healthy = status.is_success();
            let mut additional_info = serde_json::Map::new();

            if is_healthy {
                match response.json::<Value>().await {
                    Ok(models_response) => {
                        let model_count = models_response
                            .get("data")
                            .and_then(|d| d.as_array())
                            .map(|arr| arr.len())
                            .unwrap_or(0);
                        additional_info.insert("model_count".to_string(), serde_json::json!(model_count));

                        // For native API, include additional metadata
                        if endpoint.starts_with("/api/v0/") {
                            if let Some(data) = models_response.get("data").and_then(|d| d.as_array()) {
                                let loaded_count = data.iter()
                                    .filter(|model| {
                                        model.get("state")
                                            .and_then(|s| s.as_str())
                                            .map_or(false, |state| state == "loaded")
                                    })
                                    .count();
                                additional_info.insert("loaded_models".to_string(), serde_json::json!(loaded_count));
                            }
                        }
                    }
                    Err(_) => {}
                }
            }

            let mut result = serde_json::json!({
                "status": if is_healthy { "healthy" } else { "unhealthy" },
                "lmstudio_url": context.lmstudio_url,
                "http_status": status.as_u16(),
                "api_endpoint": endpoint,
                "timestamp": chrono::Utc::now().to_rfc3339()
            });

            // Add additional info if available
            if !additional_info.is_empty() {
                if let Some(result_obj) = result.as_object_mut() {
                    for (key, value) in additional_info {
                        result_obj.insert(key, value);
                    }
                }
            }

            Ok(result)
        }
        Err(_) => {
            let status_message = if endpoint.starts_with("/api/v0/") {
                "unreachable_native_api"
            } else {
                "unreachable"
            };

            Ok(serde_json::json!({
                "status": status_message,
                "lmstudio_url": context.lmstudio_url,
                "error": ERROR_LM_STUDIO_UNAVAILABLE,
                "api_endpoint": endpoint,
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "suggestion": if endpoint.starts_with("/api/v0/") {
                    "Update to LM Studio 0.3.6+ or use --legacy flag"
                } else {
                    "Check LM Studio availability"
                }
            }))
        }
    }
}

/// Helper to convert between API endpoint formats
pub fn convert_endpoint_for_api_type(
    endpoint: &str,
    target_api_type: &ModelResolverType,
) -> String {
    match target_api_type {
        ModelResolverType::Native(_) => {
            if endpoint.starts_with("/v1/") {
                endpoint.replace("/v1/", "/api/v0/")
            } else {
                endpoint.to_string()
            }
        }
        ModelResolverType::Legacy(_) => {
            if endpoint.starts_with("/api/v0/") {
                endpoint.replace("/api/v0/", "/v1/")
            } else {
                endpoint.to_string()
            }
        }
    }
}

/// Check if endpoint is supported by the given API type
pub fn is_endpoint_supported(endpoint: &str, api_type: &ModelResolverType) -> bool {
    match api_type {
        ModelResolverType::Native(_) => {
            // Native API supports both v0 and v1 endpoints (with conversion)
            endpoint.starts_with("/api/v0/") || endpoint.starts_with("/v1/")
        }
        ModelResolverType::Legacy(_) => {
            // Legacy API supports v1 endpoints and converts v0 to v1
            endpoint.starts_with("/v1/") || endpoint.starts_with("/api/v0/")
        }
    }
}
