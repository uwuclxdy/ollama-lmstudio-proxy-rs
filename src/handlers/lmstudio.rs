// src/handlers/lmstudio.rs - Unified LM Studio passthrough handlers with cancellation support

use serde_json::Value;
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::server::ProxyServer;
use crate::utils::{format_duration, ProxyError, clean_model_name};
use crate::common::{CancellableRequest, handle_json_response};
use crate::handlers::retry::with_simple_retry;
use super::retry::with_retry_and_cancellation;
use super::streaming::{is_streaming_request, handle_passthrough_streaming_response};
use super::helpers::json_response;

/// Handle direct LM Studio API passthrough with streaming and cancellation support
pub async fn handle_lmstudio_passthrough(
    server: Arc<ProxyServer>,
    method: &str,
    endpoint: &str,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    // Try to extract model name from request body for retry logic
    let model_name = body.get("model")
        .and_then(|m| m.as_str())
        .map(|m| clean_model_name(m));

    let operation = {
        let server = server.clone();
        let method = method.to_string();
        let endpoint = endpoint.to_string();
        let body = body.clone();
        let cancellation_token = cancellation_token.clone();
        move || {
            let server = server.clone();
            let method = method.clone();
            let endpoint = endpoint.clone();
            let body = body.clone();
            let cancellation_token = cancellation_token.clone();
            async move {
                let url = format!("{}{}", server.config.lmstudio_url, endpoint);
                let is_streaming = is_streaming_request(&body);

                server.logger.log(&format!("Passthrough: {} {} (stream: {})", method, url, is_streaming));

                let request_method = match method.as_str() {
                    "GET" => reqwest::Method::GET,
                    "POST" => reqwest::Method::POST,
                    "PUT" => reqwest::Method::PUT,
                    "DELETE" => reqwest::Method::DELETE,
                    _ => return Err(ProxyError::bad_request(&format!("Unsupported method: {}", method))),
                };

                // Updated CancellableRequest constructor call with logger and timeout
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
                    return Err(ProxyError::new(error_text, status.as_u16()));
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
    let result = if let Some(model) = model_name {
        server.logger.log(&format!("Using model-specific retry for passthrough with model: {}", model));
        with_retry_and_cancellation(&server, &model, operation, cancellation_token).await?
    } else {
        server.logger.log("Using simple retry for passthrough (no model specified)");
        with_simple_retry(&server, operation, cancellation_token).await?
    };

    let duration = start_time.elapsed();
    server.logger.log(&format!("LM Studio passthrough completed (took {})", format_duration(duration)));
    Ok(result)
}
