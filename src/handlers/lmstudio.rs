use serde_json::Value;
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::server::ProxyServer;
use crate::utils::{format_duration, ProxyError};
use super::retry::with_retry_and_cancellation;
use super::streaming::{is_streaming_request, handle_passthrough_streaming_response_with_cancellation};
use super::helpers::json_response;

/// Wrapper for cancellable HTTP requests (same as in ollama.rs, but we'll redefine for clarity)
pub struct CancellableRequest {
    client: reqwest::Client,
    token: CancellationToken,
}

impl CancellableRequest {
    pub fn new(client: reqwest::Client, token: CancellationToken) -> Self {
        Self { client, token }
    }

    /// Make a cancellable HTTP request
    pub async fn make_request(
        &self,
        method: reqwest::Method,
        url: &str,
        body: Option<Value>,
    ) -> Result<reqwest::Response, ProxyError> {
        let mut request_builder = self.client.request(method, url);

        if let Some(body) = body {
            request_builder = request_builder
                .header("Content-Type", "application/json")
                .json(&body);
        }

        let request_future = request_builder.send();

        // Use tokio::select to race between the request and cancellation
        tokio::select! {
            // Request completes normally
            result = request_future => {
                match result {
                    Ok(response) => Ok(response),
                    Err(err) => Err(ProxyError::internal_server_error(&format!("Failed to reach LM Studio: {}", err))),
                }
            }
            // Request was cancelled
            _ = self.token.cancelled() => {
                log::info!("HTTP request to LM Studio was cancelled");
                Err(ProxyError::request_cancelled())
            }
        }
    }
}

/// Handle direct LM Studio API passthrough with streaming and cancellation support
pub async fn handle_lmstudio_passthrough_with_cancellation(
    server: Arc<ProxyServer>,
    method: &str,
    endpoint: &str,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

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

                let request = CancellableRequest::new(server.client.clone(), cancellation_token.clone());

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
                    // For streaming requests, pass through the response directly with cancellation support
                    handle_passthrough_streaming_response_with_cancellation(response, cancellation_token.clone()).await
                } else {
                    // Handle regular JSON response with cancellation support
                    handle_non_streaming_passthrough_response(response, cancellation_token.clone()).await
                }
            }
        }
    };

    let result = with_retry_and_cancellation(&server, operation, cancellation_token).await?;
    let duration = start_time.elapsed();

    server.logger.log(&format!("LM Studio passthrough completed (took {})", format_duration(duration)));
    Ok(result)
}

/// Handle non-streaming passthrough response with cancellation support
async fn handle_non_streaming_passthrough_response(
    response: reqwest::Response,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    // Check if cancelled before processing response
    if cancellation_token.is_cancelled() {
        return Err(ProxyError::request_cancelled());
    }

    let response_future = response.json::<Value>();

    let json_data: Value = tokio::select! {
        result = response_future => {
            result.map_err(|e| ProxyError::internal_server_error(&format!("Failed to parse LM Studio response: {}", e)))?
        }
        _ = cancellation_token.cancelled() => {
            return Err(ProxyError::request_cancelled());
        }
    };

    Ok(json_response(&json_data))
}

// Backwards compatibility function without cancellation
pub async fn handle_lmstudio_passthrough(
    server: ProxyServer,
    method: &str,
    endpoint: &str,
    body: Value,
) -> Result<warp::reply::Response, ProxyError> {
    let token = CancellationToken::new(); // Never gets cancelled
    handle_lmstudio_passthrough_with_cancellation(Arc::new(server), method, endpoint, body, token).await
}
