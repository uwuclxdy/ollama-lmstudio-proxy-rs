use serde_json::Value;
use std::time::Instant;

use crate::server::ProxyServer;
use crate::utils::{format_duration, ProxyError};
use super::retry::with_retry;
use super::streaming::{is_streaming_request, handle_passthrough_streaming_response};
use super::helpers::json_response;

/// Handle direct LM Studio API passthrough with streaming support
pub async fn handle_lmstudio_passthrough(
    server: ProxyServer,
    method: &str,
    endpoint: &str,
    body: Value,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    let operation = || async {
        let url = format!("{}{}", server.config.lmstudio_url, endpoint);
        let is_streaming = is_streaming_request(&body);

        server.logger.log(&format!("Passthrough: {} {} (stream: {})", method, url, is_streaming));

        let request_builder = match method {
            "GET" => server.client.get(&url),
            "POST" => server.client.post(&url).json(&body),
            "PUT" => server.client.put(&url).json(&body),
            "DELETE" => server.client.delete(&url),
            _ => return Err(ProxyError::bad_request(&format!("Unsupported method: {}", method))),
        };

        let response = request_builder.send().await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to reach LM Studio: {}", e)))?;

        let status = response.status();

        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProxyError::new(error_text, status.as_u16()));
        }

        if is_streaming {
            // For streaming requests, pass through the response directly
            handle_passthrough_streaming_response(response).await
        } else {
            // Handle regular JSON response
            let json_data: Value = response.json().await
                .map_err(|e| ProxyError::internal_server_error(&format!("Failed to parse LM Studio response: {}", e)))?;

            Ok(json_response(&json_data))
        }
    };

    let result = with_retry(&server, operation).await?;
    let duration = start_time.elapsed();

    server.logger.log(&format!("LM Studio passthrough completed (took {})", format_duration(duration)));
    Ok(result)
}
