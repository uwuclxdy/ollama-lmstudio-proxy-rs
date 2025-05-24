// src/handlers/cancellation.rs - Core cancellation functionality
// This file adds the cancellation layer on top of your existing handlers

use serde_json::Value;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use futures_util::StreamExt;
use tokio::sync::mpsc;

use crate::server::ProxyServer;
use crate::utils::ProxyError;
use crate::handlers::streaming::{
    convert_sse_to_ollama_chat, convert_sse_to_ollama_generate, is_streaming_request,
};

/// Wrapper for cancellable HTTP requests
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

/// Enhanced streaming response handler with cancellation support
pub async fn handle_streaming_response_with_cancellation(
    response: reqwest::Response,
    is_chat: bool,
    model: &str,
    start_time: std::time::Instant,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let model = model.to_string();
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();

    // Spawn task to process the stream with cancellation support
    let model_clone = model.clone();
    let token_clone = cancellation_token.clone();
    tokio::spawn(async move {
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut chunk_count = 0u64;
        let mut partial_content = String::new();

        loop {
            tokio::select! {
                // Handle incoming chunks from LM Studio
                chunk_result = stream.next() => {
                    match chunk_result {
                        Some(Ok(chunk)) => {
                            if let Ok(chunk_str) = std::str::from_utf8(&chunk) {
                                buffer.push_str(chunk_str);

                                // Process complete SSE messages (separated by \n\n)
                                while let Some(end_pos) = buffer.find("\n\n") {
                                    let message = buffer[..end_pos].to_string();
                                    buffer = buffer[end_pos + 2..].to_string();

                                    if let Some(ollama_chunk) = if is_chat {
                                        convert_sse_to_ollama_chat(&message, &model_clone)
                                    } else {
                                        convert_sse_to_ollama_generate(&message, &model_clone)
                                    } {
                                        chunk_count += 1;
                                        
                                        // Track partial content for graceful cancellation
                                        if let Some(content) = extract_content_from_chunk(&ollama_chunk) {
                                            partial_content.push_str(&content);
                                        }
                                        
                                        let chunk_json = serde_json::to_string(&ollama_chunk).unwrap_or_default();
                                        let chunk_with_newline = format!("{}\n", chunk_json);

                                        if tx.send(Ok(bytes::Bytes::from(chunk_with_newline))).is_err() {
                                            // Client disconnected, stop processing
                                            log::info!("Client disconnected during streaming");
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                        Some(Err(e)) => {
                            // Network error during streaming - send error chunk and stop
                            let error_chunk = create_error_chunk(&model_clone, &format!("Streaming error: {}", e), is_chat);
                            send_chunk_and_exit(&tx, error_chunk).await;
                            return;
                        }
                        None => {
                            // Stream ended naturally - send final chunk
                            break;
                        }
                    }
                }
                // Handle cancellation gracefully
                _ = token_clone.cancelled() => {
                    log::info!("Stream cancelled by client disconnection");
                    
                    // Send a graceful cancellation response
                    let cancellation_chunk = create_cancellation_chunk(
                        &model_clone,
                        &partial_content,
                        start_time.elapsed(),
                        chunk_count,
                        is_chat,
                    );
                    send_chunk_and_exit(&tx, cancellation_chunk).await;
                    return;
                }
            }
        }

        // Process any remaining buffer content
        if !buffer.trim().is_empty() {
            if let Some(ollama_chunk) = if is_chat {
                convert_sse_to_ollama_chat(&buffer, &model_clone)
            } else {
                convert_sse_to_ollama_generate(&buffer, &model_clone)
            } {
                chunk_count += 1;
                let chunk_json = serde_json::to_string(&ollama_chunk).unwrap_or_default();
                let chunk_with_newline = format!("{}\n", chunk_json);
                let _ = tx.send(Ok(bytes::Bytes::from(chunk_with_newline)));
            }
        }

        // Add final completion chunk with statistics
        let final_chunk = create_final_chunk(&model_clone, start_time.elapsed(), chunk_count, is_chat);
        send_chunk_and_exit(&tx, final_chunk).await;
    });

    // Create stream from receiver
    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);

    // Create streaming response with proper headers for Ollama compatibility
    let response = warp::http::Response::builder()
        .status(warp::http::StatusCode::OK)
        .header("content-type", "application/json; charset=utf-8")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .header("access-control-allow-origin", "*")
        .header("access-control-allow-methods", "GET, POST, PUT, DELETE, OPTIONS")
        .header("access-control-allow-headers", "Content-Type, Authorization")
        .body(warp::hyper::Body::wrap_stream(stream))
        .map_err(|e| ProxyError::internal_server_error(&format!("Failed to create streaming response: {}", e)))?;

    Ok(response)
}

/// Enhanced passthrough streaming with cancellation support
pub async fn handle_passthrough_streaming_response_with_cancellation(
    response: reqwest::Response,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();

    // Spawn task to forward the stream with cancellation and error handling
    tokio::spawn(async move {
        let mut stream = response.bytes_stream();

        loop {
            tokio::select! {
                // Forward chunks from LM Studio
                chunk_result = stream.next() => {
                    match chunk_result {
                        Some(Ok(chunk)) => {
                            if tx.send(Ok(chunk)).is_err() {
                                // Client disconnected, stop forwarding
                                log::info!("Client disconnected during passthrough streaming");
                                return;
                            }
                        }
                        Some(Err(e)) => {
                            // Send error as SSE data and stop
                            let error_data = format!("data: {{\"error\": \"Streaming error: {}\"}}\n\n", e);
                            let _ = tx.send(Ok(bytes::Bytes::from(error_data)));
                            return;
                        }
                        None => {
                            // Stream ended naturally
                            break;
                        }
                    }
                }
                // Handle cancellation
                _ = cancellation_token.cancelled() => {
                    log::info!("Passthrough stream cancelled by client disconnection");
                    // Send cancellation message and stop
                    let cancel_data = "data: {\"cancelled\": true, \"message\": \"Request cancelled by client\"}\n\n";
                    let _ = tx.send(Ok(bytes::Bytes::from(cancel_data)));
                    return;
                }
            }
        }
    });

    // Create stream from receiver
    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);

    // Create streaming response that forwards chunks immediately
    let response = warp::http::Response::builder()
        .status(warp::http::StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .header("access-control-allow-origin", "*")
        .header("access-control-allow-methods", "GET, POST, PUT, DELETE, OPTIONS")
        .header("access-control-allow-headers", "Content-Type, Authorization")
        .body(warp::hyper::Body::wrap_stream(stream))
        .map_err(|e| ProxyError::internal_server_error(&format!("Failed to create passthrough streaming response: {}", e)))?;

    Ok(response)
}

// Helper functions

/// Helper function to send a chunk and close the channel
async fn send_chunk_and_exit(
    tx: &mpsc::UnboundedSender<Result<bytes::Bytes, std::io::Error>>,
    chunk: serde_json::Value,
) {
    let chunk_json = serde_json::to_string(&chunk).unwrap_or_default();
    let chunk_with_newline = format!("{}\n", chunk_json);
    let _ = tx.send(Ok(bytes::Bytes::from(chunk_with_newline)));
}

/// Extract content from a chunk for tracking partial responses
fn extract_content_from_chunk(chunk: &Value) -> Option<String> {
    // For chat format
    if let Some(content) = chunk.get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str()) {
        return Some(content.to_string());
    }

    // For generate format
    if let Some(content) = chunk.get("response")
        .and_then(|r| r.as_str()) {
        return Some(content.to_string());
    }

    None
}

/// Create an error chunk
fn create_error_chunk(model: &str, error_message: &str, is_chat: bool) -> Value {
    if is_chat {
        serde_json::json!({
            "model": model,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "message": {
                "role": "assistant",
                "content": ""
            },
            "done": true,
            "error": error_message
        })
    } else {
        serde_json::json!({
            "model": model,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "response": "",
            "done": true,
            "error": error_message
        })
    }
}

/// Create a cancellation chunk with partial response info
fn create_cancellation_chunk(
    model: &str,
    partial_content: &str,
    duration: std::time::Duration,
    tokens_generated: u64,
    is_chat: bool,
) -> Value {
    let cancellation_message = if partial_content.is_empty() {
        "[Request cancelled before any content was generated]".to_string()
    } else {
        format!("[Request cancelled after {} tokens]", tokens_generated)
    };

    if is_chat {
        serde_json::json!({
            "model": model,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "message": {
                "role": "assistant",
                "content": cancellation_message
            },
            "done": true,
            "total_duration": duration.as_nanos() as u64,
            "load_duration": 1000000u64,
            "prompt_eval_count": 10,
            "prompt_eval_duration": duration.as_nanos() as u64 / 4,
            "eval_count": tokens_generated,
            "eval_duration": duration.as_nanos() as u64 / 2,
            "cancelled": true
        })
    } else {
        serde_json::json!({
            "model": model,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "response": cancellation_message,
            "done": true,
            "context": [1, 2, 3],
            "total_duration": duration.as_nanos() as u64,
            "load_duration": 1000000u64,
            "prompt_eval_count": 10,
            "prompt_eval_duration": duration.as_nanos() as u64 / 4,
            "eval_count": tokens_generated,
            "eval_duration": duration.as_nanos() as u64 / 2,
            "cancelled": true
        })
    }
}

/// Create final completion chunk
fn create_final_chunk(
    model: &str,
    duration: std::time::Duration,
    chunk_count: u64,
    is_chat: bool,
) -> Value {
    if is_chat {
        serde_json::json!({
            "model": model,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "message": {
                "role": "assistant",
                "content": ""
            },
            "done": true,
            "total_duration": duration.as_nanos() as u64,
            "load_duration": 1000000u64,
            "prompt_eval_count": 10,
            "prompt_eval_duration": duration.as_nanos() as u64 / 4,
            "eval_count": chunk_count.max(1),
            "eval_duration": duration.as_nanos() as u64 / 2
        })
    } else {
        serde_json::json!({
            "model": model,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "response": "",
            "done": true,
            "context": [1, 2, 3],
            "total_duration": duration.as_nanos() as u64,
            "load_duration": 1000000u64,
            "prompt_eval_count": 10,
            "prompt_eval_duration": duration.as_nanos() as u64 / 4,
            "eval_count": chunk_count.max(1),
            "eval_duration": duration.as_nanos() as u64 / 2
        })
    }
}