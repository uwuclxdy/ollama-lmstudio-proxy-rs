// src/handlers/streaming.rs - Unified streaming response handling

use futures_util::StreamExt;
use serde_json::{json, Value};
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::utils::ProxyError;
use super::helpers::{
    extract_content_from_chunk, create_error_chunk, create_cancellation_chunk,
    create_final_chunk
};

/// Check if request has streaming enabled
pub fn is_streaming_request(body: &Value) -> bool {
    body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false)
}

/// Handle streaming responses from LM Studio with proper cancellation support
pub async fn handle_streaming_response(
    response: reqwest::Response,
    is_chat: bool,
    model: &str,
    start_time: Instant,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let model = model.to_string();
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();

    // Spawn task to process the stream with cancellation support
    let model_clone = model.clone();
    let token_clone = cancellation_token.clone();
    let stream_id = format!("stream_{}", chrono::Utc::now().timestamp_millis());

    tokio::spawn(async move {
        log::info!("🌊 [{}] Starting stream processing for model: {}", stream_id, model_clone);

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
                                            log::warn!("🚫 [{}] Client disconnected during streaming after {} chunks", stream_id, chunk_count);
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                        Some(Err(e)) => {
                            // Network error during streaming - send error chunk and stop
                            log::error!("❌ [{}] Streaming error: {}", stream_id, e);
                            let error_chunk = create_error_chunk(&model_clone, &format!("Streaming error: {}", e), is_chat);
                            send_chunk_and_exit(&tx, error_chunk).await;
                            return;
                        }
                        None => {
                            // Stream ended naturally - send final chunk
                            log::info!("✅ [{}] Stream ended naturally after {} chunks", stream_id, chunk_count);
                            break;
                        }
                    }
                }
                // Handle cancellation gracefully - THIS IS THE KEY PART!
                _ = token_clone.cancelled() => {
                    log::warn!("🚫 [{}] Stream cancelled by client disconnection after {} chunks with content: '{}'",
                             stream_id, chunk_count,
                             if partial_content.len() > 50 {
                                 format!("{}...", &partial_content[..50])
                             } else {
                                 partial_content.clone()
                             });

                    // Send a graceful cancellation response
                    let cancellation_chunk = create_cancellation_chunk(
                        &model_clone,
                        &partial_content,
                        start_time.elapsed(),
                        chunk_count,
                        is_chat,
                    );
                    send_chunk_and_exit(&tx, cancellation_chunk).await;

                    // Important: Return immediately to drop the stream and cancel the LM Studio request
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

        log::info!("🏁 [{}] Stream processing completed successfully", stream_id);
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

/// Handle direct streaming passthrough from LM Studio with cancellation support
pub async fn handle_passthrough_streaming_response(
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

/// Helper function to send a chunk and close the channel
async fn send_chunk_and_exit(
    tx: &mpsc::UnboundedSender<Result<bytes::Bytes, std::io::Error>>,
    chunk: Value,
) {
    let chunk_json = serde_json::to_string(&chunk).unwrap_or_default();
    let chunk_with_newline = format!("{}\n", chunk_json);
    let _ = tx.send(Ok(bytes::Bytes::from(chunk_with_newline)));
}

/// Convert SSE message to Ollama chat format
pub fn convert_sse_to_ollama_chat(sse_message: &str, model: &str) -> Option<Value> {
    for line in sse_message.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data.trim() == "[DONE]" {
                return None; // Will be handled by final chunk
            }

            if let Ok(json_data) = serde_json::from_str::<Value>(data) {
                if let Some(choices) = json_data.get("choices").and_then(|c| c.as_array()) {
                    if let Some(first_choice) = choices.first() {
                        if let Some(delta) = first_choice.get("delta") {
                            let content = delta.get("content")
                                .and_then(|c| c.as_str())
                                .unwrap_or("")
                                .to_string();

                            // Only return chunks with actual content
                            if !content.is_empty() {
                                return Some(json!({
                                    "model": model,
                                    "created_at": chrono::Utc::now().to_rfc3339(),
                                    "message": {
                                        "role": "assistant",
                                        "content": content
                                    },
                                    "done": false
                                }));
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

/// Convert SSE message to Ollama generate format
pub fn convert_sse_to_ollama_generate(sse_message: &str, model: &str) -> Option<Value> {
    for line in sse_message.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data.trim() == "[DONE]" {
                return None; // Will be handled by final chunk
            }

            if let Ok(json_data) = serde_json::from_str::<Value>(data) {
                if let Some(choices) = json_data.get("choices").and_then(|c| c.as_array()) {
                    if let Some(first_choice) = choices.first() {
                        let content = first_choice.get("text")
                            .and_then(|t| t.as_str())
                            .unwrap_or("")
                            .to_string();

                        // Only return chunks with actual content
                        if !content.is_empty() {
                            return Some(json!({
                                "model": model,
                                "created_at": chrono::Utc::now().to_rfc3339(),
                                "response": content,
                                "done": false
                            }));
                        }
                    }
                }
            }
        }
    }
    None
}
