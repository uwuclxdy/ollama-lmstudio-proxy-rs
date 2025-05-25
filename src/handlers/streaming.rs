// src/handlers/streaming.rs - Unified streaming response handling with optimized resource management

use futures_util::StreamExt;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

use crate::constants::*;
use crate::handlers::helpers::{
    create_cancellation_chunk, create_error_chunk, create_final_chunk,
    extract_content_from_chunk
};
use crate::utils::{Logger, ProxyError};

/// Global counter for stream IDs
static STREAM_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Check if streaming is enabled in request
pub fn is_streaming_request(body: &Value) -> bool {
    body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false)
}

/// Handle streaming responses from LM Studio with improved resource management
pub async fn handle_streaming_response(
    response: reqwest::Response,
    is_chat: bool,
    model: &str,
    start_time: Instant,
    cancellation_token: CancellationToken,
    logger: Logger,
    stream_timeout_seconds: u64,
) -> Result<warp::reply::Response, ProxyError> {
    let model = model.to_string();
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();

    // Generate unique stream ID
    let stream_id = STREAM_COUNTER.fetch_add(1, Ordering::Relaxed);

    let model_clone = model.clone();
    let token_clone = cancellation_token.clone();
    let logger_clone = logger.clone();

    tokio::spawn(async move {
        logger_clone.log_with_prefix(LOG_PREFIX_STREAM, &format!("[{}] Starting stream processing for model: {}", stream_id, model_clone));

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut chunk_count = 0u64;
        let mut partial_content = String::new();

        let stream_result = 'stream: loop {
            // Check limits before processing
            if chunk_count >= MAX_CHUNK_COUNT {
                logger_clone.log_warning(&format!("[{}] Reached max chunk limit: {}", stream_id, MAX_CHUNK_COUNT));
                send_error_and_close(&tx, &model_clone, ERROR_CHUNK_LIMIT, is_chat).await;
                break 'stream Err(ERROR_CHUNK_LIMIT.to_string());
            }

            tokio::select! {
                chunk_result = timeout(Duration::from_secs(stream_timeout_seconds), stream.next()) => {
                    match chunk_result {
                        Ok(Some(Ok(chunk))) => {
                            if let Ok(chunk_str) = std::str::from_utf8(&chunk) {
                                // Prevent buffer overflow
                                if buffer.len() + chunk_str.len() > MAX_BUFFER_SIZE {
                                    logger_clone.log_warning(&format!("[{}] Buffer overflow prevented", stream_id));
                                    send_error_and_close(&tx, &model_clone, ERROR_BUFFER_OVERFLOW, is_chat).await;
                                    break 'stream Err(ERROR_BUFFER_OVERFLOW.to_string());
                                }

                                buffer.push_str(chunk_str);

                                // Process complete SSE messages
                                while let Some(message_end) = find_complete_sse_message(&buffer) {
                                    let message = buffer[..message_end].to_string();
                                    buffer = buffer[message_end + 2..].to_string();

                                    if let Some(ollama_chunk) = convert_sse_to_ollama(&message, &model_clone, is_chat) {
                                        chunk_count += 1;

                                        // Track partial content (with size limit)
                                        if let Some(content) = extract_content_from_chunk(&ollama_chunk) {
                                            if partial_content.len() < MAX_PARTIAL_CONTENT_SIZE {
                                                partial_content.push_str(&content);
                                            }
                                        }

                                        if !send_chunk(&tx, &ollama_chunk).await {
                                            logger_clone.log_cancellation(&format!("[{}] Client disconnected after {} chunks", stream_id, chunk_count));
                                            break 'stream Ok(());
                                        }
                                    }
                                }
                            }
                        }
                        Ok(Some(Err(e))) => {
                            logger_clone.log_error(&format!("[{}] Streaming", stream_id), &e.to_string());
                            send_error_and_close(&tx, &model_clone, &format!("Streaming error: {}", e), is_chat).await;
                            break 'stream Err(format!("Network error: {}", e));
                        }
                        Ok(None) => {
                            logger_clone.log_success(&format!("[{}] Stream", stream_id), start_time.elapsed());
                            break 'stream Ok(());
                        }
                        Err(_) => {
                            logger_clone.log_warning(&format!("[{}] Stream timeout after {}s", stream_id, stream_timeout_seconds));
                            send_error_and_close(&tx, &model_clone, &format!("{} after {}s", ERROR_TIMEOUT, stream_timeout_seconds), is_chat).await;
                            break 'stream Err(ERROR_TIMEOUT.to_string());
                        }
                    }
                }
                _ = token_clone.cancelled() => {
                    logger_clone.log_cancellation(&format!("[{}] Stream after {} chunks", stream_id, chunk_count));
                    let cancellation_chunk = create_cancellation_chunk(
                        &model_clone,
                        &partial_content,
                        start_time.elapsed(),
                        chunk_count,
                        is_chat,
                    );
                    send_chunk_and_close(&tx, cancellation_chunk).await;
                    break 'stream Err(ERROR_CANCELLED.to_string());
                }
            }
        };

        // Handle remaining buffer content if stream ended successfully
        if stream_result.is_ok() && !token_clone.is_cancelled() && !buffer.trim().is_empty() {
            if let Some(ollama_chunk) = convert_sse_to_ollama(&buffer, &model_clone, is_chat) {
                chunk_count += 1;
                send_chunk(&tx, &ollama_chunk).await;
            }
        }

        // Send final chunk if successful
        if stream_result.is_ok() {
            let final_chunk = create_final_chunk(&model_clone, start_time.elapsed(), chunk_count, is_chat);
            send_chunk_and_close(&tx, final_chunk).await;
            logger_clone.log_success(&format!("[{}] Stream processing", stream_id), start_time.elapsed());
        }
    });

    // Create streaming response with proper headers
    create_streaming_response(rx)
}

/// Handle direct streaming passthrough from LM Studio
pub async fn handle_passthrough_streaming_response(
    response: reqwest::Response,
    cancellation_token: CancellationToken,
    logger: Logger,
    stream_timeout_seconds: u64,
) -> Result<warp::reply::Response, ProxyError> {
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
    let stream_id = STREAM_COUNTER.fetch_add(1, Ordering::Relaxed);

    let logger_clone = logger.clone();
    tokio::spawn(async move {
        let mut stream = response.bytes_stream();

        loop {
            tokio::select! {
                chunk_result = timeout(Duration::from_secs(stream_timeout_seconds), stream.next()) => {
                    match chunk_result {
                        Ok(Some(Ok(chunk))) => {
                            if tx.send(Ok(chunk)).is_err() {
                                logger_clone.log_cancellation(&format!("[{}] Passthrough stream", stream_id));
                                break;
                            }
                        }
                        Ok(Some(Err(e))) => {
                            let error_data = format!("data: {{\"error\": \"Streaming error: {}\"}}\n\n", e);
                            let _ = tx.send(Ok(bytes::Bytes::from(error_data)));
                            break;
                        }
                        Ok(None) => {
                            logger_clone.log_success(&format!("[{}] Passthrough stream", stream_id), Instant::now().elapsed());
                            break;
                        }
                        Err(_) => {
                            logger_clone.log_warning(&format!("[{}] Passthrough stream timeout after {}s", stream_id, stream_timeout_seconds));
                            let timeout_data = format!("data: {{\"error\": \"{} after {}s\"}}\n\n", ERROR_TIMEOUT, stream_timeout_seconds);
                            let _ = tx.send(Ok(bytes::Bytes::from(timeout_data)));
                            break;
                        }
                    }
                }
                _ = cancellation_token.cancelled() => {
                    logger_clone.log_cancellation(&format!("[{}] Passthrough stream", stream_id));
                    let cancel_data = format!("data: {{\"cancelled\": true, \"message\": \"{}\"}}\n\n", ERROR_CANCELLED);
                    let _ = tx.send(Ok(bytes::Bytes::from(cancel_data)));
                    break;
                }
            }
        }
    });

    create_passthrough_streaming_response(rx)
}

/// Unified SSE to Ollama conversion function
fn convert_sse_to_ollama(sse_message: &str, model: &str, is_chat: bool) -> Option<Value> {
    for line in sse_message.lines() {
        if let Some(data) = line.strip_prefix(SSE_DATA_PREFIX) {
            if data.trim() == SSE_DONE_MESSAGE {
                return None; // End of stream
            }

            if let Ok(json_data) = serde_json::from_str::<Value>(data) {
                let content = if is_chat {
                    extract_chat_content(&json_data)
                } else {
                    extract_completion_content(&json_data)
                };

                if let Some(content) = content {
                    if !content.is_empty() {
                        return Some(create_ollama_chunk(model, &content, is_chat));
                    }
                }
            }
        }
    }
    None
}

/// Extract content from LM Studio SSE chat message
fn extract_chat_content(json_data: &Value) -> Option<String> {
    json_data
        .get("choices")?
        .as_array()?
        .first()?
        .get("delta")?
        .get("content")?
        .as_str()
        .map(|s| s.to_string())
}

/// Extract content from LM Studio SSE completion message
fn extract_completion_content(json_data: &Value) -> Option<String> {
    json_data
        .get("choices")?
        .as_array()?
        .first()?
        .get("text")?
        .as_str()
        .map(|s| s.to_string())
}

/// Create Ollama-formatted streaming chunk
fn create_ollama_chunk(model: &str, content: &str, is_chat: bool) -> Value {
    let timestamp = chrono::Utc::now().to_rfc3339();

    if is_chat {
        json!({
            "model": model,
            "created_at": timestamp,
            "message": {
                "role": "assistant",
                "content": content
            },
            "done": false
        })
    } else {
        json!({
            "model": model,
            "created_at": timestamp,
            "response": content,
            "done": false
        })
    }
}

/// Find complete SSE message boundary in buffer
fn find_complete_sse_message(buffer: &str) -> Option<usize> {
    let mut pos = 0;
    while let Some(found) = buffer[pos..].find(SSE_MESSAGE_BOUNDARY) {
        let end_pos = pos + found;
        let message = &buffer[..end_pos];

        // Verify this looks like a complete SSE message
        if message.lines().any(|line| line.starts_with(SSE_DATA_PREFIX)) {
            return Some(end_pos);
        }

        pos = end_pos + 2;
        if pos >= buffer.len() {
            break;
        }
    }
    None
}

/// Send a chunk and return false if client disconnected
async fn send_chunk(tx: &mpsc::UnboundedSender<Result<bytes::Bytes, std::io::Error>>, chunk: &Value) -> bool {
    let chunk_json = serde_json::to_string(chunk).unwrap_or_default();
    let chunk_with_newline = format!("{}\n", chunk_json);
    tx.send(Ok(bytes::Bytes::from(chunk_with_newline))).is_ok()
}

/// Send a chunk and close the stream
async fn send_chunk_and_close(
    tx: &mpsc::UnboundedSender<Result<bytes::Bytes, std::io::Error>>,
    chunk: Value,
) {
    let chunk_json = serde_json::to_string(&chunk).unwrap_or_default();
    let chunk_with_newline = format!("{}\n", chunk_json);
    let _ = tx.send(Ok(bytes::Bytes::from(chunk_with_newline)));
}

/// Send error chunk and close stream
async fn send_error_and_close(
    tx: &mpsc::UnboundedSender<Result<bytes::Bytes, std::io::Error>>,
    model: &str,
    error_message: &str,
    is_chat: bool,
) {
    let error_chunk = create_error_chunk(model, error_message, is_chat);
    send_chunk_and_close(tx, error_chunk).await;
}

/// Create streaming response with proper headers for Ollama compatibility
fn create_streaming_response(
    rx: mpsc::UnboundedReceiver<Result<bytes::Bytes, std::io::Error>>,
) -> Result<warp::reply::Response, ProxyError> {
    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);

    warp::http::Response::builder()
        .status(warp::http::StatusCode::OK)
        .header("content-type", CONTENT_TYPE_JSON)
        .header("cache-control", HEADER_CACHE_CONTROL)
        .header("connection", HEADER_CONNECTION)
        .header("access-control-allow-origin", HEADER_ACCESS_CONTROL_ALLOW_ORIGIN)
        .header("access-control-allow-methods", HEADER_ACCESS_CONTROL_ALLOW_METHODS)
        .header("access-control-allow-headers", HEADER_ACCESS_CONTROL_ALLOW_HEADERS)
        .body(warp::hyper::Body::wrap_stream(stream))
        .map_err(|e| ProxyError::internal_server_error(&format!("Failed to create streaming response: {}", e)))
}

/// Create passthrough streaming response with SSE headers
fn create_passthrough_streaming_response(
    rx: mpsc::UnboundedReceiver<Result<bytes::Bytes, std::io::Error>>,
) -> Result<warp::reply::Response, ProxyError> {
    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);

    warp::http::Response::builder()
        .status(warp::http::StatusCode::OK)
        .header("content-type", CONTENT_TYPE_SSE)
        .header("cache-control", HEADER_CACHE_CONTROL)
        .header("connection", HEADER_CONNECTION)
        .header("access-control-allow-origin", HEADER_ACCESS_CONTROL_ALLOW_ORIGIN)
        .header("access-control-allow-methods", HEADER_ACCESS_CONTROL_ALLOW_METHODS)
        .header("access-control-allow-headers", HEADER_ACCESS_CONTROL_ALLOW_HEADERS)
        .body(warp::hyper::Body::wrap_stream(stream))
        .map_err(|e| ProxyError::internal_server_error(&format!("Failed to create passthrough streaming response: {}", e)))
}
