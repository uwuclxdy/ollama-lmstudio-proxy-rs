// src/handlers/streaming.rs - Optimized streaming with fixed buffer overflow

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

/// Stream counter with overflow protection
static STREAM_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Check if streaming is enabled in request
pub fn is_streaming_request(body: &Value) -> bool {
    body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false)
}

/// Optimized streaming response handler
pub async fn handle_streaming_response(
    response: reqwest::Response,
    is_chat: bool,
    model: &str,
    start_time: Instant,
    cancellation_token: CancellationToken,
    logger: &Logger,
    stream_timeout_seconds: u64,
) -> Result<warp::reply::Response, ProxyError> {
    let model = model.to_string();
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();

    // Generate stream ID with overflow protection
    let stream_id = STREAM_COUNTER.fetch_add(1, Ordering::Relaxed) % 1_000_000;

    let model_clone = model.clone();
    let token_clone = cancellation_token.clone();
    let logger_clone = logger.clone();

    tokio::spawn(async move {
        let mut stream = response.bytes_stream();
        let mut buffer = String::with_capacity(MAX_BUFFER_SIZE);
        let mut chunk_count = 0u64;
        let mut partial_content = String::with_capacity(MAX_PARTIAL_CONTENT_SIZE);

        let stream_result = 'stream: loop {
            // Check limits before processing
            if chunk_count >= MAX_CHUNK_COUNT {
                send_error_and_close(&tx, &model_clone, ERROR_CHUNK_LIMIT, is_chat).await;
                break 'stream Err(ERROR_CHUNK_LIMIT.to_string());
            }

            tokio::select! {
                chunk_result = timeout(Duration::from_secs(stream_timeout_seconds), stream.next()) => {
                    match chunk_result {
                        Ok(Some(Ok(chunk))) => {
                            if let Ok(chunk_str) = std::str::from_utf8(&chunk) {
                                // Check buffer size BEFORE appending to prevent overflow
                                if buffer.len() + chunk_str.len() > MAX_BUFFER_SIZE {
                                    send_error_and_close(&tx, &model_clone, ERROR_BUFFER_OVERFLOW, is_chat).await;
                                    break 'stream Err(ERROR_BUFFER_OVERFLOW.to_string());
                                }

                                buffer.push_str(chunk_str);

                                // Process complete SSE messages
                                while let Some(message_end) = find_complete_sse_message(&buffer) {
                                    let message = buffer[..message_end].to_string();
                                    buffer.drain(..message_end + 2); // Remove processed message

                                    if let Some(ollama_chunk) = convert_sse_to_ollama(&message, &model_clone, is_chat) {
                                        chunk_count += 1;

                                        // Track partial content efficiently
                                        if partial_content.len() < MAX_PARTIAL_CONTENT_SIZE {
                                            if let Some(content) = extract_content_from_chunk(&ollama_chunk) {
                                                partial_content.push_str(&content);
                                            }
                                        }

                                        if !send_chunk(&tx, &ollama_chunk).await {
                                            break 'stream Ok(());
                                        }
                                    }
                                }
                            }
                        }
                        Ok(Some(Err(e))) => {
                            send_error_and_close(&tx, &model_clone, &format!("Streaming error: {}", e), is_chat).await;
                            break 'stream Err(format!("Network error: {}", e));
                        }
                        Ok(None) => {
                            break 'stream Ok(());
                        }
                        Err(_) => {
                            send_error_and_close(&tx, &model_clone, ERROR_TIMEOUT, is_chat).await;
                            break 'stream Err(ERROR_TIMEOUT.to_string());
                        }
                    }
                }
                _ = token_clone.cancelled() => {
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

        // Send final chunk if successful
        if stream_result.is_ok() && !token_clone.is_cancelled() {
            let final_chunk = create_final_chunk(&model_clone, start_time.elapsed(), chunk_count, is_chat);
            send_chunk_and_close(&tx, final_chunk).await;
        }

        logger_clone.log_timed("Stream processing", &format!("[{}]", stream_id), start_time);
    });

    create_streaming_response(rx)
}

/// Optimized passthrough streaming
pub async fn handle_passthrough_streaming_response(
    response: reqwest::Response,
    cancellation_token: CancellationToken,
    logger: &Logger,
    stream_timeout_seconds: u64,
) -> Result<warp::reply::Response, ProxyError> {
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
    let stream_id = STREAM_COUNTER.fetch_add(1, Ordering::Relaxed) % 1_000_000;
    let start_time = Instant::now();

    let logger_clone = logger.clone();
    tokio::spawn(async move {
        let mut stream = response.bytes_stream();

        loop {
            tokio::select! {
                chunk_result = timeout(Duration::from_secs(stream_timeout_seconds), stream.next()) => {
                    match chunk_result {
                        Ok(Some(Ok(chunk))) => {
                            if tx.send(Ok(chunk)).is_err() {
                                break;
                            }
                        }
                        Ok(Some(Err(e))) => {
                            let error_data = format!("data: {{\"error\": \"Streaming error: {}\"}}\n\n", e);
                            let _ = tx.send(Ok(bytes::Bytes::from(error_data)));
                            break;
                        }
                        Ok(None) => break,
                        Err(_) => {
                            let timeout_data = format!("data: {{\"error\": \"{}\"}}\n\n", ERROR_TIMEOUT);
                            let _ = tx.send(Ok(bytes::Bytes::from(timeout_data)));
                            break;
                        }
                    }
                }
                _ = cancellation_token.cancelled() => {
                    let cancel_data = format!("data: {{\"cancelled\": true, \"message\": \"{}\"}}\n\n", ERROR_CANCELLED);
                    let _ = tx.send(Ok(bytes::Bytes::from(cancel_data)));
                    break;
                }
            }
        }

        logger_clone.log_timed("Passthrough stream", &format!("[{}]", stream_id), start_time);
    });

    create_passthrough_streaming_response(rx)
}

/// Fast SSE to Ollama conversion
fn convert_sse_to_ollama(sse_message: &str, model: &str, is_chat: bool) -> Option<Value> {
    for line in sse_message.lines() {
        if let Some(data) = line.strip_prefix(SSE_DATA_PREFIX) {
            if data.trim() == SSE_DONE_MESSAGE {
                return None;
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

/// Extract chat content from LM Studio response
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

/// Extract completion content from LM Studio response
fn extract_completion_content(json_data: &Value) -> Option<String> {
    json_data
        .get("choices")?
        .as_array()?
        .first()?
        .get("text")?
        .as_str()
        .map(|s| s.to_string())
}

/// Create Ollama-formatted chunk
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

/// Optimized SSE message boundary detection
fn find_complete_sse_message(buffer: &str) -> Option<usize> {
    let mut pos = 0;
    while let Some(found) = buffer[pos..].find(SSE_MESSAGE_BOUNDARY) {
        let end_pos = pos + found;
        let message = &buffer[..end_pos];

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

/// Send chunk and detect client disconnect
async fn send_chunk(tx: &mpsc::UnboundedSender<Result<bytes::Bytes, std::io::Error>>, chunk: &Value) -> bool {
    let chunk_json = serde_json::to_string(chunk).unwrap_or_default();
    let chunk_with_newline = format!("{}\n", chunk_json);
    tx.send(Ok(bytes::Bytes::from(chunk_with_newline))).is_ok()
}

/// Send chunk and close stream
async fn send_chunk_and_close(
    tx: &mpsc::UnboundedSender<Result<bytes::Bytes, std::io::Error>>,
    chunk: Value,
) {
    let chunk_json = serde_json::to_string(&chunk).unwrap_or_default();
    let chunk_with_newline = format!("{}\n", chunk_json);
    let _ = tx.send(Ok(bytes::Bytes::from(chunk_with_newline)));
}

/// Send error and close stream
async fn send_error_and_close(
    tx: &mpsc::UnboundedSender<Result<bytes::Bytes, std::io::Error>>,
    model: &str,
    error_message: &str,
    is_chat: bool,
) {
    let error_chunk = create_error_chunk(model, error_message, is_chat);
    send_chunk_and_close(tx, error_chunk).await;
}

/// Create streaming response
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
        .map_err(|_| ProxyError::internal_server_error("Failed to create streaming response"))
}

/// Create passthrough streaming response
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
        .map_err(|_| ProxyError::internal_server_error("Failed to create passthrough streaming response"))
}
