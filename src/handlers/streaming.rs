// src/handlers/streaming.rs - Enhanced streaming with chunk recovery and metrics

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
use crate::metrics::get_global_metrics;
use crate::utils::{Logger, ProxyError};

/// Stream counter for tracking active streams
static STREAM_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Stream recovery state for partial chunk handling
#[derive(Debug, Clone)]
struct StreamRecoveryState {
    partial_buffer: String,
    last_known_position: usize,
    recovery_attempts: u32,
}

impl StreamRecoveryState {
    fn new() -> Self {
        Self {
            partial_buffer: String::new(),
            last_known_position: 0,
            recovery_attempts: 0,
        }
    }

    /// Attempt to recover from partial content
    fn try_recover(&mut self, new_content: &str) -> Option<String> {
        if !get_runtime_config().enable_chunk_recovery {
            return None;
        }

        self.recovery_attempts += 1;

        // Try to find overlap between partial buffer and new content
        if let Some(overlap_pos) = find_content_overlap(&self.partial_buffer, new_content) {
            // Found overlap, merge the content
            let recovered = format!("{}{}", &self.partial_buffer[..overlap_pos], new_content);
            self.partial_buffer.clear();
            self.recovery_attempts = 0;
            Some(recovered)
        } else if self.recovery_attempts < 3 {
            // Store partial content for next attempt
            self.partial_buffer.push_str(new_content);
            None
        } else {
            // Give up recovery after 3 attempts
            self.partial_buffer.clear();
            self.recovery_attempts = 0;
            Some(new_content.to_string())
        }
    }
}

/// Check if streaming is enabled in request
pub fn is_streaming_request(body: &Value) -> bool {
    body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false)
}

/// Enhanced streaming response handler with recovery and metrics
pub async fn handle_streaming_response(
    response: reqwest::Response,
    is_chat: bool,
    model: &str,
    start_time: Instant,
    cancellation_token: CancellationToken,
    logger: &Logger,
    stream_timeout_seconds: u64,
) -> Result<warp::reply::Response, ProxyError> {
    let config = get_runtime_config();
    let model = model.to_string();
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();

    // Generate unique stream ID
    let stream_id = STREAM_COUNTER.fetch_add(1, Ordering::Relaxed) % 1_000_000;

    // Record stream start in metrics
    if let Some(metrics) = get_global_metrics() {
        metrics.record_stream_start();
    }

    let model_clone = model.clone();
    let token_clone = cancellation_token.clone();
    let logger_clone = logger.clone();

    tokio::spawn(async move {
        let mut stream = response.bytes_stream();
        let mut buffer = String::with_capacity(config.max_buffer_size);
        let mut chunk_count = 0u64;
        let mut partial_content = String::with_capacity(config.max_partial_content_size);
        let mut recovery_state = StreamRecoveryState::new();

        let stream_result = 'stream: loop {
            // Check limits before processing
            if chunk_count >= config.max_chunk_count {
                send_error_and_close(&tx, &model_clone, ERROR_CHUNK_LIMIT, is_chat).await;
                break 'stream Err(ERROR_CHUNK_LIMIT.to_string());
            }

            tokio::select! {
                chunk_result = timeout(Duration::from_secs(stream_timeout_seconds), stream.next()) => {
                    match chunk_result {
                        Ok(Some(Ok(chunk))) => {
                            // Check chunk size before processing to prevent memory issues
                            if chunk.len() > config.max_buffer_size {
                                send_error_and_close(&tx, &model_clone, ERROR_BUFFER_OVERFLOW, is_chat).await;
                                break 'stream Err(ERROR_BUFFER_OVERFLOW.to_string());
                            }

                            if let Ok(chunk_str) = std::str::from_utf8(&chunk) {
                                // Check buffer size BEFORE appending
                                if buffer.len() + chunk_str.len() > config.max_buffer_size {
                                    send_error_and_close(&tx, &model_clone, ERROR_BUFFER_OVERFLOW, is_chat).await;
                                    break 'stream Err(ERROR_BUFFER_OVERFLOW.to_string());
                                }

                                buffer.push_str(chunk_str);

                                // Process complete SSE messages with recovery
                                while let Some(message_end) = find_complete_sse_message(&buffer) {
                                    let message = buffer[..message_end].to_string();
                                    buffer.drain(..message_end + 2);

                                    if let Some(ollama_chunk) = convert_sse_to_ollama_with_recovery(
                                        &message,
                                        &model_clone,
                                        is_chat,
                                        &mut recovery_state
                                    ) {
                                        chunk_count += 1;

                                        // Track partial content efficiently
                                        if partial_content.len() < config.max_partial_content_size {
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

        // Record stream completion in metrics
        if let Some(metrics) = get_global_metrics() {
            metrics.record_stream_end(chunk_count, stream_result.is_err());
        }

        logger_clone.log_timed("Stream processing", &format!("[{}] {} chunks", stream_id, chunk_count), start_time);
    });

    create_streaming_response(rx)
}

/// Enhanced passthrough streaming with better error handling
pub async fn handle_passthrough_streaming_response(
    response: reqwest::Response,
    cancellation_token: CancellationToken,
    logger: &Logger,
    stream_timeout_seconds: u64,
) -> Result<warp::reply::Response, ProxyError> {
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
    let stream_id = STREAM_COUNTER.fetch_add(1, Ordering::Relaxed) % 1_000_000;
    let start_time = Instant::now();

    // Record stream start in metrics
    if let Some(metrics) = get_global_metrics() {
        metrics.record_stream_start();
    }

    let logger_clone = logger.clone();
    tokio::spawn(async move {
        let mut stream = response.bytes_stream();
        let mut chunk_count = 0u64;
        let mut has_error = false;

        loop {
            tokio::select! {
                chunk_result = timeout(Duration::from_secs(stream_timeout_seconds), stream.next()) => {
                    match chunk_result {
                        Ok(Some(Ok(chunk))) => {
                            chunk_count += 1;
                            if tx.send(Ok(chunk)).is_err() {
                                break;
                            }
                        }
                        Ok(Some(Err(e))) => {
                            has_error = true;
                            let error_data = format!("data: {{\"error\": \"Streaming error: {}\"}}\n\n", e);
                            let _ = tx.send(Ok(bytes::Bytes::from(error_data)));
                            break;
                        }
                        Ok(None) => break,
                        Err(_) => {
                            has_error = true;
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

        // Record stream completion in metrics
        if let Some(metrics) = get_global_metrics() {
            metrics.record_stream_end(chunk_count, has_error);
        }

        logger_clone.log_timed("Passthrough stream", &format!("[{}] {} chunks", stream_id, chunk_count), start_time);
    });

    create_passthrough_streaming_response(rx)
}

/// Enhanced SSE to Ollama conversion with recovery support
fn convert_sse_to_ollama_with_recovery(
    sse_message: &str,
    model: &str,
    is_chat: bool,
    recovery_state: &mut StreamRecoveryState
) -> Option<Value> {
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
                    // Try recovery if content seems incomplete
                    let final_content = if content.is_empty() || seems_incomplete(&content) {
                        recovery_state.try_recover(&content).unwrap_or(content)
                    } else {
                        content
                    };

                    if !final_content.is_empty() {
                        return Some(create_ollama_chunk(model, &final_content, is_chat));
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

/// Find overlap between partial content and new content for recovery
fn find_content_overlap(partial: &str, new_content: &str) -> Option<usize> {
    if partial.is_empty() || new_content.is_empty() {
        return None;
    }

    // Look for the longest suffix of partial that matches a prefix of new_content
    let partial_chars: Vec<char> = partial.chars().collect();
    let new_chars: Vec<char> = new_content.chars().collect();

    for i in (1..=partial_chars.len().min(new_chars.len())).rev() {
        let partial_suffix = &partial_chars[partial_chars.len() - i..];
        let new_prefix = &new_chars[..i];

        if partial_suffix == new_prefix {
            return Some(partial_chars.len() - i);
        }
    }

    None
}

/// Check if content seems incomplete (for recovery heuristics)
fn seems_incomplete(content: &str) -> bool {
    if content.is_empty() {
        return true;
    }

    // Simple heuristics for incomplete content
    let trimmed = content.trim();

    // Very short content might be incomplete
    if trimmed.len() < 3 {
        return true;
    }

    // Ends with incomplete word or punctuation patterns
    if trimmed.ends_with(['.', '?', '!', ' ']) {
        return false;
    }

    // Check for incomplete JSON or common incomplete patterns
    if trimmed.ends_with(['{', '[', ',', ':', '"']) {
        return true;
    }

    false
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

/// Create a streaming response with specified content type
fn create_streaming_response_with_type(
    rx: mpsc::UnboundedReceiver<Result<bytes::Bytes, std::io::Error>>,
    content_type: &str,
    error_message: &str,
) -> Result<warp::reply::Response, ProxyError> {
    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);

    warp::http::Response::builder()
        .status(warp::http::StatusCode::OK)
        .header("content-type", content_type)
        .header("cache-control", HEADER_CACHE_CONTROL)
        .header("connection", HEADER_CONNECTION)
        .header("access-control-allow-origin", HEADER_ACCESS_CONTROL_ALLOW_ORIGIN)
        .header("access-control-allow-methods", HEADER_ACCESS_CONTROL_ALLOW_METHODS)
        .header("access-control-allow-headers", HEADER_ACCESS_CONTROL_ALLOW_HEADERS)
        .body(warp::hyper::Body::wrap_stream(stream))
        .map_err(|_| ProxyError::internal_server_error(error_message))
}

/// Create streaming response with JSON content type
fn create_streaming_response(
    rx: mpsc::UnboundedReceiver<Result<bytes::Bytes, std::io::Error>>,
) -> Result<warp::reply::Response, ProxyError> {
    create_streaming_response_with_type(rx, CONTENT_TYPE_JSON, "Failed to create streaming response")
}

/// Create passthrough streaming response with SSE headers
fn create_passthrough_streaming_response(
    rx: mpsc::UnboundedReceiver<Result<bytes::Bytes, std::io::Error>>,
) -> Result<warp::reply::Response, ProxyError> {
    create_streaming_response_with_type(rx, CONTENT_TYPE_SSE, "Failed to create passthrough streaming response")
}
