/// src/handlers/streaming.rs - Enhanced streaming with model loading detection and better timing

use futures_util::StreamExt;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

use crate::constants::*;
use crate::handlers::helpers::{
    create_cancellation_chunk, create_error_chunk, create_final_chunk, create_ollama_streaming_chunk,
};
use crate::utils::{log_error, log_info, log_timed, log_warning, ProxyError};

static STREAM_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Threshold for detecting slow stream starts (likely model loading)
const STREAM_START_LOADING_THRESHOLD_MS: u128 = 500;

/// Check if request is streaming
pub fn is_streaming_request(body: &Value) -> bool {
    body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false)
}

/// Handle streaming response with model loading detection
pub async fn handle_streaming_response(
    lm_studio_response: reqwest::Response,
    is_chat_endpoint: bool,
    ollama_model_name: &str,
    start_time: Instant,
    cancellation_token: CancellationToken,
    stream_timeout_seconds: u64,
) -> Result<warp::reply::Response, ProxyError> {
    let runtime_config = get_runtime_config();
    let ollama_model_name = ollama_model_name.to_string();
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();

    let stream_id = STREAM_COUNTER.fetch_add(1, Ordering::Relaxed) % 1_000_000;

    let model_clone_for_task = ollama_model_name.clone();
    let token_clone = cancellation_token.clone();

    tokio::spawn(async move {
        let mut stream = lm_studio_response.bytes_stream();
        let mut sse_buffer = String::with_capacity(runtime_config.max_buffer_size.min(1024 * 1024));
        let mut chunk_count = 0u64;
        let mut accumulated_tool_calls: Option<Vec<Value>> = None;
        let mut first_chunk_received = false;

        let stream_result = 'stream_loop: loop {
            tokio::select! {
                biased; // Prioritize cancellation
                _ = token_clone.cancelled() => {
                    let cancellation_chunk = create_cancellation_chunk(
                        &model_clone_for_task,
                        start_time.elapsed(),
                        chunk_count,
                        is_chat_endpoint,
                    );
                    send_chunk_and_close_channel(&tx, cancellation_chunk).await;
                    break 'stream_loop Err(ERROR_CANCELLED.to_string());
                }

                chunk_result = timeout(Duration::from_secs(stream_timeout_seconds), stream.next()) => {
                    match chunk_result {
                        Ok(Some(Ok(bytes_chunk))) => {
                            // Track first chunk timing for model loading detection
                            if !first_chunk_received {
                                first_chunk_received = true;
                                let time_to_first_chunk = start_time.elapsed();

                                if time_to_first_chunk.as_millis() > STREAM_START_LOADING_THRESHOLD_MS {
                                    log_info(&format!(
                                        "{} loaded | took {}ms",
                                        model_clone_for_task, time_to_first_chunk.as_millis()
                                    ));
                                }
                            }

                            if let Ok(chunk_str) = std::str::from_utf8(&bytes_chunk) {
                                sse_buffer.push_str(chunk_str);

                                while let Some(boundary_pos) = sse_buffer.find(SSE_MESSAGE_BOUNDARY) {
                                    let message_text = sse_buffer[..boundary_pos].to_string();
                                    sse_buffer.drain(..boundary_pos + SSE_MESSAGE_BOUNDARY.len());

                                    if message_text.trim().is_empty() { continue; }

                                    if let Some(data_content) = message_text.strip_prefix(SSE_DATA_PREFIX) {
                                        if data_content.trim() == SSE_DONE_MESSAGE {
                                            break 'stream_loop Ok(());
                                        }

                                        match serde_json::from_str::<Value>(data_content) {
                                            Ok(lm_studio_json_chunk) => {
                                                let mut content_to_send = String::new();
                                                let mut tool_calls_delta: Option<Value> = None;

                                                if let Some(choices) = lm_studio_json_chunk.get("choices").and_then(|c| c.as_array()) {
                                                    if let Some(choice) = choices.first() {
                                                        if let Some(delta) = choice.get("delta") {
                                                            if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                                                                content_to_send.push_str(content);
                                                            }
                                                            if let Some(new_tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
                                                                if accumulated_tool_calls.is_none() {
                                                                    accumulated_tool_calls = Some(Vec::new());
                                                                }
                                                                tool_calls_delta = Some(json!(new_tool_calls));
                                                            }
                                                        }
                                                    }
                                                }

                                                if !content_to_send.is_empty() || tool_calls_delta.is_some() {
                                                    let ollama_chunk = create_ollama_streaming_chunk(
                                                        &model_clone_for_task,
                                                        &content_to_send,
                                                        is_chat_endpoint,
                                                        false,
                                                        tool_calls_delta.as_ref()
                                                    );
                                                    chunk_count += 1;
                                                    if !send_ollama_chunk(&tx, &ollama_chunk).await {
                                                        break 'stream_loop Ok(());
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                log_error("SSE JSON parsing", &format!("Failed to parse LM Studio SSE data: {}. Data: '{}'", e, data_content));
                                            }
                                        }
                                    } else if !message_text.trim().is_empty() {
                                         log_warning("SSE format", &format!("Received non-standard SSE line: {}", message_text));
                                    }
                                }
                            } else {
                                send_error_and_close(&tx, &model_clone_for_task, "Invalid UTF-8 in stream", is_chat_endpoint).await;
                                break 'stream_loop Err("Invalid UTF-8".to_string());
                            }
                        }
                        Ok(Some(Err(e))) => {
                            send_error_and_close(&tx, &model_clone_for_task, &format!("Streaming error: {}", e), is_chat_endpoint).await;
                            break 'stream_loop Err(format!("Network error: {}", e));
                        }
                        Ok(None) => {
                            log_warning("Stream ended prematurely", "LM Studio stream ended without [DONE]");
                            break 'stream_loop Ok(());
                        }
                        Err(_) => {
                            send_error_and_close(&tx, &model_clone_for_task, ERROR_TIMEOUT, is_chat_endpoint).await;
                            break 'stream_loop Err(ERROR_TIMEOUT.to_string());
                        }
                    }
                }
            }
        };

        if stream_result.is_ok() && !token_clone.is_cancelled() {
            let final_chunk = create_final_chunk(
                &model_clone_for_task,
                start_time.elapsed(),
                chunk_count,
                is_chat_endpoint,
            );
            send_chunk_and_close_channel(&tx, final_chunk).await;
        }

        log_timed("Stream processing", &format!("[{}] {} Ollama chunks", stream_id, chunk_count), start_time);
    });

    create_ollama_streaming_response_format(rx)
}

/// Handle passthrough streaming for direct LM Studio responses
pub async fn handle_passthrough_streaming_response(
    response: reqwest::Response,
    cancellation_token: CancellationToken,
    stream_timeout_seconds: u64,
) -> Result<warp::reply::Response, ProxyError> {
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
    let stream_id = STREAM_COUNTER.fetch_add(1, Ordering::Relaxed) % 1_000_000;
    let start_time = Instant::now();

    tokio::spawn(async move {
        let mut stream = response.bytes_stream();
        let mut chunk_count = 0u64;

        loop {
            tokio::select! {
                biased;
                _ = cancellation_token.cancelled() => {
                    let cancel_data = format!("data: {{\"error\": \"{}\", \"cancelled\": true}}\n\n", ERROR_CANCELLED);
                    let _ = tx.send(Ok(bytes::Bytes::from(cancel_data)));
                    break;
                }
                chunk_result = timeout(Duration::from_secs(stream_timeout_seconds), stream.next()) => {
                    match chunk_result {
                        Ok(Some(Ok(chunk))) => {
                            chunk_count += 1;
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
            }
        }

        log_timed(LOG_PREFIX_SUCCESS,&format!("Passthrough stream [{}] finished! | {} chunks", stream_id, chunk_count), start_time);
    });

    create_passthrough_streaming_response_format(rx)
}

/// Send Ollama chunk to client
async fn send_ollama_chunk(tx: &mpsc::UnboundedSender<Result<bytes::Bytes, std::io::Error>>, chunk: &Value) -> bool {
    let chunk_json = serde_json::to_string(chunk).unwrap_or_else(|e| {
        log_error("Chunk serialization", &format!("Failed to serialize Ollama chunk: {}", e));
        String::from("{\"error\":\"Internal proxy error: failed to serialize chunk\"}")
    });
    let chunk_with_newline = format!("{}\n", chunk_json);
    tx.send(Ok(bytes::Bytes::from(chunk_with_newline))).is_ok()
}

/// Send chunk and close channel
async fn send_chunk_and_close_channel(
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
    model_ollama_name: &str,
    error_message: &str,
    is_chat_endpoint: bool,
) {
    let error_chunk = create_error_chunk(model_ollama_name, error_message, is_chat_endpoint);
    send_chunk_and_close_channel(tx, error_chunk).await;
}

/// Create generic streaming response
fn create_generic_streaming_response(
    rx: mpsc::UnboundedReceiver<Result<bytes::Bytes, std::io::Error>>,
    content_type: &str,
    error_message_on_build_fail: &str,
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
        .map_err(|_| ProxyError::internal_server_error(error_message_on_build_fail))
}

/// Create Ollama streaming response format
fn create_ollama_streaming_response_format(
    rx: mpsc::UnboundedReceiver<Result<bytes::Bytes, std::io::Error>>,
) -> Result<warp::reply::Response, ProxyError> {
    create_generic_streaming_response(rx, "application/x-ndjson; charset=utf-8", "Failed to create Ollama streaming response")
}

/// Create passthrough SSE streaming response
fn create_passthrough_streaming_response_format(
    rx: mpsc::UnboundedReceiver<Result<bytes::Bytes, std::io::Error>>,
) -> Result<warp::reply::Response, ProxyError> {
    create_generic_streaming_response(rx, CONTENT_TYPE_SSE, "Failed to create passthrough SSE streaming response")
}
