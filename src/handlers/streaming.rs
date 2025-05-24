use futures_util::StreamExt;
use serde_json::{json, Value};
use std::time::Instant;
use tokio::sync::mpsc;

use crate::utils::ProxyError;

/// Check if request has streaming enabled
pub fn is_streaming_request(body: &Value) -> bool {
    body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false)
}

/// Handle streaming responses from LM Studio - For Ollama format conversion
pub async fn handle_streaming_response(
    response: reqwest::Response,
    is_chat: bool,
    model: &str,
    start_time: Instant,
) -> Result<warp::reply::Response, ProxyError> {
    let model = model.to_string();
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();

    // Spawn task to process the stream
    let model_clone = model.clone();
    tokio::spawn(async move {
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut chunk_count = 0u64;

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
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
                                let chunk_json = serde_json::to_string(&ollama_chunk).unwrap_or_default();
                                let chunk_with_newline = format!("{}\n", chunk_json);

                                if tx.send(Ok(bytes::Bytes::from(chunk_with_newline))).is_err() {
                                    // Client disconnected, stop processing
                                    return;
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    // Network error during streaming - send error chunk and stop
                    let error_chunk = if is_chat {
                        json!({
                            "model": model_clone,
                            "created_at": chrono::Utc::now().to_rfc3339(),
                            "message": {
                                "role": "assistant",
                                "content": ""
                            },
                            "done": true,
                            "error": format!("Streaming error: {}", e)
                        })
                    } else {
                        json!({
                            "model": model_clone,
                            "created_at": chrono::Utc::now().to_rfc3339(),
                            "response": "",
                            "done": true,
                            "error": format!("Streaming error: {}", e)
                        })
                    };

                    let error_json = serde_json::to_string(&error_chunk).unwrap_or_default();
                    let error_with_newline = format!("{}\n", error_json);
                    let _ = tx.send(Ok(bytes::Bytes::from(error_with_newline)));
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
        let total_duration = start_time.elapsed().as_nanos() as u64;
        let final_chunk = if is_chat {
            json!({
                "model": model_clone,
                "created_at": chrono::Utc::now().to_rfc3339(),
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "done": true,
                "total_duration": total_duration,
                "load_duration": 1000000u64,
                "prompt_eval_count": 10,
                "prompt_eval_duration": total_duration / 4,
                "eval_count": chunk_count.max(1),
                "eval_duration": total_duration / 2
            })
        } else {
            json!({
                "model": model_clone,
                "created_at": chrono::Utc::now().to_rfc3339(),
                "response": "",
                "done": true,
                "context": [1, 2, 3],
                "total_duration": total_duration,
                "load_duration": 1000000u64,
                "prompt_eval_count": 10,
                "prompt_eval_duration": total_duration / 4,
                "eval_count": chunk_count.max(1),
                "eval_duration": total_duration / 2
            })
        };

        let final_json = serde_json::to_string(&final_chunk).unwrap_or_default();
        let final_with_newline = format!("{}\n", final_json);
        let _ = tx.send(Ok(bytes::Bytes::from(final_with_newline)));
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

/// Handle direct streaming passthrough from LM Studio
pub async fn handle_passthrough_streaming_response(
    response: reqwest::Response,
) -> Result<warp::reply::Response, ProxyError> {
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();

    // Spawn task to forward the stream with error handling
    tokio::spawn(async move {
        let mut stream = response.bytes_stream();

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    if tx.send(Ok(chunk)).is_err() {
                        // Client disconnected, stop forwarding
                        return;
                    }
                }
                Err(e) => {
                    // Send error as SSE data and stop
                    let error_data = format!("data: {{\"error\": \"Streaming error: {}\"}}\n\n", e);
                    let _ = tx.send(Ok(bytes::Bytes::from(error_data)));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_parsing_chat() {
        let sse_data = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n";
        let result = convert_sse_to_ollama_chat(sse_data, "test-model");
        assert!(result.is_some());

        let json_result = result.unwrap();
        assert_eq!(json_result["message"]["content"], "Hello");
        assert_eq!(json_result["model"], "test-model");
        assert_eq!(json_result["done"], false);
    }

    #[test]
    fn test_sse_parsing_generate() {
        let sse_data = "data: {\"choices\":[{\"text\":\"Hello\"}]}\n\n";
        let result = convert_sse_to_ollama_generate(sse_data, "test-model");
        assert!(result.is_some());

        let json_result = result.unwrap();
        assert_eq!(json_result["response"], "Hello");
        assert_eq!(json_result["model"], "test-model");
        assert_eq!(json_result["done"], false);
    }

    #[test]
    fn test_sse_parsing_chat_with_empty_content() {
        let sse_data = "data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n\n";
        let result = convert_sse_to_ollama_chat(sse_data, "test-model");
        assert!(result.is_none()); // Empty content should not generate a chunk
    }

    #[test]
    fn test_sse_parsing_generate_with_empty_content() {
        let sse_data = "data: {\"choices\":[{\"text\":\"\"}]}\n\n";
        let result = convert_sse_to_ollama_generate(sse_data, "test-model");
        assert!(result.is_none()); // Empty content should not generate a chunk
    }

    #[test]
    fn test_sse_done_signal() {
        let sse_data = "data: [DONE]\n\n";
        let chat_result = convert_sse_to_ollama_chat(sse_data, "test-model");
        let generate_result = convert_sse_to_ollama_generate(sse_data, "test-model");

        assert!(chat_result.is_none());
        assert!(generate_result.is_none());
    }

    #[test]
    fn test_malformed_sse_data() {
        let sse_data = "data: {invalid json}\n\n";
        let chat_result = convert_sse_to_ollama_chat(sse_data, "test-model");
        let generate_result = convert_sse_to_ollama_generate(sse_data, "test-model");

        assert!(chat_result.is_none());
        assert!(generate_result.is_none());
    }

    #[test]
    fn test_streaming_request_detection() {
        let streaming_body = json!({"stream": true, "model": "test"});
        let non_streaming_body = json!({"stream": false, "model": "test"});
        let no_stream_body = json!({"model": "test"});

        assert_eq!(is_streaming_request(&streaming_body), true);
        assert_eq!(is_streaming_request(&non_streaming_body), false);
        assert_eq!(is_streaming_request(&no_stream_body), false);
    }
}