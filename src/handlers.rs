use futures_util::StreamExt;
use serde_json::{json, Value};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use warp::Reply;
use tokio::sync::mpsc;

use crate::server::ProxyServer;
use crate::utils::{clean_model_name, format_duration, is_no_models_loaded_error, ProxyError};

/// Helper function to convert JSON to Response
fn json_response(value: &Value) -> warp::reply::Response {
    warp::reply::with_status(
        warp::reply::json(value),
        warp::http::StatusCode::OK,
    ).into_response()
}

/// Auto-retry infrastructure: trigger model loading by calling /v1/models
pub async fn trigger_model_loading(server: &ProxyServer) -> Result<bool, ProxyError> {
    server.logger.log("Attempting to trigger model loading...");

    let url = format!("{}/v1/models", server.config.lmstudio_url);

    match server.client.get(&url).send().await {
        Ok(response) => {
            if response.status().is_success() {
                server.logger.log("Model loading triggered successfully");
                Ok(true)
            } else {
                server.logger.log(&format!("Failed to trigger model loading: {}", response.status()));
                Ok(false)
            }
        }
        Err(e) => {
            server.logger.log(&format!("Error triggering model loading: {}", e));
            Err(ProxyError::internal_server_error(&format!("Failed to communicate with LM Studio: {}", e)))
        }
    }
}

/// Generic retry wrapper that attempts model loading and retries operation
pub async fn with_retry<F, T, Fut>(server: &ProxyServer, operation: F) -> Result<T, ProxyError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output=Result<T, ProxyError>>,
{
    // First attempt
    match operation().await {
        Ok(result) => Ok(result),
        Err(e) => {
            if is_no_models_loaded_error(&e.message) {
                server.logger.log("Detected 'no models loaded' error, attempting retry with model loading...");

                // Trigger model loading
                if trigger_model_loading(server).await.unwrap_or(false) {
                    server.logger.log("Waiting for model to load...");
                    sleep(Duration::from_secs(server.config.load_timeout_seconds)).await;
                    
                    server.logger.log("Retrying operation after model loading...");
                    return operation().await;
                }
            }

            // If not a model loading error or retry failed, return original error
            Err(e)
        }
    }
}

/// Check if request has streaming enabled
fn is_streaming_request(body: &Value) -> bool {
    body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false)
}

/// Handle streaming responses from LM Studio - For Ollama format conversion
async fn handle_streaming_response(
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
                                let chunk_json = serde_json::to_string(&ollama_chunk).unwrap_or_default();
                                let chunk_with_newline = format!("{}\n", chunk_json);
                                
                                if tx.send(Ok(bytes::Bytes::from(chunk_with_newline))).is_err() {
                                    break;
                                }
                            }
                        }
                    }
                }
                Err(_) => break,
            }
        }

        // Process any remaining buffer content
        if !buffer.trim().is_empty() {
            if let Some(ollama_chunk) = if is_chat {
                convert_sse_to_ollama_chat(&buffer, &model_clone)
            } else {
                convert_sse_to_ollama_generate(&buffer, &model_clone)
            } {
                let chunk_json = serde_json::to_string(&ollama_chunk).unwrap_or_default();
                let chunk_with_newline = format!("{}\n", chunk_json);
                let _ = tx.send(Ok(bytes::Bytes::from(chunk_with_newline)));
            }
        }

        // Add final completion chunk
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
                "eval_count": 50,
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
                "eval_count": 50,
                "eval_duration": total_duration / 2
            })
        };
        
        let final_json = serde_json::to_string(&final_chunk).unwrap_or_default();
        let final_with_newline = format!("{}\n", final_json);
        let _ = tx.send(Ok(bytes::Bytes::from(final_with_newline)));
    });

    // Create stream from receiver
    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);

    // Create streaming response
    let response = warp::http::Response::builder()
        .status(warp::http::StatusCode::OK)
        .header("content-type", "application/json; charset=utf-8")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .body(warp::hyper::Body::wrap_stream(stream))
        .unwrap();

    Ok(response)
}

/// Handle direct streaming passthrough from LM Studio
async fn handle_passthrough_streaming_response(
    response: reqwest::Response,
) -> Result<warp::reply::Response, ProxyError> {
    let (tx, rx) = mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
    
    // Spawn task to forward the stream
    tokio::spawn(async move {
        let mut stream = response.bytes_stream();

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    if tx.send(Ok(chunk)).is_err() {
                        break;
                    }
                }
                Err(_) => break,
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
        .body(warp::hyper::Body::wrap_stream(stream))
        .unwrap();

    Ok(response)
}

/// Convert SSE message to Ollama chat format
fn convert_sse_to_ollama_chat(sse_message: &str, model: &str) -> Option<Value> {
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
    None
}

/// Convert SSE message to Ollama generate format
fn convert_sse_to_ollama_generate(sse_message: &str, model: &str) -> Option<Value> {
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
    None
}

// =============================================================================
// OLLAMA API HANDLERS (with streaming support)
// =============================================================================

/// Handle GET /api/tags - list available models
pub async fn handle_ollama_tags(server: ProxyServer) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    let operation = || async {
        let url = format!("{}/v1/models", server.config.lmstudio_url);
        server.logger.log(&format!("Calling LM Studio: GET {}", url));

        let response = server.client.get(&url).send().await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to reach LM Studio: {}", e)))?;

        if !response.status().is_success() {
            server.logger.log(&format!("LM Studio not available ({}), returning empty model list", response.status()));
            let empty_response = json!({
                "models": []
            });
            return Ok(empty_response);
        }

        let lm_response: Value = response.json().await
            .map_err(|_e| {
                ProxyError::internal_server_error("LM Studio response parsing failed")
            })?;

        // Transform LM Studio format to Ollama format with ALL required fields
        let models = if let Some(data) = lm_response.get("data").and_then(|d| d.as_array()) {
            data.iter().map(|model| {
                let model_id = model.get("id").and_then(|id| id.as_str()).unwrap_or("unknown");
                let cleaned_model = clean_model_name(model_id);
                let model_name = format!("{}:latest", cleaned_model);
                
                let (family, families) = determine_model_family(&cleaned_model);
                let parameter_size = determine_parameter_size(&cleaned_model);
                let size = estimate_model_size(parameter_size);
                
                json!({
                    "name": model_name,
                    "model": model_name,
                    "modified_at": chrono::Utc::now().to_rfc3339(),
                    "size": size,
                    "digest": format!("{:x}", md5::compute(model_id.as_bytes())),
                    "details": {
                        "parent_model": "",
                        "format": "gguf",
                        "family": family,
                        "families": families,
                        "parameter_size": parameter_size,
                        "quantization_level": "Q4_K_M"
                    }
                })
            }).collect::<Vec<_>>()
        } else {
            vec![]
        };

        let ollama_response = json!({
            "models": models
        });

        Ok(ollama_response)
    };

    let result = match with_retry(&server, operation).await {
        Ok(result) => result,
        Err(e) => {
            server.logger.log(&format!("Error in handle_ollama_tags: {}", e.message));
            json!({
                "models": []
            })
        }
    };

    let duration = start_time.elapsed();
    server.logger.log(&format!("Ollama tags response completed (took {})", format_duration(duration)));
    Ok(json_response(&result))
}

/// Handle POST /api/chat - chat completion with streaming support
pub async fn handle_ollama_chat(server: ProxyServer, body: Value) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    let operation = || async {
        // Extract fields from Ollama request
        let model = body.get("model").and_then(|m| m.as_str())
            .ok_or_else(|| ProxyError::bad_request("Missing 'model' field"))?;
        let messages = body.get("messages").and_then(|m| m.as_array())
            .ok_or_else(|| ProxyError::bad_request("Missing 'messages' field"))?;
        let stream = is_streaming_request(&body);

        let cleaned_model = clean_model_name(model);

        // Convert to LM Studio format
        let lm_request = json!({
            "model": cleaned_model,
            "messages": messages,
            "stream": stream,
            "temperature": body.get("options").and_then(|o| o.get("temperature")).unwrap_or(&json!(0.7)),
            "max_tokens": body.get("options").and_then(|o| o.get("num_predict")).unwrap_or(&json!(2048))
        });

        let url = format!("{}/v1/chat/completions", server.config.lmstudio_url);
        server.logger.log(&format!("Calling LM Studio: POST {} (stream: {})", url, stream));

        let response = server.client.post(&url)
            .json(&lm_request)
            .send().await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to reach LM Studio: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProxyError::internal_server_error(&format!("LM Studio error: {}", error_text)));
        }

        if stream {
            // Handle streaming response
            handle_streaming_response(response, true, model, start_time).await
        } else {
            // Handle non-streaming response
            handle_non_streaming_chat_response(response, model, messages, start_time).await
        }
    };

    let result = with_retry(&server, operation).await?;
    let duration = start_time.elapsed();

    server.logger.log(&format!("Ollama chat response completed (took {})", format_duration(duration)));
    Ok(result)
}

/// Handle non-streaming chat response
async fn handle_non_streaming_chat_response(
    response: reqwest::Response,
    model: &str,
    messages: &[Value],
    start_time: Instant,
) -> Result<warp::reply::Response, ProxyError> {
    let lm_response: Value = response.json().await
        .map_err(|e| ProxyError::internal_server_error(&format!("Failed to parse LM Studio response: {}", e)))?;

    // Transform response to Ollama format
    let content = if let Some(choices) = lm_response.get("choices").and_then(|c| c.as_array()) {
        if let Some(first_choice) = choices.first() {
            let mut content = first_choice.get("message")
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();

            // Merge reasoning_content if present
            if let Some(reasoning) = first_choice.get("message")
                .and_then(|m| m.get("reasoning_content"))
                .and_then(|r| r.as_str()) {
                content = format!("**Reasoning:**\n{}\n\n**Answer:**\n{}", reasoning, content);
            }

            content
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    // Calculate timing estimates
    let total_duration = start_time.elapsed().as_nanos() as u64;
    let prompt_eval_count = messages.len() as u64 * 10;
    let eval_count = content.len() as u64 / 4;

    let ollama_response = json!({
        "model": model,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "message": {
            "role": "assistant",
            "content": content
        },
        "done": true,
        "total_duration": total_duration,
        "load_duration": 1000000u64,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": total_duration / 4,
        "eval_count": eval_count,
        "eval_duration": total_duration / 2
    });

    Ok(json_response(&ollama_response))
}

/// Handle POST /api/generate - text completion with streaming support
pub async fn handle_ollama_generate(server: ProxyServer, body: Value) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    let operation = || async {
        let model = body.get("model").and_then(|m| m.as_str())
            .ok_or_else(|| ProxyError::bad_request("Missing 'model' field"))?;
        let prompt = body.get("prompt").and_then(|p| p.as_str())
            .ok_or_else(|| ProxyError::bad_request("Missing 'prompt' field"))?;
        let stream = is_streaming_request(&body);

        let cleaned_model = clean_model_name(model);

        // Convert to LM Studio completions format
        let lm_request = json!({
            "model": cleaned_model,
            "prompt": prompt,
            "stream": stream,
            "temperature": body.get("options").and_then(|o| o.get("temperature")).unwrap_or(&json!(0.7)),
            "max_tokens": body.get("options").and_then(|o| o.get("num_predict")).unwrap_or(&json!(2048))
        });

        let url = format!("{}/v1/completions", server.config.lmstudio_url);
        server.logger.log(&format!("Calling LM Studio: POST {} (stream: {})", url, stream));

        let response = server.client.post(&url)
            .json(&lm_request)
            .send().await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to reach LM Studio: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProxyError::internal_server_error(&format!("LM Studio error: {}", error_text)));
        }

        if stream {
            handle_streaming_response(response, false, model, start_time).await
        } else {
            handle_non_streaming_generate_response(response, model, prompt, start_time).await
        }
    };

    let result = with_retry(&server, operation).await?;
    let duration = start_time.elapsed();

    server.logger.log(&format!("Ollama generate response completed (took {})", format_duration(duration)));
    Ok(result)
}

/// Handle non-streaming generate response
async fn handle_non_streaming_generate_response(
    response: reqwest::Response,
    model: &str,
    prompt: &str,
    start_time: Instant,
) -> Result<warp::reply::Response, ProxyError> {
    let lm_response: Value = response.json().await
        .map_err(|e| ProxyError::internal_server_error(&format!("Failed to parse LM Studio response: {}", e)))?;

    let response_text = if let Some(choices) = lm_response.get("choices").and_then(|c| c.as_array()) {
        choices.first()
            .and_then(|choice| choice.get("text"))
            .and_then(|text| text.as_str())
            .unwrap_or("")
            .to_string()
    } else {
        String::new()
    };

    let total_duration = start_time.elapsed().as_nanos() as u64;
    let prompt_eval_count = prompt.len() as u64 / 4;
    let eval_count = response_text.len() as u64 / 4;

    let ollama_response = json!({
        "model": model,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "response": response_text,
        "done": true,
        "context": [1, 2, 3],
        "total_duration": total_duration,
        "load_duration": 1000000u64,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": total_duration / 4,
        "eval_count": eval_count,
        "eval_duration": total_duration / 2
    });

    Ok(json_response(&ollama_response))
}

/// Handle POST /api/embed or /api/embeddings - generate embeddings
pub async fn handle_ollama_embeddings(server: ProxyServer, body: Value) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    let operation = || async {
        let model = body.get("model").and_then(|m| m.as_str())
            .ok_or_else(|| ProxyError::bad_request("Missing 'model' field"))?;
        let input = body.get("input").or_else(|| body.get("prompt"))
            .ok_or_else(|| ProxyError::bad_request("Missing 'input' or 'prompt' field"))?;

        let cleaned_model = clean_model_name(model);

        let lm_request = json!({
            "model": cleaned_model,
            "input": input
        });

        let url = format!("{}/v1/embeddings", server.config.lmstudio_url);
        server.logger.log(&format!("Calling LM Studio: POST {}", url));

        let response = server.client.post(&url)
            .json(&lm_request)
            .send().await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to reach LM Studio: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProxyError::internal_server_error(&format!("LM Studio error: {}", error_text)));
        }

        let lm_response: Value = response.json().await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to parse LM Studio response: {}", e)))?;

        let embeddings = if let Some(data) = lm_response.get("data").and_then(|d| d.as_array()) {
            data.iter()
                .filter_map(|item| item.get("embedding"))
                .collect::<Vec<_>>()
        } else {
            vec![]
        };

        let total_duration = start_time.elapsed().as_nanos() as u64;

        let ollama_response = json!({
            "model": model,
            "embeddings": embeddings,
            "total_duration": total_duration,
            "load_duration": 1000000u64,
            "prompt_eval_count": 1
        });

        Ok(ollama_response)
    };

    let result = with_retry(&server, operation).await?;
    let duration = start_time.elapsed();

    server.logger.log(&format!("Ollama embeddings response completed (took {})", format_duration(duration)));
    Ok(json_response(&result))
}

// =============================================================================
// SIMPLE OLLAMA HANDLERS (no LM Studio calls needed)
// =============================================================================

/// Handle GET /api/ps - list running models
pub async fn handle_ollama_ps() -> Result<warp::reply::Response, ProxyError> {
    let response = json!({
        "models": []
    });
    Ok(json_response(&response))
}

/// Handle POST /api/show - show model info
pub async fn handle_ollama_show(body: Value) -> Result<warp::reply::Response, ProxyError> {
    let model = body.get("model").and_then(|m| m.as_str()).unwrap_or("unknown");
    let cleaned_model = clean_model_name(model);

    let architecture = match cleaned_model.to_lowercase() {
        name if name.contains("llama") => "llama",
        name if name.contains("mistral") => "mistral",
        name if name.contains("qwen") => "qwen",
        name if name.contains("deepseek") => "llama",
        name if name.contains("gemma") => "gemma",
        name if name.contains("phi") => "phi",
        name if name.contains("codellama") => "llama",
        name if name.contains("vicuna") => "llama",
        name if name.contains("alpaca") => "llama",
        _ => "llama",
    };

    let (family, parameter_size) = match cleaned_model.to_lowercase() {
        name if name.contains("7b") => ("llama", "7B"),
        name if name.contains("13b") => ("llama", "13B"),
        name if name.contains("30b") => ("llama", "30B"),
        name if name.contains("70b") => ("llama", "70B"),
        name if name.contains("8b") => ("llama", "8B"),
        name if name.contains("14b") => ("qwen", "14B"),
        name if name.contains("32b") => ("qwen", "32B"),
        name if name.contains("1.5b") => ("qwen", "1.5B"),
        name if name.contains("2b") => ("gemma", "2B"),
        name if name.contains("9b") => ("gemma", "9B"),
        name if name.contains("27b") => ("gemma", "27B"),
        _ => ("llama", "7B"),
    };

    let response = json!({
        "modelfile": format!("# Modelfile for {}\nFROM {}\nPARAMETER temperature 0.7\nPARAMETER top_p 0.9\nPARAMETER top_k 40", model, model),
        "parameters": "temperature 0.7\ntop_p 0.9\ntop_k 40\nrepeat_penalty 1.1",
        "template": "{{ if .System }}{{ .System }}\n{{ end }}{{ .Prompt }}",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": family,
            "families": [family],
            "parameter_size": parameter_size,
            "quantization_level": "Q4_K_M"
        },
        "model_info": {
            "general.architecture": architecture,
            "general.name": cleaned_model,
            "general.parameter_count": match parameter_size {
                _ => 7000000000u64,
            },
            "general.quantization_version": 2,
            "general.file_type": 2,
            "tokenizer.model": "llama",
            "tokenizer.chat_template": "{{ if .System }}{{ .System }}\n{{ end }}{{ .Prompt }}"
        },
        "capabilities": determine_model_capabilities(&cleaned_model),
        "system": format!("You are a helpful AI assistant using the {} model.", model),
        "license": "Custom license for proxy model",
        "digest": format!("{:x}", md5::compute(model.as_bytes())),
        "size": match parameter_size {
            "1.5B" => 1000000000u64,
            "2B" => 1500000000u64,
            "7B" => 4000000000u64,
            "8B" => 5000000000u64,
            "9B" => 5500000000u64,
            "13B" => 8000000000u64,
            "14B" => 8500000000u64,
            "27B" => 16000000000u64,
            "30B" => 18000000000u64,
            "32B" => 20000000000u64,
            "70B" => 40000000000u64,
            _ => 4000000000u64,
        },
        "modified_at": chrono::Utc::now().to_rfc3339()
    });

    Ok(json_response(&response))
}

/// Handle GET /api/version - return version info
pub async fn handle_ollama_version() -> Result<warp::reply::Response, ProxyError> {
    let response = json!({
        "version": "0.5.1-proxy"
    });
    Ok(json_response(&response))
}

/// Handle unsupported endpoints
pub async fn handle_unsupported(endpoint: &str) -> Result<warp::reply::Response, ProxyError> {
    Err(ProxyError::not_implemented(&format!(
        "The '{}' endpoint is not supported by this proxy. This endpoint requires direct Ollama functionality that cannot be translated to LM Studio.",
        endpoint
    )))
}

// =============================================================================
// LM STUDIO PASSTHROUGH HANDLERS (with streaming support)
// =============================================================================

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

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn determine_model_family(model_name: &str) -> (&'static str, Vec<&'static str>) {
    let lower_name = model_name.to_lowercase();

    match lower_name {
        name if name.contains("llama") => ("llama", vec!["llama"]),
        name if name.contains("mistral") => ("mistral", vec!["mistral"]),
        name if name.contains("qwen") => ("qwen2", vec!["qwen2"]),
        name if name.contains("deepseek") => ("llama", vec!["llama"]),
        name if name.contains("gemma") => ("gemma", vec!["gemma"]),
        name if name.contains("phi") => ("phi", vec!["phi"]),
        name if name.contains("codellama") => ("llama", vec!["llama"]),
        name if name.contains("vicuna") => ("llama", vec!["llama"]),
        name if name.contains("alpaca") => ("llama", vec!["llama"]),
        _ => ("llama", vec!["llama"]),
    }
}

fn determine_parameter_size(model_name: &str) -> &'static str {
    let lower_name = model_name.to_lowercase();

    if lower_name.contains("0.5b") { "0.5B" } else if lower_name.contains("1.5b") { "1.5B" } else if lower_name.contains("2b") { "2B" } else if lower_name.contains("3b") { "3B" } else if lower_name.contains("7b") { "7B" } else if lower_name.contains("8b") { "8B" } else if lower_name.contains("9b") { "9B" } else if lower_name.contains("13b") { "13B" } else if lower_name.contains("14b") { "14B" } else if lower_name.contains("27b") { "27B" } else if lower_name.contains("30b") { "30B" } else if lower_name.contains("32b") { "32B" } else if lower_name.contains("70b") { "70B" } else { "7B" }
}

fn estimate_model_size(parameter_size: &str) -> u64 {
    match parameter_size {
        "0.5B" => 500_000_000,
        "1.5B" => 1_000_000_000,
        "2B" => 1_500_000_000,
        "3B" => 2_000_000_000,
        "7B" => 4_000_000_000,
        "8B" => 5_000_000_000,
        "9B" => 5_500_000_000,
        "13B" => 8_000_000_000,
        "14B" => 8_500_000_000,
        "27B" => 16_000_000_000,
        "30B" => 18_000_000_000,
        "32B" => 20_000_000_000,
        "70B" => 40_000_000_000,
        _ => 4_000_000_000,
    }
}

fn determine_model_capabilities(model_name: &str) -> Vec<&'static str> {
    let lower_name = model_name.to_lowercase();
    let mut capabilities = vec!["completion", "chat"];

    if lower_name.contains("embed") || lower_name.contains("bge") || lower_name.contains("nomic") {
        capabilities.push("embeddings");
    }

    if lower_name.contains("llava") || lower_name.contains("vision") || lower_name.contains("multimodal") {
        capabilities.push("vision");
    }

    if lower_name.contains("llama3") || lower_name.contains("mistral") || lower_name.contains("qwen") {
        capabilities.push("tools");
    }

    capabilities
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_request_detection() {
        let streaming_body = json!({"stream": true, "model": "test"});
        let non_streaming_body = json!({"stream": false, "model": "test"});
        let no_stream_body = json!({"model": "test"});

        assert_eq!(is_streaming_request(&streaming_body), true);
        assert_eq!(is_streaming_request(&non_streaming_body), false);
        assert_eq!(is_streaming_request(&no_stream_body), false);
    }

    #[test]
    fn test_model_name_cleaning_in_handlers() {
        let test_cases = vec![
            ("llama3.2:latest", "llama3.2"),
            ("deepseek-r1:2", "deepseek-r1"),
            ("model-name:3", "model-name"),
            ("simple-model", "simple-model"),
        ];

        for (input, expected) in test_cases {
            assert_eq!(clean_model_name(input), expected);
        }
    }

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
    fn test_parameter_size_detection() {
        assert_eq!(determine_parameter_size("qwen2.5-coder-0.5b-instruct"), "0.5B");
        assert_eq!(determine_parameter_size("llama-7b-chat"), "7B");
        assert_eq!(determine_parameter_size("mistral-14b"), "14B");
        assert_eq!(determine_parameter_size("unknown-model"), "7B");
    }
}