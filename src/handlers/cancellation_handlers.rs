// src/handlers/cancellation_handlers.rs - Cancellation-aware versions of your handlers
// These wrap your existing handlers with cancellation support

use serde_json::Value;
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::server::ProxyServer;
use crate::utils::{clean_model_name, format_duration, ProxyError};
use crate::handlers::{
    retry::with_retry,
    streaming::is_streaming_request,
    helpers::json_response,
};
use super::cancellation::{
    CancellableRequest,
    handle_streaming_response_with_cancellation,
    handle_passthrough_streaming_response_with_cancellation,
};

/// Handle GET /api/tags with cancellation support
pub async fn handle_ollama_tags_with_cancellation(
    server: Arc<ProxyServer>,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    // Check if already cancelled
    if cancellation_token.is_cancelled() {
        return Err(ProxyError::request_cancelled());
    }

    let operation = || async {
        let request = CancellableRequest::new(server.client.clone(), cancellation_token.clone());
        let url = format!("{}/v1/models", server.config.lmstudio_url);
        server.logger.log(&format!("Calling LM Studio: GET {}", url));

        let response = request.make_request(reqwest::Method::GET, &url, None).await?;

        if !response.status().is_success() {
            server.logger.log(&format!("LM Studio not available ({}), returning empty model list", response.status()));
            let empty_response = serde_json::json!({
                "models": []
            });
            return Ok(empty_response);
        }

        // Check cancellation before processing response
        if cancellation_token.is_cancelled() {
            return Err(ProxyError::request_cancelled());
        }

        let response_future = response.json::<Value>();
        let lm_response: Value = tokio::select! {
            result = response_future => {
                result.map_err(|_e| ProxyError::internal_server_error("LM Studio response parsing failed"))?
            }
            _ = cancellation_token.cancelled() => {
                return Err(ProxyError::request_cancelled());
            }
        };

        // Use your existing transformation logic
        let models = if let Some(data) = lm_response.get("data").and_then(|d| d.as_array()) {
            data.iter().map(|model| {
                let model_id = model.get("id").and_then(|id| id.as_str()).unwrap_or("unknown");
                let cleaned_model = clean_model_name(model_id);
                let model_name = format!("{}:latest", cleaned_model);

                serde_json::json!({
                    "name": model_name,
                    "model": model_name,
                    "modified_at": chrono::Utc::now().to_rfc3339(),
                    "size": 4000000000u64, // 4GB default
                    "digest": format!("{:x}", md5::compute(model_id.as_bytes())),
                    "details": {
                        "parent_model": "",
                        "format": "gguf",
                        "family": "llama",
                        "families": ["llama"],
                        "parameter_size": "7B",
                        "quantization_level": "Q4_K_M"
                    }
                })
            }).collect::<Vec<_>>()
        } else {
            vec![]
        };

        Ok(serde_json::json!({
            "models": models
        }))
    };

    let result = match with_retry(&server, operation).await {
        Ok(result) => result,
        Err(e) if e.is_cancelled() => {
            server.logger.log("Tags request was cancelled");
            return Err(ProxyError::request_cancelled());
        }
        Err(e) => {
            server.logger.log(&format!("Error in handle_ollama_tags: {}", e.message));
            serde_json::json!({
                "models": []
            })
        }
    };

    let duration = start_time.elapsed();
    server.logger.log(&format!("Ollama tags response completed (took {})", format_duration(duration)));
    Ok(json_response(&result))
}

/// Handle POST /api/chat with cancellation support
pub async fn handle_ollama_chat_with_cancellation(
    server: Arc<ProxyServer>,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    // Check if already cancelled
    if cancellation_token.is_cancelled() {
        return Err(ProxyError::request_cancelled());
    }

    let operation = || async {
        // Extract fields from Ollama request
        let model = body.get("model").and_then(|m| m.as_str())
            .ok_or_else(|| ProxyError::bad_request("Missing 'model' field"))?;
        let messages = body.get("messages").and_then(|m| m.as_array())
            .ok_or_else(|| ProxyError::bad_request("Missing 'messages' field"))?;
        let stream = is_streaming_request(&body);

        let cleaned_model = clean_model_name(model);

        // Convert to LM Studio format
        let lm_request = serde_json::json!({
            "model": cleaned_model,
            "messages": messages,
            "stream": stream,
            "temperature": body.get("options").and_then(|o| o.get("temperature")).unwrap_or(&serde_json::json!(0.7)),
            "max_tokens": body.get("options").and_then(|o| o.get("num_predict")).unwrap_or(&serde_json::json!(2048))
        });

        let request = CancellableRequest::new(server.client.clone(), cancellation_token.clone());
        let url = format!("{}/v1/chat/completions", server.config.lmstudio_url);
        server.logger.log(&format!("Calling LM Studio: POST {} (stream: {})", url, stream));

        let response = request.make_request(
            reqwest::Method::POST,
            &url,
            Some(lm_request),
        ).await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProxyError::internal_server_error(&format!("LM Studio error: {}", error_text)));
        }

        if stream {
            // Handle streaming response with cancellation
            handle_streaming_response_with_cancellation(response, true, model, start_time, cancellation_token.clone()).await
        } else {
            // Handle non-streaming response with cancellation
            handle_non_streaming_chat_response_with_cancellation(response, model, messages, start_time, cancellation_token.clone()).await
        }
    };

    let result = with_retry(&server, operation).await?;
    let duration = start_time.elapsed();

    server.logger.log(&format!("Ollama chat response completed (took {})", format_duration(duration)));
    Ok(result)
}

/// Handle POST /api/generate with cancellation support
pub async fn handle_ollama_generate_with_cancellation(
    server: Arc<ProxyServer>,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    // Check if already cancelled
    if cancellation_token.is_cancelled() {
        return Err(ProxyError::request_cancelled());
    }

    let operation = || async {
        let model = body.get("model").and_then(|m| m.as_str())
            .ok_or_else(|| ProxyError::bad_request("Missing 'model' field"))?;
        let prompt = body.get("prompt").and_then(|p| p.as_str())
            .ok_or_else(|| ProxyError::bad_request("Missing 'prompt' field"))?;
        let stream = is_streaming_request(&body);

        let cleaned_model = clean_model_name(model);

        // Convert to LM Studio completions format
        let lm_request = serde_json::json!({
            "model": cleaned_model,
            "prompt": prompt,
            "stream": stream,
            "temperature": body.get("options").and_then(|o| o.get("temperature")).unwrap_or(&serde_json::json!(0.7)),
            "max_tokens": body.get("options").and_then(|o| o.get("num_predict")).unwrap_or(&serde_json::json!(2048))
        });

        let request = CancellableRequest::new(server.client.clone(), cancellation_token.clone());
        let url = format!("{}/v1/completions", server.config.lmstudio_url);
        server.logger.log(&format!("Calling LM Studio: POST {} (stream: {})", url, stream));

        let response = request.make_request(
            reqwest::Method::POST,
            &url,
            Some(lm_request),
        ).await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProxyError::internal_server_error(&format!("LM Studio error: {}", error_text)));
        }

        if stream {
            handle_streaming_response_with_cancellation(response, false, model, start_time, cancellation_token.clone()).await
        } else {
            handle_non_streaming_generate_response_with_cancellation(response, model, prompt, start_time, cancellation_token.clone()).await
        }
    };

    let result = with_retry(&server, operation).await?;
    let duration = start_time.elapsed();

    server.logger.log(&format!("Ollama generate response completed (took {})", format_duration(duration)));
    Ok(result)
}

/// Handle POST /api/embeddings with cancellation support
pub async fn handle_ollama_embeddings_with_cancellation(
    server: Arc<ProxyServer>,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    // Check if already cancelled
    if cancellation_token.is_cancelled() {
        return Err(ProxyError::request_cancelled());
    }

    let operation = || async {
        let model = body.get("model").and_then(|m| m.as_str())
            .ok_or_else(|| ProxyError::bad_request("Missing 'model' field"))?;
        let input = body.get("input").or_else(|| body.get("prompt"))
            .ok_or_else(|| ProxyError::bad_request("Missing 'input' or 'prompt' field"))?;

        let cleaned_model = clean_model_name(model);

        let lm_request = serde_json::json!({
            "model": cleaned_model,
            "input": input
        });

        let request = CancellableRequest::new(server.client.clone(), cancellation_token.clone());
        let url = format!("{}/v1/embeddings", server.config.lmstudio_url);
        server.logger.log(&format!("Calling LM Studio: POST {}", url));

        let response = request.make_request(
            reqwest::Method::POST,
            &url,
            Some(lm_request),
        ).await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProxyError::internal_server_error(&format!("LM Studio error: {}", error_text)));
        }

        let response_future = response.json::<Value>();

        let lm_response: Value = tokio::select! {
            result = response_future => {
                result.map_err(|e| ProxyError::internal_server_error(&format!("Failed to parse LM Studio response: {}", e)))?
            }
            _ = cancellation_token.cancelled() => {
                return Err(ProxyError::request_cancelled());
            }
        };

        let embeddings = if let Some(data) = lm_response.get("data").and_then(|d| d.as_array()) {
            data.iter()
                .filter_map(|item| item.get("embedding"))
                .collect::<Vec<_>>()
        } else {
            vec![]
        };

        let total_duration = start_time.elapsed().as_nanos() as u64;

        Ok(serde_json::json!({
            "model": model,
            "embeddings": embeddings,
            "total_duration": total_duration,
            "load_duration": 1000000u64,
            "prompt_eval_count": 1
        }))
    };

    let result = with_retry(&server, operation).await?;
    let duration = start_time.elapsed();

    server.logger.log(&format!("Ollama embeddings response completed (took {})", format_duration(duration)));
    Ok(json_response(&result))
}

/// Handle LM Studio passthrough with cancellation support
pub async fn handle_lmstudio_passthrough_with_cancellation(
    server: Arc<ProxyServer>,
    method: &str,
    endpoint: &str,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    // Check if already cancelled
    if cancellation_token.is_cancelled() {
        return Err(ProxyError::request_cancelled());
    }

    let operation = || async {
        let url = format!("{}{}", server.config.lmstudio_url, endpoint);
        let is_streaming = is_streaming_request(&body);

        server.logger.log(&format!("Passthrough: {} {} (stream: {})", method, url, is_streaming));

        let request_method = match method {
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
    };

    let result = with_retry(&server, operation).await?;
    let duration = start_time.elapsed();

    server.logger.log(&format!("LM Studio passthrough completed (took {})", format_duration(duration)));
    Ok(result)
}

// Helper functions for non-streaming responses with cancellation

async fn handle_non_streaming_chat_response_with_cancellation(
    response: reqwest::Response,
    model: &str,
    messages: &[Value],
    start_time: Instant,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    // Check if cancelled before processing response
    if cancellation_token.is_cancelled() {
        return Err(ProxyError::request_cancelled());
    }

    let response_future = response.json::<Value>();

    let lm_response: Value = tokio::select! {
        result = response_future => {
            result.map_err(|e| ProxyError::internal_server_error(&format!("Failed to parse LM Studio response: {}", e)))?
        }
        _ = cancellation_token.cancelled() => {
            return Err(ProxyError::request_cancelled());
        }
    };

    // Transform response to Ollama format (using your existing logic)
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

    let ollama_response = serde_json::json!({
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

async fn handle_non_streaming_generate_response_with_cancellation(
    response: reqwest::Response,
    model: &str,
    prompt: &str,
    start_time: Instant,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    // Check if cancelled before processing response
    if cancellation_token.is_cancelled() {
        return Err(ProxyError::request_cancelled());
    }

    let response_future = response.json::<Value>();

    let lm_response: Value = tokio::select! {
        result = response_future => {
            result.map_err(|e| ProxyError::internal_server_error(&format!("Failed to parse LM Studio response: {}", e)))?
        }
        _ = cancellation_token.cancelled() => {
            return Err(ProxyError::request_cancelled());
        }
    };

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

    let ollama_response = serde_json::json!({
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