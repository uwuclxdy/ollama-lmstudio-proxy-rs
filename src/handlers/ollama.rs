// src/handlers/ollama.rs - Fixed Ollama API handlers with proper model name resolution
//
// KEY FIX: Instead of complex retry/switching logic, we now simply resolve
// Ollama model names to actual LM Studio model names before sending requests.
// LM Studio will automatically switch models when we send the correct name.

use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::server::ProxyServer;
use crate::handlers::retry::with_simple_retry;
use crate::utils::{clean_model_name, format_duration, ProxyError};
use crate::common::{CancellableRequest, handle_json_response};
use super::retry::with_retry_and_cancellation;
use super::streaming::{is_streaming_request, handle_streaming_response};
use super::helpers::{
    json_response, determine_model_family, determine_parameter_size,
    estimate_model_size, determine_model_capabilities,
    transform_chat_response, transform_generate_response, transform_embeddings_response
};

/// Resolve Ollama model name to actual LM Studio model name
async fn resolve_model_name(
    server: &ProxyServer,
    ollama_model: &str,
    cancellation_token: CancellationToken,
) -> Result<String, ProxyError> {
    let cleaned_ollama = clean_model_name(ollama_model);

    // Get available models from LM Studio
    let url = format!("{}/v1/models", server.config.lmstudio_url);

    let request = CancellableRequest::new(
        server.client.clone(),
        cancellation_token,
        server.logger.clone(),
        server.config.request_timeout_seconds
    );

    let response = match request.make_request(reqwest::Method::GET, &url, None).await {
        Ok(response) => response,
        Err(_) => {
            // If we can't fetch models, use cleaned name as fallback
            server.logger.log(&format!("⚠️  Cannot fetch LM Studio models, using fallback: '{}'", cleaned_ollama));
            return Ok(cleaned_ollama);
        }
    };

    if !response.status().is_success() {
        server.logger.log(&format!("⚠️  LM Studio models endpoint returned {}, using fallback: '{}'", response.status(), cleaned_ollama));
        return Ok(cleaned_ollama);
    }

    let models_response: Value = match response.json().await {
        Ok(json) => json,
        Err(_) => {
            server.logger.log(&format!("⚠️  Cannot parse LM Studio models response, using fallback: '{}'", cleaned_ollama));
            return Ok(cleaned_ollama);
        }
    };

    let mut available_models = Vec::new();
    if let Some(data) = models_response.get("data").and_then(|d| d.as_array()) {
        for model in data {
            if let Some(model_id) = model.get("id").and_then(|id| id.as_str()) {
                available_models.push(model_id.to_string());
            }
        }
    }

    // Find the best match
    if let Some(matched_model) = find_best_model_match(&cleaned_ollama, &available_models) {
        server.logger.log(&format!("✅ Resolved '{}' -> '{}'", ollama_model, matched_model));
        Ok(matched_model)
    } else {
        server.logger.log(&format!("⚠️  No match found for '{}', using cleaned name '{}'", ollama_model, cleaned_ollama));
        Ok(cleaned_ollama)
    }
}

/// Find the best matching LM Studio model for an Ollama model name
fn find_best_model_match(ollama_name: &str, available_models: &[String]) -> Option<String> {
    let lower_ollama = ollama_name.to_lowercase();

    // Direct match first
    for model in available_models {
        if model.to_lowercase() == lower_ollama {
            return Some(model.clone());
        }
    }

    // Pattern matching for common model families
    for model in available_models {
        let lower_model = model.to_lowercase();

        // Llama family matching
        if lower_ollama.contains("llama") && lower_model.contains("llama") {
            if models_match_size(&lower_ollama, &lower_model) {
                return Some(model.clone());
            }
        }

        // Qwen family matching
        else if lower_ollama.contains("qwen") && lower_model.contains("qwen") {
            if models_match_size(&lower_ollama, &lower_model) {
                return Some(model.clone());
            }
        }

        // Mistral family matching
        else if lower_ollama.contains("mistral") && lower_model.contains("mistral") {
            if models_match_size(&lower_ollama, &lower_model) {
                return Some(model.clone());
            }
        }

        // Gemma family matching
        else if lower_ollama.contains("gemma") && lower_model.contains("gemma") {
            if models_match_size(&lower_ollama, &lower_model) {
                return Some(model.clone());
            }
        }

        // Phi family matching
        else if lower_ollama.contains("phi") && lower_model.contains("phi") {
            if models_match_size(&lower_ollama, &lower_model) {
                return Some(model.clone());
            }
        }

        // DeepSeek family matching
        else if lower_ollama.contains("deepseek") && lower_model.contains("deepseek") {
            if models_match_size(&lower_ollama, &lower_model) {
                return Some(model.clone());
            }
        }
    }

    None
}

/// Check if two model names refer to the same model size
fn models_match_size(name1: &str, name2: &str) -> bool {
    let sizes = ["0.5b", "1.5b", "2b", "3b", "7b", "8b", "9b", "11b", "13b", "14b", "27b", "30b", "32b", "70b"];

    for size in &sizes {
        if name1.contains(size) && name2.contains(size) {
            return true;
        }
    }

    // If no size found in either, consider them matching (size unknown)
    true
}

/// Handle GET /api/tags - list available models with cancellation support
pub async fn handle_ollama_tags(
    server: Arc<ProxyServer>,
    cancellation_token: CancellationToken
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    let operation = {
        let server = server.clone();
        let cancellation_token = cancellation_token.clone();
        move || {
            let server = server.clone();
            let cancellation_token = cancellation_token.clone();
            async move {
                let request = CancellableRequest::new(
                    server.client.clone(),
                    cancellation_token.clone(),
                    server.logger.clone(),
                    server.config.request_timeout_seconds
                );
                let url = format!("{}/v1/models", server.config.lmstudio_url);
                server.logger.log(&format!("Calling LM Studio: GET {}", url));

                let response = request.make_request(reqwest::Method::GET, &url, None).await?;

                if !response.status().is_success() {
                    let status = response.status();
                    return if status.as_u16() == 404 || status.as_u16() == 503 {
                        server.logger.log(&format!("LM Studio not available ({}), returning empty model list", status));
                        let empty_response = json!({
                            "models": []
                        });
                        Ok(empty_response)
                    } else {
                        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                        server.logger.log(&format!("LM Studio error ({}): {}", status, error_text));
                        Err(ProxyError::new(
                            format!("LM Studio error: {}", error_text),
                            status.as_u16()
                        ))
                    }
                }

                let lm_response = handle_json_response(response, cancellation_token).await?;

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
            }
        }
    };

    let result = with_simple_retry(operation, cancellation_token).await?;
    let duration = start_time.elapsed();
    server.logger.log(&format!("Ollama tags response completed (took {})", format_duration(duration)));
    Ok(json_response(&result))
}

/// Handle POST /api/chat - chat completion with streaming support and cancellation
pub async fn handle_ollama_chat(
    server: Arc<ProxyServer>,
    body: Value,
    cancellation_token: CancellationToken
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    // Extract model name early for retry logic
    let model = body.get("model").and_then(|m| m.as_str())
        .ok_or_else(|| ProxyError::bad_request("Missing 'model' field"))?;

    let operation = {
        let server = server.clone();
        let body = body.clone();
        let cancellation_token = cancellation_token.clone();
        move || {
            let server = server.clone();
            let body = body.clone();
            let cancellation_token = cancellation_token.clone();
            async move {
                // Extract fields from Ollama request
                let model = body.get("model").and_then(|m| m.as_str())
                    .ok_or_else(|| ProxyError::bad_request("Missing 'model' field"))?;
                let messages = body.get("messages").and_then(|m| m.as_array())
                    .ok_or_else(|| ProxyError::bad_request("Missing 'messages' field"))?;
                let stream = is_streaming_request(&body);

                // 🔑 KEY FIX: Resolve to actual LM Studio model name
                let lmstudio_model = resolve_model_name(&server, model, cancellation_token.clone()).await?;

                // Convert to LM Studio format using the resolved model name
                let lm_request = json!({
                    "model": lmstudio_model,  // 🎯 Use actual LM Studio model name here
                    "messages": messages,
                    "stream": stream,
                    "temperature": body.get("options").and_then(|o| o.get("temperature")).unwrap_or(&json!(0.7)),
                    "max_tokens": body.get("options").and_then(|o| o.get("num_predict")).unwrap_or(&json!(2048))
                });

                let request = CancellableRequest::new(
                    server.client.clone(),
                    cancellation_token.clone(),
                    server.logger.clone(),
                    server.config.request_timeout_seconds
                );
                let url = format!("{}/v1/chat/completions", server.config.lmstudio_url);
                server.logger.log(&format!("Calling LM Studio: POST {} with model '{}' (stream: {})", url, lmstudio_model, stream));

                let response = request.make_request(
                    reqwest::Method::POST,
                    &url,
                    Some(lm_request)
                ).await?;

                if !response.status().is_success() {
                    let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                    return Err(ProxyError::internal_server_error(&format!("LM Studio error: {}", error_text)));
                }

                if stream {
                    // Handle streaming response with cancellation
                    handle_streaming_response(
                        response,
                        true,
                        model,
                        start_time,
                        cancellation_token.clone(),
                        server.logger.clone(),
                        server.config.stream_timeout_seconds
                    ).await
                } else {
                    // Handle non-streaming response
                    let lm_response = handle_json_response(response, cancellation_token).await?;
                    let ollama_response = transform_chat_response(&lm_response, model, messages, start_time);
                    Ok(json_response(&ollama_response))
                }
            }
        }
    };

    let result = with_retry_and_cancellation(&server, model, operation, cancellation_token).await?;
    let duration = start_time.elapsed();

    server.logger.log(&format!("Ollama chat response completed (took {})", format_duration(duration)));
    Ok(result)
}

/// Handle POST /api/generate - text completion with streaming support and cancellation
pub async fn handle_ollama_generate(
    server: Arc<ProxyServer>,
    body: Value,
    cancellation_token: CancellationToken
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    // Extract model name early for retry logic
    let model = body.get("model").and_then(|m| m.as_str())
        .ok_or_else(|| ProxyError::bad_request("Missing 'model' field"))?;

    let operation = {
        let server = server.clone();
        let body = body.clone();
        let cancellation_token = cancellation_token.clone();
        move || {
            let server = server.clone();
            let body = body.clone();
            let cancellation_token = cancellation_token.clone();
            async move {
                let model = body.get("model").and_then(|m| m.as_str())
                    .ok_or_else(|| ProxyError::bad_request("Missing 'model' field"))?;
                let prompt = body.get("prompt").and_then(|p| p.as_str())
                    .ok_or_else(|| ProxyError::bad_request("Missing 'prompt' field"))?;
                let stream = is_streaming_request(&body);

                // 🔑 KEY FIX: Resolve to actual LM Studio model name
                let lmstudio_model = resolve_model_name(&server, model, cancellation_token.clone()).await?;

                // Convert to LM Studio completions format using the resolved model name
                let lm_request = json!({
                    "model": lmstudio_model,  // 🎯 Use actual LM Studio model name here
                    "prompt": prompt,
                    "stream": stream,
                    "temperature": body.get("options").and_then(|o| o.get("temperature")).unwrap_or(&json!(0.7)),
                    "max_tokens": body.get("options").and_then(|o| o.get("num_predict")).unwrap_or(&json!(4096))
                });

                let request = CancellableRequest::new(
                    server.client.clone(),
                    cancellation_token.clone(),
                    server.logger.clone(),
                    server.config.request_timeout_seconds
                );
                let url = format!("{}/v1/completions", server.config.lmstudio_url);
                server.logger.log(&format!("Calling LM Studio: POST {} with model '{}' (stream: {})", url, lmstudio_model, stream));

                let response = request.make_request(
                    reqwest::Method::POST,
                    &url,
                    Some(lm_request)
                ).await?;

                if !response.status().is_success() {
                    let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                    return Err(ProxyError::internal_server_error(&format!("LM Studio error: {}", error_text)));
                }

                if stream {
                    handle_streaming_response(
                        response,
                        false,
                        model,
                        start_time,
                        cancellation_token.clone(),
                        server.logger.clone(),
                        server.config.stream_timeout_seconds
                    ).await
                } else {
                    let lm_response = handle_json_response(response, cancellation_token).await?;
                    let ollama_response = transform_generate_response(&lm_response, model, prompt, start_time);
                    Ok(json_response(&ollama_response))
                }
            }
        }
    };

    let result = with_retry_and_cancellation(&server, model, operation, cancellation_token).await?;
    let duration = start_time.elapsed();

    server.logger.log(&format!("Ollama generate response completed (took {})", format_duration(duration)));
    Ok(result)
}

/// Handle POST /api/embed or /api/embeddings - generate embeddings with cancellation
pub async fn handle_ollama_embeddings(
    server: Arc<ProxyServer>,
    body: Value,
    cancellation_token: CancellationToken
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    // Extract model name early for retry logic
    let model = body.get("model").and_then(|m| m.as_str())
        .ok_or_else(|| ProxyError::bad_request("Missing 'model' field"))?;

    let operation = {
        let server = server.clone();
        let body = body.clone();
        let cancellation_token = cancellation_token.clone();
        move || {
            let server = server.clone();
            let body = body.clone();
            let cancellation_token = cancellation_token.clone();
            async move {
                let model = body.get("model").and_then(|m| m.as_str())
                    .ok_or_else(|| ProxyError::bad_request("Missing 'model' field"))?;
                let input = body.get("input").or_else(|| body.get("prompt"))
                    .ok_or_else(|| ProxyError::bad_request("Missing 'input' or 'prompt' field"))?;

                // 🔑 KEY FIX: Resolve to actual LM Studio model name
                let lmstudio_model = resolve_model_name(&server, model, cancellation_token.clone()).await?;

                let lm_request = json!({
                    "model": lmstudio_model,  // 🎯 Use actual LM Studio model name here
                    "input": input
                });

                let request = CancellableRequest::new(
                    server.client.clone(),
                    cancellation_token.clone(),
                    server.logger.clone(),
                    server.config.request_timeout_seconds
                );
                let url = format!("{}/v1/embeddings", server.config.lmstudio_url);
                server.logger.log(&format!("Calling LM Studio: POST {} with model '{}'", url, lmstudio_model));

                let response = request.make_request(
                    reqwest::Method::POST,
                    &url,
                    Some(lm_request)
                ).await?;

                if !response.status().is_success() {
                    let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                    return Err(ProxyError::internal_server_error(&format!("LM Studio error: {}", error_text)));
                }

                let lm_response = handle_json_response(response, cancellation_token).await?;
                let ollama_response = transform_embeddings_response(&lm_response, model, start_time);
                Ok(ollama_response)
            }
        }
    };

    let result = with_retry_and_cancellation(&server, model, operation, cancellation_token).await?;
    let duration = start_time.elapsed();

    server.logger.log(&format!("Ollama embeddings response completed (took {})", format_duration(duration)));
    Ok(json_response(&result))
}

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

    let size_in_bytes = if let Some(num_str) = parameter_size.trim_end_matches(&['B', 'b']).parse::<f64>().ok() {
        let billions = num_str;
        let approx_bytes = billions * 600_000_000.0;
        approx_bytes as u64
    } else {
        4_000_000_000u64
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
            "general.parameter_count": size_in_bytes,
            "general.quantization_version": 2,
            "general.file_type": 2,
            "tokenizer.model": "llama",
            "tokenizer.chat_template": "{{ if .System }}{{ .System }}\n{{ end }}{{ .Prompt }}"
        },
        "capabilities": determine_model_capabilities(&cleaned_model),
        "system": format!("You are a helpful AI assistant using the {} model.", model),
        "license": "Custom license for proxy model",
        "digest": format!("{:x}", md5::compute(model.as_bytes())),
        "size": size_in_bytes,
        "modified_at": chrono::Utc::now().to_rfc3339()
    });

    Ok(json_response(&response))
}

/// Handle GET /api/version - return version info
pub async fn handle_ollama_version() -> Result<warp::reply::Response, ProxyError> {
    let response = json!({
            "version": crate::VERSION,
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
