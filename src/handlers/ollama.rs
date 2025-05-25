// src/handlers/ollama.rs - Simplified Ollama API handlers using centralized model resolver

use serde_json::{json, Value, Map};
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::server::ProxyServer;
use crate::handlers::retry::with_simple_retry;
use crate::utils::{clean_model_name, determine_parameter_size, format_duration, ProxyError, ModelResolver};
use crate::common::{CancellableRequest, handle_json_response};
use super::retry::with_retry_and_cancellation;
use super::streaming::{is_streaming_request, handle_streaming_response};
use super::helpers::{
    json_response, determine_model_family,
    estimate_model_size, determine_model_capabilities,
    transform_chat_response, transform_generate_response, transform_embeddings_response,
    build_lm_studio_request
};

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

                // Use centralized model resolver
                let resolver = ModelResolver::new(server.clone());
                let lmstudio_model = resolver.resolve_model_name(model, cancellation_token.clone()).await?;

                // Build request with only provided parameters
                let mut base_params = Map::new();
                base_params.insert("model".to_string(), json!(lmstudio_model));
                base_params.insert("messages".to_string(), json!(messages));
                base_params.insert("stream".to_string(), json!(stream));

                let lm_request = build_lm_studio_request(base_params, body.get("options"));

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

                // Use centralized model resolver
                let resolver = ModelResolver::new(server.clone());
                let lmstudio_model = resolver.resolve_model_name(model, cancellation_token.clone()).await?;

                // Build request with only provided parameters
                let mut base_params = Map::new();
                base_params.insert("model".to_string(), json!(lmstudio_model));
                base_params.insert("prompt".to_string(), json!(prompt));
                base_params.insert("stream".to_string(), json!(stream));

                let lm_request = build_lm_studio_request(base_params, body.get("options"));

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

                // Use centralized model resolver
                let resolver = ModelResolver::new(server.clone());
                let lmstudio_model = resolver.resolve_model_name(model, cancellation_token.clone()).await?;

                // Minimal request for embeddings (no optional parameters needed)
                let lm_request = json!({
                    "model": lmstudio_model,
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

    let size_in_bytes = estimate_model_size(parameter_size);

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
