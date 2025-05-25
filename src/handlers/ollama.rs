// src/handlers/ollama.rs - Simplified Ollama API handlers using consolidated systems

use serde_json::{json, Value, Map};
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::constants::*;
use crate::server::ProxyServer;
use crate::model::{ModelInfo, ModelResolver, clean_model_name};
use crate::handlers::helpers::{json_response, ResponseTransformer, build_lm_studio_request};
use crate::handlers::retry::with_simple_retry;
use crate::handlers::streaming::{is_streaming_request, handle_streaming_response};
use crate::utils::ProxyError;
use crate::common::{CancellableRequest, handle_json_response};
use super::retry::with_retry_and_cancellation;

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
                server.logger.log_request("GET", &url, None);

                let response = request.make_request(reqwest::Method::GET, &url, None).await?;

                if !response.status().is_success() {
                    let status = response.status();
                    return if status.as_u16() == 404 || status.as_u16() == 503 {
                        server.logger.log_warning(&format!("{} ({}), returning empty model list", ERROR_LM_STUDIO_UNAVAILABLE, status));
                        Ok(json!({"models": []}))
                    } else {
                        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                        server.logger.log_error("LM Studio request", &format!("{}: {}", status, error_text));
                        Err(ProxyError::new(format!("LM Studio error: {}", error_text), status.as_u16()))
                    }
                }

                let lm_response = handle_json_response(response, cancellation_token).await?;

                // Transform LM Studio format to Ollama format using consolidated model handling
                let models = if let Some(data) = lm_response.get("data").and_then(|d| d.as_array()) {
                    data.iter().map(|model| {
                        let model_id = model.get("id").and_then(|id| id.as_str()).unwrap_or("unknown");
                        let cleaned_model = clean_model_name(model_id);
                        let model_name = format!("{}:latest", cleaned_model);
                        let model_info = ModelInfo::from_name(&cleaned_model);
                        model_info.to_ollama_model(&model_name)
                    }).collect::<Vec<_>>()
                } else {
                    vec![]
                };

                Ok(json!({"models": models}))
            }
        }
    };

    let result = with_simple_retry(operation, cancellation_token).await?;
    server.logger.log_success("Ollama tags", start_time.elapsed());
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
        .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_MODEL))?;

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
                    .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_MODEL))?;
                let messages = body.get("messages").and_then(|m| m.as_array())
                    .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_MESSAGES))?;
                let stream = is_streaming_request(&body);

                // Use centralized model resolver with caching
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
                server.logger.log_request("POST", &url, Some(&lmstudio_model));

                let response = request.make_request(reqwest::Method::POST, &url, Some(lm_request)).await?;

                if !response.status().is_success() {
                    let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                    return Err(ProxyError::internal_server_error(&format!("LM Studio error: {}", error_text)));
                }

                if stream {
                    handle_streaming_response(
                        response, true, model, start_time, cancellation_token.clone(),
                        server.logger.clone(), server.config.stream_timeout_seconds
                    ).await
                } else {
                    let lm_response = handle_json_response(response, cancellation_token).await?;
                    let ollama_response = ResponseTransformer::to_ollama_chat(&lm_response, model, messages, start_time);
                    Ok(json_response(&ollama_response))
                }
            }
        }
    };

    let result = with_retry_and_cancellation(&server, model, operation, cancellation_token).await?;
    server.logger.log_success("Ollama chat", start_time.elapsed());
    Ok(result)
}

/// Handle POST /api/generate - text completion with streaming support and cancellation
pub async fn handle_ollama_generate(
    server: Arc<ProxyServer>,
    body: Value,
    cancellation_token: CancellationToken
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    let model = body.get("model").and_then(|m| m.as_str())
        .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_MODEL))?;

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
                    .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_MODEL))?;
                let prompt = body.get("prompt").and_then(|p| p.as_str())
                    .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_PROMPT))?;
                let stream = is_streaming_request(&body);

                let resolver = ModelResolver::new(server.clone());
                let lmstudio_model = resolver.resolve_model_name(model, cancellation_token.clone()).await?;

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
                server.logger.log_request("POST", &url, Some(&lmstudio_model));

                let response = request.make_request(reqwest::Method::POST, &url, Some(lm_request)).await?;

                if !response.status().is_success() {
                    let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                    return Err(ProxyError::internal_server_error(&format!("LM Studio error: {}", error_text)));
                }

                if stream {
                    handle_streaming_response(
                        response, false, model, start_time, cancellation_token.clone(),
                        server.logger.clone(), server.config.stream_timeout_seconds
                    ).await
                } else {
                    let lm_response = handle_json_response(response, cancellation_token).await?;
                    let ollama_response = ResponseTransformer::to_ollama_generate(&lm_response, model, prompt, start_time);
                    Ok(json_response(&ollama_response))
                }
            }
        }
    };

    let result = with_retry_and_cancellation(&server, model, operation, cancellation_token).await?;
    server.logger.log_success("Ollama generate", start_time.elapsed());
    Ok(result)
}

/// Handle POST /api/embed or /api/embeddings - generate embeddings with cancellation
pub async fn handle_ollama_embeddings(
    server: Arc<ProxyServer>,
    body: Value,
    cancellation_token: CancellationToken
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    let model = body.get("model").and_then(|m| m.as_str())
        .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_MODEL))?;

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
                    .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_MODEL))?;
                let input = body.get("input").or_else(|| body.get("prompt"))
                    .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_INPUT))?;

                let resolver = ModelResolver::new(server.clone());
                let lmstudio_model = resolver.resolve_model_name(model, cancellation_token.clone()).await?;

                // Minimal request for embeddings
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
                server.logger.log_request("POST", &url, Some(&lmstudio_model));

                let response = request.make_request(reqwest::Method::POST, &url, Some(lm_request)).await?;

                if !response.status().is_success() {
                    let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                    return Err(ProxyError::internal_server_error(&format!("LM Studio error: {}", error_text)));
                }

                let lm_response = handle_json_response(response, cancellation_token).await?;
                let ollama_response = ResponseTransformer::to_ollama_embeddings(&lm_response, model, start_time);
                Ok(ollama_response)
            }
        }
    };

    let result = with_retry_and_cancellation(&server, model, operation, cancellation_token).await?;
    server.logger.log_success("Ollama embeddings", start_time.elapsed());
    Ok(json_response(&result))
}

/// Handle GET /api/ps - list running models (always returns empty)
pub async fn handle_ollama_ps() -> Result<warp::reply::Response, ProxyError> {
    let response = json!({"models": []});
    Ok(json_response(&response))
}

/// Handle POST /api/show - show model info using consolidated model handling
pub async fn handle_ollama_show(body: Value) -> Result<warp::reply::Response, ProxyError> {
    let model = body.get("model").and_then(|m| m.as_str()).unwrap_or("unknown");
    let cleaned_model = clean_model_name(model);
    let model_info = ModelInfo::from_name(&cleaned_model);
    let response = model_info.to_show_response(model);
    Ok(json_response(&response))
}

/// Handle GET /api/version - return version info
pub async fn handle_ollama_version() -> Result<warp::reply::Response, ProxyError> {
    let response = json!({
        "version": crate::VERSION,
    });
    Ok(json_response(&response))
}

/// Handle unsupported endpoints with helpful error messages
pub async fn handle_unsupported(endpoint: &str) -> Result<warp::reply::Response, ProxyError> {
    let message = match endpoint {
        "/api/create" => "Model creation is not supported. Models must be loaded directly in LM Studio.",
        "/api/pull" => "Model pulling is not supported. Download models through LM Studio interface.",
        "/api/push" => "Model pushing is not supported. This proxy does not manage model repositories.",
        "/api/delete" => "Model deletion is not supported through the proxy. Manage models in LM Studio.",
        "/api/copy" => "Model copying is not supported through the proxy. Manage models in LM Studio.",
        _ => "This endpoint requires direct Ollama functionality that cannot be translated to LM Studio.",
    };

    Err(ProxyError::not_implemented(&format!(
        "The '{}' endpoint is not supported by this proxy. {}",
        endpoint, message
    )))
}
