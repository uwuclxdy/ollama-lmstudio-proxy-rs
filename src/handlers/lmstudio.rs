// src/handlers/lmstudio.rs - LM Studio passthrough with model name resolution

use serde_json::Value;
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::server::ProxyServer;
use crate::utils::{format_duration, ProxyError, clean_model_name};
use crate::common::{CancellableRequest, handle_json_response};
use crate::handlers::retry::with_simple_retry;
use super::retry::with_retry_and_cancellation;
use super::streaming::{is_streaming_request, handle_passthrough_streaming_response};
use super::helpers::json_response;

/// Resolve model name for LM Studio passthrough (same logic as in ollama.rs)
async fn resolve_model_name_for_passthrough(
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
            server.logger.log(&format!("⚠️  Cannot fetch LM Studio models for passthrough, using fallback: '{}'", cleaned_ollama));
            return Ok(cleaned_ollama);
        }
    };

    if !response.status().is_success() {
        server.logger.log(&format!("⚠️  LM Studio models endpoint returned {} for passthrough, using fallback: '{}'", response.status(), cleaned_ollama));
        return Ok(cleaned_ollama);
    }

    let models_response: Value = match response.json().await {
        Ok(json) => json,
        Err(_) => {
            server.logger.log(&format!("⚠️  Cannot parse LM Studio models response for passthrough, using fallback: '{}'", cleaned_ollama));
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

    // Find the best match (same logic as in ollama.rs)
    if let Some(matched_model) = find_best_model_match(&cleaned_ollama, &available_models) {
        server.logger.log(&format!("✅ Resolved passthrough '{}' -> '{}'", ollama_model, matched_model));
        Ok(matched_model)
    } else {
        server.logger.log(&format!("⚠️  No passthrough match found for '{}', using cleaned name '{}'", ollama_model, cleaned_ollama));
        Ok(cleaned_ollama)
    }
}

/// Find the best matching LM Studio model for an Ollama model name (duplicate from ollama.rs)
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

/// Check if two model names refer to the same model size (duplicate from ollama.rs)
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

/// Handle direct LM Studio API passthrough with streaming, cancellation, and model name resolution
pub async fn handle_lmstudio_passthrough(
    server: Arc<ProxyServer>,
    method: &str,
    endpoint: &str,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    // Try to extract model name from request body for proper resolution
    let original_model_name = body.get("model")
        .and_then(|m| m.as_str())
        .map(|m| m.to_string());

    let operation = {
        let server = server.clone();
        let method = method.to_string();
        let endpoint = endpoint.to_string();
        let body = body.clone();
        let cancellation_token = cancellation_token.clone();
        let original_model_name = original_model_name.clone();

        move || {
            let server = server.clone();
            let method = method.clone();
            let endpoint = endpoint.clone();
            let mut body = body.clone();
            let cancellation_token = cancellation_token.clone();
            let original_model_name = original_model_name.clone();

            async move {
                // 🔑 KEY FIX: Resolve model name if present in passthrough request
                if let Some(ref model_name) = original_model_name {
                    let resolved_model = resolve_model_name_for_passthrough(&server, model_name, cancellation_token.clone()).await?;

                    // Update the request body with the resolved model name
                    if let Some(body_obj) = body.as_object_mut() {
                        body_obj.insert("model".to_string(), Value::String(resolved_model.clone()));
                        server.logger.log(&format!("🔄 Updated passthrough request model: '{}' -> '{}'", model_name, resolved_model));
                    }
                }

                let url = format!("{}{}", server.config.lmstudio_url, endpoint);
                let is_streaming = is_streaming_request(&body);

                server.logger.log(&format!("Passthrough: {} {} (stream: {})", method, url, is_streaming));

                let request_method = match method.as_str() {
                    "GET" => reqwest::Method::GET,
                    "POST" => reqwest::Method::POST,
                    "PUT" => reqwest::Method::PUT,
                    "DELETE" => reqwest::Method::DELETE,
                    _ => return Err(ProxyError::bad_request(&format!("Unsupported method: {}", method))),
                };

                let request = CancellableRequest::new(
                    server.client.clone(),
                    cancellation_token.clone(),
                    server.logger.clone(),
                    server.config.request_timeout_seconds
                );

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
                    handle_passthrough_streaming_response(
                        response,
                        cancellation_token.clone(),
                        server.logger.clone(),
                        server.config.stream_timeout_seconds
                    ).await
                } else {
                    let json_data = handle_json_response(response, cancellation_token).await?;
                    Ok(json_response(&json_data))
                }
            }
        }
    };

    // Use model-specific retry if we have a model name, otherwise use simple retry
    let result = if let Some(model) = original_model_name {
        server.logger.log(&format!("Using model-specific retry for passthrough with model: {}", model));
        with_retry_and_cancellation(&server, &model, operation, cancellation_token).await?
    } else {
        server.logger.log("Using simple retry for passthrough (no model specified)");
        with_simple_retry(operation, cancellation_token).await?
    };

    let duration = start_time.elapsed();
    server.logger.log(&format!("LM Studio passthrough completed (took {})", format_duration(duration)));
    Ok(result)
}
