// src/handlers/ollama.rs - Optimized Ollama handlers with lightweight context

use serde_json::{json, Map, Value};
use std::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::common::{extract_model_name, handle_json_response, CancellableRequest, RequestContext};
use crate::constants::*;
use crate::handle_lm_error;
use crate::handlers::helpers::{build_lm_studio_request, json_response, ResponseTransformer};
use crate::handlers::retry::{with_retry_and_cancellation, with_simple_retry};
use crate::handlers::streaming::{handle_streaming_response, is_streaming_request};
use crate::model::{clean_model_name, ModelInfo, ModelResolver};
use crate::server::Config;
use crate::utils::ProxyError;

/// Handle GET /api/tags - list available models
pub async fn handle_ollama_tags(
    context: RequestContext<'_>,
    cancellation_token: CancellationToken
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    let operation = {
        let context = context.clone();
        let cancellation_token = cancellation_token.clone();
        move || {
            let context = context.clone();
            let cancellation_token = cancellation_token.clone();
            async move {
                let request = CancellableRequest::new(context.clone(), cancellation_token.clone());
                let url = format!("{}/v1/models", context.lmstudio_url);
                context.logger.log_request("GET", &url, None);

                let response = request.make_request(reqwest::Method::GET, &url, None).await?;
                handle_lm_error!(response);

                let lm_response = handle_json_response(response, cancellation_token).await?;

                // Transform to Ollama format
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
    context.logger.log_timed(LOG_PREFIX_SUCCESS, "Ollama tags", start_time);
    Ok(json_response(&result))
}

/// Handle POST /api/chat - chat completion with streaming support
pub async fn handle_ollama_chat(
    context: RequestContext<'_>,
    body: Value,
    cancellation_token: CancellationToken,
    config: &Config,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    let model = extract_model_name(&body, "model")?;

    let operation = {
        let context = context.clone();
        let body = body.clone();
        let cancellation_token = cancellation_token.clone();
        move || {
            let context = context.clone();
            let body = body.clone();
            let cancellation_token = cancellation_token.clone();
            async move {
                let model = extract_model_name(&body, "model")?;
                let messages = body.get("messages").and_then(|m| m.as_array())
                    .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_MESSAGES))?;
                let stream = is_streaming_request(&body);

                // Resolve model name
                let resolver = ModelResolver::new(context.clone());
                let lmstudio_model = resolver.resolve_model_name(model, cancellation_token.clone()).await?;

                // Build request
                let mut base_params = Map::new();
                base_params.insert("model".to_string(), json!(lmstudio_model));
                base_params.insert("messages".to_string(), json!(messages));
                base_params.insert("stream".to_string(), json!(stream));

                let lm_request = build_lm_studio_request(base_params, body.get("options"));

                let request = CancellableRequest::new(context.clone(), cancellation_token.clone());
                let url = format!("{}/v1/chat/completions", context.lmstudio_url);
                context.logger.log_request("POST", &url, Some(&lmstudio_model));

                let response = request.make_request(reqwest::Method::POST, &url, Some(lm_request)).await?;

                if stream {
                    // Check error status before streaming
                    if !response.status().is_success() {
                        let status = response.status();
                        return Err(ProxyError::new(
                            format!("LM Studio error: {}", status),
                            status.as_u16()
                        ));
                    }
                    handle_streaming_response(
                        response, true, model, start_time, cancellation_token.clone(),
                        context.logger, config.stream_timeout_seconds
                    ).await
                } else {
                    let lm_response = handle_json_response(response, cancellation_token).await?;
                    let ollama_response = ResponseTransformer::to_ollama_chat(&lm_response, model, messages, start_time);
                    Ok(json_response(&ollama_response))
                }
            }
        }
    };

    let result = with_retry_and_cancellation(&context, model, config.load_timeout_seconds, operation, cancellation_token).await?;
    context.logger.log_timed(LOG_PREFIX_SUCCESS, "Ollama chat", start_time);
    Ok(result)
}

/// Handle POST /api/generate - text completion with streaming support
pub async fn handle_ollama_generate(
    context: RequestContext<'_>,
    body: Value,
    cancellation_token: CancellationToken,
    config: &Config,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    let model = extract_model_name(&body, "model")?;

    let operation = {
        let context = context.clone();
        let body = body.clone();
        let cancellation_token = cancellation_token.clone();
        move || {
            let context = context.clone();
            let body = body.clone();
            let cancellation_token = cancellation_token.clone();
            async move {
                let model = extract_model_name(&body, "model")?;
                let prompt = body.get("prompt").and_then(|p| p.as_str())
                    .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_PROMPT))?;
                let stream = is_streaming_request(&body);

                let resolver = ModelResolver::new(context.clone());
                let lmstudio_model = resolver.resolve_model_name(model, cancellation_token.clone()).await?;

                let mut base_params = Map::new();
                base_params.insert("model".to_string(), json!(lmstudio_model));
                base_params.insert("prompt".to_string(), json!(prompt));
                base_params.insert("stream".to_string(), json!(stream));

                let lm_request = build_lm_studio_request(base_params, body.get("options"));

                let request = CancellableRequest::new(context.clone(), cancellation_token.clone());
                let url = format!("{}/v1/completions", context.lmstudio_url);
                context.logger.log_request("POST", &url, Some(&lmstudio_model));

                let response = request.make_request(reqwest::Method::POST, &url, Some(lm_request)).await?;

                if stream {
                    // Check error status before streaming
                    if !response.status().is_success() {
                        let status = response.status();
                        return Err(ProxyError::new(
                            format!("LM Studio error: {}", status),
                            status.as_u16()
                        ));
                    }
                    handle_streaming_response(
                        response, false, model, start_time, cancellation_token.clone(),
                        context.logger, config.stream_timeout_seconds
                    ).await
                } else {
                    let lm_response = handle_json_response(response, cancellation_token).await?;
                    let ollama_response = ResponseTransformer::to_ollama_generate(&lm_response, model, prompt, start_time);
                    Ok(json_response(&ollama_response))
                }
            }
        }
    };

    let result = with_retry_and_cancellation(&context, model, config.load_timeout_seconds, operation, cancellation_token).await?;
    context.logger.log_timed(LOG_PREFIX_SUCCESS, "Ollama generate", start_time);
    Ok(result)
}

/// Handle POST /api/embed or /api/embeddings - generate embeddings
pub async fn handle_ollama_embeddings(
    context: RequestContext<'_>,
    body: Value,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    let model = extract_model_name(&body, "model")?;

    let operation = {
        let context = context.clone();
        let body = body.clone();
        let cancellation_token = cancellation_token.clone();
        move || {
            let context = context.clone();
            let body = body.clone();
            let cancellation_token = cancellation_token.clone();
            async move {
                let model = extract_model_name(&body, "model")?;
                let input = body.get("input").or_else(|| body.get("prompt"))
                    .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_INPUT))?;

                let resolver = ModelResolver::new(context.clone());
                let lmstudio_model = resolver.resolve_model_name(model, cancellation_token.clone()).await?;

                let lm_request = json!({
                    "model": lmstudio_model,
                    "input": input
                });

                let request = CancellableRequest::new(context.clone(), cancellation_token.clone());
                let url = format!("{}/v1/embeddings", context.lmstudio_url);
                context.logger.log_request("POST", &url, Some(&lmstudio_model));

                let response = request.make_request(reqwest::Method::POST, &url, Some(lm_request)).await?;
                let lm_response = handle_json_response(response, cancellation_token).await?;
                let ollama_response = ResponseTransformer::to_ollama_embeddings(&lm_response, model, start_time);
                Ok(ollama_response)
            }
        }
    };

    let result = with_retry_and_cancellation(&context, model, 3, operation, cancellation_token).await?;
    context.logger.log_timed(LOG_PREFIX_SUCCESS, "Ollama embeddings", start_time);
    Ok(json_response(&result))
}

/// Handle GET /api/ps - list running models (always empty for simplicity)
pub async fn handle_ollama_ps() -> Result<warp::reply::Response, ProxyError> {
    let response = json!({"models": []});
    Ok(json_response(&response))
}

/// Handle POST /api/show - show model info
pub async fn handle_ollama_show(body: Value) -> Result<warp::reply::Response, ProxyError> {
    let model = extract_model_name(&body, "model")?;
    let cleaned_model = clean_model_name(model);
    let model_info = ModelInfo::from_name(cleaned_model);
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

/// Handle unsupported endpoints
pub async fn handle_unsupported(endpoint: &str) -> Result<warp::reply::Response, ProxyError> {
    let message = match endpoint {
        "/api/create" => "Model creation not supported. Load models in LM Studio.",
        "/api/pull" => "Model pulling not supported. Download via LM Studio.",
        "/api/push" => "Model pushing not supported.",
        "/api/delete" => "Model deletion not supported via proxy.",
        "/api/copy" => "Model copying not supported via proxy.",
        _ => "Endpoint requires direct Ollama functionality.",
    };

    Err(ProxyError::not_implemented(&format!(
        "The '{}' endpoint is not supported by this proxy. {}",
        endpoint, message
    )))
}
