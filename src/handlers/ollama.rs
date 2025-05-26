/// src/handlers/ollama.rs - Ollama API endpoint handlers with native and legacy support
use serde_json::{json, Value};
use std::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::common::{extract_model_name, handle_json_response, CancellableRequest, RequestContext};
use crate::constants::*;
use crate::handlers::helpers::{
    build_lm_studio_request, execute_request_with_retry, json_response, LMStudioRequestType,
    ResponseTransformer,
};
use crate::handlers::retry::trigger_model_loading_for_ollama;
use crate::handlers::streaming::{handle_streaming_response, is_streaming_request};
use crate::model::ModelInfo;
use crate::model_legacy::ModelInfoLegacy;
use crate::server::{Config, ModelResolverType};
use crate::utils::{log_error, log_info, log_request, log_timed, log_warning, ProxyError};

/// Handle GET /api/tags - list available models
pub async fn handle_ollama_tags(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    let operation = || {
        let context = context.clone();
        let model_resolver = model_resolver.clone();
        let cancellation_token = cancellation_token.clone();
        async move {
            match model_resolver {
                ModelResolverType::Native(resolver) => {
                    let models = resolver.get_all_models(context.client, cancellation_token).await?;
                    let ollama_models: Vec<Value> = models
                        .iter()
                        .map(|model| model.to_ollama_tags_model())
                        .collect();
                    Ok(json!({ "models": ollama_models }))
                }
                ModelResolverType::Legacy(_) => {
                    let request = CancellableRequest::new(context.clone(), cancellation_token.clone());
                    let url = format!("{}/v1/models", context.lmstudio_url);
                    log_request("GET", &url, None);

                    let response = request
                        .make_request(reqwest::Method::GET, &url, None::<Value>)
                        .await?;

                    let lm_response_value = handle_json_response(response, cancellation_token).await?;

                    let models = if let Some(data) = lm_response_value.get("data").and_then(|d| d.as_array()) {
                        data.iter()
                            .map(|model_entry| {
                                let lm_studio_model_id = model_entry
                                    .get("id")
                                    .and_then(|id| id.as_str())
                                    .unwrap_or("unknown");
                                let model_info = ModelInfoLegacy::from_lm_studio_id_legacy(lm_studio_model_id);
                                model_info.to_ollama_tags_model_legacy()
                            })
                            .collect::<Vec<_>>()
                    } else {
                        log_warning(
                            "/v1/models response",
                            "LM Studio response missing 'data' array or not an array, returning empty models list.",
                        );
                        vec![]
                    };

                    Ok(json!({ "models": models }))
                }
            }
        }
    };

    let result = execute_request_with_retry(
        &context,
        "_system_tags_",
        operation,
        false,
        0,
        cancellation_token.clone(),
    )
        .await
        .unwrap_or_else(|e| {
            log_error("Ollama tags fetch", &e.message);
            json!({ "models": [] })
        });

    log_timed(LOG_PREFIX_SUCCESS, "Ollama tags", start_time);
    Ok(json_response(&result))
}

/// Handle GET /api/ps - list running models
pub async fn handle_ollama_ps(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    cancellation_token: CancellationToken,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    log_request("GET", "/api/ps", None);

    let operation = || {
        let context = context.clone();
        let model_resolver = model_resolver.clone();
        let cancellation_token = cancellation_token.clone();
        async move {
            match model_resolver {
                ModelResolverType::Native(resolver) => {
                    let models = resolver.get_loaded_models(context.client, cancellation_token).await?;
                    let ollama_models: Vec<Value> = models
                        .iter()
                        .map(|model| model.to_ollama_ps_model())
                        .collect();
                    Ok(json!({ "models": ollama_models }))
                }
                ModelResolverType::Legacy(_) => {
                    let request = CancellableRequest::new(context.clone(), cancellation_token.clone());
                    let url = format!("{}/v1/models", context.lmstudio_url);

                    let response = request
                        .make_request(reqwest::Method::GET, &url, None::<Value>)
                        .await?;

                    let lm_response_value = handle_json_response(response, cancellation_token).await?;

                    let models = if let Some(data) = lm_response_value.get("data").and_then(|d| d.as_array()) {
                        data.iter()
                            .map(|model_entry| {
                                let lm_studio_model_id = model_entry
                                    .get("id")
                                    .and_then(|id| id.as_str())
                                    .unwrap_or("unknown/error");
                                let model_info = ModelInfoLegacy::from_lm_studio_id_legacy(lm_studio_model_id);
                                model_info.to_ollama_ps_model_legacy()
                            })
                            .collect::<Vec<_>>()
                    } else {
                        log_warning(
                            "/v1/models response for /api/ps",
                            "LM Studio response missing 'data' array or not an array, returning empty models list.",
                        );
                        vec![]
                    };
                    Ok(json!({ "models": models }))
                }
            }
        }
    };

    let result = execute_request_with_retry(
        &context,
        "_system_ps_",
        operation,
        false,
        0,
        cancellation_token.clone(),
    )
        .await
        .unwrap_or_else(|e| {
            log_error("Ollama ps fetch", &e.message);
            json!({ "models": [] })
        });

    log_timed(LOG_PREFIX_SUCCESS, "Ollama ps", start_time);
    Ok(json_response(&result))
}

/// Handle POST /api/show - show model info
pub async fn handle_ollama_show(
    body: Value,
    model_resolver: ModelResolverType,
) -> Result<warp::reply::Response, ProxyError> {
    let ollama_model_name = extract_model_name(&body, "model")?;

    let response = match model_resolver {
        ModelResolverType::Native(_) => {
            // For native API, we could fetch real model data, but for simplicity we'll create from name
            let model_info = ModelInfo::from_native_data(&crate::model::NativeModelData {
                id: ollama_model_name.to_string(),
                object: "model".to_string(),
                model_type: "llm".to_string(),
                publisher: Some("unknown".to_string()),
                arch: "unknown".to_string(),
                compatibility_type: "gguf".to_string(),
                quantization: "Q4_K_M".to_string(),
                state: "unknown".to_string(),
                max_context_length: 4096,
            });
            model_info.to_show_response()
        }
        ModelResolverType::Legacy(_) => {
            let model_info = ModelInfoLegacy::from_lm_studio_id_legacy(ollama_model_name);
            model_info.to_show_response_legacy()
        }
    };

    Ok(json_response(&response))
}

/// Handle POST /api/chat - chat completion with streaming support
pub async fn handle_ollama_chat(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    cancellation_token: CancellationToken,
    config: &Config,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    let ollama_model_name = extract_model_name(&body, "model")?;

    let messages = body
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_MESSAGES))?;

    // Empty messages trigger
    if messages.is_empty() {
        log_info(
            &format!(
                "Empty messages for /api/chat with model '{}', treating as load hint.",
                ollama_model_name
            ),
        );
        trigger_model_loading_for_ollama(&context, ollama_model_name, cancellation_token.clone())
            .await?;
        let fabricated_response = json!({
            "model": ollama_model_name,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "message": {"role": "assistant", "content": ""},
            "done_reason": "load",
            "done": true
        });
        log_timed(LOG_PREFIX_SUCCESS, "Ollama chat (load hint)", start_time);
        return Ok(json_response(&fabricated_response));
    }

    let operation = || {
        let context = context.clone();
        let model_resolver = model_resolver.clone();
        let body_clone = body.clone();
        let cancellation_token_clone = cancellation_token.clone();
        let ollama_model_name_clone = ollama_model_name.to_string();

        async move {
            let current_ollama_model_name = extract_model_name(&body_clone, "model")?;
            let current_messages = body_clone
                .get("messages")
                .and_then(|m| m.as_array())
                .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_MESSAGES))?;
            let stream = is_streaming_request(&body_clone);
            let ollama_options = body_clone.get("options");
            let ollama_tools = body_clone.get("tools");

            let (lm_studio_model_id, endpoint_url) = match &model_resolver {
                ModelResolverType::Native(resolver) => {
                    let model_id = resolver
                        .resolve_model_name(
                            current_ollama_model_name,
                            context.client,
                            cancellation_token_clone.clone(),
                        )
                        .await?;
                    let url = format!("{}{}", context.lmstudio_url, LM_STUDIO_NATIVE_CHAT);
                    (model_id, url)
                }
                ModelResolverType::Legacy(resolver) => {
                    let model_id = resolver
                        .resolve_model_name_legacy(
                            current_ollama_model_name,
                            context.client,
                            cancellation_token_clone.clone(),
                        )
                        .await?;
                    let url = format!("{}{}", context.lmstudio_url, LM_STUDIO_LEGACY_CHAT);
                    (model_id, url)
                }
            };

            let lm_request = build_lm_studio_request(
                &lm_studio_model_id,
                LMStudioRequestType::Chat {
                    messages: &json!(current_messages),
                    stream,
                },
                ollama_options,
                ollama_tools,
            );

            let request_obj = CancellableRequest::new(context.clone(), cancellation_token_clone.clone());
            log_request("POST", &endpoint_url, Some(&lm_studio_model_id));

            let response = request_obj
                .make_request(reqwest::Method::POST, &endpoint_url, Some(lm_request))
                .await?;

            if stream {
                handle_streaming_response(
                    response,
                    true,
                    &ollama_model_name_clone,
                    start_time,
                    cancellation_token_clone.clone(),
                    60,
                )
                    .await
            } else {
                let lm_response_value = handle_json_response(response, cancellation_token_clone).await?;
                let ollama_response = ResponseTransformer::convert_to_ollama_chat(
                    &lm_response_value,
                    &ollama_model_name_clone,
                    current_messages.len(),
                    start_time,
                    matches!(model_resolver, ModelResolverType::Native(_)),
                );
                Ok(json_response(&ollama_response))
            }
        }
    };

    let result = execute_request_with_retry(
        &context,
        ollama_model_name,
        operation,
        true,
        config.load_timeout_seconds,
        cancellation_token.clone(),
    )
        .await?;

    log_timed(LOG_PREFIX_SUCCESS, "Ollama chat", start_time);
    Ok(result)
}

/// Handle POST /api/generate - text completion with streaming support
pub async fn handle_ollama_generate(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    cancellation_token: CancellationToken,
    config: &Config,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    let ollama_model_name = extract_model_name(&body, "model")?;

    let prompt = body
        .get("prompt")
        .and_then(|p| p.as_str())
        .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_PROMPT))?;
    let images = body.get("images");

    // Empty prompt trigger
    if prompt.is_empty()
        && images.map_or(true, |i| i.as_array().map_or(true, |a| a.is_empty()))
    {
        log_info(
            &format!(
                "Empty prompt for /api/generate with model '{}', treating as load hint.",
                ollama_model_name
            ),
        );
        trigger_model_loading_for_ollama(&context, ollama_model_name, cancellation_token.clone())
            .await?;
        let fabricated_response = json!({
            "model": ollama_model_name,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "response": "",
            "done": true
        });
        log_timed(LOG_PREFIX_SUCCESS, "Ollama generate (load hint)", start_time);
        return Ok(json_response(&fabricated_response));
    }

    let operation = || {
        let context = context.clone();
        let model_resolver = model_resolver.clone();
        let body_clone = body.clone();
        let cancellation_token_clone = cancellation_token.clone();
        let ollama_model_name_clone = ollama_model_name.to_string();

        async move {
            let current_ollama_model_name = extract_model_name(&body_clone, "model")?;
            let current_prompt = body_clone
                .get("prompt")
                .and_then(|p| p.as_str())
                .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_PROMPT))?;
            let current_images = body_clone.get("images");
            let stream = is_streaming_request(&body_clone);
            let ollama_options = body_clone.get("options");

            let (lm_studio_model_id, endpoint_url_base) = match &model_resolver {
                ModelResolverType::Native(resolver) => {
                    let model_id = resolver
                        .resolve_model_name(
                            current_ollama_model_name,
                            context.client,
                            cancellation_token_clone.clone(),
                        )
                        .await?;
                    (model_id, context.lmstudio_url.to_string())
                }
                ModelResolverType::Legacy(resolver) => {
                    let model_id = resolver
                        .resolve_model_name_legacy(
                            current_ollama_model_name,
                            context.client,
                            cancellation_token_clone.clone(),
                        )
                        .await?;
                    (model_id, context.lmstudio_url.to_string())
                }
            };

            // Determine endpoint based on API type and whether images are present
            let (lm_studio_target_url, lm_request_type) = if current_images.is_some()
                && current_images.unwrap().as_array().map_or(false, |a| !a.is_empty())
            {
                let chat_endpoint = match &model_resolver {
                    ModelResolverType::Native(_) => LM_STUDIO_NATIVE_CHAT,
                    ModelResolverType::Legacy(_) => LM_STUDIO_LEGACY_CHAT,
                };
                (
                    format!("{}{}", endpoint_url_base, chat_endpoint),
                    LMStudioRequestType::Completion {
                        prompt: current_prompt,
                        stream,
                        images: current_images,
                    },
                )
            } else {
                let completions_endpoint = match &model_resolver {
                    ModelResolverType::Native(_) => LM_STUDIO_NATIVE_COMPLETIONS,
                    ModelResolverType::Legacy(_) => LM_STUDIO_LEGACY_COMPLETIONS,
                };
                (
                    format!("{}{}", endpoint_url_base, completions_endpoint),
                    LMStudioRequestType::Completion {
                        prompt: current_prompt,
                        stream,
                        images: None,
                    },
                )
            };

            let lm_request = build_lm_studio_request(
                &lm_studio_model_id,
                lm_request_type,
                ollama_options,
                None,
            );

            let request_obj = CancellableRequest::new(context.clone(), cancellation_token_clone.clone());
            log_request("POST", &lm_studio_target_url, Some(&lm_studio_model_id));

            let response = request_obj
                .make_request(reqwest::Method::POST, &lm_studio_target_url, Some(lm_request))
                .await?;

            if stream {
                handle_streaming_response(
                    response,
                    false,
                    &ollama_model_name_clone,
                    start_time,
                    cancellation_token_clone.clone(),
                    60,
                )
                    .await
            } else {
                let lm_response_value = handle_json_response(response, cancellation_token_clone).await?;
                let ollama_response = ResponseTransformer::convert_to_ollama_generate(
                    &lm_response_value,
                    &ollama_model_name_clone,
                    current_prompt,
                    start_time,
                    matches!(model_resolver, ModelResolverType::Native(_)),
                );
                Ok(json_response(&ollama_response))
            }
        }
    };

    let result = execute_request_with_retry(
        &context,
        ollama_model_name,
        operation,
        true,
        config.load_timeout_seconds,
        cancellation_token.clone(),
    )
        .await?;

    log_timed(LOG_PREFIX_SUCCESS, "Ollama generate", start_time);
    Ok(result)
}

/// Handle POST /api/embed or /api/embeddings - generate embeddings
pub async fn handle_ollama_embeddings(
    context: RequestContext<'_>,
    model_resolver: ModelResolverType,
    body: Value,
    cancellation_token: CancellationToken,
    config: &Config,
) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    let ollama_model_name = extract_model_name(&body, "model")?;

    let operation = || {
        let context = context.clone();
        let model_resolver = model_resolver.clone();
        let body_clone = body.clone();
        let cancellation_token_clone = cancellation_token.clone();
        let ollama_model_name_clone = ollama_model_name.to_string();

        async move {
            let current_ollama_model_name = extract_model_name(&body_clone, "model")?;
            let input_value = body_clone
                .get("input")
                .or_else(|| body_clone.get("prompt"))
                .cloned()
                .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_INPUT))?;

            let (lm_studio_model_id, endpoint_url) = match &model_resolver {
                ModelResolverType::Native(resolver) => {
                    let model_id = resolver
                        .resolve_model_name(
                            current_ollama_model_name,
                            context.client,
                            cancellation_token_clone.clone(),
                        )
                        .await?;
                    let url = format!("{}{}", context.lmstudio_url, LM_STUDIO_NATIVE_EMBEDDINGS);
                    (model_id, url)
                }
                ModelResolverType::Legacy(resolver) => {
                    let model_id = resolver
                        .resolve_model_name_legacy(
                            current_ollama_model_name,
                            context.client,
                            cancellation_token_clone.clone(),
                        )
                        .await?;
                    let url = format!("{}{}", context.lmstudio_url, LM_STUDIO_LEGACY_EMBEDDINGS);
                    (model_id, url)
                }
            };

            let lm_request = build_lm_studio_request(
                &lm_studio_model_id,
                LMStudioRequestType::Embeddings {
                    input: &input_value,
                },
                None,
                None,
            );

            let request_obj = CancellableRequest::new(context.clone(), cancellation_token_clone.clone());
            log_request("POST", &endpoint_url, Some(&lm_studio_model_id));

            let response = request_obj
                .make_request(reqwest::Method::POST, &endpoint_url, Some(lm_request))
                .await?;
            let lm_response_value = handle_json_response(response, cancellation_token_clone).await?;

            let ollama_response = ResponseTransformer::convert_to_ollama_embeddings(
                &lm_response_value,
                &ollama_model_name_clone,
                start_time,
                matches!(model_resolver, ModelResolverType::Native(_)),
            );
            Ok(json_response(&ollama_response))
        }
    };

    let result = execute_request_with_retry(
        &context,
        ollama_model_name,
        operation,
        true,
        config.load_timeout_seconds,
        cancellation_token.clone(),
    )
        .await?;

    log_timed(LOG_PREFIX_SUCCESS, "Ollama embeddings", start_time);
    Ok(result)
}

/// Handle GET /api/version - return version info
pub async fn handle_ollama_version() -> Result<warp::reply::Response, ProxyError> {
    let response = json!({
        "version": crate::VERSION,
        "proxy_backend": "lmstudio"
    });
    Ok(json_response(&response))
}

/// Handle unsupported endpoints with helpful messages
pub async fn handle_unsupported(endpoint: &str) -> Result<warp::reply::Response, ProxyError> {
    let (message, suggestion) = match endpoint {
        "/api/create" => (
            "Model creation not supported via proxy",
            "Load models directly in LM Studio",
        ),
        "/api/pull" => (
            "Model pulling not supported via proxy",
            "Download models through LM Studio interface",
        ),
        "/api/push" => (
            "Model pushing not supported via proxy",
            "Use LM Studio for model management",
        ),
        "/api/delete" => (
            "Model deletion not supported via proxy",
            "Remove models through LM Studio",
        ),
        "/api/copy" => (
            "Model copying not supported via proxy",
            "Use LM Studio for model operations",
        ),
        _ => (
            "Endpoint requires direct Ollama functionality not available via this LM Studio proxy",
            "This proxy focuses on inference and basic model listing operations",
        ),
    };

    Err(ProxyError::not_implemented(&format!(
        "{}. Suggestion: {}.",
        message, suggestion
    )))
}

/// Handle health check that tests actual model availability
pub async fn handle_health_check(
    context: RequestContext<'_>,
    cancellation_token: CancellationToken,
) -> Result<Value, ProxyError> {
    let start_time = Instant::now();
    let url = format!("{}/v1/models", context.lmstudio_url);
    let request = CancellableRequest::new(context.clone(), cancellation_token.clone());

    match request
        .make_request(reqwest::Method::GET, &url, None::<Value>)
        .await
    {
        Ok(response) => {
            let status = response.status();
            let is_healthy = status.is_success();
            let mut model_count = 0;

            if is_healthy {
                match response.json::<Value>().await {
                    Ok(models_response) => {
                        model_count = models_response
                            .get("data")
                            .and_then(|d| d.as_array())
                            .map(|arr| arr.len())
                            .unwrap_or(0);
                    }
                    Err(_) => {}
                }
            }

            let duration_ms = start_time.elapsed().as_millis();
            Ok(json!({
                "status": if is_healthy { "healthy" } else { "unhealthy" },
                "lmstudio_url": context.lmstudio_url,
                "http_status": status.as_u16(),
                "models_known_to_lmstudio": model_count,
                "response_time_ms": duration_ms,
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "proxy_version": crate::VERSION
            }))
        }
        Err(e) if e.is_cancelled() => Err(ProxyError::request_cancelled()),
        Err(e) => {
            let duration_ms = start_time.elapsed().as_millis();
            Ok(json!({
                "status": "unreachable",
                "lmstudio_url": context.lmstudio_url,
                "error_message": e.message,
                "error_details": ERROR_LM_STUDIO_UNAVAILABLE,
                "response_time_ms": duration_ms,
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "proxy_version": crate::VERSION
            }))
        }
    }
}
