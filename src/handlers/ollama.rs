// src/handlers/ollama.rs - Enhanced Ollama handlers with metrics and consolidated patterns

use serde_json::{json, Value};
use std::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::common::{extract_model_name, handle_json_response, CancellableRequest, RequestContext};
use crate::constants::*;
use crate::handle_lm_error;
use crate::handlers::helpers::{
    json_response, ResponseTransformer, LMStudioRequestType, build_lm_studio_request,
    execute_request_with_retry
};
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

    let operation = || {
        let context = context.clone();
        let cancellation_token = cancellation_token.clone();
        async move {
            let request = CancellableRequest::new(context.clone(), cancellation_token.clone());
            let url = format!("{}/v1/models", context.lmstudio_url);
            context.logger.log_request("GET", &url, None);

            let response = request.make_request(reqwest::Method::GET, &url, None).await?;
            handle_lm_error!(response);

            let lm_response = handle_json_response(response, cancellation_token).await?;

            // Transform to Ollama format using programmatic model info
            let models = if let Some(data) = lm_response.get("data").and_then(|d| d.as_array()) {
                data.iter().map(|model| {
                    let model_id = model.get("id").and_then(|id| id.as_str()).unwrap_or("unknown");
                    let cleaned_model = clean_model_name(model_id);
                    let model_info = ModelInfo::from_name(cleaned_model);
                    model_info.to_ollama_model()
                }).collect::<Vec<_>>()
            } else {
                vec![]
            };

            Ok(json!({"models": models}))
        }
    };

    let result = execute_request_with_retry(
        &context,
        "tags", // Not model-specific, but needed for function signature
        operation,
        false, // No model-specific retry needed
        0,
        cancellation_token.clone(),
    ).await?;

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

    let operation = || {
        let context = context.clone();
        let body = body.clone();
        let cancellation_token = cancellation_token.clone();
        let config = config.clone();
        async move {
            let model = extract_model_name(&body, "model")?;
            let messages = body.get("messages").and_then(|m| m.as_array())
                .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_MESSAGES))?;
            let stream = is_streaming_request(&body);

            // Resolve model name
            let resolver = ModelResolver::new(context.clone());
            let lmstudio_model = resolver.resolve_model_name(model, cancellation_token.clone()).await?;

            // Build request using consolidated helper
            let lm_request = build_lm_studio_request(
                &lmstudio_model,
                LMStudioRequestType::Chat { messages: &json!(messages), stream },
                body.get("options")
            );

            let request = CancellableRequest::new(context.clone(), cancellation_token.clone());
            let url = format!("{}/v1/chat/completions", context.lmstudio_url);
            context.logger.log_request("POST", &url, Some(&lmstudio_model));

            let response = request.make_request(reqwest::Method::POST, &url, Some(lm_request)).await?;

            if stream {
                // Check error status before streaming
                if !response.status().is_success() {
                    let status = response.status();
                    return Err(ProxyError::new(
                        format!("LM Studio streaming error: {}", status),
                        status.as_u16()
                    ));
                }
                handle_streaming_response(
                    response, true, model, start_time, cancellation_token.clone(),
                    context.logger, config.stream_timeout_seconds
                ).await
            } else {
                let lm_response = handle_json_response(response, cancellation_token).await?;
                let ollama_response = ResponseTransformer::convert_to_ollama_chat(&lm_response, model, messages, start_time);
                Ok(json_response(&ollama_response))
            }
        }
    };

    let result = execute_request_with_retry(
        &context,
        model,
        operation,
        true, // Use model-specific retry
        config.load_timeout_seconds,
        cancellation_token.clone(),
    ).await?;

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

    let operation = || {
        let context = context.clone();
        let body = body.clone();
        let cancellation_token = cancellation_token.clone();
        let config = config.clone();
        async move {
            let model = extract_model_name(&body, "model")?;
            let prompt = body.get("prompt").and_then(|p| p.as_str())
                .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_PROMPT))?;
            let stream = is_streaming_request(&body);

            let resolver = ModelResolver::new(context.clone());
            let lmstudio_model = resolver.resolve_model_name(model, cancellation_token.clone()).await?;

            let lm_request = build_lm_studio_request(
                &lmstudio_model,
                LMStudioRequestType::Completion { prompt, stream },
                body.get("options")
            );

            let request = CancellableRequest::new(context.clone(), cancellation_token.clone());
            let url = format!("{}/v1/completions", context.lmstudio_url);
            context.logger.log_request("POST", &url, Some(&lmstudio_model));

            let response = request.make_request(reqwest::Method::POST, &url, Some(lm_request)).await?;

            if stream {
                if !response.status().is_success() {
                    let status = response.status();
                    return Err(ProxyError::new(
                        format!("LM Studio streaming error: {}", status),
                        status.as_u16()
                    ));
                }
                handle_streaming_response(
                    response, false, model, start_time, cancellation_token.clone(),
                    context.logger, config.stream_timeout_seconds
                ).await
            } else {
                let lm_response = handle_json_response(response, cancellation_token).await?;
                let ollama_response = ResponseTransformer::convert_to_ollama_generate(&lm_response, model, prompt, start_time);
                Ok(json_response(&ollama_response))
            }
        }
    };

    let result = execute_request_with_retry(
        &context,
        model,
        operation,
        true,
        config.load_timeout_seconds,
        cancellation_token.clone(),
    ).await?;

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

    let operation = || {
        let context = context.clone();
        let body = body.clone();
        let cancellation_token = cancellation_token.clone();
        async move {
            let model = extract_model_name(&body, "model")?;
            let input = body.get("input").or_else(|| body.get("prompt"))
                .ok_or_else(|| ProxyError::bad_request(ERROR_MISSING_INPUT))?;

            let resolver = ModelResolver::new(context.clone());
            let lmstudio_model = resolver.resolve_model_name(model, cancellation_token.clone()).await?;

            let lm_request = build_lm_studio_request(
                &lmstudio_model,
                LMStudioRequestType::Embeddings { input },
                None
            );

            let request = CancellableRequest::new(context.clone(), cancellation_token.clone());
            let url = format!("{}/v1/embeddings", context.lmstudio_url);
            context.logger.log_request("POST", &url, Some(&lmstudio_model));

            let response = request.make_request(reqwest::Method::POST, &url, Some(lm_request)).await?;
            let lm_response = handle_json_response(response, cancellation_token).await?;
            let ollama_response = ResponseTransformer::convert_to_ollama_embeddings(&lm_response, model, start_time);
            Ok(ollama_response)
        }
    };

    let result = execute_request_with_retry(
        &context,
        model,
        operation,
        true,
        3, // Short timeout for embeddings
        cancellation_token.clone(),
    ).await?;

    context.logger.log_timed(LOG_PREFIX_SUCCESS, "Ollama embeddings", start_time);
    Ok(json_response(&result))
}

/// Handle GET /api/ps - list running models (simplified for proxy)
pub async fn handle_ollama_ps() -> Result<warp::reply::Response, ProxyError> {
    // For a proxy, we don't track running models locally
    let response = json!({"models": []});
    Ok(json_response(&response))
}

/// Handle POST /api/show - show model info programmatically
pub async fn handle_ollama_show(body: Value) -> Result<warp::reply::Response, ProxyError> {
    let model = extract_model_name(&body, "model")?;
    let cleaned_model = clean_model_name(model);
    let model_info = ModelInfo::from_name(cleaned_model);
    let response = model_info.to_show_response();
    Ok(json_response(&response))
}

/// Handle GET /api/version - return version info
pub async fn handle_ollama_version() -> Result<warp::reply::Response, ProxyError> {
    let response = json!({
        "version": crate::VERSION,
        "proxy": true,
        "backend": "lmstudio"
    });
    Ok(json_response(&response))
}

/// Handle unsupported endpoints with helpful messages
pub async fn handle_unsupported(endpoint: &str) -> Result<warp::reply::Response, ProxyError> {
    let (message, suggestion) = match endpoint {
        "/api/create" => (
            "Model creation not supported via proxy",
            "Load models directly in LM Studio"
        ),
        "/api/pull" => (
            "Model pulling not supported via proxy",
            "Download models through LM Studio interface"
        ),
        "/api/push" => (
            "Model pushing not supported via proxy",
            "Use LM Studio for model management"
        ),
        "/api/delete" => (
            "Model deletion not supported via proxy",
            "Remove models through LM Studio"
        ),
        "/api/copy" => (
            "Model copying not supported via proxy",
            "Use LM Studio for model operations"
        ),
        _ => (
            "Endpoint requires direct Ollama functionality",
            "This proxy focuses on inference operations"
        ),
    };

    Err(ProxyError::not_implemented(&format!(
        "{}. Suggestion: {}.",
        message, suggestion
    )))
}

/// Enhanced health check that tests actual model availability
pub async fn handle_health_check(
    context: RequestContext<'_>,
    cancellation_token: CancellationToken,
) -> Result<Value, ProxyError> {
    let start_time = Instant::now();

    // Basic connectivity test
    let url = format!("{}/v1/models", context.lmstudio_url);
    let request = CancellableRequest::new(context.clone(), cancellation_token.clone());

    match request.make_request(reqwest::Method::GET, &url, None).await {
        Ok(response) => {
            let status = response.status();
            let is_healthy = status.is_success();

            // If basic connectivity works, try to get model list
            let model_count = if is_healthy {
                match response.json::<Value>().await {
                    Ok(models_response) => {
                        models_response
                            .get("data")
                            .and_then(|d| d.as_array())
                            .map(|arr| arr.len())
                            .unwrap_or(0)
                    },
                    Err(_) => 0,
                }
            } else {
                0
            };

            let duration_ms = start_time.elapsed().as_millis();

            Ok(json!({
                "status": if is_healthy { "healthy" } else { "unhealthy" },
                "lmstudio_url": context.lmstudio_url,
                "http_status": status.as_u16(),
                "models_available": model_count,
                "response_time_ms": duration_ms,
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "proxy_version": crate::VERSION
            }))
        },
        Err(e) if e.is_cancelled() => Err(ProxyError::request_cancelled()),
        Err(_) => {
            let duration_ms = start_time.elapsed().as_millis();

            Ok(json!({
                "status": "unreachable",
                "lmstudio_url": context.lmstudio_url,
                "error": ERROR_LM_STUDIO_UNAVAILABLE,
                "response_time_ms": duration_ms,
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "proxy_version": crate::VERSION
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_extract_model_name_validation() {
        let valid_body = json!({"model": "llama3.2:7b"});
        assert!(extract_model_name(&valid_body, "model").is_ok());

        let invalid_body = json!({"prompt": "hello"});
        assert!(extract_model_name(&invalid_body, "model").is_err());

        let empty_model = json!({"model": ""});
        assert!(extract_model_name(&empty_model, "model").is_err());
    }

    #[tokio::test]
    async fn test_handle_ollama_show() {
        let body = json!({"model": "llama3.2:7b"});
        let result = handle_ollama_show(body).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_ollama_version() {
        let result = handle_ollama_version().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_ollama_ps() {
        let result = handle_ollama_ps().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_unsupported() {
        let result = handle_unsupported("/api/create").await;
        assert!(result.is_err());

        if let Err(e) = result {
            assert_eq!(e.status_code, 501);
            assert!(e.message.contains("not supported"));
        }
    }
}
