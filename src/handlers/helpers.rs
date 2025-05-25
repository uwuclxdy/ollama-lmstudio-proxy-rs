// src/handlers/helpers.rs - Consolidated helper functions with reduced duplication

use serde_json::{json, Value};
use std::time::{Duration, Instant};
use warp::Reply;

use crate::common::{RequestBuilder, map_ollama_to_lmstudio_params};
use crate::constants::*;
use crate::metrics::get_global_metrics;

/// Create JSON response with proper headers
pub fn json_response(value: &Value) -> warp::reply::Response {
    warp::reply::with_status(
        warp::reply::json(value),
        warp::http::StatusCode::OK,
    ).into_response()
}

/// Consolidated timing information calculator
#[derive(Debug, Clone)]
pub struct TimingInfo {
    pub total_duration: u64,
    pub load_duration: u64,
    pub prompt_eval_count: u64,
    pub prompt_eval_duration: u64,
    pub eval_count: u64,
    pub eval_duration: u64,
}

impl TimingInfo {
    /// Calculate timing from request parameters
    pub fn calculate(start_time: Instant, input_tokens: u64, output_tokens: u64) -> Self {
        let total_duration = start_time.elapsed().as_nanos() as u64;

        Self {
            total_duration,
            load_duration: DEFAULT_LOAD_DURATION_NS,
            prompt_eval_count: input_tokens.max(1),
            prompt_eval_duration: total_duration / TIMING_PROMPT_RATIO,
            eval_count: output_tokens.max(1),
            eval_duration: total_duration / TIMING_EVAL_RATIO,
        }
    }

    /// Calculate timing from text content
    pub fn from_text_content(start_time: Instant, input_text: &str, output_text: &str) -> Self {
        let input_tokens = estimate_token_count(input_text);
        let output_tokens = estimate_token_count(output_text);
        Self::calculate(start_time, input_tokens, output_tokens)
    }

    /// Calculate timing from message count
    pub fn from_message_count(start_time: Instant, message_count: usize, output_text: &str) -> Self {
        let input_tokens = (message_count * 10).max(1) as u64; // Rough estimate
        let output_tokens = estimate_token_count(output_text);
        Self::calculate(start_time, input_tokens, output_tokens)
    }
}

/// Enhanced response transformer with consolidated patterns
pub struct ResponseTransformer;

impl ResponseTransformer {
    /// Transform LM Studio chat response to Ollama format
    pub fn convert_to_ollama_chat(
        lm_response: &Value,
        model: &str,
        messages: &[Value],
        start_time: Instant,
    ) -> Value {
        let content = Self::extract_content_with_reasoning(lm_response, true);
        let timing = TimingInfo::from_message_count(start_time, messages.len(), &content);

        // Record metrics if enabled
        if let Some(metrics) = get_global_metrics() {
            tokio::spawn({
                let model = model.to_string();
                let content_len = content.len() as u64;
                async move {
                    metrics.record_model_usage(&model, content_len / 4).await; // Rough token estimate
                }
            });
        }

        json!({
            "model": model,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "message": {
                "role": "assistant",
                "content": content
            },
            "done": true,
            "total_duration": timing.total_duration,
            "load_duration": timing.load_duration,
            "prompt_eval_count": timing.prompt_eval_count,
            "prompt_eval_duration": timing.prompt_eval_duration,
            "eval_count": timing.eval_count,
            "eval_duration": timing.eval_duration
        })
    }

    /// Transform LM Studio completion response to Ollama format
    pub fn convert_to_ollama_generate(
        lm_response: &Value,
        model: &str,
        prompt: &str,
        start_time: Instant,
    ) -> Value {
        let content = Self::extract_completion_content(lm_response);
        let timing = TimingInfo::from_text_content(start_time, prompt, &content);

        // Record metrics if enabled
        if let Some(metrics) = get_global_metrics() {
            tokio::spawn({
                let model = model.to_string();
                let content_len = content.len() as u64;
                async move {
                    metrics.record_model_usage(&model, content_len / 4).await;
                }
            });
        }

        json!({
            "model": model,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "response": content,
            "done": true,
            "context": DEFAULT_CONTEXT,
            "total_duration": timing.total_duration,
            "load_duration": timing.load_duration,
            "prompt_eval_count": timing.prompt_eval_count,
            "prompt_eval_duration": timing.prompt_eval_duration,
            "eval_count": timing.eval_count,
            "eval_duration": timing.eval_duration
        })
    }

    /// Transform LM Studio embeddings response to Ollama format
    pub fn convert_to_ollama_embeddings(
        lm_response: &Value,
        model: &str,
        start_time: Instant,
    ) -> Value {
        let embeddings = Self::extract_embeddings(lm_response);
        let timing = TimingInfo::calculate(start_time, 1, 1); // Minimal timing for embeddings

        json!({
            "model": model,
            "embeddings": embeddings,
            "total_duration": timing.total_duration,
            "load_duration": timing.load_duration,
            "prompt_eval_count": timing.prompt_eval_count,
            "prompt_eval_duration": timing.prompt_eval_duration
        })
    }

    /// Extract content with reasoning support
    fn extract_content_with_reasoning(lm_response: &Value, is_chat: bool) -> String {
        let base_content = if is_chat {
            Self::extract_chat_content(lm_response)
        } else {
            Self::extract_completion_content(lm_response)
        };

        // Check for reasoning content
        if let Some(reasoning) = lm_response
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|choices| choices.first())
            .and_then(|choice| {
                choice
                    .get("message")
                    .and_then(|m| m.get("reasoning_content"))
                    .and_then(|r| r.as_str())
            })
        {
            if !reasoning.is_empty() {
                return format!("**Reasoning:**\n{}\n\n**Answer:**\n{}", reasoning, base_content);
            }
        }

        base_content
    }

    /// Extract chat content from LM Studio response
    fn extract_chat_content(lm_response: &Value) -> String {
        lm_response
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|choices| choices.first())
            .and_then(|choice| {
                choice
                    .get("message")
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
            })
            .unwrap_or("")
            .to_string()
    }

    /// Extract completion content from LM Studio response
    fn extract_completion_content(lm_response: &Value) -> String {
        lm_response
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("text"))
            .and_then(|text| text.as_str())
            .unwrap_or("")
            .to_string()
    }

    /// Extract embeddings from LM Studio response
    fn extract_embeddings(lm_response: &Value) -> Vec<Value> {
        lm_response
            .get("data")
            .and_then(|d| d.as_array())
            .map(|data| {
                data.iter()
                    .filter_map(|item| item.get("embedding"))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Build LM Studio request using consolidated pattern
pub fn build_lm_studio_request(
    model: &str,
    request_type: LMStudioRequestType,
    options: Option<&Value>,
) -> Value {
    let mut builder = RequestBuilder::new()
        .add_required("model", model);

    // Add type-specific fields
    match request_type {
        LMStudioRequestType::Chat { messages, stream } => {
            builder = builder
                .add_required("messages", messages.clone())
                .add_required("stream", stream);
        }
        LMStudioRequestType::Completion { prompt, stream } => {
            builder = builder
                .add_required("prompt", prompt)
                .add_required("stream", stream);
        }
        LMStudioRequestType::Embeddings { input } => {
            builder = builder.add_required("input", input.clone());
        }
    }

    // Add common parameters from Ollama options
    let lm_params = map_ollama_to_lmstudio_params(options);
    let mut request_json = builder.build();

    if let Some(request_obj) = request_json.as_object_mut() {
        for (key, value) in lm_params {
            request_obj.insert(key, value);
        }
    }

    request_json
}

/// Request type enumeration for cleaner code
pub enum LMStudioRequestType<'a> {
    Chat { messages: &'a Value, stream: bool },
    Completion { prompt: &'a str, stream: bool },
    Embeddings { input: &'a Value },
}

/// Extract content from streaming chunk
pub fn extract_content_from_chunk(chunk: &Value) -> Option<String> {
    // Try chat format first
    chunk
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            // Try generate format
            chunk
                .get("response")
                .and_then(|r| r.as_str())
                .map(|s| s.to_string())
        })
}

/// Create streaming chunk templates
pub fn create_streaming_chunk(model: &str, content: &str, is_chat: bool, done: bool) -> Value {
    let timestamp = chrono::Utc::now().to_rfc3339();

    if is_chat {
        json!({
            "model": model,
            "created_at": timestamp,
            "message": {
                "role": "assistant",
                "content": content
            },
            "done": done
        })
    } else {
        json!({
            "model": model,
            "created_at": timestamp,
            "response": content,
            "done": done,
            "context": if done { Some(DEFAULT_CONTEXT.to_vec()) } else { None }
        })
    }
}

/// Create error chunk for streaming responses
pub fn create_error_chunk(model: &str, error_message: &str, is_chat: bool) -> Value {
    let mut chunk = create_streaming_chunk(model, "", is_chat, true);

    if let Some(chunk_obj) = chunk.as_object_mut() {
        chunk_obj.insert("error".to_string(), json!(error_message));
    }

    chunk
}

/// Create cancellation chunk with timing info
pub fn create_cancellation_chunk(
    model: &str,
    duration: Duration,
    tokens_generated: u64,
    is_chat: bool,
) -> Value {
    let total_duration = duration.as_nanos() as u64;
    let mut chunk = create_streaming_chunk(model, "", is_chat, true);

    if let Some(chunk_obj) = chunk.as_object_mut() {
        if is_chat {
            chunk_obj.insert("message".to_string(), json!({
                "role": "system",
                "content": ERROR_CANCELLED
            }));
        } else {
            chunk_obj.insert("response".to_string(), json!(ERROR_CANCELLED));
        }

        chunk_obj.insert("total_duration".to_string(), json!(total_duration));
        chunk_obj.insert("eval_count".to_string(), json!(tokens_generated));
        chunk_obj.insert("cancelled".to_string(), json!(true));
    }

    chunk
}

/// Create final completion chunk
pub fn create_final_chunk(model: &str, duration: Duration, chunk_count: u64, is_chat: bool) -> Value {
    let timing = TimingInfo::calculate(Instant::now() - duration, 10, chunk_count.max(1));
    let mut chunk = create_streaming_chunk(model, "", is_chat, true);

    if let Some(chunk_obj) = chunk.as_object_mut() {
        chunk_obj.insert("total_duration".to_string(), json!(timing.total_duration));
        chunk_obj.insert("load_duration".to_string(), json!(timing.load_duration));
        chunk_obj.insert("prompt_eval_count".to_string(), json!(timing.prompt_eval_count));
        chunk_obj.insert("prompt_eval_duration".to_string(), json!(timing.prompt_eval_duration));
        chunk_obj.insert("eval_count".to_string(), json!(timing.eval_count));
        chunk_obj.insert("eval_duration".to_string(), json!(timing.eval_duration));
    }

    chunk
}

/// Estimate token count from text (rough approximation)
fn estimate_token_count(text: &str) -> u64 {
    // Rough estimation: 1 token ≈ 4 characters on average
    ((text.len() as f64) * TOKEN_TO_CHAR_RATIO) as u64
}

/// Common request pattern for handlers
pub async fn execute_request_with_retry<F, Fut, T>(
    context: &crate::common::RequestContext<'_>,
    model_name: &str,
    operation: F,
    use_model_retry: bool,
    load_timeout_seconds: u64,
    cancellation_token: tokio_util::sync::CancellationToken,
) -> Result<T, crate::utils::ProxyError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output=Result<T, crate::utils::ProxyError>>,
{
    if use_model_retry {
        crate::handlers::retry::with_retry_and_cancellation(
            context,
            model_name,
            load_timeout_seconds,
            operation,
            cancellation_token,
        ).await
    } else {
        crate::handlers::retry::with_simple_retry(operation, cancellation_token).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_timing_calculation() {
        let start = Instant::now();
        std::thread::sleep(Duration::from_millis(10));
        let timing = TimingInfo::calculate(start, 100, 50);

        assert!(timing.total_duration > 0);
        assert_eq!(timing.prompt_eval_count, 100);
        assert_eq!(timing.eval_count, 50);
    }

    #[test]
    fn test_extract_content_from_chunk() {
        let chat_chunk = json!({
            "message": {
                "content": "Hello world"
            }
        });

        assert_eq!(
            extract_content_from_chunk(&chat_chunk).unwrap(),
            "Hello world"
        );

        let generate_chunk = json!({
            "response": "Hello world"
        });

        assert_eq!(
            extract_content_from_chunk(&generate_chunk).unwrap(),
            "Hello world"
        );
    }

    #[test]
    fn test_build_lm_studio_request() {
        let messages = json!([{"role": "user", "content": "Hello"}]);
        let request = build_lm_studio_request(
            "test-model",
            LMStudioRequestType::Chat { messages: &messages, stream: false },
            None,
        );

        assert_eq!(request.get("model").unwrap().as_str().unwrap(), "test-model");
        assert_eq!(request.get("messages").unwrap(), &messages);
        assert_eq!(request.get("stream").unwrap().as_bool().unwrap(), false);
    }

    #[test]
    fn test_estimate_token_count() {
        assert_eq!(estimate_token_count(""), 0);
        assert_eq!(estimate_token_count("hello"), 1); // 5 * 0.25 = 1.25 -> 1
        assert_eq!(estimate_token_count("hello world test"), 3); // 17 * 0.25 = 4.25 -> 4, but let's check
    }
}
