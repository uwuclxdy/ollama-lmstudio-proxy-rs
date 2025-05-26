/// src/handlers/helpers.rs - Helper functions for request/response transformation and timing.

use serde_json::{json, Value};
use std::time::{Duration, Instant};

use crate::common::{map_ollama_to_lmstudio_params, RequestBuilder};
use crate::constants::*;

/// Create JSON response with proper headers
pub fn json_response(value: &Value) -> warp::reply::Response {
    let json_string = serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string());
    let content_length = json_string.len();

    warp::http::Response::builder()
        .status(warp::http::StatusCode::OK)
        .header("Content-Type", CONTENT_TYPE_JSON)
        .header("Content-Length", content_length.to_string())
        .header("Cache-Control", HEADER_CACHE_CONTROL)
        .header("Access-Control-Allow-Origin", HEADER_ACCESS_CONTROL_ALLOW_ORIGIN)
        .header("Access-Control-Allow-Methods", HEADER_ACCESS_CONTROL_ALLOW_METHODS)
        .header("Access-Control-Allow-Headers", HEADER_ACCESS_CONTROL_ALLOW_HEADERS)
        .body(json_string.into())
        .unwrap_or_else(|_| {
            warp::http::Response::builder()
                .status(warp::http::StatusCode::INTERNAL_SERVER_ERROR)
                .body("Internal Server Error".into())
                .unwrap()
        })
}

/// Consolidated timing information for Ollama responses
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
    /// Calculate timing from token counts and duration
    pub fn calculate(
        start_time: Instant,
        input_tokens_estimate: u64,
        output_tokens_estimate: u64,
        actual_prompt_tokens: Option<u64>,
        actual_completion_tokens: Option<u64>,
    ) -> Self {
        let total_duration_ns = start_time.elapsed().as_nanos() as u64;

        let final_prompt_tokens = actual_prompt_tokens.unwrap_or(input_tokens_estimate).max(1);
        let final_eval_tokens = actual_completion_tokens.unwrap_or(output_tokens_estimate).max(1);

        // Proportional split
        let prompt_eval_duration_ns = if final_prompt_tokens + final_eval_tokens > 0 && total_duration_ns > 1000 {
            (total_duration_ns as f64 * (final_prompt_tokens as f64 / (final_prompt_tokens + final_eval_tokens) as f64)) as u64
        } else {
            total_duration_ns / TIMING_PROMPT_RATIO
        };

        let eval_duration_ns = if final_prompt_tokens + final_eval_tokens > 0 && total_duration_ns > 1000 {
            total_duration_ns - prompt_eval_duration_ns
        } else {
            total_duration_ns / TIMING_EVAL_RATIO
        };

        Self {
            total_duration: total_duration_ns,
            load_duration: DEFAULT_LOAD_DURATION_NS,
            prompt_eval_count: final_prompt_tokens,
            prompt_eval_duration: prompt_eval_duration_ns.max(1),
            eval_count: final_eval_tokens,
            eval_duration: eval_duration_ns.max(1),
        }
    }

    /// Calculate timing from text content
    pub fn from_text_content(start_time: Instant, input_text: &str, output_text: &str) -> Self {
        let input_tokens = estimate_token_count(input_text);
        let output_tokens = estimate_token_count(output_text);
        Self::calculate(start_time, input_tokens, output_tokens, None, None)
    }

    /// Calculate timing from message count
    pub fn from_message_count(start_time: Instant, message_count: usize, output_text: &str) -> Self {
        let input_tokens = (message_count * 10).max(1) as u64;
        let output_tokens = estimate_token_count(output_text);
        Self::calculate(start_time, input_tokens, output_tokens, None, None)
    }
}

/// Transform LM Studio responses to Ollama format
pub struct ResponseTransformer;

impl ResponseTransformer {
    /// Transform LM Studio chat response to Ollama format
    pub fn convert_to_ollama_chat(
        lm_response: &Value,
        model_ollama_name: &str,
        message_count_for_estimation: usize,
        start_time: Instant,
    ) -> Value {
        let content = Self::extract_chat_content_with_reasoning(lm_response);

        let actual_prompt_tokens = lm_response.get("usage").and_then(|u| u.get("prompt_tokens")).and_then(|t| t.as_u64());
        let actual_completion_tokens = lm_response.get("usage").and_then(|u| u.get("completion_tokens")).and_then(|t| t.as_u64());

        let timing = TimingInfo::calculate(
            start_time,
            (message_count_for_estimation * 10).max(1) as u64,
            estimate_token_count(&content),
            actual_prompt_tokens,
            actual_completion_tokens,
        );

        let mut ollama_message = json!({
            "role": "assistant",
            "content": content
        });

        if let Some(tool_calls) = lm_response.get("choices")
            .and_then(|c| c.as_array()?.first())
            .and_then(|choice| choice.get("message")?.get("tool_calls"))
            .and_then(|tc| tc.as_array())
        {
            if !tool_calls.is_empty() {
                if let Some(msg_obj) = ollama_message.as_object_mut() {
                    msg_obj.insert("tool_calls".to_string(), json!(tool_calls));
                }
            }
        }

        json!({
            "model": model_ollama_name,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "message": ollama_message,
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
        model_ollama_name: &str,
        prompt_for_estimation: &str,
        start_time: Instant,
    ) -> Value {
        let content = Self::extract_completion_content(lm_response);

        let actual_prompt_tokens = lm_response.get("usage").and_then(|u| u.get("prompt_tokens")).and_then(|t| t.as_u64());
        let actual_completion_tokens = lm_response.get("usage").and_then(|u| u.get("completion_tokens")).and_then(|t| t.as_u64());

        let timing = TimingInfo::calculate(
            start_time,
            estimate_token_count(prompt_for_estimation),
            estimate_token_count(&content),
            actual_prompt_tokens,
            actual_completion_tokens,
        );

        json!({
            "model": model_ollama_name,
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
        model_ollama_name: &str,
        start_time: Instant,
    ) -> Value {
        let embeddings = Self::extract_embeddings(lm_response);

        let actual_prompt_tokens = lm_response.get("usage").and_then(|u| u.get("prompt_tokens")).and_then(|t| t.as_u64());

        let estimated_input_tokens = 10;
        let estimated_output_tokens = embeddings.len().max(1) as u64;

        let timing = TimingInfo::calculate(
            start_time,
            estimated_input_tokens,
            estimated_output_tokens,
            actual_prompt_tokens,
            None,
        );

        json!({
            "model": model_ollama_name,
            "embeddings": embeddings,
            "total_duration": timing.total_duration,
            "load_duration": timing.load_duration,
            "prompt_eval_count": timing.prompt_eval_count,
            "prompt_eval_duration": timing.prompt_eval_duration
        })
    }

    /// Extract chat content including reasoning
    fn extract_chat_content_with_reasoning(lm_response: &Value) -> String {
        let base_content = lm_response
            .get("choices")
            .and_then(|c| c.as_array()?.first())
            .and_then(|choice| choice.get("message")?.get("content")?.as_str())
            .unwrap_or("")
            .to_string();

        if let Some(reasoning) = lm_response
            .get("choices")
            .and_then(|c| c.as_array()?.first())
            .and_then(|choice| choice.get("message")?.get("reasoning_content")?.as_str())
        {
            if !reasoning.is_empty() {
                return format!("**Reasoning:**\n{}\n\n**Answer:**\n{}", reasoning, base_content);
            }
        }
        base_content
    }

    /// Extract completion content from response
    fn extract_completion_content(lm_response: &Value) -> String {
        lm_response
            .get("choices")
            .and_then(|c| c.as_array()?.first())
            .and_then(|choice| choice.get("text")?.as_str())
            .unwrap_or("")
            .to_string()
    }

    /// Extract embeddings from response
    fn extract_embeddings(lm_response: &Value) -> Vec<Value> {
        lm_response
            .get("data")
            .and_then(|d| d.as_array())
            .map(|data_array| {
                data_array.iter()
                    .filter_map(|item| item.get("embedding").cloned())
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Build LM Studio request from Ollama parameters
pub fn build_lm_studio_request(
    model_lm_studio_id: &str,
    request_type: LMStudioRequestType,
    ollama_options: Option<&Value>,
    ollama_tools: Option<&Value>,
) -> Value {
    let mut builder = RequestBuilder::new()
        .add_required("model", model_lm_studio_id);

    match request_type {
        LMStudioRequestType::Chat { messages, stream } => {
            builder = builder
                .add_required("messages", messages.clone())
                .add_required("stream", stream);
            if let Some(tools_val) = ollama_tools {
                if tools_val.is_array() && !tools_val.as_array().unwrap().is_empty() {
                    builder = builder.add_required("tools", tools_val.clone());
                }
            }
        }
        LMStudioRequestType::Completion { prompt, stream, images } => {
            // Vision support
            if let Some(img_array) = images {
                let chat_messages = json!([{
                    "role": "user",
                    "content": prompt,
                    "images": img_array
                }]);
                builder = builder
                    .add_required("messages", chat_messages)
                    .add_required("stream", stream);
            } else {
                builder = builder
                    .add_required("prompt", prompt)
                    .add_required("stream", stream);
            }
        }
        LMStudioRequestType::Embeddings { input } => {
            builder = builder.add_required("input", input.clone());
        }
    }

    let lm_studio_mapped_params = map_ollama_to_lmstudio_params(ollama_options);
    let mut request_json = builder.build();

    if let Some(request_obj) = request_json.as_object_mut() {
        for (key, value) in lm_studio_mapped_params {
            request_obj.insert(key, value);
        }
    }

    request_json
}

/// Request type enumeration
pub enum LMStudioRequestType<'a> {
    Chat { messages: &'a Value, stream: bool },
    Completion { prompt: &'a str, stream: bool, images: Option<&'a Value> },
    Embeddings { input: &'a Value },
}

/// Extract content from streaming chunk
pub fn extract_content_from_chunk(chunk: &Value) -> Option<String> {
    // Chat format
    chunk
        .get("choices")
        .and_then(|c| c.as_array()?.first())
        .and_then(|choice| choice.get("delta")?.get("content")?.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            // Completion format
            chunk
                .get("choices")
                .and_then(|c| c.as_array()?.first())
                .and_then(|choice| choice.get("text")?.as_str())
                .map(|s| s.to_string())
        })
        .or_else(|| {
            // Ollama fallback
            chunk
                .get("response")
                .and_then(|r| r.as_str())
                .map(|s| s.to_string())
        })
}

/// Create Ollama streaming chunk
pub fn create_ollama_streaming_chunk(
    model_ollama_name: &str,
    content: &str,
    is_chat_endpoint: bool,
    done: bool,
    tool_calls_delta: Option<&Value>,
) -> Value {
    let timestamp = chrono::Utc::now().to_rfc3339();

    if is_chat_endpoint {
        let mut message_obj = json!({
            "role": "assistant",
            "content": content
        });
        if let Some(tc_delta) = tool_calls_delta {
            if let Some(msg_map) = message_obj.as_object_mut() {
                msg_map.insert("tool_calls".to_string(), tc_delta.clone());
            }
        }

        json!({
            "model": model_ollama_name,
            "created_at": timestamp,
            "message": message_obj,
            "done": done
        })
    } else {
        json!({
            "model": model_ollama_name,
            "created_at": timestamp,
            "response": content,
            "done": done,
            "context": if done { Some(DEFAULT_CONTEXT.to_vec()) } else { None }
        })
    }
}

/// Create error chunk for streaming
pub fn create_error_chunk(model_ollama_name: &str, error_message: &str, is_chat_endpoint: bool) -> Value {
    let mut chunk = create_ollama_streaming_chunk(model_ollama_name, "", is_chat_endpoint, true, None);
    if let Some(chunk_obj) = chunk.as_object_mut() {
        chunk_obj.insert("error".to_string(), json!(error_message));
        if is_chat_endpoint {
            if let Some(msg) = chunk_obj.get_mut("message").and_then(|m| m.as_object_mut()) {
                msg.insert("content".to_string(), json!(""));
            }
        }
    }
    chunk
}

/// Create cancellation chunk with timing
pub fn create_cancellation_chunk(
    model_ollama_name: &str,
    duration: Duration,
    tokens_generated_estimate: u64,
    is_chat_endpoint: bool,
) -> Value {
    let timing = TimingInfo::calculate(Instant::now() - duration, 10, tokens_generated_estimate, None, Some(tokens_generated_estimate));

    let mut chunk = create_ollama_streaming_chunk(model_ollama_name, "", is_chat_endpoint, true, None);

    if let Some(chunk_obj) = chunk.as_object_mut() {
        let content_field_value = if tokens_generated_estimate > 0 {
            format!("[Request cancelled after {} tokens generated (estimated)]", tokens_generated_estimate)
        } else {
            ERROR_CANCELLED.to_string()
        };

        if is_chat_endpoint {
            if let Some(msg) = chunk_obj.get_mut("message").and_then(|m| m.as_object_mut()) {
                msg.insert("content".to_string(), json!(content_field_value));
            }
        } else {
            chunk_obj.insert("response".to_string(), json!(content_field_value));
        }

        chunk_obj.insert("total_duration".to_string(), json!(timing.total_duration));
        chunk_obj.insert("load_duration".to_string(), json!(timing.load_duration));
        chunk_obj.insert("prompt_eval_count".to_string(), json!(timing.prompt_eval_count));
        chunk_obj.insert("prompt_eval_duration".to_string(), json!(timing.prompt_eval_duration));
        chunk_obj.insert("eval_count".to_string(), json!(timing.eval_count));
        chunk_obj.insert("eval_duration".to_string(), json!(timing.eval_duration));
        chunk_obj.insert("done_reason".to_string(), json!("cancelled"));
    }
    chunk
}

/// Create final completion chunk for streaming
pub fn create_final_chunk(
    model_ollama_name: &str,
    duration: Duration,
    chunk_count_for_token_estimation: u64,
    is_chat_endpoint: bool,
) -> Value {
    let timing = TimingInfo::calculate(
        Instant::now() - duration,
        10,
        chunk_count_for_token_estimation.max(1),
        None,
        None,
    );

    let mut chunk = create_ollama_streaming_chunk(model_ollama_name, "", is_chat_endpoint, true, None);

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

/// Estimate token count from text
fn estimate_token_count(text: &str) -> u64 {
    if text.is_empty() { return 0; }
    ((text.len() as f64) * TOKEN_TO_CHAR_RATIO).ceil() as u64
}

/// Execute request with optional retry logic
pub async fn execute_request_with_retry<F, Fut, T>(
    context: &crate::common::RequestContext<'_>,
    model_name_for_retry_logic: &str,
    operation: F,
    use_model_retry: bool,
    load_timeout_seconds: u64,
    cancellation_token: tokio_util::sync::CancellationToken,
) -> Result<T, crate::utils::ProxyError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, crate::utils::ProxyError>>,
{
    if use_model_retry {
        crate::handlers::retry::with_retry_and_cancellation(
            context,
            model_name_for_retry_logic,
            load_timeout_seconds,
            operation,
            cancellation_token,
        ).await
    } else {
        crate::handlers::retry::with_simple_retry(operation, cancellation_token).await
    }
}
