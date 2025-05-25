// src/handlers/helpers.rs - Simplified response transformation utilities

use serde_json::{json, Map, Value};
use std::time::{Duration, Instant};
use warp::Reply;

use crate::constants::*;

/// Helper function to convert JSON to Response
pub fn json_response(value: &Value) -> warp::reply::Response {
    warp::reply::with_status(
        warp::reply::json(value),
        warp::http::StatusCode::OK,
    ).into_response()
}

/// Timing information for response generation
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
    /// Calculate timing from message count and content
    pub fn from_chat(start_time: Instant, message_count: usize, content: &str) -> Self {
        let total_duration = start_time.elapsed().as_nanos() as u64;
        let prompt_eval_count = (message_count * 10).max(1) as u64;
        let eval_count = ((content.len() as f64 * TOKEN_TO_CHAR_RATIO) as u64).max(1);

        Self {
            total_duration,
            load_duration: DEFAULT_LOAD_DURATION_NS,
            prompt_eval_count,
            prompt_eval_duration: total_duration / TIMING_PROMPT_RATIO,
            eval_count,
            eval_duration: total_duration / TIMING_EVAL_RATIO,
        }
    }

    /// Calculate timing from prompt and content
    pub fn from_generate(start_time: Instant, prompt: &str, content: &str) -> Self {
        let total_duration = start_time.elapsed().as_nanos() as u64;
        let prompt_eval_count = ((prompt.len() as f64 * TOKEN_TO_CHAR_RATIO) as u64).max(1);
        let eval_count = ((content.len() as f64 * TOKEN_TO_CHAR_RATIO) as u64).max(1);

        Self {
            total_duration,
            load_duration: DEFAULT_LOAD_DURATION_NS,
            prompt_eval_count,
            prompt_eval_duration: total_duration / TIMING_PROMPT_RATIO,
            eval_count,
            eval_duration: total_duration / TIMING_EVAL_RATIO,
        }
    }

    /// Calculate timing for embeddings
    pub fn from_embeddings(start_time: Instant) -> Self {
        let total_duration = start_time.elapsed().as_nanos() as u64;

        Self {
            total_duration,
            load_duration: DEFAULT_LOAD_DURATION_NS,
            prompt_eval_count: 1,
            prompt_eval_duration: total_duration / TIMING_PROMPT_RATIO,
            eval_count: 1,
            eval_duration: total_duration / TIMING_EVAL_RATIO,
        }
    }
}

/// Unified response transformer
pub struct ResponseTransformer;

impl ResponseTransformer {
    /// Transform LM Studio chat response to Ollama format
    pub fn to_ollama_chat(
        lm_response: &Value,
        model: &str,
        messages: &[Value],
        start_time: Instant
    ) -> Value {
        let content = Self::extract_chat_content(lm_response);
        let timing = TimingInfo::from_chat(start_time, messages.len(), &content);

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
    pub fn to_ollama_generate(
        lm_response: &Value,
        model: &str,
        prompt: &str,
        start_time: Instant
    ) -> Value {
        let content = Self::extract_completion_content(lm_response);
        let timing = TimingInfo::from_generate(start_time, prompt, &content);

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
    pub fn to_ollama_embeddings(
        lm_response: &Value,
        model: &str,
        start_time: Instant
    ) -> Value {
        let embeddings = Self::extract_embeddings(lm_response);
        let timing = TimingInfo::from_embeddings(start_time);

        json!({
            "model": model,
            "embeddings": embeddings,
            "total_duration": timing.total_duration,
            "load_duration": timing.load_duration,
            "prompt_eval_count": timing.prompt_eval_count
        })
    }

    /// Extract content from LM Studio chat response
    fn extract_chat_content(lm_response: &Value) -> String {
        lm_response
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|choices| choices.first())
            .and_then(|choice| {
                let mut content = choice
                    .get("message")
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                    .unwrap_or("")
                    .to_string();

                // Handle reasoning content if present (for models like DeepSeek R1)
                if let Some(reasoning) = choice
                    .get("message")
                    .and_then(|m| m.get("reasoning_content"))
                    .and_then(|r| r.as_str())
                {
                    if !reasoning.is_empty() {
                        content = format!("**Reasoning:**\n{}\n\n**Answer:**\n{}", reasoning, content);
                    }
                }

                Some(content)
            })
            .unwrap_or_default()
    }

    /// Extract content from LM Studio completion response
    fn extract_completion_content(lm_response: &Value) -> String {
        lm_response
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("text"))
            .and_then(|text| text.as_str())
            .map(|s| s.to_string())
            .unwrap_or_default()
    }

    /// Extract embeddings from LM Studio response
    fn extract_embeddings(lm_response: &Value) -> Vec<&Value> {
        lm_response
            .get("data")
            .and_then(|d| d.as_array())
            .map(|data| {
                data.iter()
                    .filter_map(|item| item.get("embedding"))
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Build LM Studio request with only provided parameters
pub fn build_lm_studio_request(base_params: Map<String, Value>, ollama_options: Option<&Value>) -> Value {
    let mut request = base_params;

    if let Some(options) = ollama_options {
        // Only add parameters that were explicitly provided by the client

        if let Some(temp) = options.get("temperature") {
            request.insert("temperature".to_string(), temp.clone());
        }

        // Map Ollama's num_predict to LM Studio's max_tokens
        if let Some(max_tokens) = options.get("num_predict") {
            request.insert("max_tokens".to_string(), max_tokens.clone());
        }

        if let Some(top_p) = options.get("top_p") {
            request.insert("top_p".to_string(), top_p.clone());
        }

        if let Some(top_k) = options.get("top_k") {
            request.insert("top_k".to_string(), top_k.clone());
        }

        if let Some(presence_penalty) = options.get("presence_penalty") {
            request.insert("presence_penalty".to_string(), presence_penalty.clone());
        }

        if let Some(frequency_penalty) = options.get("frequency_penalty") {
            request.insert("frequency_penalty".to_string(), frequency_penalty.clone());
        }

        // Map Ollama's repeat_penalty to frequency_penalty if not already set
        if let Some(repeat_penalty) = options.get("repeat_penalty") {
            if !request.contains_key("frequency_penalty") {
                request.insert("frequency_penalty".to_string(), repeat_penalty.clone());
            }
        }

        if let Some(seed) = options.get("seed") {
            request.insert("seed".to_string(), seed.clone());
        }

        if let Some(stop) = options.get("stop") {
            request.insert("stop".to_string(), stop.clone());
        }
    }

    Value::Object(request)
}

/// Extract content from a streaming chunk for tracking partial responses
pub fn extract_content_from_chunk(chunk: &Value) -> Option<String> {
    // Try chat format first
    if let Some(content) = chunk
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
    {
        return Some(content.to_string());
    }

    // Try generate format
    if let Some(content) = chunk
        .get("response")
        .and_then(|r| r.as_str())
    {
        return Some(content.to_string());
    }

    None
}

/// Create an error chunk for streaming responses
pub fn create_error_chunk(model: &str, error_message: &str, is_chat: bool) -> Value {
    let timestamp = chrono::Utc::now().to_rfc3339();

    if is_chat {
        json!({
            "model": model,
            "created_at": timestamp,
            "message": {
                "role": "assistant",
                "content": ""
            },
            "done": true,
            "error": error_message
        })
    } else {
        json!({
            "model": model,
            "created_at": timestamp,
            "response": "",
            "done": true,
            "context": DEFAULT_CONTEXT,
            "error": error_message
        })
    }
}

/// Create a cancellation chunk (simplified message)
pub fn create_cancellation_chunk(
    model: &str,
    _partial_content: &str,
    duration: Duration,
    tokens_generated: u64,
    is_chat: bool
) -> Value {
    let timestamp = chrono::Utc::now().to_rfc3339();
    let total_duration = duration.as_nanos() as u64;

    if is_chat {
        json!({
            "model": model,
            "created_at": timestamp,
            "message": {
                "role": "system",
                "content": ERROR_CANCELLED
            },
            "done": true,
            "total_duration": total_duration,
            "eval_count": tokens_generated,
            "cancelled": true
        })
    } else {
        json!({
            "model": model,
            "created_at": timestamp,
            "response": ERROR_CANCELLED,
            "done": true,
            "context": DEFAULT_CONTEXT,
            "total_duration": total_duration,
            "eval_count": tokens_generated,
            "cancelled": true
        })
    }
}

/// Create final completion chunk for streaming responses
pub fn create_final_chunk(model: &str, duration: Duration, chunk_count: u64, is_chat: bool) -> Value {
    let timestamp = chrono::Utc::now().to_rfc3339();
    let total_duration = duration.as_nanos() as u64;
    let eval_count = chunk_count.max(1);

    if is_chat {
        json!({
            "model": model,
            "created_at": timestamp,
            "message": {
                "role": "assistant",
                "content": ""
            },
            "done": true,
            "total_duration": total_duration,
            "load_duration": DEFAULT_LOAD_DURATION_NS,
            "prompt_eval_count": 10,
            "prompt_eval_duration": total_duration / TIMING_PROMPT_RATIO,
            "eval_count": eval_count,
            "eval_duration": total_duration / TIMING_EVAL_RATIO
        })
    } else {
        json!({
            "model": model,
            "created_at": timestamp,
            "response": "",
            "done": true,
            "context": DEFAULT_CONTEXT,
            "total_duration": total_duration,
            "load_duration": DEFAULT_LOAD_DURATION_NS,
            "prompt_eval_count": 10,
            "prompt_eval_duration": total_duration / TIMING_PROMPT_RATIO,
            "eval_count": eval_count,
            "eval_duration": total_duration / TIMING_EVAL_RATIO
        })
    }
}
