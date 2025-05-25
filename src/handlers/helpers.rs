// src/handlers/helpers.rs - Helper functions and utilities

use once_cell::sync::Lazy;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use warp::Reply;

/// Helper function to convert JSON to Response
pub fn json_response(value: &Value) -> warp::reply::Response {
    warp::reply::with_status(
        warp::reply::json(value),
        warp::http::StatusCode::OK,
    ).into_response()
}

// ===== OPTIMIZED MODEL METADATA LOOKUPS =====

/// Model family lookup table for O(1) access
static MODEL_FAMILIES: Lazy<HashMap<&'static str, (&'static str, Vec<&'static str>)>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert("llama", ("llama", vec!["llama"]));
    map.insert("mistral", ("mistral", vec!["mistral"]));
    map.insert("qwen", ("qwen2", vec!["qwen2"]));
    map.insert("deepseek", ("llama", vec!["llama"]));
    map.insert("gemma", ("gemma", vec!["gemma"]));
    map.insert("phi", ("phi", vec!["phi"]));
    map.insert("codellama", ("llama", vec!["llama"]));
    map.insert("vicuna", ("llama", vec!["llama"]));
    map.insert("alpaca", ("llama", vec!["llama"]));
    map
});

/// Model size estimates for parameter counts
static SIZE_ESTIMATES: Lazy<HashMap<&'static str, u64>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert("0.5B", 500_000_000);
    map.insert("1.5B", 1_000_000_000);
    map.insert("2B", 1_500_000_000);
    map.insert("3B", 2_000_000_000);
    map.insert("7B", 4_000_000_000);
    map.insert("8B", 5_000_000_000);
    map.insert("9B", 5_500_000_000);
    map.insert("13B", 8_000_000_000);
    map.insert("14B", 8_500_000_000);
    map.insert("27B", 16_000_000_000);
    map.insert("30B", 18_000_000_000);
    map.insert("32B", 20_000_000_000);
    map.insert("70B", 40_000_000_000);
    map
});

// ===== MODEL METADATA FUNCTIONS =====

/// Determine model family and families array based on model name
pub fn determine_model_family(model_name: &str) -> (&'static str, Vec<&'static str>) {
    let lower_name = model_name.to_lowercase();

    for (pattern, (family, families)) in MODEL_FAMILIES.iter() {
        if lower_name.contains(pattern) {
            return (*family, families.clone());
        }
    }

    ("llama", vec!["llama"]) // Default fallback
}

/// Estimate model size in bytes based on parameter size
pub fn estimate_model_size(parameter_size: &str) -> u64 {
    SIZE_ESTIMATES.get(parameter_size).copied().unwrap_or(4_000_000_000)
}

/// Determine model capabilities based on model name
pub fn determine_model_capabilities(model_name: &str) -> Vec<&'static str> {
    let lower_name = model_name.to_lowercase();
    let mut capabilities = vec!["completion", "chat"];

    if lower_name.contains("embed") || lower_name.contains("bge") || lower_name.contains("nomic") {
        capabilities.push("embeddings");
    }

    if lower_name.contains("llava") || lower_name.contains("vision") || lower_name.contains("multimodal") {
        capabilities.push("vision");
    }

    if lower_name.contains("llama3") || lower_name.contains("mistral") || lower_name.contains("qwen") {
        capabilities.push("tools");
    }

    capabilities
}

// ===== REQUEST BUILDING UTILITIES =====

/// Simplified request builder for LM Studio API calls
/// Only includes parameters that were explicitly provided by the client
pub fn build_lm_studio_request(base_params: Map<String, Value>, ollama_options: Option<&Value>) -> Value {
    let mut request = base_params;

    if let Some(options) = ollama_options {
        // Temperature
        if let Some(temp) = options.get("temperature") {
            request.insert("temperature".to_string(), temp.clone());
        }

        // Max tokens (Ollama uses "num_predict", LM Studio uses "max_tokens")
        if let Some(max_tokens) = options.get("num_predict") {
            request.insert("max_tokens".to_string(), max_tokens.clone());
        }

        // Top-p
        if let Some(top_p) = options.get("top_p") {
            request.insert("top_p".to_string(), top_p.clone());
        }

        // Top-k
        if let Some(top_k) = options.get("top_k") {
            request.insert("top_k".to_string(), top_k.clone());
        }

        // Presence penalty
        if let Some(presence_penalty) = options.get("presence_penalty") {
            request.insert("presence_penalty".to_string(), presence_penalty.clone());
        }

        // Frequency penalty
        if let Some(frequency_penalty) = options.get("frequency_penalty") {
            request.insert("frequency_penalty".to_string(), frequency_penalty.clone());
        }

        // Repeat penalty (Ollama-specific, map to frequency_penalty if not already set)
        if let Some(repeat_penalty) = options.get("repeat_penalty") {
            if !request.contains_key("frequency_penalty") {
                request.insert("frequency_penalty".to_string(), repeat_penalty.clone());
            }
        }

        // Seed
        if let Some(seed) = options.get("seed") {
            request.insert("seed".to_string(), seed.clone());
        }

        // Stop sequences
        if let Some(stop) = options.get("stop") {
            request.insert("stop".to_string(), stop.clone());
        }
    }

    Value::Object(request)
}

// ===== CHUNK CREATION UTILITIES =====

/// Extract content from a chunk for tracking partial responses
pub fn extract_content_from_chunk(chunk: &Value) -> Option<String> {
    // For chat format
    if let Some(content) = chunk.get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str()) {
        return Some(content.to_string());
    }

    // For generate format
    if let Some(content) = chunk.get("response")
        .and_then(|r| r.as_str()) {
        return Some(content.to_string());
    }

    None
}

/// Create an error chunk for streaming responses
pub fn create_error_chunk(model: &str, error_message: &str, is_chat: bool) -> Value {
    if is_chat {
        json!({
            "model": model,
            "created_at": chrono::Utc::now().to_rfc3339(),
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
            "created_at": chrono::Utc::now().to_rfc3339(),
            "response": "",
            "done": true,
            "error": error_message
        })
    }
}

/// Create a cancellation chunk that indicates cancellation, not generated content
pub fn create_cancellation_chunk(model: &str, partial_content: &str, duration: Duration, tokens_generated: u64, is_chat: bool) -> Value {
    let cancellation_message = if partial_content.is_empty() {
        "🚫 Request cancelled - no content was generated due to client disconnection".to_string()
    } else {
        format!("🚫 Request cancelled - {} tokens were generated before client disconnection. This is a cancellation notice, not generated content.", tokens_generated)
    };

    if is_chat {
        json!({
            "model": model,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "message": {
                "role": "system",
                "content": cancellation_message
            },
            "done": true,
            "total_duration": duration.as_nanos() as u64,
            "load_duration": 1000000u64,
            "prompt_eval_count": 10,
            "prompt_eval_duration": duration.as_nanos() as u64 / 4,
            "eval_count": tokens_generated,
            "eval_duration": duration.as_nanos() as u64 / 2,
            "cancelled": true,
            "partial_response": !partial_content.is_empty()
        })
    } else {
        json!({
            "model": model,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "response": cancellation_message,
            "done": true,
            "context": [1, 2, 3],
            "total_duration": duration.as_nanos() as u64,
            "load_duration": 1000000u64,
            "prompt_eval_count": 10,
            "prompt_eval_duration": duration.as_nanos() as u64 / 4,
            "eval_count": tokens_generated,
            "eval_duration": duration.as_nanos() as u64 / 2,
            "cancelled": true,
            "partial_response": !partial_content.is_empty()
        })
    }
}

/// Create final completion chunk for streaming responses
pub fn create_final_chunk(model: &str, duration: Duration, chunk_count: u64, is_chat: bool) -> Value {
    if is_chat {
        json!({
            "model": model,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "message": {
                "role": "assistant",
                "content": ""
            },
            "done": true,
            "total_duration": duration.as_nanos() as u64,
            "load_duration": 1000000u64,
            "prompt_eval_count": 10,
            "prompt_eval_duration": duration.as_nanos() as u64 / 4,
            "eval_count": chunk_count.max(1),
            "eval_duration": duration.as_nanos() as u64 / 2
        })
    } else {
        json!({
            "model": model,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "response": "",
            "done": true,
            "context": [1, 2, 3],
            "total_duration": duration.as_nanos() as u64,
            "load_duration": 1000000u64,
            "prompt_eval_count": 10,
            "prompt_eval_duration": duration.as_nanos() as u64 / 4,
            "eval_count": chunk_count.max(1),
            "eval_duration": duration.as_nanos() as u64 / 2
        })
    }
}

// ===== RESPONSE TRANSFORMATION UTILITIES =====

/// Transform LM Studio chat response to Ollama format
pub fn transform_chat_response(lm_response: &Value, model: &str, messages: &[Value], start_time: Instant) -> Value {
    let content = if let Some(choices) = lm_response.get("choices").and_then(|c| c.as_array()) {
        if let Some(first_choice) = choices.first() {
            let mut content = first_choice.get("message")
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();

            // Merge reasoning_content if present
            // todo: implement reasoning_content once ollama officially supports it
            if let Some(reasoning) = first_choice.get("message")
                .and_then(|m| m.get("reasoning_content"))
                .and_then(|r| r.as_str()) {
                content = format!("**Reasoning:**\n{}\n\n**Answer:**\n{}", reasoning, content);
            }

            content
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    // Calculate timing estimates
    let total_duration = start_time.elapsed().as_nanos() as u64;
    let prompt_eval_count = messages.len() as u64 * 10;
    let eval_count = content.len() as u64 / 4;

    json!({
        "model": model,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "message": {
            "role": "assistant",
            "content": content
        },
        "done": true,
        "total_duration": total_duration,
        "load_duration": 1000000u64,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": total_duration / 4,
        "eval_count": eval_count,
        "eval_duration": total_duration / 2
    })
}

/// Transform LM Studio completion response to Ollama format
pub fn transform_generate_response(lm_response: &Value, model: &str, prompt: &str, start_time: Instant) -> Value {
    let response_text = if let Some(choices) = lm_response.get("choices").and_then(|c| c.as_array()) {
        choices.first()
            .and_then(|choice| choice.get("text"))
            .and_then(|text| text.as_str())
            .unwrap_or("")
            .to_string()
    } else {
        String::new()
    };

    let total_duration = start_time.elapsed().as_nanos() as u64;
    let prompt_eval_count = prompt.len() as u64 / 4;
    let eval_count = response_text.len() as u64 / 4;

    json!({
        "model": model,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "response": response_text,
        "done": true,
        "context": [1, 2, 3],
        "total_duration": total_duration,
        "load_duration": 1000000u64,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": total_duration / 4,
        "eval_count": eval_count,
        "eval_duration": total_duration / 2
    })
}

/// Transform LM Studio embeddings response to Ollama format
pub fn transform_embeddings_response(lm_response: &Value, model: &str, start_time: Instant) -> Value {
    let embeddings = if let Some(data) = lm_response.get("data").and_then(|d| d.as_array()) {
        data.iter()
            .filter_map(|item| item.get("embedding"))
            .collect::<Vec<_>>()
    } else {
        vec![]
    };

    let total_duration = start_time.elapsed().as_nanos() as u64;

    json!({
        "model": model,
        "embeddings": embeddings,
        "total_duration": total_duration,
        "load_duration": 1000000u64,
        "prompt_eval_count": 1
    })
}
