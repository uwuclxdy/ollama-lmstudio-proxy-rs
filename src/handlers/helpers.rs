// src/handlers/helpers.rs - Consolidated helper functions and utilities

use serde_json::{json, Value};
use std::time::{Duration, Instant};
use warp::Reply;

/// Helper function to convert JSON to Response
pub fn json_response(value: &Value) -> warp::reply::Response {
    warp::reply::with_status(
        warp::reply::json(value),
        warp::http::StatusCode::OK,
    ).into_response()
}

/// Determine model family and families array based on model name
pub fn determine_model_family(model_name: &str) -> (&'static str, Vec<&'static str>) {
    let lower_name = model_name.to_lowercase();

    match lower_name {
        name if name.contains("llama") => ("llama", vec!["llama"]),
        name if name.contains("mistral") => ("mistral", vec!["mistral"]),
        name if name.contains("qwen") => ("qwen2", vec!["qwen2"]),
        name if name.contains("deepseek") => ("llama", vec!["llama"]),
        name if name.contains("gemma") => ("gemma", vec!["gemma"]),
        name if name.contains("phi") => ("phi", vec!["phi"]),
        name if name.contains("codellama") => ("llama", vec!["llama"]),
        name if name.contains("vicuna") => ("llama", vec!["llama"]),
        name if name.contains("alpaca") => ("llama", vec!["llama"]),
        _ => ("llama", vec!["llama"]),
    }
}

/// Parameter size detection lookup table
/// todo: replace with a more efficient lookup mechanism
pub const PARAMETER_SIZES: &[(&str, &str)] = &[
    ("0.5b", "0.5B"),
    ("1.5b", "1.5B"),
    ("2b", "2B"),
    ("3b", "3B"),
    ("7b", "7B"),
    ("8b", "8B"),
    ("9b", "9B"),
    ("13b", "13B"),
    ("14b", "14B"),
    ("27b", "27B"),
    ("30b", "30B"),
    ("32b", "32B"),
    ("70b", "70B"),
];

/// Determine parameter size based on model name
pub fn determine_parameter_size(model_name: &str) -> &'static str {
    let lower_name = model_name.to_lowercase();

    // Use lookup table instead of long if-else chain
    for (pattern, size) in PARAMETER_SIZES {
        if lower_name.contains(pattern) {
            return size;
        }
    }

    "7B" // Fallback
}

/// Estimate model size in bytes based on parameter size
pub fn estimate_model_size(parameter_size: &str) -> u64 {
    match parameter_size {
        "0.5B" => 500_000_000,
        "1.5B" => 1_000_000_000,
        "2B" => 1_500_000_000,
        "3B" => 2_000_000_000,
        "7B" => 4_000_000_000,
        "8B" => 5_000_000_000,
        "9B" => 5_500_000_000,
        "13B" => 8_000_000_000,
        "14B" => 8_500_000_000,
        "27B" => 16_000_000_000,
        "30B" => 18_000_000_000,
        "32B" => 20_000_000_000,
        "70B" => 40_000_000_000,
        _ => 4_000_000_000,
    }
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

/// Find the best matching LM Studio model for an Ollama model name (duplicate from ollama.rs)
pub fn find_best_model_match(ollama_name: &str, available_models: &[String]) -> Option<String> {
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
pub fn models_match_size(name1: &str, name2: &str) -> bool {
    let sizes = ["0.5b", "1.5b", "2b", "3b", "7b", "8b", "9b", "11b", "13b", "14b", "27b", "30b", "32b", "70b"];

    for size in &sizes {
        if name1.contains(size) && name2.contains(size) {
            return true;
        }
    }

    // If no size found in either, consider them matching (size unknown)
    true
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

/// Create a clearer cancellation chunk that indicates cancellation, not generated content
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
            // todo: workaround until ollama supports reasoning_content
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
