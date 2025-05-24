use serde_json::{json, Value};
use std::time::Instant;

use crate::server::ProxyServer;
use crate::utils::{clean_model_name, format_duration, ProxyError};
use super::retry::with_retry;
use super::streaming::{is_streaming_request, handle_streaming_response};
use super::helpers::{
    json_response, determine_model_family, determine_parameter_size,
    estimate_model_size, determine_model_capabilities,
};

/// Handle GET /api/tags - list available models
pub async fn handle_ollama_tags(server: ProxyServer) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    let operation = || async {
        let url = format!("{}/v1/models", server.config.lmstudio_url);
        server.logger.log(&format!("Calling LM Studio: GET {}", url));

        let response = server.client.get(&url).send().await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to reach LM Studio: {}", e)))?;

        if !response.status().is_success() {
            server.logger.log(&format!("LM Studio not available ({}), returning empty model list", response.status()));
            let empty_response = json!({
                "models": []
            });
            return Ok(empty_response);
        }

        let lm_response: Value = response.json().await
            .map_err(|_e| {
                ProxyError::internal_server_error("LM Studio response parsing failed")
            })?;

        // Transform LM Studio format to Ollama format with ALL required fields
        let models = if let Some(data) = lm_response.get("data").and_then(|d| d.as_array()) {
            data.iter().map(|model| {
                let model_id = model.get("id").and_then(|id| id.as_str()).unwrap_or("unknown");
                let cleaned_model = clean_model_name(model_id);
                let model_name = format!("{}:latest", cleaned_model);

                let (family, families) = determine_model_family(&cleaned_model);
                let parameter_size = determine_parameter_size(&cleaned_model);
                let size = estimate_model_size(parameter_size);

                json!({
                    "name": model_name,
                    "model": model_name,
                    "modified_at": chrono::Utc::now().to_rfc3339(),
                    "size": size,
                    "digest": format!("{:x}", md5::compute(model_id.as_bytes())),
                    "details": {
                        "parent_model": "",
                        "format": "gguf",
                        "family": family,
                        "families": families,
                        "parameter_size": parameter_size,
                        "quantization_level": "Q4_K_M"
                    }
                })
            }).collect::<Vec<_>>()
        } else {
            vec![]
        };

        let ollama_response = json!({
            "models": models
        });

        Ok(ollama_response)
    };

    let result = match with_retry(&server, operation).await {
        Ok(result) => result,
        Err(e) => {
            server.logger.log(&format!("Error in handle_ollama_tags: {}", e.message));
            json!({
                "models": []
            })
        }
    };

    let duration = start_time.elapsed();
    server.logger.log(&format!("Ollama tags response completed (took {})", format_duration(duration)));
    Ok(json_response(&result))
}

/// Handle POST /api/chat - chat completion with streaming support
pub async fn handle_ollama_chat(server: ProxyServer, body: Value) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    let operation = || async {
        // Extract fields from Ollama request
        let model = body.get("model").and_then(|m| m.as_str())
            .ok_or_else(|| ProxyError::bad_request("Missing 'model' field"))?;
        let messages = body.get("messages").and_then(|m| m.as_array())
            .ok_or_else(|| ProxyError::bad_request("Missing 'messages' field"))?;
        let stream = is_streaming_request(&body);

        let cleaned_model = clean_model_name(model);

        // Convert to LM Studio format
        let lm_request = json!({
            "model": cleaned_model,
            "messages": messages,
            "stream": stream,
            "temperature": body.get("options").and_then(|o| o.get("temperature")).unwrap_or(&json!(0.7)),
            "max_tokens": body.get("options").and_then(|o| o.get("num_predict")).unwrap_or(&json!(2048))
        });

        let url = format!("{}/v1/chat/completions", server.config.lmstudio_url);
        server.logger.log(&format!("Calling LM Studio: POST {} (stream: {})", url, stream));

        let response = server.client.post(&url)
            .json(&lm_request)
            .send().await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to reach LM Studio: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProxyError::internal_server_error(&format!("LM Studio error: {}", error_text)));
        }

        if stream {
            // Handle streaming response
            handle_streaming_response(response, true, model, start_time).await
        } else {
            // Handle non-streaming response
            handle_non_streaming_chat_response(response, model, messages, start_time).await
        }
    };

    let result = with_retry(&server, operation).await?;
    let duration = start_time.elapsed();

    server.logger.log(&format!("Ollama chat response completed (took {})", format_duration(duration)));
    Ok(result)
}

/// Handle non-streaming chat response
async fn handle_non_streaming_chat_response(
    response: reqwest::Response,
    model: &str,
    messages: &[Value],
    start_time: Instant,
) -> Result<warp::reply::Response, ProxyError> {
    let lm_response: Value = response.json().await
        .map_err(|e| ProxyError::internal_server_error(&format!("Failed to parse LM Studio response: {}", e)))?;

    // Transform response to Ollama format
    let content = if let Some(choices) = lm_response.get("choices").and_then(|c| c.as_array()) {
        if let Some(first_choice) = choices.first() {
            let mut content = first_choice.get("message")
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();

            // Merge reasoning_content if present
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

    let ollama_response = json!({
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
    });

    Ok(json_response(&ollama_response))
}

/// Handle POST /api/generate - text completion with streaming support
pub async fn handle_ollama_generate(server: ProxyServer, body: Value) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    let operation = || async {
        let model = body.get("model").and_then(|m| m.as_str())
            .ok_or_else(|| ProxyError::bad_request("Missing 'model' field"))?;
        let prompt = body.get("prompt").and_then(|p| p.as_str())
            .ok_or_else(|| ProxyError::bad_request("Missing 'prompt' field"))?;
        let stream = is_streaming_request(&body);

        let cleaned_model = clean_model_name(model);

        // Convert to LM Studio completions format
        let lm_request = json!({
            "model": cleaned_model,
            "prompt": prompt,
            "stream": stream,
            "temperature": body.get("options").and_then(|o| o.get("temperature")).unwrap_or(&json!(0.7)),
            "max_tokens": body.get("options").and_then(|o| o.get("num_predict")).unwrap_or(&json!(2048))
        });

        let url = format!("{}/v1/completions", server.config.lmstudio_url);
        server.logger.log(&format!("Calling LM Studio: POST {} (stream: {})", url, stream));

        let response = server.client.post(&url)
            .json(&lm_request)
            .send().await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to reach LM Studio: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProxyError::internal_server_error(&format!("LM Studio error: {}", error_text)));
        }

        if stream {
            handle_streaming_response(response, false, model, start_time).await
        } else {
            handle_non_streaming_generate_response(response, model, prompt, start_time).await
        }
    };

    let result = with_retry(&server, operation).await?;
    let duration = start_time.elapsed();

    server.logger.log(&format!("Ollama generate response completed (took {})", format_duration(duration)));
    Ok(result)
}

/// Handle non-streaming generate response
async fn handle_non_streaming_generate_response(
    response: reqwest::Response,
    model: &str,
    prompt: &str,
    start_time: Instant,
) -> Result<warp::reply::Response, ProxyError> {
    let lm_response: Value = response.json().await
        .map_err(|e| ProxyError::internal_server_error(&format!("Failed to parse LM Studio response: {}", e)))?;

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

    let ollama_response = json!({
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
    });

    Ok(json_response(&ollama_response))
}

/// Handle POST /api/embed or /api/embeddings - generate embeddings
pub async fn handle_ollama_embeddings(server: ProxyServer, body: Value) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();

    let operation = || async {
        let model = body.get("model").and_then(|m| m.as_str())
            .ok_or_else(|| ProxyError::bad_request("Missing 'model' field"))?;
        let input = body.get("input").or_else(|| body.get("prompt"))
            .ok_or_else(|| ProxyError::bad_request("Missing 'input' or 'prompt' field"))?;

        let cleaned_model = clean_model_name(model);

        let lm_request = json!({
            "model": cleaned_model,
            "input": input
        });

        let url = format!("{}/v1/embeddings", server.config.lmstudio_url);
        server.logger.log(&format!("Calling LM Studio: POST {}", url));

        let response = server.client.post(&url)
            .json(&lm_request)
            .send().await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to reach LM Studio: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProxyError::internal_server_error(&format!("LM Studio error: {}", error_text)));
        }

        let lm_response: Value = response.json().await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to parse LM Studio response: {}", e)))?;

        let embeddings = if let Some(data) = lm_response.get("data").and_then(|d| d.as_array()) {
            data.iter()
                .filter_map(|item| item.get("embedding"))
                .collect::<Vec<_>>()
        } else {
            vec![]
        };

        let total_duration = start_time.elapsed().as_nanos() as u64;

        let ollama_response = json!({
            "model": model,
            "embeddings": embeddings,
            "total_duration": total_duration,
            "load_duration": 1000000u64,
            "prompt_eval_count": 1
        });

        Ok(ollama_response)
    };

    let result = with_retry(&server, operation).await?;
    let duration = start_time.elapsed();

    server.logger.log(&format!("Ollama embeddings response completed (took {})", format_duration(duration)));
    Ok(json_response(&result))
}

/// Handle GET /api/ps - list running models
pub async fn handle_ollama_ps() -> Result<warp::reply::Response, ProxyError> {
    let response = json!({
        "models": []
    });
    Ok(json_response(&response))
}

/// Handle POST /api/show - show model info
pub async fn handle_ollama_show(body: Value) -> Result<warp::reply::Response, ProxyError> {
    let model = body.get("model").and_then(|m| m.as_str()).unwrap_or("unknown");
    let cleaned_model = clean_model_name(model);

    let architecture = match cleaned_model.to_lowercase() {
        name if name.contains("llama") => "llama",
        name if name.contains("mistral") => "mistral",
        name if name.contains("qwen") => "qwen",
        name if name.contains("deepseek") => "llama",
        name if name.contains("gemma") => "gemma",
        name if name.contains("phi") => "phi",
        name if name.contains("codellama") => "llama",
        name if name.contains("vicuna") => "llama",
        name if name.contains("alpaca") => "llama",
        _ => "llama",
    };

    let (family, parameter_size) = match cleaned_model.to_lowercase() {
        name if name.contains("7b") => ("llama", "7B"),
        name if name.contains("13b") => ("llama", "13B"),
        name if name.contains("30b") => ("llama", "30B"),
        name if name.contains("70b") => ("llama", "70B"),
        name if name.contains("8b") => ("llama", "8B"),
        name if name.contains("14b") => ("qwen", "14B"),
        name if name.contains("32b") => ("qwen", "32B"),
        name if name.contains("1.5b") => ("qwen", "1.5B"),
        name if name.contains("2b") => ("gemma", "2B"),
        name if name.contains("9b") => ("gemma", "9B"),
        name if name.contains("27b") => ("gemma", "27B"),
        _ => ("llama", "7B"),
    };

    let response = json!({
        "modelfile": format!("# Modelfile for {}\nFROM {}\nPARAMETER temperature 0.7\nPARAMETER top_p 0.9\nPARAMETER top_k 40", model, model),
        "parameters": "temperature 0.7\ntop_p 0.9\ntop_k 40\nrepeat_penalty 1.1",
        "template": "{{ if .System }}{{ .System }}\n{{ end }}{{ .Prompt }}",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": family,
            "families": [family],
            "parameter_size": parameter_size,
            "quantization_level": "Q4_K_M"
        },
        "model_info": {
            "general.architecture": architecture,
            "general.name": cleaned_model,
            "general.parameter_count": match parameter_size {
                _ => 7000000000u64,
            },
            "general.quantization_version": 2,
            "general.file_type": 2,
            "tokenizer.model": "llama",
            "tokenizer.chat_template": "{{ if .System }}{{ .System }}\n{{ end }}{{ .Prompt }}"
        },
        "capabilities": determine_model_capabilities(&cleaned_model),
        "system": format!("You are a helpful AI assistant using the {} model.", model),
        "license": "Custom license for proxy model",
        "digest": format!("{:x}", md5::compute(model.as_bytes())),
        "size": match parameter_size {
            "1.5B" => 1000000000u64,
            "2B" => 1500000000u64,
            "7B" => 4000000000u64,
            "8B" => 5000000000u64,
            "9B" => 5500000000u64,
            "13B" => 8000000000u64,
            "14B" => 8500000000u64,
            "27B" => 16000000000u64,
            "30B" => 18000000000u64,
            "32B" => 20000000000u64,
            "70B" => 40000000000u64,
            _ => 4000000000u64,
        },
        "modified_at": chrono::Utc::now().to_rfc3339()
    });

    Ok(json_response(&response))
}

/// Handle GET /api/version - return version info
pub async fn handle_ollama_version() -> Result<warp::reply::Response, ProxyError> {
    let response = json!({
        "version": "0.5.1-proxy"
    });
    Ok(json_response(&response))
}

/// Handle unsupported endpoints
pub async fn handle_unsupported(endpoint: &str) -> Result<warp::reply::Response, ProxyError> {
    Err(ProxyError::not_implemented(&format!(
        "The '{}' endpoint is not supported by this proxy. This endpoint requires direct Ollama functionality that cannot be translated to LM Studio.",
        endpoint
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_name_cleaning_in_handlers() {
        let test_cases = vec![
            ("llama3.2:latest", "llama3.2"),
            ("deepseek-r1:2", "deepseek-r1"),
            ("model-name:3", "model-name"),
            ("simple-model", "simple-model"),
        ];

        for (input, expected) in test_cases {
            assert_eq!(clean_model_name(input), expected);
        }
    }
}