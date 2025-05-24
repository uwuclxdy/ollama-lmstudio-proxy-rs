use clap::Parser;
use reqwest::Client;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::time::Instant;
use warp::{Filter, Reply};

#[derive(Parser, Debug)]
#[command(name = "ollama-lmstudio-proxy")]
#[command(about = "A reliable proxy server that bridges Ollama API to LM Studio")]
struct Args {
    #[arg(long, default_value = "0.0.0.0:11434")]
    listen: String,

    #[arg(long, default_value = "http://localhost:1234")]
    lmstudio_url: String,

    #[arg(long)]
    no_log: bool,

    #[arg(long, default_value = "5")]
    load_timeout_seconds: u64,
}

#[derive(Clone)]
struct ProxyServer {
    client: Client,
    lmstudio_url: String,
    logging: bool,
    load_timeout_seconds: u64,
}

// Custom error type for better error handling
#[derive(Debug)]
struct ProxyError {
    message: String,
    status_code: u16,
}

impl ProxyError {
    fn new(message: &str, status_code: u16) -> Self {
        Self {
            message: message.to_string(),
            status_code,
        }
    }

    fn internal_server_error(message: &str) -> Self {
        Self::new(message, 500)
    }

    fn bad_request(message: &str) -> Self {
        Self::new(message, 400)
    }

    fn not_found(message: &str) -> Self {
        Self::new(message, 404)
    }

    fn not_implemented(message: &str) -> Self {
        Self::new(message, 501)
    }
}

impl std::fmt::Display for ProxyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ProxyError {}

impl warp::reject::Reject for ProxyError {}

impl ProxyServer {
    fn new(lmstudio_url: String, logging: bool, load_timeout_seconds: u64) -> Self {
        Self {
            client: Client::new(),
            lmstudio_url,
            logging,
            load_timeout_seconds,
        }
    }

    fn log(&self, message: &str) {
        if self.logging {
            println!("[PROXY] {}", message);
        }
    }

    fn clean_model_name(&self, name: &str) -> String {
        // Remove ":latest" and ":number" suffixes
        let cleaned = name.replace(":latest", "");
        if let Some(colon_pos) = cleaned.rfind(':') {
            let suffix = &cleaned[colon_pos + 1..];
            if suffix.chars().all(|c| c.is_ascii_digit()) {
                return cleaned[..colon_pos].to_string();
            }
        }
        cleaned
    }

    // Check if the error response indicates no models are loaded
    fn is_no_models_loaded_error(&self, response_text: &str) -> bool {
        let response_text_lower = response_text.to_lowercase();
        response_text_lower.contains("no models loaded") ||
            response_text_lower.contains("model_not_found") ||
            response_text_lower.contains("please load a model")
    }

    // Attempt to trigger model loading by making a request to /v1/models
    async fn trigger_model_loading(&self) -> Result<bool, ProxyError> {
        self.log("Attempting to trigger model loading...");

        let url = format!("{}/v1/models", self.lmstudio_url);

        // First, try to get models list which might trigger loading
        match self.client.get(&url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    if let Ok(json_response) = response.json::<Value>().await {
                        if let Some(data) = json_response.get("data") {
                            if let Some(models) = data.as_array() {
                                if !models.is_empty() {
                                    self.log("Models are now available!");
                                    return Ok(true);
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                self.log(&format!("Failed to check models: {}", e));
            }
        }

        // If no models found, wait a bit and try again
        self.log("No models found, waiting for auto-loading...");
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        // Try again after waiting
        match self.client.get(&url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    if let Ok(json_response) = response.json::<Value>().await {
                        if let Some(data) = json_response.get("data") {
                            if let Some(models) = data.as_array() {
                                if !models.is_empty() {
                                    self.log("Models loaded successfully!");
                                    return Ok(true);
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                self.log(&format!("Failed to check models after waiting: {}", e));
            }
        }

        Ok(false)
    }

    // Enhanced passthrough with auto-retry on model loading error
    async fn passthrough_to_lmstudio_with_retry(&self, method: &str, endpoint: &str, body: Value) -> Result<warp::reply::Response, ProxyError> {
        // First attempt
        match self.passthrough_to_lmstudio(method, endpoint, body.clone()).await {
            Ok(response) => Ok(response),
            Err(error) => {
                // Check if this is a "no models loaded" error
                if self.is_no_models_loaded_error(&error.message) {
                    self.log("Detected 'no models loaded' error, attempting to trigger model loading...");

                    // Try to trigger model loading
                    match self.trigger_model_loading().await {
                        Ok(true) => {
                            self.log("Model loading successful, retrying original request...");

                            // Retry the original request
                            match self.passthrough_to_lmstudio(method, endpoint, body).await {
                                Ok(response) => {
                                    self.log("Retry successful!");
                                    Ok(response)
                                }
                                Err(retry_error) => {
                                    self.log(&format!("Retry failed: {}", retry_error));
                                    Err(retry_error)
                                }
                            }
                        }
                        Ok(false) => {
                            self.log("Could not load models automatically");
                            Err(ProxyError::internal_server_error(
                                "No models loaded in LM Studio and auto-loading failed. Please load a model manually."
                            ))
                        }
                        Err(load_error) => {
                            self.log(&format!("Model loading attempt failed: {}", load_error));
                            Err(error) // Return original error
                        }
                    }
                } else {
                    // Not a model loading error, return original error
                    Err(error)
                }
            }
        }
    }

    // Original passthrough function (unchanged)
    async fn passthrough_to_lmstudio(&self, method: &str, endpoint: &str, body: Value) -> Result<warp::reply::Response, ProxyError> {
        let start_time = Instant::now();
        let url = format!("{}{}", self.lmstudio_url, endpoint);

        self.log(&format!("{} {} - Request: {}", method, url,
                          if body.is_object() && !body.as_object().unwrap().is_empty() {
                              serde_json::to_string_pretty(&body).unwrap_or_else(|_| "{}".to_string())
                          } else {
                              "{}".to_string()
                          }
        ));

        // Build request based on method
        let request_builder = match method {
            "GET" => self.client.get(&url),
            "POST" => self.client.post(&url).json(&body),
            _ => return Err(ProxyError::bad_request("Unsupported HTTP method")),
        };

        // Execute request
        let response = request_builder
            .send()
            .await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to connect to LM Studio: {}", e)))?;

        // Check response status
        if !response.status().is_success() {
            let status = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            self.log(&format!("LM Studio error ({}): {}", status, error_text));
            return Err(ProxyError::new(&error_text, status));
        }

        // Check if this is a streaming response
        let content_type = response.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/json");

        let is_streaming = content_type.contains("text/event-stream") ||
            content_type.contains("text/plain") ||
            body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

        if is_streaming {
            self.handle_streaming_response(response, start_time).await
        } else {
            self.handle_json_response(response, start_time).await
        }
    }

    async fn handle_streaming_response(&self, response: reqwest::Response, start_time: Instant) -> Result<warp::reply::Response, ProxyError> {
        self.log("Handling streaming response from LM Studio");

        // Convert response to text stream
        let text = response.text().await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to read streaming response: {}", e)))?;

        self.log(&format!("Streaming response received (took {:?})", start_time.elapsed()));

        // Return the streaming response as plain text with appropriate headers
        let reply = warp::reply::with_header(
            warp::reply::with_header(
                warp::reply::with_header(
                    text,
                    "content-type",
                    "text/event-stream"
                ),
                "cache-control",
                "no-cache"
            ),
            "connection",
            "keep-alive"
        );

        Ok(reply.into_response())
    }

    async fn handle_json_response(&self, response: reqwest::Response, start_time: Instant) -> Result<warp::reply::Response, ProxyError> {
        // Handle JSON response
        let json_response: Value = response.json().await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to parse JSON response: {}", e)))?;

        let duration = start_time.elapsed();
        self.log(&format!("LM Studio response: {} (took {:?})",
                          serde_json::to_string_pretty(&json_response).unwrap_or_else(|_| "{}".to_string()),
                          duration
        ));

        Ok(warp::reply::json(&json_response).into_response())
    }

    // Enhanced Ollama API translation functions with retry logic
    async fn handle_ollama_tags(&self) -> Result<warp::reply::Response, ProxyError> {
        match self.handle_ollama_tags_internal().await {
            Ok(response) => Ok(response),
            Err(error) => {
                if self.is_no_models_loaded_error(&error.message) {
                    self.log("No models loaded for tags request, attempting to trigger loading...");

                    if self.trigger_model_loading().await.unwrap_or(false) {
                        self.log("Model loading successful, retrying tags request...");
                        self.handle_ollama_tags_internal().await
                    } else {
                        Err(error)
                    }
                } else {
                    Err(error)
                }
            }
        }
    }

    async fn handle_ollama_tags_internal(&self) -> Result<warp::reply::Response, ProxyError> {
        let start_time = Instant::now();
        let url = format!("{}/v1/models", self.lmstudio_url);

        self.log(&format!("GET {}", url));

        let response = self.client.get(&url).send().await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to connect to LM Studio: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProxyError::internal_server_error(&format!("LM Studio error: {}", error_text)));
        }

        let lm_response: Value = response.json().await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to parse LM Studio response: {}", e)))?;

        // Transform to Ollama format
        let models = lm_response["data"].as_array().unwrap_or(&vec![])
            .iter()
            .map(|model| {
                let name = model["id"].as_str().unwrap_or("unknown");
                let clean_name = self.clean_model_name(name);
                json!({
                    "name": format!("{}:latest", clean_name),
                    "model": format!("{}:latest", clean_name),
                    "modified_at": chrono::Utc::now().to_rfc3339(),
                    "size": 1000000000i64,
                    "digest": format!("sha256:{:x}", md5::compute(name.as_bytes())),
                    "details": {
                        "parent_model": "",
                        "format": "gguf",
                        "family": "llama",
                        "families": ["llama"],
                        "parameter_size": "7B",
                        "quantization_level": "Q4_K_M"
                    }
                })
            })
            .collect::<Vec<_>>();

        let result = json!({ "models": models });
        let duration = start_time.elapsed();

        self.log(&format!("Ollama tags response (took {:?})", duration));
        Ok(warp::reply::json(&result).into_response())
    }

    async fn handle_ollama_chat(&self, body: Value) -> Result<warp::reply::Response, ProxyError> {
        match self.handle_ollama_chat_internal(body.clone()).await {
            Ok(response) => Ok(response),
            Err(error) => {
                if self.is_no_models_loaded_error(&error.message) {
                    self.log("No models loaded for chat request, attempting to trigger loading...");

                    if self.trigger_model_loading().await.unwrap_or(false) {
                        self.log("Model loading successful, retrying chat request...");
                        self.handle_ollama_chat_internal(body).await
                    } else {
                        Err(error)
                    }
                } else {
                    Err(error)
                }
            }
        }
    }

    async fn handle_ollama_chat_internal(&self, body: Value) -> Result<warp::reply::Response, ProxyError> {
        let start_time = Instant::now();
        let model_name = body["model"].as_str().unwrap_or("").replace(":latest", "");
        let clean_model = self.clean_model_name(&model_name);

        // Transform to LM Studio format
        let lm_request = json!({
            "model": clean_model,
            "messages": body["messages"],
            "temperature": body["temperature"].as_f64().unwrap_or(0.7),
            "max_tokens": body["max_tokens"].as_u64().unwrap_or(2048),
            "stream": body["stream"].as_bool().unwrap_or(false),
            "top_p": body["top_p"].as_f64().unwrap_or(1.0),
            "frequency_penalty": body["frequency_penalty"].as_f64().unwrap_or(0.0),
            "presence_penalty": body["presence_penalty"].as_f64().unwrap_or(0.0)
        });

        let url = format!("{}/v1/chat/completions", self.lmstudio_url);
        self.log(&format!("POST {} - Ollama chat request", url));

        let response = self.client.post(&url)
            .json(&lm_request)
            .send()
            .await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to connect to LM Studio: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProxyError::internal_server_error(&format!("LM Studio error: {}", error_text)));
        }

        let lm_response: Value = response.json().await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to parse LM Studio response: {}", e)))?;

        // Transform back to Ollama format
        let duration = start_time.elapsed();
        let choice = &lm_response["choices"][0];
        let message = &choice["message"];

        // Combine content and reasoning_content if present
        let mut content = message["content"].as_str().unwrap_or("").to_string();
        if let Some(reasoning) = message["reasoning_content"].as_str() {
            if !reasoning.is_empty() {
                content = format!("{}\n{}", reasoning, content);
            }
        }

        let total_duration_ns = duration.as_nanos() as u64;
        let prompt_tokens = lm_response["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
        let completion_tokens = lm_response["usage"]["completion_tokens"].as_u64().unwrap_or(0);

        let result = json!({
            "model": format!("{}:latest", clean_model),
            "created_at": chrono::Utc::now().to_rfc3339(),
            "message": {
                "role": "assistant",
                "content": content
            },
            "done": true,
            "total_duration": total_duration_ns,
            "load_duration": total_duration_ns / 10,
            "prompt_eval_count": prompt_tokens,
            "prompt_eval_duration": total_duration_ns / 4,
            "eval_count": completion_tokens,
            "eval_duration": total_duration_ns / 2
        });

        self.log(&format!("Ollama chat response (took {:?})", duration));
        Ok(warp::reply::json(&result).into_response())
    }

    async fn handle_ollama_generate(&self, body: Value) -> Result<warp::reply::Response, ProxyError> {
        match self.handle_ollama_generate_internal(body.clone()).await {
            Ok(response) => Ok(response),
            Err(error) => {
                if self.is_no_models_loaded_error(&error.message) {
                    self.log("No models loaded for generate request, attempting to trigger loading...");

                    if self.trigger_model_loading().await.unwrap_or(false) {
                        self.log("Model loading successful, retrying generate request...");
                        self.handle_ollama_generate_internal(body).await
                    } else {
                        Err(error)
                    }
                } else {
                    Err(error)
                }
            }
        }
    }

    async fn handle_ollama_generate_internal(&self, body: Value) -> Result<warp::reply::Response, ProxyError> {
        let start_time = Instant::now();
        let model_name = body["model"].as_str().unwrap_or("").replace(":latest", "");
        let clean_model = self.clean_model_name(&model_name);

        let lm_request = json!({
            "model": clean_model,
            "prompt": body["prompt"],
            "temperature": body["temperature"].as_f64().unwrap_or(0.7),
            "max_tokens": body["max_tokens"].as_u64().unwrap_or(2048),
            "stream": body["stream"].as_bool().unwrap_or(false),
            "top_p": body["top_p"].as_f64().unwrap_or(1.0),
            "frequency_penalty": body["frequency_penalty"].as_f64().unwrap_or(0.0),
            "presence_penalty": body["presence_penalty"].as_f64().unwrap_or(0.0)
        });

        let url = format!("{}/v1/completions", self.lmstudio_url);
        self.log(&format!("POST {} - Ollama generate request", url));

        let response = self.client.post(&url)
            .json(&lm_request)
            .send()
            .await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to connect to LM Studio: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProxyError::internal_server_error(&format!("LM Studio error: {}", error_text)));
        }

        let lm_response: Value = response.json().await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to parse LM Studio response: {}", e)))?;

        let duration = start_time.elapsed();
        let choice = &lm_response["choices"][0];
        let content = choice["text"].as_str().unwrap_or("");

        let total_duration_ns = duration.as_nanos() as u64;
        let prompt_tokens = lm_response["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
        let completion_tokens = lm_response["usage"]["completion_tokens"].as_u64().unwrap_or(0);

        let result = json!({
            "model": format!("{}:latest", clean_model),
            "created_at": chrono::Utc::now().to_rfc3339(),
            "response": content,
            "done": true,
            "context": [1, 2, 3],
            "total_duration": total_duration_ns,
            "load_duration": total_duration_ns / 10,
            "prompt_eval_count": prompt_tokens,
            "prompt_eval_duration": total_duration_ns / 4,
            "eval_count": completion_tokens,
            "eval_duration": total_duration_ns / 2
        });

        self.log(&format!("Ollama generate response (took {:?})", duration));
        Ok(warp::reply::json(&result).into_response())
    }

    async fn handle_ollama_embeddings(&self, body: Value) -> Result<warp::reply::Response, ProxyError> {
        match self.handle_ollama_embeddings_internal(body.clone()).await {
            Ok(response) => Ok(response),
            Err(error) => {
                if self.is_no_models_loaded_error(&error.message) {
                    self.log("No models loaded for embeddings request, attempting to trigger loading...");

                    if self.trigger_model_loading().await.unwrap_or(false) {
                        self.log("Model loading successful, retrying embeddings request...");
                        self.handle_ollama_embeddings_internal(body).await
                    } else {
                        Err(error)
                    }
                } else {
                    Err(error)
                }
            }
        }
    }

    async fn handle_ollama_embeddings_internal(&self, body: Value) -> Result<warp::reply::Response, ProxyError> {
        let start_time = Instant::now();
        let model_name = body["model"].as_str().unwrap_or("").replace(":latest", "");
        let clean_model = self.clean_model_name(&model_name);

        let lm_request = json!({
            "model": clean_model,
            "input": body["input"]
        });

        let url = format!("{}/v1/embeddings", self.lmstudio_url);
        self.log(&format!("POST {} - Ollama embeddings request", url));

        let response = self.client.post(&url)
            .json(&lm_request)
            .send()
            .await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to connect to LM Studio: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProxyError::internal_server_error(&format!("LM Studio embeddings error: {}", error_text)));
        }

        let lm_response: Value = response.json().await
            .map_err(|e| ProxyError::internal_server_error(&format!("Failed to parse LM Studio response: {}", e)))?;

        let duration = start_time.elapsed();
        let embeddings: Vec<Vec<f64>> = lm_response["data"].as_array().unwrap_or(&vec![])
            .iter()
            .map(|item| {
                item["embedding"].as_array().unwrap_or(&vec![])
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0))
                    .collect()
            })
            .collect();

        let result = json!({
            "model": format!("{}:latest", clean_model),
            "embeddings": embeddings,
            "total_duration": duration.as_nanos() as u64,
            "load_duration": duration.as_nanos() as u64 / 10,
            "prompt_eval_count": body["input"].as_str().map(|s| s.len()).unwrap_or(0)
        });

        self.log(&format!("Ollama embeddings response (took {:?})", duration));
        Ok(warp::reply::json(&result).into_response())
    }

    async fn handle_ollama_ps(&self) -> Result<warp::reply::Response, ProxyError> {
        // Get models from LM Studio and treat them as running
        let result = json!({
            "models": []
        });

        self.log("PS response generated");
        Ok(warp::reply::json(&result).into_response())
    }

    async fn handle_ollama_show(&self, body: Value) -> Result<warp::reply::Response, ProxyError> {
        let model_name = body["model"].as_str().unwrap_or("").replace(":latest", "");
        let clean_model = self.clean_model_name(&model_name);

        let result = json!({
            "modelfile": format!("# Modelfile for {}\nFROM {}\n", clean_model, clean_model),
            "parameters": "temperature 0.7\ntop_p 0.9\nstop \"<|eot_id|>\"",
            "template": "{{ if .System }}{{ .System }}{{ end }}{{ if .Prompt }}{{ .Prompt }}{{ end }}{{ .Response }}",
            "details": {
                "format": "gguf",
                "family": "llama",
                "parameter_size": "7B"
            },
            "model_info": {
                "general.architecture": "llama",
                "general.parameter_count": 7000000000u64,
                "llama.context_length": 4096,
                "llama.embedding_length": 4096
            },
            "capabilities": ["completion", "chat"]
        });

        self.log(&format!("Show response for model: {}", clean_model));
        Ok(warp::reply::json(&result).into_response())
    }

    async fn handle_ollama_version(&self) -> Result<warp::reply::Response, ProxyError> {
        let result = json!({
            "version": "0.5.1-proxy"
        });

        self.log("Version response generated");
        Ok(warp::reply::json(&result).into_response())
    }

    async fn handle_unsupported(&self, endpoint: &str) -> Result<warp::reply::Response, ProxyError> {
        let error_msg = match endpoint {
            "/api/create" => "Model creation not supported via LM Studio proxy",
            "/api/pull" => "Model pulling not supported via LM Studio proxy",
            "/api/push" => "Model pushing not supported via LM Studio proxy",
            "/api/delete" => "Model deletion not supported via LM Studio proxy",
            "/api/copy" => "Model copying not supported via LM Studio proxy",
            _ => "Endpoint not supported via LM Studio proxy"
        };

        Err(ProxyError::not_implemented(error_msg))
    }
}

// Request handler
async fn handle_request(
    method: warp::http::Method,
    path: warp::path::FullPath,
    body: bytes::Bytes,
    proxy: ProxyServer,
) -> Result<warp::reply::Response, warp::Rejection> {
    let method_str = method.to_string();
    let path_str = path.as_str();

    proxy.log(&format!("{} {} - Body: {}", method_str, path_str,
                       if body.is_empty() {
                           "{}".to_string()
                       } else {
                           String::from_utf8_lossy(&body).to_string()
                       }
    ));

    let body_json: Value = if body.is_empty() {
        json!({})
    } else {
        serde_json::from_slice(&body).unwrap_or_else(|_| json!({}))
    };

    let result: Result<warp::reply::Response, ProxyError> = match (method_str.as_str(), path_str) {
        // Ollama API endpoints (translated to LM Studio format with retry logic)
        ("GET", "/api/tags") => proxy.handle_ollama_tags().await,
        ("POST", "/api/chat") => proxy.handle_ollama_chat(body_json).await,
        ("POST", "/api/generate") => proxy.handle_ollama_generate(body_json).await,
        ("POST", "/api/embed" | "/api/embeddings") => proxy.handle_ollama_embeddings(body_json).await,
        ("GET", "/api/ps") => proxy.handle_ollama_ps().await,
        ("POST", "/api/show") => proxy.handle_ollama_show(body_json).await,
        ("GET", "/api/version") => proxy.handle_ollama_version().await,

        // Unsupported Ollama endpoints
        (_, "/api/create" | "/api/pull" | "/api/push" | "/api/delete" | "/api/copy") => {
            proxy.handle_unsupported(path_str).await
        }

        // Direct LM Studio API endpoints (passthrough with retry logic)
        ("GET", "/v1/models") | ("POST", "/v1/chat/completions") | ("POST", "/v1/completions") | ("POST", "/v1/embeddings") => {
            proxy.passthrough_to_lmstudio_with_retry(&method_str, path_str, body_json).await
        }

        // Unknown endpoints
        _ => {
            proxy.log(&format!("Unknown endpoint: {} {}", method_str, path_str));
            Err(ProxyError::not_found("Unknown endpoint"))
        }
    };

    match result {
        Ok(response) => Ok(response),
        Err(proxy_error) => {
            proxy.log(&format!("Error: {}", proxy_error));
            let status_code = match proxy_error.status_code {
                400 => warp::http::StatusCode::BAD_REQUEST,
                404 => warp::http::StatusCode::NOT_FOUND,
                501 => warp::http::StatusCode::NOT_IMPLEMENTED,
                _ => warp::http::StatusCode::INTERNAL_SERVER_ERROR,
            };

            Ok(warp::reply::with_status(
                warp::reply::json(&json!({"error": proxy_error.message})),
                status_code,
            ).into_response())
        }
    }
}

// Custom rejection handler
async fn handle_rejection(err: warp::Rejection) -> Result<warp::reply::Response, Infallible> {
    if let Some(proxy_error) = err.find::<ProxyError>() {
        let status_code = match proxy_error.status_code {
            400 => warp::http::StatusCode::BAD_REQUEST,
            404 => warp::http::StatusCode::NOT_FOUND,
            501 => warp::http::StatusCode::NOT_IMPLEMENTED,
            _ => warp::http::StatusCode::INTERNAL_SERVER_ERROR,
        };

        Ok(warp::reply::with_status(
            warp::reply::json(&json!({"error": proxy_error.message})),
            status_code,
        ).into_response())
    } else {
        Ok(warp::reply::with_status(
            warp::reply::json(&json!({"error": "Internal server error"})),
            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
        ).into_response())
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    println!("🚀 Starting Enhanced Ollama-LMStudio Proxy Server");
    println!("📡 Listening on: {}", args.listen);
    println!("🎯 LM Studio URL: {}", args.lmstudio_url);
    println!("📝 Logging: {}", !args.no_log);
    println!("⏱️  Auto-load timeout: {} seconds", args.load_timeout_seconds);
    println!("🔄 Supporting both Ollama (/api/*) and LM Studio (/v1/*) endpoints");
    println!("🤖 Auto-retry on 'no models loaded' errors");

    let proxy = ProxyServer::new(args.lmstudio_url, !args.no_log, args.load_timeout_seconds);

    let routes = warp::any()
        .and(warp::method())
        .and(warp::path::full())
        .and(warp::body::bytes())
        .and_then(move |method, path, body| {
            let proxy = proxy.clone();
            handle_request(method, path, body, proxy)
        })
        .recover(handle_rejection);

    let addr: std::net::SocketAddr = args.listen
        .parse()
        .expect("Invalid listen address");

    println!("✅ Enhanced proxy server running on http://{}", addr);
    println!("💡 Test with: curl http://{}/api/tags", addr);
    println!("🔧 The proxy will automatically attempt to load models if none are available");

    warp::serve(routes).run(addr).await;
}
