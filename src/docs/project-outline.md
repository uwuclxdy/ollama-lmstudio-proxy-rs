# 🔧 Ollama-LM Studio Proxy - Project Outline

## 📋 Project Purpose

**Translation proxy** that bridges the gap between Ollama API clients and LM Studio backend, enabling Ollama-compatible tools to work with LM Studio models.

- **Input**: Ollama API requests (`/api/*`)
- **Translation**: Converts to OpenAI/LM Studio format (`/v1/*`)
- **Output**: Ollama-compatible responses with proper metadata
- **Passthrough**: Direct LM Studio API access via `/v1/*` endpoints

## 🏗️ Architecture Overview

```
[Client] → [Proxy Server] → [LM Studio]
         ↑               ↓
    [API Translation] [Auto-Retry] [Response Enhancement]
```

**Core Features:**
- API format translation (Ollama ↔ OpenAI)
- Streaming support for real-time responses
- Auto-retry with model loading detection
- Enhanced response metadata for compatibility

## 📁 Project Structure

```
src/
├── main.rs              # Entry point & CLI setup (~20 lines)
├── lib.rs               # Module declarations (~20 lines)
├── server.rs            # HTTP server & routing (~200 lines)
├── utils.rs             # Utilities & error handling (~150 lines)
└── handlers/            # Modular request handlers (~400+ lines)
    ├── mod.rs           # Module organization & exports
    ├── retry.rs         # Auto-retry infrastructure
    ├── streaming.rs     # Streaming response handling
    ├── helpers.rs       # Model metadata utilities
    ├── ollama.rs        # Ollama API endpoint handlers
    └── lmstudio.rs      # LM Studio passthrough handlers
```

## 🔌 API Endpoints

### Ollama API (Translated)
- `GET /api/tags` → `GET /v1/models` - List models
- `POST /api/chat` → `POST /v1/chat/completions` - Chat completion
- `POST /api/generate` → `POST /v1/completions` - Text completion
- `POST /api/embed[dings]` → `POST /v1/embeddings` - Generate embeddings
- `POST /api/show` - Model info (fake response)
- `GET /api/ps` - Running models (empty response)
- `GET /api/version` - Version info

### LM Studio API (Passthrough)
- `GET /v1/models` - Direct passthrough
- `POST /v1/chat/completions` - Direct passthrough
- `POST /v1/completions` - Direct passthrough
- `POST /v1/embeddings` - Direct passthrough

### Unsupported (501 responses)
- `/api/create`, `/api/pull`, `/api/push`, `/api/delete`, `/api/copy`

## 🔄 Key Components

### 1. Auto-Retry System (`handlers/retry.rs`)
```
trigger_model_loading(server: &ProxyServer) -> Result<bool, ProxyError>
// Calls GET /v1/models to wake up LM Studio and trigger model loading

with_retry<F, T, Fut>(server: &ProxyServer, operation: F) -> Result<T, ProxyError>
// Generic retry wrapper that detects model loading errors and retries operations
```

### 2. Streaming Support (`handlers/streaming.rs`)
```
is_streaming_request(body: &Value) -> bool
// Check if request has "stream": true enabled

handle_streaming_response(response, is_chat, model, start_time) -> Result<Response, ProxyError>
// Convert LM Studio SSE stream to Ollama format with proper chunking and final stats

handle_passthrough_streaming_response(response) -> Result<Response, ProxyError>
// Direct passthrough of LM Studio streaming responses (for /v1/* endpoints)

convert_sse_to_ollama_chat(sse_message: &str, model: &str) -> Option<Value>
// Parse SSE data and convert to Ollama chat chunk format

convert_sse_to_ollama_generate(sse_message: &str, model: &str) -> Option<Value>
// Parse SSE data and convert to Ollama generate chunk format
```

### 3. Response Helpers (`handlers/helpers.rs`)
```
json_response(value: &Value) -> warp::reply::Response
// Convert JSON Value to proper HTTP Response with 200 status

determine_model_family(model_name: &str) -> (&'static str, Vec<&'static str>)
// Detect model family (llama, mistral, qwen, etc.) from model name

determine_parameter_size(model_name: &str) -> &'static str
// Extract parameter size (7B, 13B, 70B, etc.) from model name

estimate_model_size(parameter_size: &str) -> u64
// Convert parameter size to estimated file size in bytes

determine_model_capabilities(model_name: &str) -> Vec<&'static str>
// Determine model capabilities (chat, completion, embeddings, vision, tools)
```

### 4. Ollama API Handlers (`handlers/ollama.rs`)
```
handle_ollama_tags(server: ProxyServer) -> Result<Response, ProxyError>
// GET /api/tags - List available models with Ollama-compatible metadata

handle_ollama_chat(server: ProxyServer, body: Value) -> Result<Response, ProxyError>
// POST /api/chat - Chat completion with streaming support and reasoning integration

handle_non_streaming_chat_response(response, model, messages, start_time) -> Result<Response, ProxyError>
// Process non-streaming chat responses from LM Studio to Ollama format

handle_ollama_generate(server: ProxyServer, body: Value) -> Result<Response, ProxyError>
// POST /api/generate - Text completion with streaming support

handle_non_streaming_generate_response(response, model, prompt, start_time) -> Result<Response, ProxyError>
// Process non-streaming completion responses from LM Studio to Ollama format

handle_ollama_embeddings(server: ProxyServer, body: Value) -> Result<Response, ProxyError>
// POST /api/embed[dings] - Generate text embeddings

handle_ollama_ps() -> Result<Response, ProxyError>
// GET /api/ps - Return empty running models list (compatibility)

handle_ollama_show(body: Value) -> Result<Response, ProxyError>
// POST /api/show - Return fake but realistic model information

handle_ollama_version() -> Result<Response, ProxyError>
// GET /api/version - Return proxy version info

handle_unsupported(endpoint: &str) -> Result<Response, ProxyError>
// Return 501 Not Implemented for unsupported Ollama endpoints
```

### 5. LM Studio Passthrough (`handlers/lmstudio.rs`)
```
handle_lmstudio_passthrough(server, method, endpoint, body) -> Result<Response, ProxyError>
// Direct passthrough for /v1/* endpoints with streaming support and retry logic
```

### 6. Utilities (`utils.rs`)
```
clean_model_name(name: &str) -> String
// Remove :latest suffix and numeric version tags (deepseek-r1:2 → deepseek-r1)

is_no_models_loaded_error(message: &str) -> bool
// Detect error messages indicating LM Studio needs model loading

format_duration(duration: Duration) -> String
// Format Duration as human-readable string (1500ms → "1.50s")

validate_model_name(name: &str) -> (bool, Option<String>)
// Validate model name and return warnings for potential issues
```

### 7. Server Core (`server.rs`)
```
ProxyServer::new(config: Config) -> Self
// Initialize proxy server with HTTP client and logger

ProxyServer::run(self) -> Result<(), Box<dyn std::error::Error>>
// Start HTTP server and handle requests with routing

handle_request(server, method, path, body) -> Result<Response, Rejection>
// Main request router that dispatches to appropriate handlers

handle_rejection(err: Rejection) -> Result<impl Reply, Infallible>
// Convert rejections to proper HTTP error responses (Ollama vs standard format)
```

### 8. Response Enhancement Features
- **Reasoning Integration**: Merges `reasoning_content` with main content in chat responses
- **Timing Estimates**: Calculates realistic `total_duration`, `eval_count`, `prompt_eval_count`
- **Metadata Generation**: Provides fake but consistent digests, sizes, and model details
- **Format Translation**: Converts between OpenAI and Ollama response structures
