# 🔧 Ollama-LM Studio Proxy - Project Outline

## 📋 Project Purpose

**Translation proxy** that bridges the gap between Ollama API clients and LM Studio backend, enabling Ollama-compatible
tools to work with LM Studio models.

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

- API format translation (Ollama ↔ OpenAI/LM Studio)
- Streaming support for real-time responses (SSE to JSON-lines)
- Auto-retry with model loading detection for LM Studio
- Enhanced response metadata for Ollama compatibility (timing, model details)
- **Advanced Request Cancellation:**
    - **Client Disconnection Detection:** Actively monitors client connections via `ConnectionTracker`.
    - **Cancellable Backend Requests:** HTTP requests to LM Studio (`CancellableRequest`) are aborted if the client
      disconnects.
    - **Cancellable Streaming:** Both Ollama-emulated and passthrough streaming operations respect cancellation tokens.
    - **Graceful Stream Termination:** Sends a specific cancellation message/chunk if a stream is cancelled.
    - **Cancellation-Aware Retries:** Retry logic for model loading also respects cancellation signals.
    - **HTTP 499 for Cancellations:** Uses status code 499 (Client Closed Request) for responses to cancelled requests.

## 📁 Project Structure

```
src/
├── main.rs              # Entry point & CLI setup (~20 lines)
├── lib.rs               # Module declarations & re-exports (~20 lines)
├── server.rs            # HTTP server, routing, cancellation tracking (~250 lines)
├── utils.rs             # Utilities, ProxyError, Logger, model name helpers (~200 lines)
└── handlers/            # Modular request handlers (~600+ lines)
    ├── cancelation_handlers.rs # Cancellation-aware request handlers
    ├── cancelation.rs   # Cancellation token management & request cancellation logic
    ├── helpers.rs       # Model metadata & response utilities
    ├── mod.rs           # Module organization & exports
    ├── retry.rs         # Auto-retry infrastructure with cancellation
    ├── streaming.rs     # Streaming response handling with cancellation
    ├── ollama.rs        # Ollama API endpoint handlers with cancellation
    └── lmstudio.rs      # LM Studio passthrough handlers with cancellation
```

## 🔌 API Endpoints

### Ollama API (Translated & Cancellable)

- `GET /api/tags` → `GET /v1/models` - List models
- `POST /api/chat` → `POST /v1/chat/completions` - Chat completion
- `POST /api/generate` → `POST /v1/completions` - Text completion
- `POST /api/embed[dings]` → `POST /v1/embeddings` - Generate embeddings
- `POST /api/show` - Model info (enhanced fake response)
- `GET /api/ps` - Running models (static empty response)
- `GET /api/version` - Version info (proxy version)

### LM Studio API (Passthrough & Cancellable)

- `GET /v1/models` - Direct passthrough
- `POST /v1/chat/completions` - Direct passthrough
- `POST /v1/completions` - Direct passthrough
- `POST /v1/embeddings` - Direct passthrough
- Other `/v1/*` endpoints can be passed through.

### Unsupported (501 responses)

- `/api/create`, `/api/pull`, `/api/push`, `/api/delete`, `/api/copy`

## 🔄 Key Components

### 1. Auto-Retry System (`handlers/retry.rs`)

```
trigger_model_loading_with_cancellation(
    server: &ProxyServer,
    cancellation_token: CancellationToken
) -> Result<bool, ProxyError>
// Calls GET /v1/models to wake up LM Studio and trigger model loading, respecting cancellation.

with_retry_and_cancellation<F, Fut, T>(
    server: &ProxyServer,
    operation: F,
    cancellation_token: CancellationToken
) -> Result<T, ProxyError>
// Generic retry wrapper that detects model loading errors and retries operations,
// respecting cancellation throughout the process (including waits).
```

### 2. Streaming Support (`handlers/streaming.rs`)

```
is_streaming_request(body: &Value) -> bool
// Check if request has "stream": true enabled.

handle_streaming_response_with_cancellation(
    response: reqwest::Response,
    is_chat: bool,
    model: &str,
    start_time: Instant,
    cancellation_token: CancellationToken
) -> Result<warp::reply::Response, ProxyError>
// Converts LM Studio SSE stream to Ollama format with proper chunking, final stats,
// and graceful handling of cancellation (sends cancellation chunk).

handle_passthrough_streaming_response_with_cancellation(
    response: reqwest::Response,
    cancellation_token: CancellationToken
) -> Result<warp::reply::Response, ProxyError>
// Direct passthrough of LM Studio streaming responses, cancellable.

convert_sse_to_ollama_chat(sse_message: &str, model: &str) -> Option<Value>
// Parses SSE data and converts to Ollama chat chunk format.

convert_sse_to_ollama_generate(sse_message: &str, model: &str) -> Option<Value>
// Parses SSE data and converts to Ollama generate chunk format.

create_cancellation_chunk(model, partial_content, duration, tokens, is_chat) -> Value
// Creates a specific JSON chunk to send when a stream is cancelled.

create_error_chunk(model, error_message, is_chat) -> Value
// Creates a specific JSON chunk for streaming errors.

create_final_chunk(model, duration, chunk_count, is_chat) -> Value
// Creates the final summary chunk for an Ollama stream.
```

### 3. Response Helpers (`handlers/helpers.rs`)

```
json_response(value: &Value) -> warp::reply::Response
// Convert JSON Value to proper HTTP Response with 200 OK status.

determine_model_family(model_name: &str) -> (&'static str, Vec<&'static str>)
// Detect model family (llama, mistral, qwen, etc.) from model name.

determine_parameter_size(model_name: &str) -> &'static str
// Extract parameter size (7B, 13B, 70B, etc.) from model name.

estimate_model_size(parameter_size: &str) -> u64
// Convert parameter size to estimated file size in bytes.

determine_model_capabilities(model_name: &str) -> Vec<&'static str>
// Determine model capabilities (chat, completion, embeddings, vision, tools).
```

### 4. Ollama API Handlers (`handlers/ollama.rs`)

Contains `CancellableRequest` struct (with `request_id`) for making cancellable HTTP calls.

```
handle_ollama_tags_with_cancellation(
    server: Arc<ProxyServer>,
    cancellation_token: CancellationToken
) -> Result<warp::reply::Response, ProxyError>
// GET /api/tags - List available models with Ollama-compatible metadata, cancellable.

handle_ollama_chat_with_cancellation(
    server: Arc<ProxyServer>,
    body: Value,
    cancellation_token: CancellationToken
) -> Result<warp::reply::Response, ProxyError>
// POST /api/chat - Chat completion with streaming/non-streaming support,
// reasoning integration, and cancellation.

handle_non_streaming_chat_response_with_cancellation(
    response: reqwest::Response,
    model: &str,
    messages: &[Value],
    start_time: Instant,
    cancellation_token: CancellationToken
) -> Result<warp::reply::Response, ProxyError>
// Process non-streaming chat responses, cancellable during response parsing.

handle_ollama_generate_with_cancellation(
    server: Arc<ProxyServer>,
    body: Value,
    cancellation_token: CancellationToken
) -> Result<warp::reply::Response, ProxyError>
// POST /api/generate - Text completion with streaming/non-streaming support and cancellation.

handle_non_streaming_generate_response_with_cancellation(
    response: reqwest::Response,
    model: &str,
    prompt: &str,
    start_time: Instant,
    cancellation_token: CancellationToken
) -> Result<warp::reply::Response, ProxyError>
// Process non-streaming completion responses, cancellable during response parsing.

handle_ollama_embeddings_with_cancellation(
    server: Arc<ProxyServer>,
    body: Value,
    cancellation_token: CancellationToken
) -> Result<warp::reply::Response, ProxyError>
// POST /api/embed[dings] - Generate text embeddings, cancellable.

// Also includes non-cancellable legacy versions and static handlers:
handle_ollama_ps() -> Result<warp::reply::Response, ProxyError>
handle_ollama_show(body: Value) -> Result<warp::reply::Response, ProxyError>
handle_ollama_version() -> Result<warp::reply::Response, ProxyError>
handle_unsupported(endpoint: &str) -> Result<warp::reply::Response, ProxyError>
```

### 5. LM Studio Passthrough (`handlers/lmstudio.rs`)

Contains `CancellableRequest` struct for making cancellable HTTP calls.

```
handle_lmstudio_passthrough_with_cancellation(
    server: Arc<ProxyServer>,
    method: &str,
    endpoint: &str,
    body: Value,
    cancellation_token: CancellationToken
) -> Result<warp::reply::Response, ProxyError>
// Direct passthrough for /v1/* endpoints with streaming/non-streaming support,
// retry logic, and cancellation.

handle_non_streaming_passthrough_response(
    response: reqwest::Response,
    cancellation_token: CancellationToken
) -> Result<warp::reply::Response, ProxyError>
// Handles non-streaming passthrough responses, cancellable during JSON parsing.
```

### 6. Utilities (`utils.rs`)

```
// ProxyError struct and related methods
ProxyError::request_cancelled() -> Self // Creates a 499 error
ProxyError::is_cancelled(&self) -> bool // Checks if error is due to cancellation

clean_model_name(name: &str) -> String
// Remove :latest suffix and numeric version tags (deepseek-r1:2 → deepseek-r1).

is_no_models_loaded_error(message: &str) -> bool
// Detect error messages indicating LM Studio needs model loading.

format_duration(duration: Duration) -> String
// Format Duration as human-readable string (1500ms → "1.50s").

validate_model_name(name: &str) -> (bool, Option<String>)
// Validate model name and return warnings for potential issues.
```

### 7. Server Core (`server.rs`)

```
// Config struct for CLI arguments

// CancellationTokenFactory for creating CancellationToken instances per request
CancellationTokenFactory::create_token(&self) -> CancellationToken

// ProxyServer struct with client, config, logger, cancellation_factory
ProxyServer::new(config: Config) -> Self
ProxyServer::run(self) -> Result<(), Box<dyn std::error::Error>>

// ConnectionTracker struct to monitor client connection and trigger cancellation on drop
ConnectionTracker::new(token: CancellationToken) -> Self
ConnectionTracker::mark_completed(&self) // Prevents cancellation if request finishes normally

handle_request_with_cancellation(
    server: Arc<ProxyServer>,
    method: String,
    path: String,
    body: Value
) -> Result<warp::reply::Response, Rejection>
// Main request router. Sets up ConnectionTracker, passes CancellationToken to handlers,
// and handles ProxyError::request_cancelled() specifically.

handle_rejection(err: Rejection) -> Result<impl Reply, Infallible>
// Convert rejections (including ProxyError) to proper HTTP error responses.
```

### 8. Response Enhancement Features

- **Reasoning Integration**: Merges `reasoning_content` with main content in chat responses (if provided by LM Studio).
- **Timing Estimates**: Calculates realistic `total_duration`, `eval_count`, `prompt_eval_count` for Ollama responses.
- **Metadata Generation**: Provides consistent digests, sizes, and model details for `/api/tags` and `/api/show`.
- **Format Translation**: Converts between LM Studio/OpenAI and Ollama response structures, including streaming chunks.
- **Cancellation Information**: Includes `cancelled: true` and `partial_response: bool` fields in the final chunk of a
  cancelled stream.
