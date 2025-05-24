# Ollama LM Studio Proxy - Project Outline

## Project Description
A Rust-based proxy server that bridges the Ollama API and LM Studio API, enabling Ollama clients to communicate with LM Studio backends. The key feature is comprehensive **cancellation support** - when clients disconnect, all related LM Studio requests are immediately cancelled to prevent resource waste.

## Key Features
- **API Translation**: Converts Ollama API calls to LM Studio format and vice versa
- **Streaming Support**: Handles both streaming and non-streaming responses
- **Client Disconnection Detection**: Automatically cancels LM Studio requests when clients disconnect
- **Auto-retry Logic**: Automatically triggers model loading in LM Studio when needed
- **Comprehensive Error Handling**: Proper error propagation and transformation

## File Structure
```
src/
├── main.rs                 # Entry point
├── lib.rs                  # Library root with re-exports
├── server.rs               # Main server implementation
├── utils.rs                # Utilities (errors, logging, model name handling)
├── common.rs               # Core cancellable request infrastructure
└── handlers/
    ├── mod.rs              # Handler module organization
    ├── retry.rs            # Auto-retry with model loading
    ├── streaming.rs        # Streaming response handling
    ├── lmstudio.rs         # LM Studio API passthrough
    ├── ollama.rs           # Ollama API handlers
    └── helpers.rs          # Response transformation utilities
```

## Core Components

### main.rs
**Purpose**: Application entry point
- `main()` - Parses CLI arguments using clap and starts the proxy server

### lib.rs
**Purpose**: Library root module
- Module declarations and public re-exports
- Exposes `Config`, `ProxyServer`, `ProxyError` publicly
- Version constants

### server.rs
**Purpose**: Main HTTP server with cancellation support

#### Key Structures:
- `Config` - CLI configuration (listen address, LM Studio URL, timeouts)
- `CancellationTokenFactory` - Creates cancellation tokens for each request
- `ProxyServer` - Main server instance with HTTP client and configuration
- `ConnectionTracker` - Tracks connection lifecycle and triggers cancellation on drop

#### Key Functions:
- `ProxyServer::new(config)` - Creates server instance
- `ProxyServer::run()` - Starts HTTP server on configured address
- `handle_request_with_cancellation()` - Routes requests to appropriate handlers with cancellation tokens
- `handle_rejection()` - Converts errors to proper HTTP responses

#### Request Routing:
- `/api/*` → Ollama API handlers (translated to LM Studio)
- `/v1/*` → LM Studio API direct passthrough
- Unsupported endpoints return proper error responses

### utils.rs
**Purpose**: Core utilities and error handling

#### Key Structures:
- `ProxyError` - Custom error type with status codes and cancellation detection
- `Logger` - Simple logging utility with enable/disable support

#### Key Functions:
- `ProxyError::*()` - Various error constructors (bad_request, not_found, request_cancelled, etc.)
- `clean_model_name(name)` - Removes `:latest` and numeric suffixes from model names
- `is_no_models_loaded_error(message)` - Detects LM Studio "no model loaded" errors
- `format_duration(duration)` - Formats durations for logging
- `validate_model_name(name)` - Validates model names and returns warnings

### common.rs
**Purpose**: Core cancellable request infrastructure

#### Key Structures:
- `CancellableRequest` - Wrapper for HTTP requests that can be cancelled mid-flight

#### Key Functions:
- `CancellableRequest::new()` - Creates request wrapper with cancellation token
- `CancellableRequest::make_request()` - Makes HTTP request that can be cancelled via tokio::select!
- `handle_cancellable_json_response()` - Parses JSON responses with cancellation support
- `handle_cancellable_text_response()` - Parses text responses with cancellation support

### handlers/retry.rs
**Purpose**: Auto-retry logic with model loading

#### Key Functions:
- `trigger_model_loading(server, token)` - Calls /v1/models to trigger LM Studio model loading
- `with_retry_and_cancellation(server, operation, token)` - Generic retry wrapper that:
  - Tries operation once
  - If "no models loaded" error, triggers model loading and retries
  - Handles cancellation at every step

### handlers/streaming.rs
**Purpose**: Streaming response handling with cancellation

#### Key Functions:
- `is_streaming_request(body)` - Checks if request has `"stream": true`
- `handle_streaming_response()` - Converts LM Studio SSE to Ollama streaming format with cancellation
- `handle_passthrough_streaming_response()` - Direct SSE passthrough with cancellation
- `convert_sse_to_ollama_chat()` - Converts LM Studio chat SSE to Ollama format
- `convert_sse_to_ollama_generate()` - Converts LM Studio completion SSE to Ollama format

#### Streaming Process:
1. Spawns async task to process LM Studio SSE stream
2. Uses `tokio::select!` to handle either new chunks or cancellation
3. On cancellation, sends graceful cancellation chunk and stops processing
4. Tracks partial content for meaningful cancellation messages

### handlers/lmstudio.rs
**Purpose**: Direct LM Studio API passthrough

#### Key Functions:
- `handle_lmstudio_passthrough(server, method, endpoint, body, token)` - Direct API passthrough with:
  - Request method conversion (GET/POST/PUT/DELETE)
  - Streaming vs non-streaming detection
  - Cancellation support throughout
  - Auto-retry integration via `with_retry_and_cancellation`

### handlers/ollama.rs
**Purpose**: Ollama API handlers that translate to LM Studio format

#### Key Functions:
- `handle_ollama_tags()` - GET /api/tags → /v1/models (lists available models)
- `handle_ollama_chat()` - POST /api/chat → /v1/chat/completions (chat with streaming support)
- `handle_ollama_generate()` - POST /api/generate → /v1/completions (text completion with streaming)
- `handle_ollama_embeddings()` - POST /api/embeddings → /v1/embeddings (text embeddings)
- `handle_ollama_show()` - POST /api/show (model info, static response)
- `handle_ollama_ps()` - GET /api/ps (running models, returns empty list)
- `handle_ollama_version()` - GET /api/version (version info)
- `handle_unsupported()` - Handles unsupported endpoints (create, pull, push, delete, copy)

#### Translation Process:
1. Extract Ollama request parameters
2. Clean and transform model names
3. Convert request format to LM Studio API
4. Make cancellable request to LM Studio
5. Transform response back to Ollama format
6. Handle streaming vs non-streaming appropriately

### handlers/helpers.rs
**Purpose**: Response transformation and utility functions

#### Response Transformation:
- `transform_chat_response()` - LM Studio chat → Ollama chat format
- `transform_generate_response()` - LM Studio completion → Ollama generate format
- `transform_embeddings_response()` - LM Studio embeddings → Ollama embeddings format

#### Model Analysis:
- `determine_model_family()` - Detects model family (llama, mistral, qwen, etc.)
- `determine_parameter_size()` - Extracts parameter size (7B, 13B, etc.) from model name
- `estimate_model_size()` - Estimates model size in bytes
- `determine_model_capabilities()` - Determines model capabilities (chat, completion, embeddings, vision, tools)

#### Streaming Utilities:
- `extract_content_from_chunk()` - Extracts text content from streaming chunks
- `create_error_chunk()` - Creates error response chunks for streaming
- `create_cancellation_chunk()` - Creates graceful cancellation response with partial content info
- `create_final_chunk()` - Creates completion chunks for streaming responses

#### General Utilities:
- `json_response()` - Creates JSON HTTP responses

## Cancellation Architecture

The entire system is built around **CancellationToken** from `tokio_util::sync`:

1. **Connection Tracking**: `ConnectionTracker` automatically cancels requests when dropped (client disconnect)
2. **Request Cancellation**: All HTTP requests use `tokio::select!` to race between completion and cancellation
3. **Streaming Cancellation**: Streaming handlers monitor cancellation token and send graceful termination
4. **Retry Cancellation**: Retry logic checks cancellation at every step
5. **Graceful Handling**: Cancelled requests return proper HTTP 499 responses

## Key Design Patterns

- **Unified Error Handling**: All functions return `Result<T, ProxyError>` with consistent error types
- **Cancellation-First Design**: Every async operation can be cancelled cleanly
- **Generic Retry Logic**: `with_retry_and_cancellation` wraps any operation with auto-retry
- **Streaming Abstraction**: Common streaming utilities handle both chat and completion formats
- **Model Name Normalization**: Consistent model name cleaning across all handlers

## Dependencies
- `tokio` - Async runtime
- `warp` - HTTP server framework
- `reqwest` - HTTP client
- `serde_json` - JSON handling
- `tokio-util` - Cancellation tokens
- `clap` - CLI argument parsing
- `chrono` - Timestamp handling
