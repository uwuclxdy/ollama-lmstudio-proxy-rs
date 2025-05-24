# Ollama-LMStudio Proxy - Project Outline

## Project Description

A high-performance Rust-based proxy server that acts as a bridge between Ollama API clients and LM Studio backend. The
proxy translates Ollama API requests into LM Studio-compatible formats while maintaining full compatibility with Ollama
clients. Key features include request cancellation on client disconnect, streaming response support, automatic retry
with model loading, and comprehensive error handling.

## Architecture Overview

The proxy operates by:

1. Accepting Ollama API requests on standard endpoints (`/api/*`)
2. Translating request/response formats between Ollama and LM Studio APIs
3. Forwarding requests to LM Studio backend (`/v1/*` endpoints)
4. Supporting direct LM Studio API passthrough for advanced use cases
5. Managing connection lifecycles with graceful cancellation

## File Structure

```
src/
├── main.rs                    # Entry point
├── lib.rs                     # Library root with re-exports
├── server.rs                  # Main server implementation
├── common.rs                  # Shared infrastructure for cancellable operations
├── utils.rs                   # Utility functions and error handling
└── handlers/
    ├── mod.rs                 # Handler module organization
    ├── helpers.rs             # Response transformation and utility functions
    ├── ollama.rs              # Ollama API endpoint handlers
    ├── lmstudio.rs            # LM Studio direct passthrough handlers
    ├── retry.rs               # Auto-retry logic with model loading
    └── streaming.rs           # Streaming response management
```

## Core Components

### Entry Point & Configuration

- **`main.rs`**: Entry point that parses CLI arguments and starts the server
- **`lib.rs`**: Library root with public re-exports and version constants

### Server Infrastructure

- **`server.rs`**: Main server implementation using Warp web framework
    - `ProxyServer` struct: Core server with HTTP client, config, logger
    - `Config` struct: CLI configuration with clap parser
    - `handle_request_with_cancellation()`: Main request router with connection tracking
    - `ConnectionTracker`: RAII-style connection lifecycle management
    - Route definitions for all supported endpoints

### Core Infrastructure

- **`common.rs`**: Shared cancellable operation infrastructure
    - `CancellableRequest`: HTTP request wrapper with client disconnect detection
    - `handle_json_response()`: Cancellable JSON response parser
    - `handle_text_response()`: Cancellable text response parser

### Utilities & Error Handling

- **`utils.rs`**: Core utilities and error management
    - `ProxyError`: Custom error type with HTTP status codes and cancellation support
    - `Logger`: Simple logging utility with enable/disable flag
    - `clean_model_name()`: Model name normalization (removes `:latest`, numeric suffixes)
    - `validate_model_name()`: Model name validation with warnings
    - `is_no_models_loaded_error()`: Error pattern detection for retry logic
    - `format_duration()`: Human-readable duration formatting

## Handler Modules

### Ollama API Handlers (`handlers/ollama.rs`)

Translates Ollama API calls to LM Studio format:

- **`handle_ollama_tags()`**: GET `/api/tags` - Lists available models with metadata
- **`handle_ollama_chat()`**: POST `/api/chat` - Chat completions with streaming support
- **`handle_ollama_generate()`**: POST `/api/generate` - Text completions with streaming
- **`handle_ollama_embeddings()`**: POST `/api/embed|embeddings` - Generate embeddings
- **`handle_ollama_show()`**: POST `/api/show` - Model information and capabilities
- **`handle_ollama_ps()`**: GET `/api/ps` - Running models (returns empty list)
- **`handle_ollama_version()`**: GET `/api/version` - Version information
- **`handle_unsupported()`**: Endpoints that cannot be translated to LM Studio

### Helper Functions (`handlers/helpers.rs`)

Response transformation and utility functions:

**Model Detection & Metadata:**

- `determine_model_family()`: Detects model family (llama, mistral, qwen, etc.)
- `determine_parameter_size()`: Extracts parameter count from model name
- `estimate_model_size()`: Estimates model size in bytes
- `determine_model_capabilities()`: Determines model capabilities (chat, vision, tools, etc.)

**Response Transformation:**

- `transform_chat_response()`: LM Studio → Ollama chat format
- `transform_generate_response()`: LM Studio → Ollama generate format
- `transform_embeddings_response()`: LM Studio → Ollama embeddings format

**Streaming Utilities:**

- `extract_content_from_chunk()`: Extracts text content from streaming chunks
- `create_error_chunk()`: Creates error response chunks
- `create_cancellation_chunk()`: Creates cancellation notification chunks
- `create_final_chunk()`: Creates stream completion chunks

**General Utilities:**

- `json_response()`: Converts JSON Value to HTTP response

### Streaming Support (`handlers/streaming.rs`)

Manages streaming responses with cancellation:

- **`handle_streaming_response()`**: Processes LM Studio streams, converts to Ollama format with timeout and
  cancellation
- **`handle_passthrough_streaming_response()`**: Direct LM Studio stream forwarding
- **`is_streaming_request()`**: Detects if request wants streaming
- **`convert_sse_to_ollama_chat()`**: Converts SSE chat chunks to Ollama format
- **`convert_sse_to_ollama_generate()`**: Converts SSE completion chunks to Ollama format
- **`send_chunk_and_close()`**: Utility to send final chunk and close stream

### Retry Logic (`handlers/retry.rs`)

Auto-retry with model loading:

- **`with_retry_and_cancellation()`**: Generic retry wrapper that detects "no models loaded" errors
- **`trigger_model_loading()`**: Attempts to wake up LM Studio by calling `/v1/models`

### LM Studio Passthrough (`handlers/lmstudio.rs`)

Direct API forwarding:

- **`handle_lmstudio_passthrough()`**: Forwards requests directly to LM Studio with streaming and cancellation support

### Module Organization (`handlers/mod.rs`)

Exports all handler functions for easy access.

## Key Design Patterns

### Cancellation Support

- All async operations use `CancellationToken` for graceful cancellation
- Connection tracking with RAII pattern automatically cancels on client disconnect
- Streaming operations respect cancellation mid-stream

### Error Handling

- Custom `ProxyError` type with HTTP status codes
- Graceful degradation (empty model lists vs errors)
- Cancellation-aware error propagation

### Response Format Translation

- Comprehensive transformation between Ollama and LM Studio JSON formats
- Maintains timing metadata and token counts
- Handles special cases like reasoning content merging

### Streaming Architecture

- Tokio channels for async stream processing
- Timeout handling for stuck streams
- Graceful cancellation with partial content tracking
- SSE to JSON chunk conversion

## Configuration Options

- `--listen`: Server bind address (default: 0.0.0.0:11434)
- `--lmstudio-url`: LM Studio backend URL (default: http://localhost:1234)
- `--no-log`: Disable logging
- `--load-timeout-seconds`: Model loading wait time (default: 5s)
- `--request-timeout-seconds`: HTTP request timeout (default: 300s)
- `--stream-timeout-seconds`: Streaming chunk timeout (default: 60s)

## Code Style Guidelines

### General Principles

- Use explicit error handling with `Result<T, ProxyError>`
- Prefer composition to inheritance
- Use `Arc<T>` for shared ownership in async contexts
- Clone data structures rather than passing references across async boundaries

### Async Patterns

- Always use `CancellationToken` for cancellable operations
- Use `tokio::select!` for racing futures with cancellation
- Prefer `tokio::spawn` for background tasks
- Use channels (`mpsc`) for stream processing

### Error Handling

- Use `ProxyError` for all application errors
- Check cancellation state before expensive operations
- Provide meaningful error messages with context
- Log errors at appropriate levels

### Naming Conventions

- Functions: `snake_case` with descriptive names
- Structs: `PascalCase`
- Constants: `SCREAMING_SNAKE_CASE`
- Modules: `snake_case`

### Documentation

- Document public functions with `///` comments
- Include `todo:` comments for future improvements
- Use short inline comments for complex logic (3-4 words)
- Maintain this outline document for architecture changes
