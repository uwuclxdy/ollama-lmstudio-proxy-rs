# 🔧 Ollama-LM Studio Proxy - Technical Documentation

**Comprehensive technical documentation for developers working on the Ollama-LM Studio proxy server.**

---

## 📐 **Project Architecture**

### **Core Concept**
The proxy serves as a translation layer between two different LLM API formats:
- **Ollama API** (`/api/*`) - Requires format translation and enhancement
- **OpenAI/LM Studio API** (`/v1/*`) - Direct passthrough with retry logic

### **High-Level Data Flow**
```
[Client] → [Proxy Server] → [LM Studio]
         ↑               ↓
    [API Translation] [Response Enhancement]
```

---

## 📁 **Code Structure**

### **File Organization (5 Files Total)**
```
src/
├── main.rs           # Entry point and CLI setup (~20 lines)
├── server.rs         # HTTP server setup and routing (~200 lines)
├── handlers.rs       # API endpoint implementations (~400+ lines)
├── utils.rs          # Utilities and helper functions (~150 lines)
└── lib.rs            # Module organization and exports (~20 lines)
```

### **Dependency Management**
```toml
[dependencies]
clap = { version = "4.4", features = ["derive"] }    # CLI argument parsing
reqwest = { version = "0.11", features = ["json"] }  # HTTP client
serde = { version = "1.0", features = ["derive"] }   # JSON serialization
serde_json = "1.0"                                   # JSON manipulation
tokio = { version = "1.0", features = ["full"] }     # Async runtime
warp = "0.3"                                         # HTTP server framework
bytes = "1.5"                                        # Byte buffer handling
chrono = { version = "0.4", features = ["serde"] }   # Date/time handling
md5 = "0.7"                                          # Hash generation for fake digests
```

---

## 🏗️ **Module Breakdown**

### **`main.rs` - Entry Point**
**Purpose**: Application bootstrap and CLI integration
**Size**: ~20 lines
**Key Functions**:
- `main()` - Parse CLI args and start server

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::parse();
    let server = ProxyServer::new(config);
    server.run().await?;
    Ok(())
}
```

---

### **`utils.rs` - Utilities and Helpers**
**Purpose**: Shared utilities, error handling, and helper functions
**Size**: ~150 lines

#### **Core Types**
```rust
pub struct ProxyError {
    pub message: String,
    pub status_code: u16,
}

pub struct Logger {
    pub enabled: bool,
}
```

#### **Key Functions**

**Error Constructors**:
- `ProxyError::new(message, status_code)` - Custom error
- `ProxyError::internal_server_error(msg)` - 500 errors
- `ProxyError::bad_request(msg)` - 400 errors
- `ProxyError::not_found(msg)` - 404 errors
- `ProxyError::not_implemented(msg)` - 501 errors

**Utility Functions**:
- `clean_model_name(name: &str) -> String` - Removes `:latest` and numeric suffixes
- `is_no_models_loaded_error(message: &str) -> bool` - Detects model loading errors
- `format_duration(duration: Duration) -> String` - Human-readable duration formatting

**Logging**:
- `Logger::new(enabled: bool)` - Create logger instance
- `Logger::log(&self, message: &str)` - Log with `[PROXY]` prefix

---

### **`server.rs` - HTTP Server and Routing**
**Purpose**: HTTP server setup, configuration, and request routing
**Size**: ~200 lines

#### **Configuration Structure**
```rust
#[derive(Parser, Debug, Clone)]
pub struct Config {
    #[arg(long, default_value = "0.0.0.0:11434")]
    pub listen: String,
    
    #[arg(long, default_value = "http://localhost:1234")]
    pub lmstudio_url: String,
    
    #[arg(long)]
    pub no_log: bool,
    
    #[arg(long, default_value = "5")]
    pub load_timeout_seconds: u64,
}
```

#### **Main Server Structure**
```rust
#[derive(Clone)]
pub struct ProxyServer {
    pub client: reqwest::Client,
    pub config: Config,
    pub logger: Logger,
}
```

#### **Key Functions**

**Server Management**:
- `ProxyServer::new(config: Config) -> Self` - Initialize server
- `ProxyServer::run(self) -> Result<(), Error>` - Start HTTP server
- `print_startup_banner(&self)` - Display configuration at startup

**Request Routing**:
- `handle_request(server, method, path, body) -> Result<Response, Rejection>` - Main router
- `handle_rejection(err: Rejection) -> Result<Response, Infallible>` - Error handling

#### **Routing Logic**
```rust
match (method.as_str(), path.as_str()) {
    // Ollama API (translated)
    ("GET", "/api/tags") => handlers::handle_ollama_tags(server).await,
    ("POST", "/api/chat") => handlers::handle_ollama_chat(server, body).await,
    ("POST", "/api/generate") => handlers::handle_ollama_generate(server, body).await,
    ("POST", "/api/embed") | ("POST", "/api/embeddings") => 
        handlers::handle_ollama_embeddings(server, body).await,
    
    // Unsupported Ollama endpoints
    (_, "/api/create" | "/api/pull" | "/api/push" | "/api/delete" | "/api/copy") => 
        handlers::handle_unsupported(endpoint).await,
    
    // LM Studio API (passthrough)
    ("GET", "/v1/models") | ("POST", "/v1/chat/completions") | 
    ("POST", "/v1/completions") | ("POST", "/v1/embeddings") => 
        handlers::handle_lmstudio_passthrough(server, method, path, body).await,
    
    _ => Err(ProxyError::not_found("Unknown endpoint")),
}
```

---

### **`handlers.rs` - API Implementation**
**Purpose**: All API endpoint implementations and business logic
**Size**: ~400+ lines

#### **Auto-Retry Infrastructure**

**Core Functions**:
- `trigger_model_loading(server: &ProxyServer) -> Result<bool, ProxyError>`
  - Calls `/v1/models` to wake up LM Studio
  - Returns success/failure status
  
- `with_retry<F, T, Fut>(server: &ProxyServer, operation: F) -> Result<T, ProxyError>`
  - Generic retry wrapper
  - Detects "no models loaded" errors
  - Automatically triggers model loading and retries

#### **Ollama API Handlers (With Translation)**

**`handle_ollama_tags(server: ProxyServer) -> Result<Response, ProxyError>`**
- **Purpose**: List available models in Ollama format
- **LM Studio Call**: `GET /v1/models`
- **Transformation**: 
  - Converts OpenAI model list to Ollama format
  - Adds fake metadata (digest, size, quantization info)
  - Appends `:latest` suffix to model names
- **Response Format**:
```json
{
  "models": [{
    "name": "model-name:latest",
    "model": "model-name:latest", 
    "modified_at": "2025-01-20T...",
    "size": 4000000000,
    "digest": "md5hash...",
    "details": {
      "format": "gguf",
      "family": "llama",
      "parameter_size": "7B",
      "quantization_level": "Q4_K_M"
    }
  }]
}
```

**`handle_ollama_chat(server: ProxyServer, body: Value) -> Result<Response, ProxyError>`**
- **Purpose**: Chat completion with conversation history
- **LM Studio Call**: `POST /v1/chat/completions`
- **Input Processing**:
  - Extracts `model`, `messages`, `stream` from Ollama request
  - Cleans model name using `clean_model_name()`
  - Maps `options.temperature` and `options.num_predict`
- **Response Enhancement**:
  - Merges `reasoning_content` with main `content` if present
  - Calculates timing estimates (`total_duration`, `eval_count`, etc.)
  - Provides Ollama-compatible metadata
- **Reasoning Integration**:
```rust
if let Some(reasoning) = first_choice.get("message")
    .and_then(|m| m.get("reasoning_content")) {
    content = format!("**Reasoning:**\n{}\n\n**Answer:**\n{}", reasoning, content);
}
```

**`handle_ollama_generate(server: ProxyServer, body: Value) -> Result<Response, ProxyError>`**
- **Purpose**: Text completion from prompt
- **LM Studio Call**: `POST /v1/completions`
- **Key Differences**: Uses completions endpoint instead of chat
- **Response Format**: Ollama's generate format with `response` field

**`handle_ollama_embeddings(server: ProxyServer, body: Value) -> Result<Response, ProxyError>`**
- **Purpose**: Generate embeddings for text
- **LM Studio Call**: `POST /v1/embeddings`
- **Input Handling**: Accepts both `input` and `prompt` fields
- **Output Format**: Ollama embeddings format with timing data

#### **Simple Ollama Handlers (No LM Studio Calls)**

**`handle_ollama_ps() -> Result<Response, ProxyError>`**
- Returns empty models array (simulates no running models)

**`handle_ollama_show(body: Value) -> Result<Response, ProxyError>`**
- Returns fake modelfile information
- Uses model name from request body

**`handle_ollama_version() -> Result<Response, ProxyError>`**
- Returns hardcoded version "0.5.1-proxy"

**`handle_unsupported(endpoint: &str) -> Result<Response, ProxyError>`**
- Returns 501 Not Implemented for unsupported endpoints
- Provides helpful error messages

#### **LM Studio Passthrough Handler**

**`handle_lmstudio_passthrough(server, method, endpoint, body) -> Result<Response, ProxyError>`**
- **Purpose**: Direct forwarding of `/v1/*` requests to LM Studio
- **HTTP Methods**: Supports GET, POST, PUT, DELETE
- **Response Handling**: 
  - Detects content-type for streaming vs JSON
  - Preserves original response format
  - Includes retry logic for model loading
- **Streaming Support**: Placeholder for Server-Sent Events (marked as TODO)

#### **Response Processing Helpers**

**`json_response(value: &Value) -> warp::reply::Response`**
- Converts JSON values to concrete HTTP responses
- Ensures type compatibility across all handlers

**`handle_streaming_response(response: reqwest::Response) -> Result<Value, ProxyError>`**
- Placeholder for streaming response handling
- Currently returns "not implemented" error

---

### **`lib.rs` - Module Organization**
**Purpose**: Module declarations and public exports
**Size**: ~20 lines

```rust
pub mod server;
pub mod handlers; 
pub mod utils;

pub use server::{Config, ProxyServer};
pub use utils::ProxyError;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");
```

---

## 🔄 **API Translation Details**

### **Request Transformations**

#### **Ollama Chat → OpenAI Chat**
```rust
// Input (Ollama)
{
  "model": "llama3.2:latest",
  "messages": [...],
  "options": {"temperature": 0.7, "num_predict": 1000}
}

// Output (OpenAI)
{
  "model": "llama3.2",  // cleaned name
  "messages": [...],    // preserved
  "temperature": 0.7,   // extracted from options
  "max_tokens": 1000    // mapped from num_predict
}
```

#### **Ollama Generate → OpenAI Completions**
```rust
// Input (Ollama)
{
  "model": "llama3.2:2",
  "prompt": "The capital of France is",
  "options": {"temperature": 0.5}
}

// Output (OpenAI)
{
  "model": "llama3.2",  // cleaned name
  "prompt": "The capital of France is",
  "temperature": 0.5
}
```

### **Response Transformations**

#### **OpenAI Chat → Ollama Chat**
```rust
// Input (OpenAI)
{
  "choices": [{
    "message": {
      "content": "Paris",
      "reasoning_content": "France is a country..."
    }
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 5}
}

// Output (Ollama)
{
  "model": "llama3.2:latest",
  "message": {
    "role": "assistant", 
    "content": "**Reasoning:**\nFrance is a country...\n\n**Answer:**\nParis"
  },
  "done": true,
  "total_duration": 1500000000,  // calculated
  "prompt_eval_count": 10,       // from usage
  "eval_count": 5                // from usage
}
```

### **Model Name Processing**
```rust
fn clean_model_name(name: &str) -> String {
    // Remove :latest suffix
    let name = name.strip_suffix(":latest").unwrap_or(name);
    
    // Remove numeric suffixes (:2, :3, etc.)
    if let Some(colon_pos) = name.rfind(':') {
        let suffix = &name[colon_pos + 1..];
        if suffix.chars().all(|c| c.is_ascii_digit()) {
            return name[..colon_pos].to_string();
        }
    }
    
    name.to_string()
}
```

---

## ⚡ **Auto-Retry Logic**

### **Error Detection**
```rust
fn is_no_models_loaded_error(message: &str) -> bool {
    let lower_msg = message.to_lowercase();
    lower_msg.contains("no model") 
        || lower_msg.contains("model not loaded")
        || lower_msg.contains("no models loaded")
        || lower_msg.contains("model loading")
        || lower_msg.contains("load a model")
        || lower_msg.contains("model is not loaded")
}
```

### **Retry Flow**
1. **First Attempt**: Execute original operation
2. **Error Analysis**: Check if error indicates missing model
3. **Model Loading**: Call `GET /v1/models` to wake up LM Studio
4. **Wait Period**: Sleep for `load_timeout_seconds`
5. **Retry**: Execute original operation again
6. **Result**: Return success or original error

### **Generic Retry Wrapper**
```rust
async fn with_retry<F, T, Fut>(server: &ProxyServer, operation: F) -> Result<T, ProxyError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, ProxyError>>,
{
    // First attempt
    match operation().await {
        Ok(result) => return Ok(result),
        Err(e) => {
            if is_no_models_loaded_error(&e.message) {
                // Trigger loading and retry
                if trigger_model_loading(server).await.unwrap_or(false) {
                    sleep(Duration::from_secs(server.config.load_timeout_seconds)).await;
                    return operation().await;
                }
            }
            Err(e)
        }
    }
}
```

---

## 🏃‍♂️ **Performance Optimizations**

### **Timing Calculations**
- **Measurement**: Uses `std::time::Instant` for accurate timing
- **Estimation**: Provides realistic timing data for Ollama compatibility
- **Token Counting**: Rough estimation based on character count / 4

### **Memory Management**
- **Clone Strategy**: `ProxyServer` is `Clone` for sharing across async tasks
- **String Handling**: Efficient string operations with minimal allocations
- **JSON Processing**: Uses `serde_json::Value` for flexible JSON manipulation

### **HTTP Client Reuse**
- **Single Client**: Reuses `reqwest::Client` across all requests
- **Connection Pooling**: Automatic HTTP connection reuse
- **Timeout Handling**: Configurable timeouts for model loading

---

## 🚨 **Error Handling Strategy**

### **Error Types**
1. **Network Errors**: Connection failures to LM Studio
2. **API Errors**: Invalid requests or LM Studio errors  
3. **Format Errors**: JSON parsing or serialization issues
4. **Model Errors**: Model not loaded or not found
5. **Configuration Errors**: Invalid listen address or URLs

### **Error Response Formats**

**Ollama Format**:
```json
{
  "error": {
    "type": "proxy_error",
    "message": "Detailed error description"
  }
}
```

**HTTP Status Codes**:
- `400` - Bad Request (missing fields, invalid JSON)
- `404` - Not Found (unknown endpoints, missing models)
- `500` - Internal Server Error (network issues, LM Studio errors)
- `501` - Not Implemented (unsupported Ollama endpoints)

### **Logging Strategy**
- **Request Logging**: Method, path, and body size
- **Response Logging**: Status, timing, and error details
- **Debug Information**: Retry attempts and model loading events
- **Configurable**: Can be disabled with `--no-log`

---

## 🧪 **Testing Approach**

### **Unit Tests**
- **Utility Functions**: Model name cleaning, error detection, duration formatting
- **Configuration**: Default values and parsing
- **Error Constructors**: Proper status codes and messages

### **Integration Testing**
```bash
# Build and test
cargo test

# Test specific module
cargo test utils::tests

# Test with output
cargo test -- --nocapture
```

### **Manual Testing Commands**
```bash
# Test Ollama endpoints
curl http://localhost:11434/api/tags
curl http://localhost:11434/api/version
curl -X POST http://localhost:11434/api/chat -d '{"model":"test","messages":[{"role":"user","content":"hello"}]}'

# Test OpenAI endpoints  
curl http://localhost:11434/v1/models
curl -X POST http://localhost:11434/v1/chat/completions -d '{"model":"test","messages":[{"role":"user","content":"hello"}]}'

# Test error handling
curl http://localhost:11434/api/unknown
curl http://localhost:11434/api/create
```

---

## 🚀 **Build and Deployment**

### **Build Process**
```bash
# Development build
cargo build

# Release build (optimized)
cargo build --release

# Check without building
cargo check

# Format code
cargo fmt

# Lint code
cargo clippy
```

### **Cross-Platform Compilation**
```bash
# Linux
cargo build --release --target x86_64-unknown-linux-gnu

# Windows  
cargo build --release --target x86_64-pc-windows-gnu

# macOS
cargo build --release --target x86_64-apple-darwin
```

### **Deployment Considerations**
- **Binary Size**: ~10-20MB after release compilation
- **Memory Usage**: ~50-100MB runtime (depends on request volume)
- **CPU Usage**: Minimal overhead, mostly I/O bound
- **Network**: Requires access to LM Studio port (default 1234)

---

## 🔧 **Configuration Details**

### **Environment Variables**
Currently not supported, but could be added:
```rust
// Potential future enhancement
let listen = env::var("PROXY_LISTEN").unwrap_or_else(|_| "0.0.0.0:11434".to_string());
```

### **Runtime Configuration**
- **No Config File**: All configuration via CLI arguments
- **Live Reload**: Not supported (requires restart for changes)
- **Validation**: Basic validation on startup (address parsing)

### **Default Values**
```rust
Config {
    listen: "0.0.0.0:11434".to_string(),      // Ollama's default port
    lmstudio_url: "http://localhost:1234".to_string(),  // LM Studio default
    no_log: false,                            // Logging enabled by default
    load_timeout_seconds: 5,                  // 5 second model loading timeout
}
```

---

## 🐛 **Known Issues and TODOs**

### **Current Limitations**
1. **Streaming Responses**: Not fully implemented (placeholder exists)
2. **WebSocket Support**: Not supported
3. **Authentication**: No auth support for either API
4. **Rate Limiting**: No built-in rate limiting
5. **Metrics**: No Prometheus/metrics endpoint

### **Future Enhancements**
1. **Server-Sent Events**: Full streaming support
2. **Configuration File**: YAML/TOML config support
3. **Health Checks**: `/health` endpoint
4. **Metrics**: Request counting and performance metrics
5. **Docker Support**: Containerization
6. **TLS Support**: HTTPS endpoints

### **Type System Issues**
- **Fixed**: `impl Reply` incompatibility resolved with concrete `warp::reply::Response`
- **Potential**: Generic type constraints in retry system

---

## 📚 **Code Examples**

### **Adding a New Ollama Endpoint**
```rust
// In server.rs routing
("POST", "/api/my-endpoint") => handlers::handle_my_endpoint(server, body).await,

// In handlers.rs
pub async fn handle_my_endpoint(server: ProxyServer, body: Value) -> Result<warp::reply::Response, ProxyError> {
    let start_time = Instant::now();
    
    let operation = || async {
        // Your LM Studio API call here
        let url = format!("{}/v1/my-lm-endpoint", server.config.lmstudio_url);
        let response = server.client.post(&url).json(&body).send().await
            .map_err(|e| ProxyError::internal_server_error(&format!("LM Studio error: {}", e)))?;
        
        // Transform response
        let result = json!({"transformed": "response"});
        Ok(result)
    };
    
    let result = with_retry(&server, operation).await?;
    let duration = start_time.elapsed();
    
    server.logger.log(&format!("My endpoint completed (took {})", format_duration(duration)));
    Ok(json_response(&result))
}
```

### **Adding Configuration Options**
```rust
// In Config struct
#[arg(long, default_value = "60")]
pub my_timeout_seconds: u64,

// Usage in handlers
tokio::time::timeout(
    Duration::from_secs(server.config.my_timeout_seconds),
    operation()
).await
```

---

## 🔍 **Debugging Guide**

### **Common Debug Scenarios**

**Enable Verbose Logging**:
```bash
./target/release/ollama-lmstudio-proxy  # Logging enabled by default
```

**Check Request/Response Flow**:
```bash
# Terminal 1: Start proxy with logging
./target/release/ollama-lmstudio-proxy

# Terminal 2: Make test request
curl -v http://localhost:11434/api/tags

# Check logs in Terminal 1 for:
# [PROXY] GET /api/tags
# [PROXY] Calling LM Studio: GET http://localhost:1234/v1/models  
# [PROXY] Ollama tags response completed (took 45ms)
```

**Debug Model Loading Issues**:
```bash
# Check if LM Studio has models loaded
curl http://localhost:1234/v1/models

# Test proxy model detection
curl http://localhost:11434/api/tags

# Look for retry logs:
# [PROXY] Detected 'no models loaded' error, attempting retry...
# [PROXY] Attempting to trigger model loading...
# [PROXY] Retrying operation after model loading...
```

### **Log Message Meanings**
- `"Calling LM Studio: ..."` - Outbound request to LM Studio
- `"...response completed (took Xms)"` - Request finished successfully
- `"Detected 'no models loaded' error"` - Auto-retry triggered
- `"Attempting to trigger model loading..."` - Calling `/v1/models` to wake up LM Studio
- `"Retrying operation after model loading..."` - Second attempt after model loading

---

This comprehensive technical documentation should provide all the context needed to understand, maintain, and extend the Ollama-LM Studio proxy server codebase.