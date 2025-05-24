# Ollama-LMStudio Proxy Server

A Rust-based proxy server that bridges between Ollama API and LM Studio, supporting **both directions**:
- Ollama clients → LM Studio backend (with API translation)
- Direct LM Studio/OpenAI clients → LM Studio backend (passthrough)

## Features

- **Bidirectional API Support**: Handles both Ollama format (`/api/*`) and LM Studio format (`/v1/*`) endpoints
- **Full API Translation**: Converts Ollama API requests to LM Studio format and responses back to Ollama format
- **Direct Passthrough**: Forwards LM Studio/OpenAI API calls directly to LM Studio
- **Streaming Support**: Handles both streaming and non-streaming responses
- **Smart Model Mapping**: Automatically cleans model names (removes `:2` suffixes, etc.)
- **Comprehensive Logging**: Detailed request/response logging with toggle option
- **Timing Estimation**: Computes approximate timing values for Ollama compatibility
- **Error Handling**: Graceful handling of unsupported endpoints with informative messages
- **VS Code/Copilot Compatible**: Works with Visual Studio Code, GitHub Copilot, and other OpenAI-compatible clients

## Supported Endpoints

### Ollama API Endpoints (Translated)
| Ollama Endpoint      | LM Studio Equivalent        | Status          |
|----------------------|-----------------------------|-----------------|
| `GET /api/tags`      | `GET /v1/models`            | ✅ Supported     |
| `POST /api/chat`     | `POST /v1/chat/completions` | ✅ Supported     |
| `POST /api/generate` | `POST /v1/completions`      | ✅ Supported     |
| `POST /api/embed`    | `POST /v1/embeddings`       | ✅ Supported     |
| `GET /api/ps`        | `GET /v1/models` (mapped)   | ✅ Supported     |
| `POST /api/show`     | Combined requests           | ✅ Supported     |
| `GET /api/version`   | Hardcoded response          | ✅ Supported     |
| `POST /api/create`   | Not supported               | ❌ Returns error |
| `POST /api/pull`     | Not supported               | ❌ Returns error |
| `POST /api/push`     | Not supported               | ❌ Returns error |
| `DELETE /api/delete` | Not supported               | ❌ Returns error |
| `POST /api/copy`     | Not supported               | ❌ Returns error |

### Direct LM Studio API Endpoints (Passthrough)
| LM Studio Endpoint          | Status      | Notes              |
|-----------------------------|-------------|--------------------|
| `GET /v1/models`            | ✅ Supported | Direct passthrough |
| `POST /v1/chat/completions` | ✅ Supported | Direct passthrough |
| `POST /v1/completions`      | ✅ Supported | Direct passthrough |
| `POST /v1/embeddings`       | ✅ Supported | Direct passthrough |

## Installation

### Prerequisites

- Rust 1.70+ installed
- LM Studio running with a model loaded
- Access to the internet for downloading dependencies

### Build from Source

```bash
# Clone the repository
git clone <repository-url>
cd ollama-lmstudio-proxy-rust

# Build the project
cargo build --release

# Run the proxy
./target/release/ollama-lmstudio-proxy-rust
```

## Usage

### Basic Usage

```bash
# Start with default settings
./ollama-lmstudio-proxy-rust

# The proxy will:
# - Listen on 0.0.0.0:11434 (Ollama's default port)
# - Forward requests to http://localhost:1234 (LM Studio's default)
# - Enable logging by default
# - Support both Ollama (/api/*) and LM Studio (/v1/*) endpoints
```

### Configuration Options

```bash
# Custom addresses
./ollama-lmstudio-proxy-rust --listen 127.0.0.1:8080 --lmstudio-url http://192.168.1.100:1234

# Disable logging
./ollama-lmstudio-proxy-rust --no-log

# Help
./ollama-lmstudio-proxy-rust --help
```

### Command Line Arguments

- `--listen <ADDRESS:PORT>`: Address and port to listen on (default: `0.0.0.0:11434`)
- `--lmstudio-url <URL>`: LM Studio API URL (default: `http://localhost:1234`)
- `--no-log`: Disable request/response logging
- `--help`: Show help information

## Example Usage

### With Ollama Clients

Once the proxy is running, you can use any Ollama-compatible client:

```bash
# List available models
curl http://localhost:11434/api/tags

# Chat with a model
curl http://localhost:11434/api/chat -d '{
  "model": "deepseek-r1-distill-qwen-14b",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}'

# Generate text completion
curl http://localhost:11434/api/generate -d '{
  "model": "deepseek-r1-distill-qwen-14b",
  "prompt": "The capital of France is"
}'
```

### With OpenAI/LM Studio Clients

You can also use the proxy as a drop-in replacement for LM Studio:

```bash
# List available models (OpenAI format)
curl http://localhost:11434/v1/models

# Chat completion (OpenAI format)
curl http://localhost:11434/v1/chat/completions -d '{
  "model": "codero1-deepseekr1-coder-14b-preview",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.1,
  "stream": true
}'

# Text completion (OpenAI format)
curl http://localhost:11434/v1/completions -d '{
  "model": "codero1-deepseekr1-coder-14b-preview",
  "prompt": "The capital of France is",
  "temperature": 0.7,
  "max_tokens": 100
}'
```

### With Visual Studio Code

Configure VS Code or your IDE to use the proxy as the API endpoint:

1. In VS Code settings, set the API endpoint to `http://localhost:11434`
2. The proxy will automatically handle the `/v1/*` endpoints that VS Code expects
3. Your requests will be forwarded directly to LM Studio

## API Translation vs Passthrough

### Translation Mode (`/api/*` endpoints)
- **Input**: Ollama format requests
- **Processing**: Converts request format, cleans model names, transforms responses
- **Output**: Ollama format responses with timing estimates and metadata

### Passthrough Mode (`/v1/*` endpoints)
- **Input**: OpenAI/LM Studio format requests
- **Processing**: Direct forwarding to LM Studio (minimal processing)
- **Output**: Native LM Studio responses

## Model Name Mapping

The proxy automatically cleans model names:
- `deepseek-r1-distill-qwen-14b:2` → `deepseek-r1-distill-qwen-14b`
- Adds `:latest` suffix for Ollama compatibility when needed

## Response Enhancement (Translation Mode Only)

- **Reasoning Content**: LM Studio's `reasoning_content` is merged with main `content`
- **Timing Data**: Estimates `total_duration`, `load_duration`, `prompt_eval_duration`, etc.
- **Context**: Provides placeholder context arrays for compatibility
- **Token Counts**: Preserves usage statistics from LM Studio

## Error Handling

- **Model Not Found**: Returns appropriate format-specific error responses
- **Unsupported Endpoints**: Returns HTTP 501 with descriptive error messages
- **LM Studio Errors**: Forwards and translates error responses appropriately

## Logging

When logging is enabled (default), the proxy logs:

```
[PROXY] POST /v1/chat/completions - Body: {...}
[PROXY] POST http://localhost:1234/v1/chat/completions - Request: {...}
[PROXY] LM Studio response: {...} (took 45ms)
[PROXY] GET /api/tags
[PROXY] Ollama response: {...} (took 32ms)
```

## Troubleshooting

### Visual Studio Code Issues

1. **"Unknown endpoint" errors**
   - ✅ **Fixed in this version** - VS Code calls are now supported via `/v1/*` endpoints
   - Make sure you're using the updated proxy code

2. **Model not found in VS Code**
   - Check that your model name in VS Code matches what's loaded in LM Studio
   - Use `curl http://localhost:11434/v1/models` to see available models

### Common Issues

1. **Connection Refused**
   - Ensure LM Studio is running and accessible
   - Check the `--lmstudio-url` parameter
   - Verify no firewall is blocking the connection

2. **Model Not Found**
   - Make sure a model is loaded in LM Studio
   - Check model names with `curl http://localhost:11434/api/tags` or `curl http://localhost:11434/v1/models`

3. **Embedding Errors**
   - LM Studio model must support embeddings
   - Load an embedding model like `text-embedding-nomic-embed-text-v1.5`

4. **Port Already in Use**
   - Use `--listen` to specify a different port
   - Kill any existing Ollama/proxy processes

### Debug Mode

Enable verbose logging to see all request/response details:

```bash
# Logging is enabled by default
./ollama-lmstudio-proxy-rust

# To disable logging
./ollama-lmstudio-proxy-rust --no-log
```

## Development

### Project Structure

```
src/
├── main.rs          # Main proxy server implementation
├── lib.rs           # Library functions (if needed)
└── ...
Cargo.toml           # Dependencies and project config
README.md            # This file
```

### Key Components

- **ProxyServer**: Main struct handling API translation and passthrough
- **Translation Handlers**: Functions for Ollama → LM Studio conversion
- **Passthrough Handler**: Direct forwarding for `/v1/*` endpoints
- **Response Transformers**: Convert LM Studio responses to Ollama format
- **Error Handling**: Graceful error responses and logging

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Anthropic](https://www.anthropic.com/) for creating Claude that coded this entire thing lol
- [Ollama](https://ollama.ai/)
- [LM Studio](https://lmstudio.ai/)
- The Rust community
