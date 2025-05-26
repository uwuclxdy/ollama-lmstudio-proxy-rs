# Ollama ↔ LM Studio Proxy

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge)](LICENSE)

Proxy server that bridges **Ollama API** and **LM Studio** - written in Rust! This is my first Rust project and was
(almost) entirely vibe coded with Claude 4 Sonnet as I wanted to test what it could do and hopefully learn Rust a bit
lol. The rest of this readme is not written by me (lazy fuck ikr).

## 🚀 Features

- **Dual API Support**: Native LM Studio REST API (`/api/v0/`) or legacy OpenAI endpoints (`/v1/`)
- **Smart Model Resolution**: Automatic model name mapping with fuzzy matching and caching
- **High-Performance Streaming**: Optimized SSE processing with chunk recovery and cancellation
- **Model Loading Detection**: Automatic retry logic with intelligent model loading triggers
- **Production Ready**: Enhanced error handling, health monitoring, structured logging, and CORS support

## ⚙️ Configuration

### Command Line Options

| Flag                                   | Default                 | Description                    |
|----------------------------------------|-------------------------|--------------------------------|
| `--listen`                             | `0.0.0.0:11434`         | Server bind address            |
| `--lmstudio_url`                       | `http://localhost:1234` | LM Studio backend URL          |
| `--legacy`                             | `false`                 | Use legacy OpenAI API mode     |
| `--no_log`                             | `false`                 | Disable logging output         |
| `--load_timeout_seconds`               | `15`                    | Model loading timeout          |
| `--model_resolution_cache_ttl_seconds` | `300`                   | Cache TTL for model resolution |
| `--max_buffer_size`                    | `262144`                | SSE buffer size (bytes)        |
| `--enable_chunk_recovery`              | `false`                 | Enable stream chunk recovery   |

### API Mode Comparison

| Feature                   | Native Mode    | Legacy Mode  |
|---------------------------|----------------|--------------|
| **LM Studio Version**     | 0.3.6+         | 0.2.0+       |
| **Model Loading State**   | ✅ Real-time    | ❌ Estimated  |
| **Context Length Limits** | ✅ Accurate     | ❌ Generic    |
| **Performance Metrics**   | ✅ Native stats | ❌ Calculated |
| **Model Metadata**        | ✅ Rich details | ❌ Basic info |
| **Publisher Info**        | ✅ Available    | ❌ Unknown    |

### Endpoint Support

| Ollama Endpoint      | Legacy Mode              | Native Mode                  | Notes                              |
|----------------------|--------------------------|------------------------------|------------------------------------|
| `GET /api/tags`      | ✅ `/v1/models`           | ✅ `/api/v0/models`           |                                    |
| `GET /api/ps`        | ✅ `/v1/models`           | ✅ `/api/v0/models`           | Shows loaded models only           |
| `POST /api/show`     | ✅ *Fabricated*           | ✅ *Fabricated*               | Generated from model name          |
| `POST /api/chat`     | ✅ `/v1/chat/completions` | ✅ `/api/v0/chat/completions` |                                    |
| `POST /api/generate` | ✅ `/v1/completions`      | ✅ `/api/v0/completions`      | Vision support via chat endpoint   |
| `POST /api/embed`    | ✅ `/v1/embeddings`       | ✅ `/api/v0/embeddings`       | Also supports `/api/embeddings`    |
| `GET /api/version`   | ✅ *Proxy response*       | ✅ *Proxy response*           |                                    |
| `GET /health`        | ✅ *Health check*         | ✅ *Health check*             |                                    |
| `POST /v1/*`         | ✅ *Direct passthrough*   | ✅ *Converts to /api/v0/*     |                                    |
| `POST /api/create`   | ❌                        | ❌                            | Use LM Studio for model management |
| `POST /api/pull`     | ❌                        | ❌                            |                                    |
| `POST /api/push`     | ❌                        | ❌                            |                                    |
| `POST /api/delete`   | ❌                        | ❌                            |                                    |
| `POST /api/copy`     | ❌                        | ❌                            |                                    |

## 📋 Requirements

- **Rust**: 1.70+ (2021 edition)
- **LM Studio**:
    - 0.3.6+ for native mode (recommended)
    - 0.2.0+ for legacy mode

## 🔧 Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/uwuclxdy/ollama-lmstudio-proxy-rust.git
cd ollama-lmstudio-proxy-rust

# Build release version
cargo build --release

# Run the proxy
./target/release/ollama-lmstudio-proxy-rust
```

### Using Cargo

```bash
cargo install --git https://github.com/uwuclxdy/ollama-lmstudio-proxy-rust.git
```

## 🚀 Quick Start

### Basic Usage

```bash
# Start with default settings (native mode)
ollama-lmstudio-proxy-rust

# Use legacy mode for older LM Studio versions
ollama-lmstudio-proxy-rust --legacy

# Custom configuration
ollama-lmstudio-proxy-rust \
  --listen 0.0.0.0:11434 \
  --lmstudio_url http://localhost:1234 \
  --load_timeout_seconds 30
```

### Test the Connection

```bash
# Check health status
curl http://localhost:11434/health

# List available models
curl http://localhost:11434/api/tags

# Send a chat request
curl http://localhost:11434/api/chat -d '{
  "model": "llama2",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

---

**Made with Claude 4 Sonnet <3**
