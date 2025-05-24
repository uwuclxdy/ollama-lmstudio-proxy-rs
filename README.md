# 🔄 Ollama ↔ LM Studio Proxy Server

**A powerful Rust-based proxy server that seamlessly bridges Ollama API and LM Studio, enabling bidirectional
communication and universal compatibility.**

---

## ✨ What This Does

Transform your LM Studio setup into a **universal AI backend** that works with:

- 🦙 **Ollama clients** (with full API translation)
- 🤖 **OpenAI-compatible tools** (VS Code, GitHub Copilot, etc.)
- 🛠️ **Any HTTP client** expecting either API format

## 🚀 Key Features

### 🎯 **Dual API Support**

- **Ollama API** (`/api/*`) - Full translation to LM Studio format
- **OpenAI API** (`/v1/*`) - Direct passthrough to LM Studio
- **Smart Routing** - Automatically detects and handles both formats

### 🧠 **Intelligent Model Management**

- **Auto-Retry Logic** - Automatically loads models when needed
- **Smart Name Mapping** - Handles model name variations (`model:2` → `model`)
- **Format Translation** - Seamless conversion between API formats

### 🎨 **Enhanced Responses**

- **Reasoning Integration** - Merges LM Studio's reasoning content
- **Timing Estimates** - Provides Ollama-compatible performance metrics
- **Token Counting** - Accurate usage statistics
- **Error Handling** - Graceful degradation with helpful messages

### ⚡ **Performance & Reliability**

- **Built in Rust** - Memory-safe, fast, and reliable
- **Concurrent Handling** - Multiple requests simultaneously
- **Comprehensive Logging** - Detailed request/response tracking
- **Configurable Timeouts** - Customizable retry behavior

---

## 🎪 **Supported Endpoints**

| **Ollama API**       | **LM Studio Equivalent**    | **Status**            |
|----------------------|-----------------------------|-----------------------|
| `GET /api/tags`      | `GET /v1/models`            | ✅ **Full Support**    |
| `POST /api/chat`     | `POST /v1/chat/completions` | ✅ **Full Support**    |
| `POST /api/generate` | `POST /v1/completions`      | ✅ **Full Support**    |
| `POST /api/embed`    | `POST /v1/embeddings`       | ✅ **Full Support**    |
| `GET /api/ps`        | *(simulated)*               | ✅ **Simulated**       |
| `POST /api/show`     | *(simulated)*               | ✅ **Simulated**       |
| `GET /api/version`   | *(hardcoded)*               | ✅ **Static Response** |

| **OpenAI/LM Studio API**    | **Handling**              |
|-----------------------------|---------------------------|
| `GET /v1/models`            | 🔄 **Direct Passthrough** |
| `POST /v1/chat/completions` | 🔄 **Direct Passthrough** |
| `POST /v1/completions`      | 🔄 **Direct Passthrough** |
| `POST /v1/embeddings`       | 🔄 **Direct Passthrough** |

---

## 🏃‍♂️ **Quick Start**

### **1. Prerequisites**

- 🦀 Rust 1.70+ installed
- 🧠 LM Studio running with a model loaded
- 🌐 Network access between components

### **2. Installation**

```bash
# Clone and build
git clone <your-repo-url>
cd ollama-lmstudio-proxy
cargo build --release

# Run with defaults
./target/release/ollama-lmstudio-proxy
```

### **3. Configuration**

```bash
# Custom configuration
./target/release/ollama-lmstudio-proxy \
  --listen 0.0.0.0:11434 \
  --lmstudio-url http://localhost:1234 \
  --load-timeout-seconds 10

# Disable logging
./target/release/ollama-lmstudio-proxy --no-log

# Help
./target/release/ollama-lmstudio-proxy --help
```

---

## 🧪 **Usage Examples**

### **With Ollama Clients**

```bash
# List models (Ollama format)
curl http://localhost:11434/api/tags

# Chat completion (Ollama format)
curl http://localhost:11434/api/chat -d '{
  "model": "your-model-name",
  "messages": [
    {"role": "user", "content": "Explain quantum computing"}
  ]
}'

# Text generation (Ollama format)
curl http://localhost:11434/api/generate -d '{
  "model": "your-model-name",
  "prompt": "The future of AI is"
}'
```

### **With OpenAI/VS Code**

```bash
# List models (OpenAI format)
curl http://localhost:11434/v1/models

# Chat completion (OpenAI format)
curl http://localhost:11434/v1/chat/completions -d '{
  "model": "your-model-name",
  "messages": [
    {"role": "user", "content": "Write a Python function"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000
}'
```

### **VS Code Integration**

1. Install an OpenAI-compatible extension
2. Set API endpoint to: `http://localhost:11434`
3. Use any model name from your LM Studio
4. Start coding with AI assistance! 🎉

---

## 🎛️ **Configuration Options**

| **Flag**                 | **Default**             | **Description**                  |
|--------------------------|-------------------------|----------------------------------|
| `--listen`               | `0.0.0.0:11434`         | Address and port to listen on    |
| `--lmstudio-url`         | `http://localhost:1234` | LM Studio API endpoint           |
| `--no-log`               | `false`                 | Disable request/response logging |
| `--load-timeout-seconds` | `5`                     | Model loading retry timeout      |

---

## 🔧 **Troubleshooting**

### **Common Issues**

**🔌 "Connection Refused"**

- Ensure LM Studio is running and accessible
- Check firewall settings
- Verify the `--lmstudio-url` parameter

**🤖 "Model Not Found"**

- Load a model in LM Studio first
- Check available models: `curl http://localhost:11434/api/tags`
- Verify model names match between tools

**📝 "VS Code Not Working"**

- Ensure you're using `/v1/*` endpoints in VS Code settings
- Check that the API endpoint is set to `http://localhost:11434`
- Verify the model name exists in LM Studio

**⏱️ "Slow Responses"**

- Increase `--load-timeout-seconds` for slow model loading
- Check LM Studio performance and hardware resources
- Monitor logs for retry attempts

---

## 🌟 **Why This Proxy?**

### **🎯 Universal Compatibility**

- Works with **any Ollama client** without modification
- Compatible with **VS Code, GitHub Copilot, and OpenAI tools**
- **Single endpoint** for multiple API formats

### **🧠 Smart Translation**

- **Preserves reasoning** from advanced models
- **Accurate timing** and token counting
- **Proper error handling** with helpful messages

### **⚡ Performance**

- **Rust-powered** for maximum speed and safety
- **Automatic retries** for seamless experience
- **Concurrent processing** for multiple clients

### **🛠️ Developer Friendly**

- **Comprehensive logging** for debugging
- **Flexible configuration** for any setup
- **Open source** and easily extendable

---

## 📊 **Performance**

- **🚀 Low Latency**: Minimal overhead between client and LM Studio
- **🔄 Auto-Recovery**: Intelligent retry logic for model loading
- **📈 Scalable**: Handles multiple concurrent connections
- **💾 Memory Efficient**: Rust's zero-cost abstractions

---

## 🤝 **Contributing**

We welcome contributions! Whether it's:

- 🐛 Bug fixes
- ✨ New features
- 📚 Documentation improvements
- 🧪 Test coverage

---

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE]() file for details.

---

## 🙏 **Acknowledgments**

- **[Ollama](https://ollama.ai/)** - For the excellent API design
- **[LM Studio](https://lmstudio.ai/)** - For the powerful local LLM platform
- **[Rust Community](https://www.rust-lang.org/)** - For the amazing ecosystem
- **[Anthropic](https://www.anthropic.com/)** - For Claude's development assistance

---

**🎉 Transform your LM Studio into a universal AI backend today!**

[⭐ Star this repo](https://github.com/your-username/ollama-lmstudio-proxy) • [🐛 Report Bug](https://github.com/your-username/ollama-lmstudio-proxy/issues) • [💡 Request Feature](https://github.com/your-username/ollama-lmstudio-proxy/issues)
