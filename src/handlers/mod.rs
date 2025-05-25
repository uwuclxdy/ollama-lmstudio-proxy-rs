// src/handlers/mod.rs - Module organization

pub mod retry;
pub mod streaming;
pub mod helpers;
pub mod ollama;
pub mod lmstudio;

// Ollama handlers
pub use ollama::{
    handle_ollama_tags,
    handle_ollama_chat,
    handle_ollama_generate,
    handle_ollama_embeddings,
    handle_ollama_ps,
    handle_ollama_show,
    handle_ollama_version,
    handle_unsupported,
};

// LM Studio handlers
pub use lmstudio::{
    handle_lmstudio_passthrough,
};

// Streaming functions
pub use streaming::{
    handle_streaming_response,
    handle_passthrough_streaming_response,
    is_streaming_request,
};

// Retry functions
pub use retry::{
    trigger_model_loading,
    with_retry_and_cancellation,
    with_simple_retry,
};

// Helper functions
pub use helpers::*;