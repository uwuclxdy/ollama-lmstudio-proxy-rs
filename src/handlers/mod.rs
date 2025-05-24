// handlers/mod.rs - Module organization with cancellation support

pub mod retry;
pub mod streaming;
pub mod helpers;
pub mod ollama;
pub mod lmstudio;
mod cancellation_handlers;
mod cancellation;
// Re-export all the handler functions for easy access

// Ollama handlers with cancellation support
pub use ollama::{
    handle_ollama_tags_with_cancellation,
    handle_ollama_chat_with_cancellation,
    handle_ollama_generate_with_cancellation,
    handle_ollama_embeddings_with_cancellation,

    // Backwards compatibility (non-cancellation versions)
    handle_ollama_tags,
    handle_ollama_chat,
    handle_ollama_generate,
    handle_ollama_embeddings,
    handle_ollama_ps,
    handle_ollama_show,
    handle_ollama_version,
    handle_unsupported,
};

// LM Studio handlers with cancellation support
pub use lmstudio::{
    handle_lmstudio_passthrough_with_cancellation,

    // Backwards compatibility
    handle_lmstudio_passthrough,
};

// Streaming functions with cancellation support
pub use streaming::{
    handle_streaming_response_with_cancellation,
    handle_passthrough_streaming_response_with_cancellation,
    is_streaming_request,

    // Backwards compatibility
    handle_streaming_response,
    handle_passthrough_streaming_response,
};

// Retry functions with cancellation support
pub use retry::{
    with_retry_and_cancellation,
    trigger_model_loading_with_cancellation,

    // Backwards compatibility
    with_retry,
    trigger_model_loading,
};

// Helper functions
pub use helpers::*;
