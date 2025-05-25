// src/handlers/mod.rs - Updated module organization with consolidated systems

pub mod retry;
pub mod streaming;
pub mod helpers;
pub mod ollama;
pub mod lmstudio;

// Ollama API handlers
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
    validate_lmstudio_endpoint,
    get_lmstudio_status,
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

// Helper functions and utilities
pub use helpers::{
    json_response,
    ResponseTransformer,
    TimingInfo,
    build_lm_studio_request,
    extract_content_from_chunk,
    create_error_chunk,
    create_cancellation_chunk,
    create_final_chunk,
};
