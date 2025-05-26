/// src/handlers/mod.rs - Module exports for API endpoint handlers and utilities

pub mod retry;
pub mod streaming;
pub mod helpers;
pub mod ollama;
pub mod lmstudio;

pub use ollama::{
    handle_ollama_tags,
    handle_ollama_chat,
    handle_ollama_generate,
    handle_ollama_embeddings,
    handle_ollama_ps,
    handle_ollama_show,
    handle_ollama_version,
    handle_unsupported,
    handle_health_check,
};

pub use lmstudio::{
    handle_lmstudio_passthrough,
    get_lmstudio_status,
};

pub use streaming::{
    handle_streaming_response,
    handle_passthrough_streaming_response,
    is_streaming_request,
};

pub use retry::{
    trigger_model_loading,
    with_retry_and_cancellation,
    with_simple_retry,
    trigger_model_loading_for_ollama,
};

pub use helpers::{
    json_response,
    ResponseTransformer,
    TimingInfo,
    build_lm_studio_request,
    extract_content_from_chunk,
    create_error_chunk,
    create_cancellation_chunk,
    create_final_chunk,
    execute_request_with_retry,
    LMStudioRequestType,
};
