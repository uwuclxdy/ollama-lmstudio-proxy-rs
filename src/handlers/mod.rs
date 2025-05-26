/// src/handlers/mod.rs - Module exports for API endpoint handlers and utilities

pub mod retry;
pub mod streaming;
pub mod helpers;
pub mod ollama;
pub mod lmstudio;

pub use ollama::{
    handle_health_check,
    handle_ollama_chat,
    handle_ollama_embeddings,
    handle_ollama_generate,
    handle_ollama_ps,
    handle_ollama_show,
    handle_ollama_tags,
    handle_ollama_version,
    handle_unsupported,
};

pub use lmstudio::{
    get_lmstudio_status,
    handle_lmstudio_passthrough,
};

pub use streaming::{
    handle_passthrough_streaming_response,
    handle_streaming_response,
    is_streaming_request,
};

pub use retry::{
    calculate_backoff_delay,
    check_lm_studio_availability,
    should_retry_error,
    trigger_model_loading,
    trigger_model_loading_for_ollama,
    with_health_check_and_retry,
    with_retry_and_cancellation,
    with_simple_retry,
};

pub use helpers::{
    build_lm_studio_request,
    create_cancellation_chunk,
    create_error_chunk,
    create_final_chunk,
    execute_request_with_retry,
    extract_content_from_chunk,
    json_response,
    LMStudioRequestType,
    ResponseTransformer,
    TimingInfo,
};
