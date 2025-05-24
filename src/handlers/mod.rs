// Main handlers module - organizes submodules and provides public API

pub mod retry;
pub mod streaming;
pub mod ollama;
pub mod lmstudio;
pub mod helpers;

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

pub use lmstudio::handle_lmstudio_passthrough;
pub use helpers::json_response;