// src/model.rs - Optimized model handling without caching for single-client use

use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::common::{CancellableRequest, RequestContext};
use crate::constants::*;
use crate::utils::ProxyError;

/// Optimized model information with lookup table
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub family: &'static str,
    pub families: &'static [&'static str],
    pub parameter_size: &'static str,
    pub size_bytes: u64,
    pub capabilities: &'static [&'static str],
    pub architecture: &'static str,
    pub quantization_level: &'static str,
}

/// Pre-computed model patterns for fast lookup
static MODEL_PATTERNS: &[(&str, ModelInfo)] = &[
    ("llama", ModelInfo {
        family: "llama",
        families: &["llama"],
        parameter_size: "7B",
        size_bytes: 4_000_000_000,
        capabilities: &["completion", "chat", "tools"],
        architecture: "llama",
        quantization_level: "Q4_K_M",
    }),
    ("qwen", ModelInfo {
        family: "qwen2",
        families: &["qwen2"],
        parameter_size: "7B",
        size_bytes: 4_000_000_000,
        capabilities: &["completion", "chat", "tools"],
        architecture: "qwen",
        quantization_level: "Q4_K_M",
    }),
    ("mistral", ModelInfo {
        family: "mistral",
        families: &["mistral"],
        parameter_size: "7B",
        size_bytes: 4_000_000_000,
        capabilities: &["completion", "chat", "tools"],
        architecture: "mistral",
        quantization_level: "Q4_K_M",
    }),
    ("deepseek", ModelInfo {
        family: "llama",
        families: &["llama"],
        parameter_size: "7B",
        size_bytes: 4_000_000_000,
        capabilities: &["completion", "chat", "reasoning"],
        architecture: "llama",
        quantization_level: "Q4_K_M",
    }),
    ("gemma", ModelInfo {
        family: "gemma",
        families: &["gemma"],
        parameter_size: "7B",
        size_bytes: 4_000_000_000,
        capabilities: &["completion", "chat"],
        architecture: "gemma",
        quantization_level: "Q4_K_M",
    }),
    ("phi", ModelInfo {
        family: "phi",
        families: &["phi"],
        parameter_size: "3B",
        size_bytes: 2_000_000_000,
        capabilities: &["completion", "chat"],
        architecture: "phi",
        quantization_level: "Q4_K_M",
    }),
    ("embed", ModelInfo {
        family: "embedding",
        families: &["embedding"],
        parameter_size: "335M",
        size_bytes: 1_000_000_000,
        capabilities: &["embeddings"],
        architecture: "transformer",
        quantization_level: "F16",
    }),
];

/// Parameter size lookup table
static SIZE_PATTERNS: &[(&str, &str, u64)] = &[
    ("0.5b", "0.5B", 500_000_000),
    ("1.5b", "1.5B", 1_000_000_000),
    ("2b", "2B", 1_500_000_000),
    ("3b", "3B", 2_000_000_000),
    ("7b", "7B", 4_000_000_000),
    ("8b", "8B", 5_000_000_000),
    ("9b", "9B", 5_500_000_000),
    ("13b", "13B", 8_000_000_000),
    ("14b", "14B", 8_500_000_000),
    ("70b", "70B", 40_000_000_000),
    ("72b", "72B", 42_000_000_000),
];

impl ModelInfo {
    /// Fast model info extraction using lookup table
    pub fn from_name(name: &str) -> Self {
        let lower = name.to_lowercase();

        // Find matching pattern
        let mut info = MODEL_PATTERNS
            .iter()
            .find(|(pattern, _)| lower.contains(pattern))
            .map(|(_, info)| info.clone())
            .unwrap_or_else(|| MODEL_PATTERNS[0].1.clone()); // Default to llama

        // Update size if found in name
        for (pattern, size_str, size_bytes) in SIZE_PATTERNS {
            if lower.contains(pattern) {
                info.parameter_size = size_str;
                info.size_bytes = *size_bytes;
                break;
            }
        }

        info
    }

    /// Generate Ollama-compatible model entry
    pub fn to_ollama_model(&self, name: &str) -> Value {
        json!({
            "name": name,
            "model": name,
            "modified_at": chrono::Utc::now().to_rfc3339(),
            "size": self.size_bytes,
            "digest": format!("{:x}", md5::compute(name.as_bytes())),
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": self.family,
                "families": self.families,
                "parameter_size": self.parameter_size,
                "quantization_level": self.quantization_level
            }
        })
    }

    /// Generate model show response
    pub fn to_show_response(&self, name: &str) -> Value {
        // Generate architecture-specific model_info
        let model_info = match self.architecture {
            "llama" => json!({
                "general.architecture": "llama",
                "general.file_type": 2,
                "general.parameter_count": self.size_bytes / 4, // Rough estimate based on model size
                "general.quantization_version": 2,
                "llama.attention.head_count": 32,
                "llama.attention.head_count_kv": 8,
                "llama.attention.layer_norm_rms_epsilon": 0.00001,
                "llama.block_count": 32,
                "llama.context_length": 8192,
                "llama.embedding_length": 4096,
                "llama.feed_forward_length": 14336,
                "llama.rope.dimension_count": 128,
                "llama.rope.freq_base": 500000,
                "llama.vocab_size": 128256,
                "tokenizer.ggml.bos_token_id": 128000,
                "tokenizer.ggml.eos_token_id": 128009,
                "tokenizer.ggml.model": "gpt2",
                "tokenizer.ggml.pre": "llama-bpe",
                "tokenizer.ggml.merges": [],
                "tokenizer.ggml.tokens": [],
                "tokenizer.ggml.token_type": []
            }),
            "qwen" => json!({
                "general.architecture": "qwen2",
                "general.file_type": 2,
                "general.parameter_count": self.size_bytes / 4,
                "general.quantization_version": 2,
                "qwen2.attention.head_count": 32,
                "qwen2.attention.head_count_kv": 32,
                "qwen2.attention.layer_norm_rms_epsilon": 0.000001,
                "qwen2.block_count": 28,
                "qwen2.context_length": 32768,
                "qwen2.embedding_length": 3584,
                "qwen2.feed_forward_length": 18944,
                "qwen2.rope.dimension_count": 128,
                "qwen2.rope.freq_base": 1000000,
                "qwen2.vocab_size": 151936,
                "tokenizer.ggml.bos_token_id": 151643,
                "tokenizer.ggml.eos_token_id": 151645,
                "tokenizer.ggml.model": "gpt2",
                "tokenizer.ggml.pre": "qwen2",
                "tokenizer.ggml.merges": [],
                "tokenizer.ggml.tokens": [],
                "tokenizer.ggml.token_type": []
            }),
            "mistral" => json!({
                "general.architecture": "mistral",
                "general.file_type": 2,
                "general.parameter_count": self.size_bytes / 4,
                "general.quantization_version": 2,
                "mistral.attention.head_count": 32,
                "mistral.attention.head_count_kv": 8,
                "mistral.attention.layer_norm_rms_epsilon": 0.00001,
                "mistral.block_count": 32,
                "mistral.context_length": 32768,
                "mistral.embedding_length": 4096,
                "mistral.feed_forward_length": 14336,
                "mistral.rope.dimension_count": 128,
                "mistral.rope.freq_base": 1000000,
                "mistral.vocab_size": 32000,
                "tokenizer.ggml.bos_token_id": 1,
                "tokenizer.ggml.eos_token_id": 2,
                "tokenizer.ggml.model": "llama",
                "tokenizer.ggml.pre": "default",
                "tokenizer.ggml.merges": [],
                "tokenizer.ggml.tokens": [],
                "tokenizer.ggml.token_type": []
            }),
            _ => json!({
                "general.architecture": self.architecture,
                "general.file_type": 2,
                "general.parameter_count": self.size_bytes / 4,
                "general.quantization_version": 2,
                "tokenizer.ggml.bos_token_id": 1,
                "tokenizer.ggml.eos_token_id": 2,
                "tokenizer.ggml.model": "gpt2",
                "tokenizer.ggml.pre": "default",
                "tokenizer.ggml.merges": [],
                "tokenizer.ggml.tokens": [],
                "tokenizer.ggml.token_type": []
            })
        };

        json!({
            "modelfile": format!("# Modelfile for {}\nFROM {}", name, name),
            "parameters": format!("temperature {}\ntop_p {}\ntop_k {}",
                DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K),
            "template": "{{ if .System }}{{ .System }}\n{{ end }}{{ .Prompt }}",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": self.family,
                "families": self.families,
                "parameter_size": self.parameter_size,
                "quantization_level": self.quantization_level
            },
            "model_info": model_info,
            "capabilities": self.capabilities,
            "digest": format!("{:x}", md5::compute(name.as_bytes())),
            "size": self.size_bytes,
            "modified_at": chrono::Utc::now().to_rfc3339()
        })
    }
}

/// Fast model name cleaning
pub fn clean_model_name(name: &str) -> &str {
    if name.is_empty() {
        return name;
    }

    // Remove :latest suffix
    let after_latest = if let Some(pos) = name.rfind(":latest") {
        &name[..pos]
    } else {
        name
    };

    // Remove numeric suffixes like :1, :2, etc.
    if let Some(colon_pos) = after_latest.rfind(':') {
        let suffix = &after_latest[colon_pos + 1..];
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) && colon_pos > 0 {
            return &after_latest[..colon_pos];
        }
    }

    after_latest
}

/// Simplified model resolver without caching (optimized for single client)
pub struct ModelResolver<'a> {
    context: RequestContext<'a>,
}

impl<'a> ModelResolver<'a> {
    pub fn new(context: RequestContext<'a>) -> Self {
        Self { context }
    }

    /// Direct model resolution without caching
    pub async fn resolve_model_name(
        &self,
        ollama_model: &str,
        cancellation_token: CancellationToken,
    ) -> Result<String, ProxyError> {
        let cleaned_ollama = clean_model_name(ollama_model);

        // Get available models from LM Studio
        let available_models = self.get_available_models(cancellation_token).await?;

        // Find the best match
        if let Some(matched_model) = self.find_best_match(cleaned_ollama, &available_models) {
            Ok(matched_model)
        } else {
            // Return cleaned name if no match found
            Ok(cleaned_ollama.to_string())
        }
    }

    /// Get available models from LM Studio (simplified)
    async fn get_available_models(&self, cancellation_token: CancellationToken) -> Result<Vec<String>, ProxyError> {
        let url = format!("{}/v1/models", self.context.lmstudio_url);
        let request = CancellableRequest::new(self.context.clone(), cancellation_token);

        let response = match request.make_request(reqwest::Method::GET, &url, None).await {
            Ok(response) => response,
            Err(_) => return Ok(vec![]), // Return empty list if LM Studio unavailable
        };

        if !response.status().is_success() {
            return Ok(vec![]);
        }

        let models_response = match response.json::<Value>().await {
            Ok(json) => json,
            Err(_) => return Ok(vec![]),
        };

        let mut model_names = Vec::new();
        if let Some(data) = models_response.get("data").and_then(|d| d.as_array()) {
            for model in data {
                if let Some(model_id) = model.get("id").and_then(|id| id.as_str()) {
                    model_names.push(model_id.to_string());
                }
            }
        }

        Ok(model_names)
    }

    /// Fast model matching using simple scoring
    fn find_best_match(&self, ollama_name: &str, available_models: &[String]) -> Option<String> {
        let lower_ollama = ollama_name.to_lowercase();

        // 1. Exact match
        for model in available_models {
            if model.to_lowercase() == lower_ollama {
                return Some(model.clone());
            }
        }

        // 2. Find best scoring match
        let mut best_match = None;
        let mut best_score = 0;

        for model in available_models {
            let score = self.calculate_match_score(&lower_ollama, &model.to_lowercase());
            if score > best_score && score >= 2 {
                best_score = score;
                best_match = Some(model.clone());
            }
        }

        best_match
    }

    /// Simple match scoring
    fn calculate_match_score(&self, ollama_name: &str, lm_name: &str) -> usize {
        let ollama_parts: Vec<&str> = ollama_name
            .split(&['-', '_', ':', '.', ' '])
            .filter(|s| s.len() > 2)
            .collect();

        ollama_parts.iter()
            .map(|part| if lm_name.contains(part) { part.len() } else { 0 })
            .sum()
    }
}
