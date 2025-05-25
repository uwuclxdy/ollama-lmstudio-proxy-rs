// src/model.rs - Simplified programmatic model handling without hardcoded JSON

use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::common::{CancellableRequest, RequestContext};
use crate::constants::*;
use crate::metrics::get_global_metrics;
use crate::utils::ProxyError;

/// Simplified model information extracted programmatically
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub family: String,
    pub parameter_size: String,
    pub size_bytes: u64,
    pub architecture: String,
    pub quantization_level: String,
}

impl ModelInfo {
    /// Create model info programmatically from name
    pub fn from_name(name: &str) -> Self {
        let lower = name.to_lowercase();

        // Extract family from model name
        let family = extract_model_family(&lower);

        // Extract parameter size
        let (parameter_size, size_bytes) = extract_model_size(&lower);

        // Determine architecture based on family
        let architecture = match family.as_str() {
            "llama" | "deepseek" => "llama",
            "qwen" => "qwen2",
            "mistral" => "mistral",
            "gemma" => "gemma",
            "phi" => "phi",
            _ => "transformer",
        }.to_string();

        // Determine quantization level
        let quantization_level = extract_quantization_level(&lower);

        Self {
            name: name.to_string(),
            family,
            parameter_size,
            size_bytes,
            architecture,
            quantization_level,
        }
    }

    /// Generate Ollama-compatible model entry programmatically
    pub fn to_ollama_model(&self) -> Value {
        // Ensure model name has :latest suffix for Ollama compatibility
        let ollama_name = if self.name.contains(':') {
            self.name.clone()
        } else {
            format!("{}:latest", self.name)
        };
        
        json!({
            "name": ollama_name,
            "model": ollama_name,
            "modified_at": chrono::Utc::now().to_rfc3339(),
            "size": self.size_bytes,
            "digest": format!("{:x}", md5::compute(ollama_name.as_bytes())),
            "details": {
                "parent_model": "",
                "format": "gguf", 
                "family": self.family,
                "families": [self.family],
                "parameter_size": self.parameter_size,
                "quantization_level": self.quantization_level
            }
        })
    }

    /// Generate model show response programmatically
    pub fn to_show_response(&self) -> Value {
        // Ensure model name has :latest suffix for Ollama compatibility
        let ollama_name = if self.name.contains(':') {
            self.name.clone()
        } else {
            format!("{}:latest", self.name)
        };
        
        let model_info = self.create_model_info();

        json!({
            "modelfile": format!("# Modelfile for {}\nFROM {}", ollama_name, ollama_name),
            "parameters": format!("temperature {}\ntop_p {}\ntop_k {}",
                DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K),
            "template": "{{ if .System }}{{ .System }}\n{{ end }}{{ .Prompt }}",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": self.family,
                "families": [self.family],
                "parameter_size": self.parameter_size,
                "quantization_level": self.quantization_level
            },
            "model_info": model_info,
            "digest": format!("{:x}", md5::compute(ollama_name.as_bytes())),
            "size": self.size_bytes,
            "modified_at": chrono::Utc::now().to_rfc3339()
        })
    }

    /// Create architecture-specific model info programmatically
    fn create_model_info(&self) -> Value {
        let base_params = self.get_base_architecture_params();
        let mut model_info = json!({
            "general.architecture": self.architecture,
            "general.file_type": 2,
            "general.parameter_count": self.size_bytes / 4,
            "general.quantization_version": 2,
        });

        // Add architecture-specific parameters
        if let Some(obj) = model_info.as_object_mut() {
            if let Some(base_obj) = base_params.as_object() {
                for (key, value) in base_obj {
                    obj.insert(key.clone(), value.clone());
                }
            }
        }

        model_info
    }

    /// Get base parameters for specific architectures
    fn get_base_architecture_params(&self) -> Value {
        match self.architecture.as_str() {
            "llama" => json!({
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
                "tokenizer.ggml.pre": "llama-bpe"
            }),
            "qwen2" => json!({
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
                "tokenizer.ggml.pre": "qwen2"
            }),
            "mistral" => json!({
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
                "tokenizer.ggml.pre": "default"
            }),
            _ => json!({
                "tokenizer.ggml.bos_token_id": 1,
                "tokenizer.ggml.eos_token_id": 2,
                "tokenizer.ggml.model": "gpt2",
                "tokenizer.ggml.pre": "default"
            })
        }
    }
}

/// Extract model family from name programmatically
fn extract_model_family(name: &str) -> String {
    const FAMILY_PATTERNS: &[(&str, &str)] = &[
        ("llama", "llama"),
        ("qwen", "qwen2"),
        ("mistral", "mistral"),
        ("deepseek", "llama"), // DeepSeek uses Llama architecture
        ("gemma", "gemma"),
        ("phi", "phi"),
        ("embed", "embedding"),
        ("nomic", "embedding"),
    ];

    for (pattern, family) in FAMILY_PATTERNS {
        if name.contains(pattern) {
            return family.to_string();
        }
    }

    "llama".to_string() // Default fallback
}

/// Extract model size programmatically
fn extract_model_size(name: &str) -> (String, u64) {
    const SIZE_PATTERNS: &[(&str, &str, u64)] = &[
        ("0.5b", "0.5B", 500_000_000),
        ("1b", "1B", 1_000_000_000),
        ("1.5b", "1.5B", 1_500_000_000),
        ("2b", "2B", 2_000_000_000),
        ("3b", "3B", 3_000_000_000),
        ("7b", "7B", 7_000_000_000),
        ("8b", "8B", 8_000_000_000),
        ("9b", "9B", 9_000_000_000),
        ("13b", "13B", 13_000_000_000),
        ("14b", "14B", 14_000_000_000),
        ("70b", "70B", 70_000_000_000),
        ("72b", "72B", 72_000_000_000),
        ("405b", "405B", 405_000_000_000),
        // Handle common variations
        ("1.8b", "1.8B", 1_800_000_000),
        ("3.2b", "3.2B", 3_200_000_000),
        ("11b", "11B", 11_000_000_000),
        ("22b", "22B", 22_000_000_000),
        ("34b", "34B", 34_000_000_000),
    ];

    for (pattern, size_str, size_bytes) in SIZE_PATTERNS {
        if name.contains(pattern) {
            return (size_str.to_string(), *size_bytes);
        }
    }

    ("7B".to_string(), DEFAULT_MODEL_SIZE_BYTES) // Default fallback
}

/// Extract quantization level programmatically
fn extract_quantization_level(name: &str) -> String {
    const QUANT_PATTERNS: &[(&str, &str)] = &[
        ("q2_k", "Q2_K"),
        ("q3_k", "Q3_K"),
        ("q4_k", "Q4_K"),
        ("q4_k_m", "Q4_K_M"),
        ("q4_k_s", "Q4_K_S"),
        ("q5_k", "Q5_K"),
        ("q5_k_m", "Q5_K_M"),
        ("q5_k_s", "Q5_K_S"),
        ("q6_k", "Q6_K"),
        ("q8_0", "Q8_0"),
        ("f16", "F16"),
        ("f32", "F32"),
        ("iq1", "IQ1"),
        ("iq2", "IQ2"),
        ("iq3", "IQ3"),
        ("iq4", "IQ4"),
    ];

    for (pattern, quant) in QUANT_PATTERNS {
        if name.contains(pattern) {
            return quant.to_string();
        }
    }

    "Q4_K_M".to_string() // Default fallback
}

/// Optimized model name cleaning
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

/// ModelResolver for handling model resolution with LM Studio
pub struct ModelResolver<'a> {
    context: RequestContext<'a>,
}

impl<'a> ModelResolver<'a> {
    pub fn new(context: RequestContext<'a>) -> Self {
        Self { context }
    }

    /// Direct model resolution with fail-fast approach
    pub async fn resolve_model_name(
        &self,
        ollama_model: &str,
        cancellation_token: CancellationToken,
    ) -> Result<String, ProxyError> {
        let cleaned_ollama = clean_model_name(ollama_model);

        // Record model load attempt in metrics
        if let Some(metrics) = get_global_metrics() {
            metrics.record_model_load(cleaned_ollama, true).await;
        }

        // Get available models from LM Studio with fail-fast
        match self.get_available_models(cancellation_token).await {
            Ok(available_models) => {
                // Find the best match
                if let Some(matched_model) = self.find_best_match(cleaned_ollama, &available_models) {
                    Ok(matched_model)
                } else {
                    // Return cleaned name if no match found
                    Ok(cleaned_ollama.to_string())
                }
            },
            Err(e) => {
                // Fail fast - don't continue if LM Studio is unavailable
                if let Some(metrics) = get_global_metrics() {
                    metrics.record_model_load(cleaned_ollama, false).await;
                }
                Err(e)
            }
        }
    }

    /// Get available models from LM Studio with fail-fast strategy
    async fn get_available_models(&self, cancellation_token: CancellationToken) -> Result<Vec<String>, ProxyError> {
        let url = format!("{}/v1/models", self.context.lmstudio_url);
        let request = CancellableRequest::new(self.context.clone(), cancellation_token);

        let response = request.make_request(reqwest::Method::GET, &url, None).await?;

        if !response.status().is_success() {
            return Err(ProxyError::new(
                ERROR_LM_STUDIO_UNAVAILABLE.to_string(),
                response.status().as_u16()
            ));
        }

        let models_response = response.json::<Value>().await
            .map_err(|_| ProxyError::internal_server_error("Invalid JSON from LM Studio models endpoint"))?;

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

    /// Enhanced model matching with better scoring
    fn find_best_match(&self, ollama_name: &str, available_models: &[String]) -> Option<String> {
        let lower_ollama = ollama_name.to_lowercase();

        // 1. Exact match (highest priority)
        for model in available_models {
            if model.to_lowercase() == lower_ollama {
                return Some(model.clone());
            }
        }

        // 2. Find best scoring match with improved algorithm
        let mut best_match = None;
        let mut best_score = 0;

        for model in available_models {
            let score = self.calculate_enhanced_match_score(&lower_ollama, &model.to_lowercase());
            if score > best_score && score >= 3 { // Raised threshold for better matches
                best_score = score;
                best_match = Some(model.clone());
            }
        }

        best_match
    }

    /// Enhanced match scoring algorithm
    fn calculate_enhanced_match_score(&self, ollama_name: &str, lm_name: &str) -> usize {
        let ollama_parts: Vec<&str> = ollama_name
            .split(&['-', '_', ':', '.', ' '])
            .filter(|s| s.len() > 1) // Filter very short parts
            .collect();

        let lm_parts: Vec<&str> = lm_name
            .split(&['-', '_', ':', '.', ' '])
            .filter(|s| s.len() > 1)
            .collect();

        let mut score = 0;

        // Score exact part matches
        for ollama_part in &ollama_parts {
            for lm_part in &lm_parts {
                if ollama_part == lm_part {
                    score += ollama_part.len() * 2; // Exact matches get double score
                } else if lm_part.contains(ollama_part) || ollama_part.contains(lm_part) {
                    score += ollama_part.len().min(lm_part.len()); // Partial matches
                }
            }
        }

        // Bonus for model family matches
        let ollama_family = extract_model_family(ollama_name);
        let lm_family = extract_model_family(lm_name);
        if ollama_family == lm_family {
            score += 5;
        }

        // Bonus for size matches
        let (ollama_size, _) = extract_model_size(ollama_name);
        let (lm_size, _) = extract_model_size(lm_name);
        if ollama_size == lm_size {
            score += 3;
        }

        score
    }
}
