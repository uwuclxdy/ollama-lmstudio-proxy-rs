/// src/model-legacy.rs - Legacy model handling with programmatic calculations (OpenAI-compatible endpoints)
use moka::future::Cache;
use serde_json::{json, Value};
use std::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::common::CancellableRequest;
use crate::constants::*;
use crate::utils::{log_info, ProxyError};
use crate::{log_timed, log_warning};

/// Legacy model information with calculated estimates
#[derive(Debug, Clone)]
pub struct ModelInfoLegacy {
    pub id_from_lm_studio: String,
    pub ollama_name: String,
    pub family: String,
    pub parameter_size_str: String,
    pub size_bytes: u64,
    pub architecture: String,
    pub quantization_level: String,
}

impl ModelInfoLegacy {
    /// Create model info programmatically from an LM Studio model ID
    pub fn from_lm_studio_id_legacy(lm_studio_id: &str) -> Self {
        let lower_id = lm_studio_id.to_lowercase();

        let family = extract_model_family_legacy(&lower_id);
        let (parameter_size_str, size_bytes) = extract_model_size_legacy(&lower_id);
        let architecture = extract_architecture_legacy(&lower_id, &family);
        let quantization_level = extract_quantization_level_legacy(&lower_id);

        let ollama_name = if lm_studio_id.contains(':') {
            lm_studio_id.to_string()
        } else {
            format!("{}:latest", lm_studio_id)
        };

        Self {
            id_from_lm_studio: lm_studio_id.to_string(),
            ollama_name,
            family,
            parameter_size_str,
            size_bytes,
            architecture,
            quantization_level,
        }
    }

    /// Determine model capabilities based on name and family
    fn determine_capabilities_legacy(&self) -> Vec<String> {
        let mut caps = Vec::new();
        let lower_name = self.ollama_name.to_lowercase();
        let lower_family = self.family.to_lowercase();

        caps.push("completion".to_string());

        if lower_name.contains("instruct")
            || lower_name.contains("chat")
            || lower_family.contains("instruct")
            || lower_family.contains("chat")
        {
            if !caps.contains(&"chat".to_string()) {
                caps.push("chat".to_string());
            }
        }

        if lower_name.contains("llava")
            || lower_name.contains("vision")
            || lower_name.contains("bakllava")
            || lower_family.contains("llava")
            || lower_family.contains("vision")
            || lower_family.contains("bakllava")
        {
            if !caps.contains(&"vision".to_string()) {
                caps.push("vision".to_string());
            }
        }

        if lower_family == "embedding" || lower_name.contains("embed") {
            if !caps.contains(&"embedding".to_string()) {
                caps.push("embedding".to_string());
            }
        }

        if caps.contains(&"chat".to_string()) && !caps.contains(&"completion".to_string()) {
            caps.push("completion".to_string());
        }

        if caps.is_empty() {
            caps.push("completion".to_string());
        }

        caps
    }

    /// Generate Ollama-compatible model entry for /api/tags
    pub fn to_ollama_tags_model_legacy(&self) -> Value {
        json!({
            "name": self.ollama_name,
            "model": self.ollama_name,
            "modified_at": chrono::Utc::now().to_rfc3339(),
            "size": self.size_bytes,
            "digest": format!("{:x}", md5::compute(self.ollama_name.as_bytes())),
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": self.family,
                "families": if self.family.is_empty() { json!([]) } else { json!([self.family]) },
                "parameter_size": self.parameter_size_str,
                "quantization_level": self.quantization_level
            }
        })
    }

    /// Generate Ollama-compatible model entry for /api/ps
    pub fn to_ollama_ps_model_legacy(&self) -> Value {
        json!({
            "name": self.ollama_name,
            "model": self.ollama_name,
            "size": self.size_bytes,
            "digest": format!("{:x}", md5::compute(self.ollama_name.as_bytes())),
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": self.family,
                "families": if self.family.is_empty() { json!([]) } else { json!([self.family]) },
                "parameter_size": self.parameter_size_str,
                "quantization_level": self.quantization_level
            },
            "expires_at": (chrono::Utc::now() + chrono::Duration::minutes(DEFAULT_KEEP_ALIVE_MINUTES)).to_rfc3339(),
            "size_vram": self.size_bytes
        })
    }

    /// Generate model show response for /api/show
    pub fn to_show_response_legacy(&self) -> Value {
        let model_info_details = self.create_fabricated_model_info_details_legacy();
        let capabilities = self.determine_capabilities_legacy();

        json!({
            "modelfile": format!("# Modelfile for {}\nFROM {} # (Fabricated by proxy)\n\nPARAMETER temperature {}\nPARAMETER top_p {}\nPARAMETER top_k {}\n\nTEMPLATE \"\"\"{{ if .System }}{{ .System }} {{ end }}{{ .Prompt }}\"\"\"",
                self.ollama_name, self.ollama_name, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K
            ),
            "parameters": format!("temperature {}\ntop_p {}\ntop_k {}\nrepeat_penalty {}",
                DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K, DEFAULT_REPEAT_PENALTY),
            "template": "{{ if .System }}{{ .System }}\\n{{ end }}{{ .Prompt }}",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": self.family,
                "families": if self.family.is_empty() { json!([]) } else { json!([self.family]) },
                "parameter_size": self.parameter_size_str,
                "quantization_level": self.quantization_level
            },
            "model_info": model_info_details,
            "capabilities": capabilities,
            "digest": format!("{:x}", md5::compute(self.ollama_name.as_bytes())),
            "size": self.size_bytes,
            "modified_at": chrono::Utc::now().to_rfc3339()
        })
    }

    /// Create the nested "model_info" object for /api/show with fabricated details
    fn create_fabricated_model_info_details_legacy(&self) -> Value {
        let mut model_info = json!({
            "general.architecture": self.architecture,
            "general.file_type": 2,
            "general.parameter_count": self.size_bytes / estimate_bytes_per_parameter_legacy(&self.quantization_level),
            "general.quantization_version": 2,
        });

        let arch_params = self.get_base_architecture_params_legacy();
        if let Some(obj) = model_info.as_object_mut() {
            if let Some(base_obj) = arch_params.as_object() {
                for (key, value) in base_obj {
                    obj.insert(key.clone(), value.clone());
                }
            }
        }

        if let Some(obj) = model_info.as_object_mut() {
            obj.insert(
                "tokenizer.ggml.model".to_string(),
                json!(self.family.split('-').next().unwrap_or("unknown")),
            );
            obj.insert("tokenizer.ggml.tokens_count".to_string(), json!(32000));
            obj.insert("tokenizer.ggml.token_type_count".to_string(), json!(1));
            obj.insert("tokenizer.ggml.bos_token_id".to_string(), json!(1));
            obj.insert("tokenizer.ggml.eos_token_id".to_string(), json!(2));
            obj.insert("tokenizer.ggml.unknown_token_id".to_string(), json!(0));
            obj.insert("tokenizer.ggml.padding_token_id".to_string(), json!(null));
            obj.insert("tokenizer.ggml.merges".to_string(), json!([]));
            obj.insert("tokenizer.ggml.tokens".to_string(), json!([]));
            obj.insert("tokenizer.ggml.token_type".to_string(), json!([]));
            obj.insert(
                "tokenizer.ggml.pre".to_string(),
                json!(self.architecture.to_lowercase()),
            );
        }

        model_info
    }

    /// Get base parameters for specific architectures
    fn get_base_architecture_params_legacy(&self) -> Value {
        match self.architecture.as_str() {
            "llama" => json!({
                "llama.attention.head_count": 32,
                "llama.attention.head_count_kv": 8,
                "llama.block_count": 32,
                "llama.embedding_length": 4096,
                "llama.context_length": if self.parameter_size_str.contains("8b") || self.parameter_size_str.contains("3.1") || self.parameter_size_str.contains("32k") || self.parameter_size_str.contains("128k") || self.parameter_size_str.contains("8192") { 8192 } else { 4096 },
                "llama.rope.dimension_count": 128,
                "llama.attention.layer_norm_rms_epsilon": 1e-5,
                "llama.rope.freq_base": if self.ollama_name.contains("codellama") { 1_000_000.0 } else { 10_000.0 },
                "llama.vocab_size": 32000,
            }),
            "qwen2" => json!({
                "qwen2.attention.head_count": 32,
                "qwen2.attention.head_count_kv": if self.parameter_size_str.contains("0.5b") { 4 } else { 8 },
                "qwen2.block_count": if self.parameter_size_str.contains("0.5b") { 24 } else { 28 },
                "qwen2.embedding_length": if self.parameter_size_str.contains("0.5b") { 1024 } else { 3584 },
                "qwen2.context_length": 32768,
                "qwen2.attention.layer_norm_rms_epsilon": 1e-6,
                "qwen2.feed_forward_length": if self.parameter_size_str.contains("0.5b") { 2816 } else { 9728 },
                "qwen2.rope.freq_base": 1_000_000.0,
                "qwen2.vocab_size": 151936,
            }),
            "mistral" | "mixtral" => json!({
                "mistral.attention.head_count": 32,
                "mistral.attention.head_count_kv": 8,
                "mistral.block_count": 32,
                "mistral.embedding_length": 4096,
                "mistral.context_length": 32768,
                "mistral.attention.layer_norm_rms_epsilon": 1e-5,
                "mistral.feed_forward_length": 14336,
                "mistral.rope.dimension_count": 128,
                "mistral.rope.freq_base": 1_000_000.0,
                "mistral.vocab_size": 32000,
            }),
            "gemma" => json!({
                "gemma.attention.head_count": if self.parameter_size_str.contains("2b") {8} else {16},
                "gemma.attention.head_count_kv": 1,
                "gemma.block_count": if self.parameter_size_str.contains("2b") {18} else {28},
                "gemma.embedding_length": if self.parameter_size_str.contains("2b") {2048} else {3072},
                "gemma.context_length": 8192,
                "gemma.attention.layer_norm_rms_epsilon": 1e-6,
                "gemma.feed_forward_length": if self.parameter_size_str.contains("2b") {16384} else {24576},
                "gemma.vocab_size": 256000,
            }),
            _ => json!({})
        }
    }
}

/// Helper to estimate bytes per parameter based on quantization
fn estimate_bytes_per_parameter_legacy(quant_level: &str) -> u64 {
    let q_lower = quant_level.to_lowercase();
    if q_lower.contains("q2") {
        3
    } else if q_lower.contains("q3") {
        4
    } else if q_lower.contains("q4") {
        5
    } else if q_lower.contains("q5") {
        6
    } else if q_lower.contains("q6") {
        7
    } else if q_lower.contains("q8") {
        9
    } else if q_lower.contains("f16") {
        16
    } else if q_lower.contains("f32") {
        32
    } else {
        5
    }
}

/// Extract model family from name programmatically
fn extract_model_family_legacy(name: &str) -> String {
    const FAMILY_PATTERNS: &[(&str, &str)] = &[
        ("llama", "llama"),
        ("codellama", "llama"),
        ("qwen", "qwen2"),
        ("mistral", "mistral"),
        ("mixtral", "mixtral"),
        ("deepseek-coder", "deepseek"),
        ("deepseek-moe", "deepseek"),
        ("deepseek-llm", "deepseek"),
        ("deepseek", "deepseek"),
        ("gemma", "gemma"),
        ("phi", "phi"),
        ("starcoder", "starcoder"),
        ("stablelm", "stablelm"),
        ("command-r", "cohere"),
        ("cohere", "cohere"),
        ("all-minilm", "embedding"),
        ("nomic-embed", "embedding"),
        ("bge-", "embedding"),
        ("gte-", "embedding"),
        ("embed", "embedding"),
    ];

    for (pattern, family) in FAMILY_PATTERNS {
        if name.contains(pattern) {
            return family.to_string();
        }
    }
    name.split(&['-', ':', '/'][..])
        .next()
        .unwrap_or("unknown")
        .to_string()
}

/// Extract model architecture from name and family
fn extract_architecture_legacy(name: &str, family: &str) -> String {
    match family {
        "llama" | "deepseek" | "mixtral" | "stablelm" => return "llama".to_string(),
        "qwen2" => return "qwen2".to_string(),
        "mistral" => return "mistral".to_string(),
        "gemma" => return "gemma".to_string(),
        "phi" => return "phi".to_string(),
        "starcoder" => return "gpt_bigcode".to_string(),
        "cohere" => return "cohere".to_string(),
        "embedding" => return "bert".to_string(),
        _ => {}
    }
    if name.contains("qwen") {
        return "qwen2".to_string();
    }
    if name.contains("llama") || name.contains("codellama") {
        return "llama".to_string();
    }
    if name.contains("mistral") || name.contains("mixtral") {
        return "mistral".to_string();
    }

    if !family.is_empty() && family != "unknown" {
        family.to_string()
    } else {
        "transformer".to_string()
    }
}

/// Extract model size programmatically
fn extract_model_size_legacy(name: &str) -> (String, u64) {
    const SIZE_PATTERNS: &[(&str, &str, u64)] = &[
        ("0.5b", "0.5B", 500_000_000),
        ("500m", "0.5B", 500_000_000),
        ("1.5b", "1.5B", 1_500_000_000),
        ("1b5", "1.5B", 1_500_000_000),
        ("1.6b", "1.6B", 1_600_000_000),
        ("1.8b", "1.8B", 1_800_000_000),
        ("2.7b", "2.7B", 2_700_000_000),
        ("2b7", "2.7B", 2_700_000_000),
        ("3.1b", "3.1B", 3_100_000_000),
        ("3.8b", "3.8B", 3_800_000_000),
        ("1b", "1B", 1_000_000_000),
        ("2b", "2B", 2_000_000_000),
        ("3b", "3B", 3_000_000_000),
        ("4b", "4B", 4_000_000_000),
        ("6b", "6B", 6_000_000_000),
        ("7b", "7B", 7_000_000_000),
        ("8b", "8B", 8_000_000_000),
        ("9b", "9B", 9_000_000_000),
        ("11b", "11B", 11_000_000_000),
        ("13b", "13B", 13_000_000_000),
        ("14b", "14B", 14_000_000_000),
        ("15b", "15B", 15_000_000_000),
        ("16b", "16B", 16_000_000_000),
        ("20b", "20B", 20_000_000_000),
        ("22b", "22B", 22_000_000_000),
        ("30b", "30B", 30_000_000_000),
        ("32b", "32B", 32_000_000_000),
        ("34b", "34B", 34_000_000_000),
        ("40b", "40B", 40_000_000_000),
        ("65b", "65B", 65_000_000_000),
        ("70b", "70B", 70_000_000_000),
        ("72b", "72B", 72_000_000_000),
        ("8x7b", "56B", 56_000_000_000),
        ("8x22b", "176B", 176_000_000_000),
        ("120b", "120B", 120_000_000_000),
        ("128b", "128B", 128_000_000_000),
        ("175b", "175B", 175_000_000_000),
        ("180b", "180B", 180_000_000_000),
        ("405b", "405B", 405_000_000_000),
    ];

    for (pattern, size_str, size_bytes_val) in SIZE_PATTERNS {
        if name.contains(pattern) {
            let quant = extract_quantization_level_legacy(name);
            let multiplier = match quant.as_str() {
                "Q2_K" | "Q2_K_S" => 0.35,
                "Q3_K_S" | "Q3_K_M" | "Q3_K_L" => 0.45,
                "Q4_0" | "Q4_1" => 0.5,
                "Q4_K_S" | "Q4_K_M" => 0.55,
                "Q5_0" | "Q5_1" => 0.625,
                "Q5_K_S" | "Q5_K_M" => 0.675,
                "Q6_K" => 0.75,
                "Q8_0" => 1.0,
                "F16" => 2.0,
                "F32" => 4.0,
                _ => 0.55,
            };
            let estimated_file_size = (*size_bytes_val as f64 * multiplier) as u64;

            return (size_str.to_string(), estimated_file_size.max(100_000_000));
        }
    }
    ("unknown".to_string(), DEFAULT_MODEL_SIZE_BYTES)
}

/// Extract quantization level programmatically
fn extract_quantization_level_legacy(name: &str) -> String {
    const QUANT_PATTERNS: &[(&str, &str)] = &[
        ("q2_k_s", "Q2_K_S"),
        ("q2_k", "Q2_K"),
        ("q3_k_s", "Q3_K_S"),
        ("q3_k_m", "Q3_K_M"),
        ("q3_k_l", "Q3_K_L"),
        ("q3_k", "Q3_K"),
        ("q4_0", "Q4_0"),
        ("q4_1", "Q4_1"),
        ("q4_k_s", "Q4_K_S"),
        ("q4_k_m", "Q4_K_M"),
        ("q4_k", "Q4_K"),
        ("q5_0", "Q5_0"),
        ("q5_1", "Q5_1"),
        ("q5_k_s", "Q5_K_S"),
        ("q5_k_m", "Q5_K_M"),
        ("q5_k", "Q5_K"),
        ("q6_k", "Q6_K"),
        ("q8_0", "Q8_0"),
        ("q8_1", "Q8_1"),
        ("q8_k_s", "Q8_K_S"),
        ("q8_k", "Q8_K"),
        ("iq1_s", "IQ1_S"),
        ("iq1_m", "IQ1_M"),
        ("iq2_xs", "IQ2_XS"),
        ("iq2_s", "IQ2_S"),
        ("iq2_m", "IQ2_M"),
        ("iq2_xxs", "IQ2_XXS"),
        ("iq3_s", "IQ3_S"),
        ("iq3_m", "IQ3_M"),
        ("iq3_xs", "IQ3_XS"),
        ("iq3_xxs", "IQ3_XXS"),
        ("iq4_xs", "IQ4_XS"),
        ("iq4_nl", "IQ4_NL"),
        ("bpw", "BPW"),
        ("f16", "F16"),
        ("fp16", "F16"),
        ("f32", "F32"),
        ("fp32", "F32"),
        ("gguf", "GGUF"),
    ];

    for (pattern, quant) in QUANT_PATTERNS {
        if name.contains(pattern) {
            return quant.to_string();
        }
    }
    if name.contains("gguf") {
        "Q4_K_M".to_string()
    } else {
        "unknown".to_string()
    }
}

/// Optimized model name cleaning
pub fn clean_model_name_legacy(name: &str) -> &str {
    if name.is_empty() {
        return name;
    }
    let after_latest = if let Some(pos) = name.rfind(":latest") {
        &name[..pos]
    } else {
        name
    };
    if let Some(colon_pos) = after_latest.rfind(':') {
        let suffix = &after_latest[colon_pos + 1..];
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) && colon_pos > 0 {
            return &after_latest[..colon_pos];
        }
    }
    after_latest
}

/// Legacy ModelResolver for handling model resolution with OpenAI-compatible endpoints
pub struct ModelResolverLegacy {
    lmstudio_url: String,
    cache: Cache<String, String>,
}

impl ModelResolverLegacy {
    /// Create new legacy model resolver
    pub fn new_legacy(lmstudio_url: String, cache: Cache<String, String>) -> Self {
        Self {
            lmstudio_url,
            cache,
        }
    }

    /// Direct model resolution with fail-fast approach and caching
    pub async fn resolve_model_name_legacy(
        &self,
        ollama_model_name_requested: &str,
        client: &reqwest::Client,
        cancellation_token: CancellationToken,
    ) -> Result<String, ProxyError> {
        let start_time = Instant::now();
        let cleaned_ollama_request = clean_model_name_legacy(ollama_model_name_requested).to_string();

        if let Some(cached_lm_studio_id) = self.cache.get(&cleaned_ollama_request).await {
            log_timed(LOG_PREFIX_SUCCESS, &format!(
                "Cache hit - legacy: '{}' -> '{}'",
                cleaned_ollama_request, cached_lm_studio_id
            ), start_time);
            return Ok(cached_lm_studio_id);
        }

        log_warning("Fetching from LM Studio - legacy.", &format!("Cache miss: '{}'.", cleaned_ollama_request));

        match self
            .get_available_lm_studio_models_legacy(client, cancellation_token)
            .await
        {
            Ok(available_lm_studio_ids) => {
                if let Some(matched_lm_studio_id) =
                    self.find_best_match_legacy(&cleaned_ollama_request, &available_lm_studio_ids)
                {
                    self.cache
                        .insert(cleaned_ollama_request.clone(), matched_lm_studio_id.clone())
                        .await;
                    log_info(&format!(
                        "Resolved and cached (legacy): '{}' -> '{}'",
                        cleaned_ollama_request, matched_lm_studio_id
                    ));
                    Ok(matched_lm_studio_id)
                } else {
                    Ok(cleaned_ollama_request)
                }
            }
            Err(e) => Err(e),
        }
    }

    /// Get available models from LM Studio legacy endpoints
    async fn get_available_lm_studio_models_legacy(
        &self,
        client: &reqwest::Client,
        cancellation_token: CancellationToken,
    ) -> Result<Vec<String>, ProxyError> {
        let url = format!("{}/v1/models", self.lmstudio_url);

        let temp_context = crate::common::RequestContext {
            client,
            lmstudio_url: &self.lmstudio_url,
        };
        let request = CancellableRequest::new(temp_context, cancellation_token);

        let response = request
            .make_request(reqwest::Method::GET, &url, None::<Value>)
            .await?;

        if !response.status().is_success() {
            return Err(ProxyError::new(
                format!(
                    "{}: {}",
                    ERROR_LM_STUDIO_UNAVAILABLE,
                    response.status()
                ),
                response.status().as_u16(),
            ));
        }

        let models_response = response
            .json::<Value>()
            .await
            .map_err(|e| {
                ProxyError::internal_server_error(&format!(
                    "Invalid JSON from LM Studio /v1/models: {}",
                    e
                ))
            })?;

        let mut model_ids = Vec::new();
        if let Some(data) = models_response.get("data").and_then(|d| d.as_array()) {
            for model_entry in data {
                if let Some(model_id) = model_entry.get("id").and_then(|id| id.as_str()) {
                    model_ids.push(model_id.to_string());
                }
            }
        }
        Ok(model_ids)
    }

    /// Enhanced model matching with better scoring
    fn find_best_match_legacy(
        &self,
        ollama_name_cleaned: &str,
        available_lm_studio_ids: &[String],
    ) -> Option<String> {
        let lower_ollama = ollama_name_cleaned.to_lowercase();

        for lm_id in available_lm_studio_ids {
            if lm_id.to_lowercase() == lower_ollama {
                return Some(lm_id.clone());
            }
        }

        for lm_id in available_lm_studio_ids {
            if lm_id.to_lowercase().contains(&lower_ollama) {
                if lower_ollama.len() > lm_id.len() / 2 || lower_ollama.len() > 10 {
                    return Some(lm_id.clone());
                }
            }
        }

        let mut best_match = None;
        let mut best_score = 0;
        for lm_id in available_lm_studio_ids {
            let score = self.calculate_enhanced_match_score_legacy(&lower_ollama, &lm_id.to_lowercase());
            if score > best_score && score >= 3 {
                best_score = score;
                best_match = Some(lm_id.clone());
            }
        }

        if best_match.is_none()
            && ollama_name_cleaned.contains('/')
            && ollama_name_cleaned.to_lowercase().ends_with(".gguf")
        {
            return Some(ollama_name_cleaned.to_string());
        }
        best_match
    }

    /// Enhanced match scoring algorithm
    fn calculate_enhanced_match_score_legacy(&self, ollama_name: &str, lm_name: &str) -> usize {
        let ollama_parts: Vec<&str> = ollama_name
            .split(&['-', '_', ':', '.', '/', ' '])
            .filter(|s| !s.is_empty() && s.len() > 1)
            .collect();
        let lm_parts: Vec<&str> = lm_name
            .split(&['-', '_', ':', '.', '/', ' '])
            .filter(|s| !s.is_empty() && s.len() > 1)
            .collect();

        let mut score = 0;

        for ollama_part in &ollama_parts {
            for lm_part in &lm_parts {
                if ollama_part == lm_part {
                    score += ollama_part.len() * 2;
                } else if lm_part.contains(ollama_part) || ollama_part.contains(lm_part) {
                    score += ollama_part.len().min(lm_part.len());
                }
            }
        }

        let ollama_family = extract_model_family_legacy(ollama_name);
        let lm_family = extract_model_family_legacy(lm_name);
        if ollama_family == lm_family && ollama_family != "unknown" {
            score += 5;
        }

        let (ollama_size_str, _) = extract_model_size_legacy(ollama_name);
        let (lm_size_str, _) = extract_model_size_legacy(lm_name);
        if ollama_size_str == lm_size_str && ollama_size_str != "unknown" {
            score += 3;
        }

        let cleaned_lm_name = lm_name.split('/').last().unwrap_or(lm_name);
        if cleaned_lm_name.starts_with(ollama_name) {
            score += ollama_name.len();
        }

        score
    }
}
