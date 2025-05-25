// src/model.rs - Unified model metadata and resolution handling

use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::common::CancellableRequest;
use crate::constants::*;
use crate::server::ProxyServer;
use crate::utils::ProxyError;

/// Complete model information structure
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub family: &'static str,
    pub families: Vec<&'static str>,
    pub parameter_size: &'static str,
    pub size_bytes: u64,
    pub capabilities: Vec<&'static str>,
    pub architecture: &'static str,
    pub quantization_level: &'static str,
}

impl Default for ModelInfo {
    fn default() -> Self {
        Self {
            family: "llama",
            families: vec!["llama"],
            parameter_size: "7B",
            size_bytes: DEFAULT_MODEL_SIZE_BYTES,
            capabilities: vec!["completion", "chat"],
            architecture: "llama",
            quantization_level: "Q4_K_M",
        }
    }
}

impl ModelInfo {
    /// Extract complete model information from name using optimized pattern matching
    pub fn from_name(name: &str) -> Self {
        let lower = name.to_lowercase();

        match () {
            // Llama family models (including derivatives)
            _ if lower.contains("llama") || lower.contains("alpaca") ||
                lower.contains("vicuna") || lower.contains("codellama") => {
                let size = Self::extract_parameter_size(&lower);
                Self {
                    family: "llama",
                    families: vec!["llama"],
                    parameter_size: size,
                    size_bytes: Self::size_to_bytes(size),
                    capabilities: Self::llama_capabilities(&lower),
                    architecture: "llama",
                    quantization_level: "Q4_K_M",
                }
            },

            // Qwen family models
            _ if lower.contains("qwen") => {
                let size = Self::extract_parameter_size(&lower);
                Self {
                    family: "qwen2",
                    families: vec!["qwen2"],
                    parameter_size: size,
                    size_bytes: Self::size_to_bytes(size),
                    capabilities: Self::qwen_capabilities(&lower),
                    architecture: "qwen",
                    quantization_level: "Q4_K_M",
                }
            },

            // Mistral family models
            _ if lower.contains("mistral") => {
                let size = Self::extract_parameter_size(&lower);
                Self {
                    family: "mistral",
                    families: vec!["mistral"],
                    parameter_size: size,
                    size_bytes: Self::size_to_bytes(size),
                    capabilities: Self::mistral_capabilities(&lower),
                    architecture: "mistral",
                    quantization_level: "Q4_K_M",
                }
            },

            // DeepSeek models (uses llama architecture)
            _ if lower.contains("deepseek") => {
                let size = Self::extract_parameter_size(&lower);
                Self {
                    family: "llama",
                    families: vec!["llama"],
                    parameter_size: size,
                    size_bytes: Self::size_to_bytes(size),
                    capabilities: Self::deepseek_capabilities(&lower),
                    architecture: "llama",
                    quantization_level: "Q4_K_M",
                }
            },

            // Gemma family models
            _ if lower.contains("gemma") => {
                let size = Self::extract_parameter_size(&lower);
                Self {
                    family: "gemma",
                    families: vec!["gemma"],
                    parameter_size: size,
                    size_bytes: Self::size_to_bytes(size),
                    capabilities: Self::gemma_capabilities(&lower),
                    architecture: "gemma",
                    quantization_level: "Q4_K_M",
                }
            },

            // Phi family models
            _ if lower.contains("phi") => {
                let size = Self::extract_parameter_size(&lower);
                Self {
                    family: "phi",
                    families: vec!["phi"],
                    parameter_size: size,
                    size_bytes: Self::size_to_bytes(size),
                    capabilities: Self::phi_capabilities(&lower),
                    architecture: "phi",
                    quantization_level: "Q4_K_M",
                }
            },

            // Embedding models
            _ if lower.contains("embed") || lower.contains("bge") ||
                lower.contains("nomic") || lower.contains("e5") => {
                Self {
                    family: "embedding",
                    families: vec!["embedding"],
                    parameter_size: "335M",
                    size_bytes: 1_000_000_000,
                    capabilities: vec!["embeddings"],
                    architecture: "transformer",
                    quantization_level: "F16",
                }
            },

            // Default fallback
            _ => Self::default(),
        }
    }

    /// Extract parameter size from model name using optimized matching
    fn extract_parameter_size(name: &str) -> &'static str {
        match () {
            _ if name.contains("0.5b") => "0.5B",
            _ if name.contains("1.5b") => "1.5B",
            _ if name.contains("2b") => "2B",
            _ if name.contains("3b") => "3B",
            _ if name.contains("7b") => "7B",
            _ if name.contains("8b") => "8B",
            _ if name.contains("9b") => "9B",
            _ if name.contains("13b") => "13B",
            _ if name.contains("14b") => "14B",
            _ if name.contains("27b") => "27B",
            _ if name.contains("30b") => "30B",
            _ if name.contains("32b") => "32B",
            _ if name.contains("70b") => "70B",
            _ if name.contains("72b") => "72B",
            _ if name.contains("405b") => "405B",
            _ => "7B",
        }
    }

    /// Convert parameter size to estimated bytes
    fn size_to_bytes(size: &str) -> u64 {
        match size {
            "0.5B" => 500_000_000,
            "1.5B" => 1_000_000_000,
            "2B" => 1_500_000_000,
            "3B" => 2_000_000_000,
            "7B" => 4_000_000_000,
            "8B" => 5_000_000_000,
            "9B" => 5_500_000_000,
            "13B" => 8_000_000_000,
            "14B" => 8_500_000_000,
            "27B" => 16_000_000_000,
            "30B" => 18_000_000_000,
            "32B" => 20_000_000_000,
            "70B" => 40_000_000_000,
            "72B" => 42_000_000_000,
            "405B" => 200_000_000_000,
            _ => DEFAULT_MODEL_SIZE_BYTES,
        }
    }

    /// Determine Llama model capabilities
    fn llama_capabilities(name: &str) -> Vec<&'static str> {
        let mut caps = vec!["completion", "chat"];
        if name.contains("3") || name.contains("instruct") {
            caps.push("tools");
        }
        if name.contains("vision") || name.contains("llava") {
            caps.push("vision");
        }
        if name.contains("code") || name.contains("codellama") {
            caps.push("code");
        }
        caps
    }

    /// Determine Qwen model capabilities
    fn qwen_capabilities(name: &str) -> Vec<&'static str> {
        let mut caps = vec!["completion", "chat"];
        if name.contains("2") || name.contains("instruct") {
            caps.push("tools");
        }
        if name.contains("vl") || name.contains("vision") {
            caps.push("vision");
        }
        if name.contains("coder") || name.contains("code") {
            caps.push("code");
        }
        caps
    }

    /// Determine Mistral model capabilities
    fn mistral_capabilities(name: &str) -> Vec<&'static str> {
        let mut caps = vec!["completion", "chat"];
        if name.contains("instruct") || name.contains("v0.3") {
            caps.push("tools");
        }
        if name.contains("code") {
            caps.push("code");
        }
        caps
    }

    /// Determine DeepSeek model capabilities
    fn deepseek_capabilities(name: &str) -> Vec<&'static str> {
        let mut caps = vec!["completion", "chat"];
        if name.contains("coder") || name.contains("code") {
            caps.push("code");
        }
        if name.contains("r1") || name.contains("reasoning") {
            caps.push("reasoning");
        }
        if name.contains("instruct") {
            caps.push("tools");
        }
        caps
    }

    /// Determine Gemma model capabilities
    fn gemma_capabilities(name: &str) -> Vec<&'static str> {
        let mut caps = vec!["completion", "chat"];
        if name.contains("code") || name.contains("codegemma") {
            caps.push("code");
        }
        if name.contains("instruct") {
            caps.push("tools");
        }
        caps
    }

    /// Determine Phi model capabilities
    fn phi_capabilities(name: &str) -> Vec<&'static str> {
        let mut caps = vec!["completion", "chat"];
        if name.contains("vision") {
            caps.push("vision");
        }
        if name.contains("instruct") {
            caps.push("tools");
        }
        caps
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
        json!({
            "modelfile": format!("# Modelfile for {}\nFROM {}\nPARAMETER temperature {}\nPARAMETER top_p {}\nPARAMETER top_k {}",
                name, name, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K),
            "parameters": format!("temperature {}\ntop_p {}\ntop_k {}\nrepeat_penalty {}",
                DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K, DEFAULT_REPEAT_PENALTY),
            "template": "{{ if .System }}{{ .System }}\n{{ end }}{{ .Prompt }}",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": self.family,
                "families": self.families,
                "parameter_size": self.parameter_size,
                "quantization_level": self.quantization_level
            },
            "model_info": {
                "general.architecture": self.architecture,
                "general.name": clean_model_name(name),
                "general.parameter_count": self.size_bytes,
                "general.quantization_version": 2,
                "general.file_type": 2,
                "tokenizer.model": self.family,
                "tokenizer.chat_template": "{{ if .System }}{{ .System }}\n{{ end }}{{ .Prompt }}"
            },
            "capabilities": self.capabilities,
            "system": format!("You are a helpful AI assistant using the {} model.", name),
            "license": "Custom license for proxy model",
            "digest": format!("{:x}", md5::compute(name.as_bytes())),
            "size": self.size_bytes,
            "modified_at": chrono::Utc::now().to_rfc3339()
        })
    }
}

/// Clean model name by removing :latest and numeric suffixes
pub fn clean_model_name(name: &str) -> String {
    if name.is_empty() {
        return name.to_string();
    }

    let after_latest = if name.ends_with(":latest") {
        &name[..name.len() - 7]
    } else {
        name
    };

    if let Some(colon_pos) = after_latest.rfind(':') {
        let suffix = &after_latest[colon_pos + 1..];
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) && colon_pos > 0 {
            return after_latest[..colon_pos].to_string();
        }
    }

    after_latest.to_string()
}

/// Cache entry for model resolution
#[derive(Debug, Clone)]
struct CacheEntry {
    resolved_name: String,
    timestamp: Instant,
}

/// Centralized model resolver with caching
pub struct ModelResolver {
    server: Arc<ProxyServer>,
    cache: Arc<Mutex<HashMap<String, CacheEntry>>>,
}

impl ModelResolver {
    pub fn new(server: Arc<ProxyServer>) -> Self {
        Self {
            server,
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Resolve Ollama model name to LM Studio model name with caching
    pub async fn resolve_model_name(
        &self,
        ollama_model: &str,
        cancellation_token: CancellationToken,
    ) -> Result<String, ProxyError> {
        let cleaned_ollama = clean_model_name(ollama_model);

        // Check cache first
        {
            let cache = self.cache.lock().await;
            if let Some(entry) = cache.get(&cleaned_ollama) {
                if entry.timestamp.elapsed() < Duration::from_secs(MODEL_CACHE_TTL_SECONDS) {
                    self.server.logger.log(&format!("✅ Cache hit: '{}' -> '{}'", ollama_model, entry.resolved_name));
                    return Ok(entry.resolved_name.clone());
                }
            }
        }

        // Get available models from LM Studio
        let available_models = self.get_available_models(cancellation_token).await?;

        // Find the best match
        let resolved_name = if let Some(matched_model) = self.find_best_match(&cleaned_ollama, &available_models) {
            self.server.logger.log(&format!("✅ Resolved '{}' -> '{}'", ollama_model, matched_model));
            matched_model
        } else {
            self.server.logger.log(&format!("⚠️ No match found for '{}', using cleaned name '{}'", ollama_model, cleaned_ollama));
            cleaned_ollama.clone()
        };

        // Update cache
        {
            let mut cache = self.cache.lock().await;
            // Clean old entries if cache is getting large
            if cache.len() >= MAX_CACHE_ENTRIES {
                let cutoff = Instant::now() - Duration::from_secs(MODEL_CACHE_TTL_SECONDS);
                cache.retain(|_, entry| entry.timestamp > cutoff);
            }

            cache.insert(cleaned_ollama, CacheEntry {
                resolved_name: resolved_name.clone(),
                timestamp: Instant::now(),
            });
        }

        Ok(resolved_name)
    }

    /// Get available models from LM Studio
    async fn get_available_models(&self, cancellation_token: CancellationToken) -> Result<Vec<String>, ProxyError> {
        let url = format!("{}/v1/models", self.server.config.lmstudio_url);

        let request = CancellableRequest::new(
            self.server.client.clone(),
            cancellation_token,
            self.server.logger.clone(),
            self.server.config.request_timeout_seconds
        );

        let response = match request.make_request(reqwest::Method::GET, &url, None).await {
            Ok(response) => response,
            Err(_) => return Ok(vec![]),
        };

        if !response.status().is_success() {
            return Ok(vec![]);
        }

        let models_response: Value = match response.json().await {
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

    /// Find the best matching LM Studio model using simplified fuzzy matching
    fn find_best_match(&self, ollama_name: &str, available_models: &[String]) -> Option<String> {
        let lower_ollama = ollama_name.to_lowercase();

        // 1. Direct exact match
        for model in available_models {
            if model.to_lowercase() == lower_ollama {
                return Some(model.clone());
            }
        }

        // 2. Family and size matching
        for model in available_models {
            if self.models_match_family_and_size(&lower_ollama, &model.to_lowercase()) {
                return Some(model.clone());
            }
        }

        // 3. Partial name matching with scoring
        let mut best_match = None;
        let mut best_score = 0;

        for model in available_models {
            let score = self.calculate_match_score(&lower_ollama, &model.to_lowercase());
            if score > best_score && score >= 2 { // Minimum threshold
                best_score = score;
                best_match = Some(model.clone());
            }
        }

        best_match
    }

    /// Check if two model names match in family and size
    fn models_match_family_and_size(&self, ollama_name: &str, lm_name: &str) -> bool {
        let families = ["llama", "qwen", "mistral", "gemma", "phi", "deepseek"];
        let sizes = ["0.5b", "1.5b", "2b", "3b", "7b", "8b", "9b", "13b", "14b", "27b", "30b", "32b", "70b", "72b"];

        // Check family match
        let family_match = families.iter().any(|family| {
            ollama_name.contains(family) && lm_name.contains(family)
        });

        if !family_match {
            return false;
        }

        // Check size match
        for size in &sizes {
            if ollama_name.contains(size) && lm_name.contains(size) {
                return true;
            }
        }

        true // If no size found in either, consider them matching
    }

    /// Calculate simple match score based on common substrings
    fn calculate_match_score(&self, ollama_name: &str, lm_name: &str) -> usize {
        let ollama_parts: Vec<&str> = ollama_name
            .split(&['-', '_', ':', '.', ' '])
            .filter(|s| s.len() > 2)
            .collect();

        let mut score = 0;
        for part in &ollama_parts {
            if lm_name.contains(part) {
                score += part.len(); // Weight by part length
            }
        }

        score
    }
}
