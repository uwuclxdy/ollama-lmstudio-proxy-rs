use serde_json::Value;
use warp::Reply;

/// Helper function to convert JSON to Response
pub fn json_response(value: &Value) -> warp::reply::Response {
    warp::reply::with_status(
        warp::reply::json(value),
        warp::http::StatusCode::OK,
    ).into_response()
}

/// Determine model family and families array based on model name
pub fn determine_model_family(model_name: &str) -> (&'static str, Vec<&'static str>) {
    let lower_name = model_name.to_lowercase();

    match lower_name {
        name if name.contains("llama") => ("llama", vec!["llama"]),
        name if name.contains("mistral") => ("mistral", vec!["mistral"]),
        name if name.contains("qwen") => ("qwen2", vec!["qwen2"]),
        name if name.contains("deepseek") => ("llama", vec!["llama"]),
        name if name.contains("gemma") => ("gemma", vec!["gemma"]),
        name if name.contains("phi") => ("phi", vec!["phi"]),
        name if name.contains("codellama") => ("llama", vec!["llama"]),
        name if name.contains("vicuna") => ("llama", vec!["llama"]),
        name if name.contains("alpaca") => ("llama", vec!["llama"]),
        _ => ("llama", vec!["llama"]),
    }
}

/// Determine parameter size based on model name
pub fn determine_parameter_size(model_name: &str) -> &'static str {
    let lower_name = model_name.to_lowercase();

    if lower_name.contains("0.5b") { "0.5B" } else if lower_name.contains("1.5b") { "1.5B" } else if lower_name.contains("2b") { "2B" } else if lower_name.contains("3b") { "3B" } else if lower_name.contains("7b") { "7B" } else if lower_name.contains("8b") { "8B" } else if lower_name.contains("9b") { "9B" } else if lower_name.contains("13b") { "13B" } else if lower_name.contains("14b") { "14B" } else if lower_name.contains("27b") { "27B" } else if lower_name.contains("30b") { "30B" } else if lower_name.contains("32b") { "32B" } else if lower_name.contains("70b") { "70B" } else { "7B" }
}

/// Estimate model size in bytes based on parameter size
pub fn estimate_model_size(parameter_size: &str) -> u64 {
    match parameter_size {
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
        _ => 4_000_000_000,
    }
}

/// Determine model capabilities based on model name
pub fn determine_model_capabilities(model_name: &str) -> Vec<&'static str> {
    let lower_name = model_name.to_lowercase();
    let mut capabilities = vec!["completion", "chat"];

    if lower_name.contains("embed") || lower_name.contains("bge") || lower_name.contains("nomic") {
        capabilities.push("embeddings");
    }

    if lower_name.contains("llava") || lower_name.contains("vision") || lower_name.contains("multimodal") {
        capabilities.push("vision");
    }

    if lower_name.contains("llama3") || lower_name.contains("mistral") || lower_name.contains("qwen") {
        capabilities.push("tools");
    }

    capabilities
}