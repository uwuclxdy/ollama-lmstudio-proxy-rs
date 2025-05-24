use clap::Parser;
use serde_json::Value;
use std::convert::Infallible;
use std::net::SocketAddr;
use warp::{Filter, Rejection, Reply};

use crate::handlers;
use crate::utils::{Logger, ProxyError};

/// Configuration struct with command line arguments
#[derive(Parser, Debug, Clone)]
#[command(name = "ollama-lmstudio-proxy")]
#[command(about = "Proxy server bridging Ollama API and LM Studio")]
pub struct Config {
    /// Address and port to listen on
    #[arg(long, default_value = "0.0.0.0:11434")]
    pub listen: String,

    /// LM Studio API URL
    #[arg(long, default_value = "http://localhost:1234")]
    pub lmstudio_url: String,

    /// Disable request/response logging
    #[arg(long)]
    pub no_log: bool,

    /// Timeout for model loading in seconds
    #[arg(long, default_value = "5")]
    pub load_timeout_seconds: u64,
}

/// Main proxy server struct
#[derive(Clone)]
pub struct ProxyServer {
    pub client: reqwest::Client,
    pub config: Config,
    pub logger: Logger,
}

impl ProxyServer {
    /// Create a new ProxyServer instance
    pub fn new(config: Config) -> Self {
        let client = reqwest::Client::new();
        let logger = Logger::new(!config.no_log);

        Self {
            client,
            config,
            logger,
        }
    }

    /// Start the proxy server
    pub async fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        // Print startup banner
        self.print_startup_banner();

        // Parse listen address
        let addr: SocketAddr = self.config.listen.parse()
            .map_err(|e| format!("Invalid listen address '{}': {}", self.config.listen, e))?;

        // Create the main route handler
        let server = self.clone();
        let routes = warp::method()
            .and(warp::path::full())
            .and(warp::body::json().or(warp::any().map(|| Value::Null)).unify())
            .and_then(move |method: warp::http::Method, path: warp::path::FullPath, body: Value| {
                let server = server.clone();
                async move {
                    handle_request(server, method.to_string(), path.as_str().to_string(), body).await
                }
            })
            .recover(handle_rejection);

        // Start the server
        self.logger.log(&format!("🚀 Server starting on {}", addr));

        warp::serve(routes)
            .run(addr)
            .await;

        Ok(())
    }

    /// Print startup information
    fn print_startup_banner(&self) {
        if self.logger.enabled {
            println!("╭─────────────────────────────────────────────────────────────╮");
            println!("│                  🔄 Ollama ↔ LM Studio Proxy                │");
            println!("╰─────────────────────────────────────────────────────────────╯");
            println!();
            println!("📡 Listen Address:    {}", self.config.listen);
            println!("🎯 LM Studio URL:     {}", self.config.lmstudio_url);
            println!("📝 Logging:           {}", if self.logger.enabled { "✅ Enabled" } else { "❌ Disabled" });
            println!("⏱️  Load Timeout:     {}s", self.config.load_timeout_seconds);
            println!();
            println!("🔌 Supported Endpoints:");
            println!("   • Ollama API:      /api/* (translated to LM Studio)");
            println!("   • LM Studio API:   /v1/* (direct passthrough)");
            println!();
            println!("🚀 Ready to handle requests!");
            println!("─────────────────────────────────────────────────────────────");
        }
    }
}

/// Handle rejections and convert them to proper HTTP responses
/// Enhanced to never return undefined responses that break VS Code extensions
async fn handle_rejection(err: Rejection) -> Result<impl Reply, Infallible> {
    let (code, message, ollama_compatible) = if err.is_not_found() {
        (404, "Not Found".to_string(), false)
    } else if let Some(proxy_error) = err.find::<ProxyError>() {
        (proxy_error.status_code, proxy_error.message.clone(), true)
    } else if err.find::<warp::reject::MethodNotAllowed>().is_some() {
        (405, "Method Not Allowed".to_string(), false)
    } else if err.find::<warp::reject::PayloadTooLarge>().is_some() {
        (413, "Payload Too Large".to_string(), false)
    } else {
        eprintln!("Unhandled rejection: {:?}", err);
        (500, "Internal Server Error".to_string(), false)
    };

    // For Ollama API endpoints, return Ollama-style errors
    let json = if ollama_compatible {
        warp::reply::json(&serde_json::json!({
            "error": {
                "type": "api_error",
                "message": message
            }
        }))
    } else {
        // For other endpoints, return standard error format
        warp::reply::json(&serde_json::json!({
            "error": {
                "type": "proxy_error",
                "message": message
            }
        }))
    };

    Ok(warp::reply::with_status(json, warp::http::StatusCode::from_u16(code).unwrap()))
}

/// Main request handler that routes to appropriate handlers
async fn handle_request(
    server: ProxyServer,
    method: String,
    path: String,
    body: Value,
) -> Result<warp::reply::Response, Rejection> {
    server.logger.log(&format!("{} {} - Body size: {}", method, path,
                               if body.is_null() { 0 } else { body.to_string().len() }));

    let result = match (method.as_str(), path.as_str()) {
        // Ollama API endpoints (translated to LM Studio)
        ("GET", "/api/tags") => handlers::handle_ollama_tags(server).await,
        ("POST", "/api/chat") => handlers::handle_ollama_chat(server, body).await,
        ("POST", "/api/generate") => handlers::handle_ollama_generate(server, body).await,
        ("POST", "/api/embed") | ("POST", "/api/embeddings") =>
            handlers::handle_ollama_embeddings(server, body).await,
        ("POST", "/api/show") => handlers::handle_ollama_show(body).await,

        // Simple Ollama endpoints (no LM Studio calls needed)
        ("GET", "/api/ps") => handlers::handle_ollama_ps().await,
        ("GET", "/api/version") => handlers::handle_ollama_version().await,

        // Unsupported Ollama endpoints
        ("POST", "/api/create") | ("POST", "/api/pull") | ("POST", "/api/push") |
        ("DELETE", "/api/delete") | ("POST", "/api/copy") =>
            handlers::handle_unsupported(&*path).await,

        // LM Studio API (direct passthrough)
        ("GET", "/v1/models") | ("POST", "/v1/chat/completions") |
        ("POST", "/v1/completions") | ("POST", "/v1/embeddings") =>
            handlers::handle_lmstudio_passthrough(server, &method, &path, body).await,

        // Unknown endpoint
        _ => Err(ProxyError::not_found(&format!("Unknown endpoint: {} {}", method, path))),
    };

    match result {
        Ok(response) => Ok(response),
        Err(e) => Err(warp::reject::custom(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = Config {
            listen: "0.0.0.0:11434".to_string(),
            lmstudio_url: "http://localhost:1234".to_string(),
            no_log: false,
            load_timeout_seconds: 5,
        };

        assert_eq!(config.listen, "0.0.0.0:11434");
        assert_eq!(config.lmstudio_url, "http://localhost:1234");
        assert!(!config.no_log);
        assert_eq!(config.load_timeout_seconds, 5);
    }

    #[test]
    fn test_proxy_server_creation() {
        let config = Config {
            listen: "127.0.0.1:8080".to_string(),
            lmstudio_url: "http://localhost:1234".to_string(),
            no_log: true,
            load_timeout_seconds: 10,
        };

        let server = ProxyServer::new(config.clone());
        assert_eq!(server.config.listen, config.listen);
        assert_eq!(server.config.lmstudio_url, config.lmstudio_url);
        assert!(!server.logger.enabled); // no_log = true means logging disabled
    }
}
