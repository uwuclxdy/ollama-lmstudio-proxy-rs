// src/server.rs - Optimized server for single-client use

use clap::Parser;
use serde_json::Value;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;
use warp::log::Info as LogInfo;
use warp::{Filter, Rejection, Reply};

use crate::common::{validate_request_size, RequestContext};
use crate::constants::*;
use crate::handlers;
use crate::utils::{validate_config, Logger, ProxyError};

#[derive(Parser, Debug, Clone)]
#[command(name = "ollama-lmstudio-proxy")]
#[command(about = "High-performance proxy server bridging Ollama API and LM Studio")]
pub struct Config {
    #[arg(long, default_value = "0.0.0.0:11434", help = "Server listen address")]
    pub listen: String,

    #[arg(long, default_value = "http://localhost:1234", help = "LM Studio backend URL")]
    pub lmstudio_url: String,

    #[arg(long, help = "Disable logging output")]
    pub no_log: bool,

    #[arg(long, default_value = "5", help = "Model loading wait timeout in seconds")]
    pub load_timeout_seconds: u64,

    #[arg(long, default_value = "120", help = "HTTP request timeout in seconds")]
    pub request_timeout_seconds: u64,

    #[arg(long, default_value = "30", help = "Streaming timeout in seconds")]
    pub stream_timeout_seconds: u64,
}

/// Lightweight proxy server for single client
#[derive(Clone)]
pub struct ProxyServer {
    pub client: reqwest::Client,
    pub config: Config,
    pub logger: Logger,
}

impl ProxyServer {
    pub fn new(config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        validate_config(&config)?;

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.request_timeout_seconds + 5))
            .build()?;

        let logger = Logger::new(!config.no_log);

        Ok(Self {
            client,
            config,
            logger,
        })
    }

    pub async fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        self.print_startup_banner();

        let addr: SocketAddr = self.config.listen.parse()
            .map_err(|e| format!("Invalid listen address '{}': {}", self.config.listen, e))?;

        // Create shared server reference (minimal Arc usage)
        let server = Arc::new(self);

        // Simplified logging
        let log = warp::log::custom({
            let logger = server.logger.clone();
            move |info: LogInfo| {
                if logger.enabled {
                    let status_icon = match info.status().as_u16() {
                        200..=299 => "✅",
                        400..=499 => "⚠️",
                        500..=599 => "❌",
                        _ => "🔄",
                    };
                    println!("[{}] {} {} {} | {} | {}",
                             chrono::Local::now().format("%H:%M:%S"),
                             status_icon, info.method(), info.path(), info.status(),
                             crate::utils::format_duration(info.elapsed()));
                }
            }
        });

        // Request size validation
        let routes = warp::method()
            .and(warp::path::full())
            .and(warp::body::json().or(warp::any().map(|| Value::Null)).unify())
            .and_then(move |method: warp::http::Method, path: warp::path::FullPath, body: Value| {
                let server = server.clone();
                async move {
                    handle_request_optimized(server, method.to_string(), path.as_str().to_string(), body).await
                }
            })
            .recover(handle_rejection)
            .with(log);

        warp::serve(routes).run(addr).await;
        Ok(())
    }

    fn print_startup_banner(&self) {
        if self.logger.enabled {
            println!();
            println!("Ollama ↔ LM Studio Proxy (written in Rust btw :3)");
            println!("------------------------------------------------------");
            println!("v{}", crate::VERSION);
            println!("Listen Address: {}", self.config.listen);
            println!("LM Studio URL: {}", self.config.lmstudio_url);
            println!("Logging: {}", if self.logger.enabled { "Enabled" } else { "Disabled" });
            println!("Load Timeout: {}s", self.config.load_timeout_seconds);
            println!("Request Timeout: {}s", self.config.request_timeout_seconds);
            println!("Stream Timeout: {}s", self.config.stream_timeout_seconds);
            println!();
        }
    }
}

/// Fixed connection tracker (no race condition)
struct ConnectionTracker {
    token: CancellationToken,
    completed: Arc<AtomicBool>,
}

impl ConnectionTracker {
    fn new(token: CancellationToken) -> Self {
        Self {
            token,
            completed: Arc::new(AtomicBool::new(false)),
        }
    }

    fn mark_completed(&self) {
        self.completed.store(true, Ordering::SeqCst);
    }
}

impl Drop for ConnectionTracker {
    fn drop(&mut self) {
        // Use compare_exchange (not weak) to avoid spurious failures
        if self.completed.compare_exchange(
            false,
            true,
            Ordering::SeqCst,
            Ordering::SeqCst
        ).is_ok() {
            self.token.cancel();
        }
    }
}

/// Optimized request handler with lightweight context
async fn handle_request_optimized(
    server: Arc<ProxyServer>,
    method: String,
    path: String,
    body: Value,
) -> Result<warp::reply::Response, Rejection> {
    let _ = Instant::now();

    // Validate request body size early (only for requests with bodies)
    if !body.is_null() {
        if let Err(e) = validate_request_size(&body) {
            return Err(warp::reject::custom(e));
        }
    }

    let cancellation_token = CancellationToken::new();
    let connection_tracker = ConnectionTracker::new(cancellation_token.clone());

    // Create lightweight request context
    let context = RequestContext {
        client: &server.client,
        logger: &server.logger,
        lmstudio_url: &server.config.lmstudio_url,
        timeout_seconds: server.config.request_timeout_seconds,
    };

    let result = match (method.as_str(), path.as_str()) {
        // Ollama API endpoints
        ("GET", "/api/tags") => {
            handlers::handle_ollama_tags(context, cancellation_token).await
        },
        ("POST", "/api/chat") => {
            handlers::handle_ollama_chat(context, body.clone(), cancellation_token, &server.config).await
        },
        ("POST", "/api/generate") => {
            handlers::handle_ollama_generate(context, body.clone(), cancellation_token, &server.config).await
        },
        ("POST", "/api/embed") | ("POST", "/api/embeddings") => {
            handlers::handle_ollama_embeddings(context, body.clone(), cancellation_token).await
        },
        ("POST", "/api/show") => handlers::handle_ollama_show(body).await,
        ("GET", "/api/ps") => handlers::handle_ollama_ps().await,
        ("GET", "/api/version") => handlers::handle_ollama_version().await,

        // Unsupported Ollama endpoints
        ("POST", "/api/create") | ("POST", "/api/pull") | ("POST", "/api/push") |
        ("DELETE", "/api/delete") | ("POST", "/api/copy") =>
            handlers::handle_unsupported(&path).await,

        // LM Studio API passthrough
        ("GET", "/v1/models") | ("POST", "/v1/chat/completions") |
        ("POST", "/v1/completions") | ("POST", "/v1/embeddings") => {
            match handlers::validate_lmstudio_endpoint(&path) {
                Ok(_) => handlers::handle_lmstudio_passthrough(context, &method, &path, body.clone(), cancellation_token).await,
                Err(e) => Err(e),
            }
        },

        // Health check
        ("GET", "/health") => {
            match handlers::get_lmstudio_status(context, cancellation_token).await {
                Ok(status) => Ok(warp::reply::json(&status).into_response()),
                Err(e) => Err(e),
            }
        },

        // Unknown endpoints
        _ => Err(ProxyError::not_found(&format!("Unknown endpoint: {} {}", method, path))),
    };

    match result {
        Ok(response) => {
            connection_tracker.mark_completed();
            Ok(response)
        },
        Err(e) if e.is_cancelled() => {
            let error_response = serde_json::json!({
                "error": {
                    "type": "request_cancelled",
                    "message": ERROR_CANCELLED
                }
            });
            Ok(warp::reply::with_status(
                warp::reply::json(&error_response),
                warp::http::StatusCode::REQUEST_TIMEOUT,
            ).into_response())
        },
        Err(e) => {
            connection_tracker.mark_completed();
            Err(warp::reject::custom(e))
        }
    }
}

/// Simplified error handling
async fn handle_rejection(err: Rejection) -> Result<impl Reply, Infallible> {
    let (code, message) = if err.is_not_found() {
        (404, "Not Found".to_string())
    } else if let Some(proxy_error) = err.find::<ProxyError>() {
        (proxy_error.status_code, proxy_error.message.clone())
    } else if err.find::<warp::reject::MethodNotAllowed>().is_some() {
        (405, "Method Not Allowed".to_string())
    } else if err.find::<warp::reject::PayloadTooLarge>().is_some() {
        (413, format!("Payload Too Large (max: {}MB)", MAX_REQUEST_SIZE_BYTES / (1024 * 1024)))
    } else {
        (500, "Internal Server Error".to_string())
    };

    let json = warp::reply::json(&serde_json::json!({
        "error": {
            "type": "api_error",
            "message": message,
            "timestamp": chrono::Utc::now().to_rfc3339()
        }
    }));

    Ok(warp::reply::with_status(
        json,
        warp::http::StatusCode::from_u16(code).unwrap_or(warp::http::StatusCode::INTERNAL_SERVER_ERROR)
    ))
}
