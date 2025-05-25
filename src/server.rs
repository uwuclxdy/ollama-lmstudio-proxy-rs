// src/server.rs - Updated server using consolidated systems and constants

use clap::Parser;
use serde_json::Value;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use warp::{Filter, Rejection, Reply};
use warp::log::Info as LogInfo;

use crate::constants::*;
use crate::handlers;
use crate::utils::{Logger, ProxyError, validate_config};
use crate::common::validate_request_size;

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

    #[arg(long, default_value = "300", help = "HTTP request timeout in seconds")]
    pub request_timeout_seconds: u64,

    #[arg(long, default_value = "60", help = "Streaming chunk timeout in seconds")]
    pub stream_timeout_seconds: u64,
}

#[derive(Clone)]
pub struct CancellationTokenFactory;

impl CancellationTokenFactory {
    pub fn new() -> Self {
        Self
    }

    pub fn create_token(&self) -> CancellationToken {
        CancellationToken::new()
    }
}

#[derive(Clone)]
pub struct ProxyServer {
    pub client: reqwest::Client,
    pub config: Config,
    pub logger: Logger,
    pub cancellation_factory: CancellationTokenFactory,
}

impl ProxyServer {
    pub fn new(config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        // Validate configuration using centralized validation
        validate_config(&config)?;

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.request_timeout_seconds + 10)) // Add buffer
            .build()?;

        let logger = Logger::new(!config.no_log);

        Ok(Self {
            client,
            config,
            logger,
            cancellation_factory: CancellationTokenFactory::new(),
        })
    }

    pub async fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        self.print_startup_banner();

        let addr: SocketAddr = self.config.listen.parse()
            .map_err(|e| format!("Invalid listen address '{}': {}", self.config.listen, e))?;

        let server = Arc::new(self.clone());

        // Custom logging with improved formatting
        let log = warp::log::custom({
            let server = server.clone();
            move |info: LogInfo| {
                let logger = &server.logger;
                let status_icon = match info.status().as_u16() {
                    200..=299 => LOG_PREFIX_SUCCESS,
                    400..=499 => LOG_PREFIX_WARNING,
                    500..=599 => LOG_PREFIX_ERROR,
                    _ => LOG_PREFIX_REQUEST,
                };

                let log_line = format!(
                    "{} {} {} | {} | {:?}",
                    status_icon,
                    info.method(),
                    info.path(),
                    info.status(),
                    info.elapsed()
                );
                logger.log(&log_line);
            }
        });

        // Request size validation filter
        let request_size_filter = warp::body::content_length_limit(MAX_REQUEST_SIZE_BYTES as u64);

        let routes = warp::method()
            .and(warp::path::full())
            .and(request_size_filter)
            .and(warp::body::json().or(warp::any().map(|| Value::Null)).unify())
            .and_then(move |method: warp::http::Method, path: warp::path::FullPath, body: Value| {
                let server = server.clone();
                async move {
                    handle_request_with_cancellation(server, method.to_string(), path.as_str().to_string(), body).await
                }
            })
            .recover(handle_rejection)
            .with(log);

        self.logger.log_with_prefix(LOG_PREFIX_SUCCESS, &format!("Server starting on {}", addr));

        warp::serve(routes)
            .run(addr)
            .await;

        Ok(())
    }

    fn print_startup_banner(&self) {
        if self.logger.enabled {
            println!("╭─────────────────────────────────────────────────────────────╮");
            println!("│              🔄 Ollama ↔ LM Studio Proxy v{}               │", crate::VERSION);
            println!("│                 High-Performance Edition                    │");
            println!("╰─────────────────────────────────────────────────────────────╯");
            println!();
            println!("📡 Listen Address:     {}", self.config.listen);
            println!("🎯 LM Studio URL:      {}", self.config.lmstudio_url);
            println!("📝 Logging:            {}", if self.logger.enabled { "✅ Enabled" } else { "❌ Disabled" });
            println!("⏱️  Load Timeout:      {}s", self.config.load_timeout_seconds);
            println!("🌐 Request Timeout:    {}s", self.config.request_timeout_seconds);
            println!("🌊 Stream Timeout:     {}s", self.config.stream_timeout_seconds);
            println!("🚫 Cancellation:      ✅ Enabled (client disconnect detection)");
            println!("💾 Max Request Size:   {}MB", MAX_REQUEST_SIZE_BYTES / (1024 * 1024));
            println!("🔄 Max Stream Chunks:  {}", MAX_CHUNK_COUNT);
            println!();
            println!("🔌 Supported Endpoints:");
            println!("   • Ollama API:       /api/* (translated to LM Studio)");
            println!("   • LM Studio API:    /v1/* (direct passthrough with model resolution)");
            println!("   • Health Check:     GET /health");
            println!();
            println!("🚀 Ready to handle requests!");
            println!("─────────────────────────────────────────────────────────────");
        }
    }
}

/// Improved connection tracker with atomic operations
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
        // Use compare_exchange to avoid race condition
        if self.completed.compare_exchange_weak(
            false,
            true,
            Ordering::SeqCst,
            Ordering::Relaxed
        ).is_ok() {
            self.token.cancel();
            if std::env::var("RUST_LOG").is_ok() {
                eprintln!("{} Client disconnected - cancelling LM Studio request", LOG_PREFIX_CANCEL);
            }
        }
    }
}

async fn handle_request_with_cancellation(
    server: Arc<ProxyServer>,
    method: String,
    path: String,
    body: Value,
) -> Result<warp::reply::Response, Rejection> {
    server.logger.log_with_prefix(LOG_PREFIX_REQUEST, &format!("Connection established: {} {}", method, path));

    // Validate request body size
    if let Err(e) = validate_request_size(&body) {
        return Err(warp::reject::custom(e));
    }

    let cancellation_token = server.cancellation_factory.create_token();
    let connection_tracker = ConnectionTracker::new(cancellation_token.clone());

    let result = match (method.as_str(), path.as_str()) {
        // Ollama API endpoints
        ("GET", "/api/tags") => {
            handlers::handle_ollama_tags(server.clone(), cancellation_token).await
        },
        ("POST", "/api/chat") => {
            handlers::handle_ollama_chat(server.clone(), body.clone(), cancellation_token).await
        },
        ("POST", "/api/generate") => {
            handlers::handle_ollama_generate(server.clone(), body.clone(), cancellation_token).await
        },
        ("POST", "/api/embed") | ("POST", "/api/embeddings") => {
            handlers::handle_ollama_embeddings(server.clone(), body.clone(), cancellation_token).await
        },
        ("POST", "/api/show") => handlers::handle_ollama_show(body).await,
        ("GET", "/api/ps") => handlers::handle_ollama_ps().await,
        ("GET", "/api/version") => handlers::handle_ollama_version().await,

        // Unsupported Ollama endpoints with helpful messages
        ("POST", "/api/create") | ("POST", "/api/pull") | ("POST", "/api/push") |
        ("DELETE", "/api/delete") | ("POST", "/api/copy") =>
            handlers::handle_unsupported(&path).await,

        // LM Studio API passthrough with validation
        ("GET", "/v1/models") | ("POST", "/v1/chat/completions") |
        ("POST", "/v1/completions") | ("POST", "/v1/embeddings") => {
            match handlers::validate_lmstudio_endpoint(&path) {
                Ok(_) => handlers::handle_lmstudio_passthrough(server.clone(), &method, &path, body.clone(), cancellation_token).await,
                Err(e) => Err(e),
            }
        },

        // Health check endpoint
        ("GET", "/health") => {
            match handlers::get_lmstudio_status(server.clone(), cancellation_token).await {
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
            server.logger.log_with_prefix(LOG_PREFIX_CANCEL, &format!("Connection lost: {} {} - Request cancelled by client disconnection", method, path));
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

async fn handle_rejection(err: Rejection) -> Result<impl Reply, Infallible> {
    let (code, message, ollama_compatible) = if err.is_not_found() {
        (404, "Not Found".to_string(), false)
    } else if let Some(proxy_error) = err.find::<ProxyError>() {
        (proxy_error.status_code, proxy_error.message.clone(), true)
    } else if err.find::<warp::reject::MethodNotAllowed>().is_some() {
        (405, "Method Not Allowed".to_string(), false)
    } else if err.find::<warp::reject::PayloadTooLarge>().is_some() {
        (413, format!("Payload Too Large (max: {}MB)", MAX_REQUEST_SIZE_BYTES / (1024 * 1024)), false)
    } else if err.find::<warp::reject::LengthRequired>().is_some() {
        (411, "Length Required".to_string(), false)
    } else {
        eprintln!("Unhandled rejection: {:?}", err);
        (500, "Internal Server Error".to_string(), false)
    };

    let json = if ollama_compatible {
        warp::reply::json(&serde_json::json!({
            "error": {
                "type": "api_error",
                "message": message
            }
        }))
    } else {
        warp::reply::json(&serde_json::json!({
            "error": {
                "type": "proxy_error",
                "message": message,
                "timestamp": chrono::Utc::now().to_rfc3339()
            }
        }))
    };

    Ok(warp::reply::with_status(
        json,
        warp::http::StatusCode::from_u16(code).unwrap_or(warp::http::StatusCode::INTERNAL_SERVER_ERROR)
    ))
}
