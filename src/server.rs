// src/server.rs - High-performance server with concurrent request support

use clap::Parser;
use serde_json::Value;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use warp::log::Info as LogInfo;
use warp::{Filter, Rejection, Reply};

use crate::common::{validate_request_size, RequestContext};
use crate::constants::*;
use crate::handlers;
use crate::metrics::{get_global_metrics, init_global_metrics};
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

    #[arg(long, help = "Enable metrics collection")]
    pub enable_metrics: bool,

    #[arg(long, default_value = "262144", help = "Maximum buffer size in bytes")]
    pub max_buffer_size: usize,

    #[arg(long, default_value = "50000", help = "Maximum chunks per stream")]
    pub max_chunk_count: u64,

    #[arg(long, default_value = "524288000", help = "Maximum request size in bytes")]
    pub max_request_size: usize,

    #[arg(long, help = "Enable partial chunk recovery for streams")]
    pub enable_chunk_recovery: bool,
}

/// Production-ready proxy server with concurrent request support
#[derive(Clone)]
pub struct ProxyServer {
    pub client: reqwest::Client,
    pub config: Config,
    pub logger: Logger,
}

impl ProxyServer {
    pub fn new(config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        validate_config(&config)?;

        // Initialize runtime configuration
        let runtime_config = RuntimeConfig {
            max_buffer_size: config.max_buffer_size,
            max_chunk_count: config.max_chunk_count,
            max_partial_content_size: config.max_buffer_size / 4,
            max_request_size_bytes: config.max_request_size,
            string_buffer_size: 2048,
            enable_metrics: config.enable_metrics,
            enable_chunk_recovery: config.enable_chunk_recovery,
        };
        init_runtime_config(runtime_config);

        // Initialize global metrics
        init_global_metrics(config.enable_metrics);

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

        // Create shared server reference for concurrent access
        let server = Arc::new(self);

        // Enhanced logging with metrics
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

        // Concurrent request handling with proper body size validation
        let routes = warp::method()
            .and(warp::path::full())
            .and(warp::body::json().or(warp::any().map(|| Value::Null)).unify())
            .and_then(move |method: warp::http::Method, path: warp::path::FullPath, body: Value| {
                let server = server.clone();
                async move {
                    handle_concurrent_request(server, method.to_string(), path.as_str().to_string(), body).await
                }
            })
            .recover(handle_rejection)
            .with(log.clone())
            .boxed();

        // Add metrics endpoint if enabled
        let metrics_route = warp::path("metrics")
            .and(warp::get())
            .and_then(handle_metrics_endpoint)
            .recover(handle_rejection)
            .with(log)
            .boxed();

        let combined_routes = routes.or(metrics_route).boxed();

        warp::serve(combined_routes).run(addr).await;
        Ok(())
    }

    fn print_startup_banner(&self) {
        if self.logger.enabled {
            println!();
            println!("Ollama ↔ LM Studio Proxy");
            println!("------------------------------------------------------");
            println!("Version: {}", crate::VERSION);
            println!("Listen Address: {}", self.config.listen);
            println!("LM Studio URL: {}", self.config.lmstudio_url);
            println!("Logging: {}", if self.logger.enabled { "Enabled" } else { "Disabled" });
            println!("Metrics: {}", if self.config.enable_metrics { "Enabled" } else { "Disabled" });
            println!("Load Timeout: {}s", self.config.load_timeout_seconds);
            println!("Request Timeout: {}s", self.config.request_timeout_seconds);
            println!("Stream Timeout: {}s", self.config.stream_timeout_seconds);
            println!("Max Buffer Size: {} bytes", self.config.max_buffer_size);
            println!("Max Request Size: {} bytes", self.config.max_request_size);
            println!("Chunk Recovery: {}", if self.config.enable_chunk_recovery { "Enabled" } else { "Disabled" });
            println!();
        }
    }
}

/// Thread-safe connection tracker with proper synchronization
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

    /// Mark request as completed successfully
    fn mark_completed(&self) {
        self.completed.store(true, Ordering::Release);
    }
}

impl Drop for ConnectionTracker {
    fn drop(&mut self) {
        // Use atomic compare-and-swap to prevent race conditions
        if self.completed.compare_exchange(
            false,
            true,
            Ordering::AcqRel,
            Ordering::Acquire,
        ).is_ok() {
            // Only cancel if we successfully marked as completed
            self.token.cancel();
        }
    }
}

/// Concurrent request handler with metrics integration
async fn handle_concurrent_request(
    server: Arc<ProxyServer>,
    method: String,
    path: String,
    body: Value,
) -> Result<warp::reply::Response, Rejection> {
    let endpoint_key = format!("{} {}", method, path);

    // Start metrics tracking if enabled
    let timer = if let Some(metrics) = get_global_metrics() {
        metrics.record_request_start(&endpoint_key)
    } else {
        None
    };

    // Validate request body size early
    if !body.is_null() {
        if let Err(e) = validate_request_size(&body) {
            if let Some(timer) = timer {
                timer.complete_failure().await;
            }
            return Err(warp::reject::custom(e));
        }
    }

    let cancellation_token = CancellationToken::new();
    let connection_tracker = ConnectionTracker::new(cancellation_token.clone());

    // Create request context for concurrent access
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
        }
        ("POST", "/api/chat") => {
            handlers::handle_ollama_chat(context, body.clone(), cancellation_token, &server.config).await
        }
        ("POST", "/api/generate") => {
            handlers::handle_ollama_generate(context, body.clone(), cancellation_token, &server.config).await
        }
        ("POST", "/api/embed") | ("POST", "/api/embeddings") => {
            handlers::handle_ollama_embeddings(context, body.clone(), cancellation_token).await
        }
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
        }

        // Health check
        ("GET", "/health") => {
            match handlers::get_lmstudio_status(context, cancellation_token).await {
                Ok(status) => Ok(warp::reply::json(&status).into_response()),
                Err(e) => Err(e),
            }
        }

        // Unknown endpoints
        _ => Err(ProxyError::not_found(&format!("Unknown endpoint: {} {}", method, path))),
    };

    // Handle result with proper completion tracking
    match result {
        Ok(response) => {
            connection_tracker.mark_completed();
            if let Some(timer) = timer {
                timer.complete_success().await;
            }
            Ok(response)
        }
        Err(e) if e.is_cancelled() => {
            if let Some(timer) = timer {
                timer.complete_cancelled();
            }
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
        }
        Err(e) => {
            connection_tracker.mark_completed();
            if let Some(timer) = timer {
                timer.complete_failure().await;
            }
            Err(warp::reject::custom(e))
        }
    }
}

/// Handle metrics endpoint
async fn handle_metrics_endpoint() -> Result<impl Reply, Rejection> {
    match get_global_metrics() {
        Some(metrics) => {
            let metrics_data = metrics.get_metrics().await;
            Ok(warp::reply::json(&metrics_data))
        }
        None => {
            let disabled_response = serde_json::json!({
                "error": "Metrics collection is disabled"
            });
            Ok(warp::reply::json(&disabled_response))
        }
    }
}

/// Enhanced error handling with proper status codes
async fn handle_rejection(err: Rejection) -> Result<impl Reply, Infallible> {
    let (code, message) = if err.is_not_found() {
        (404, "Not Found".to_string())
    } else if let Some(proxy_error) = err.find::<ProxyError>() {
        (proxy_error.status_code, proxy_error.message.clone())
    } else if err.find::<warp::reject::MethodNotAllowed>().is_some() {
        (405, "Method Not Allowed".to_string())
    } else if err.find::<warp::reject::PayloadTooLarge>().is_some() {
        let max_size_mb = get_runtime_config().max_request_size_bytes / (1024 * 1024);
        (413, format!("Payload Too Large (max: {}MB)", max_size_mb))
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
        warp::http::StatusCode::from_u16(code).unwrap_or(warp::http::StatusCode::INTERNAL_SERVER_ERROR),
    ))
}
