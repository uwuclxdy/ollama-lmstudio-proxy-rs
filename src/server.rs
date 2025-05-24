use clap::Parser;
use serde_json::Value;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::task::AbortHandle;
use tokio_util::sync::CancellationToken;
use warp::{Filter, Rejection, Reply};
use warp::log::Info as LogInfo;

use crate::handlers;
use crate::utils::{Logger, ProxyError};

#[derive(Parser, Debug, Clone)]
#[command(name = "ollama-lmstudio-proxy")]
#[command(about = "Proxy server bridging Ollama API and LM Studio")]
pub struct Config {
    #[arg(long, default_value = "0.0.0.0:11434")]
    pub listen: String,

    #[arg(long, default_value = "http://localhost:1234")]
    pub lmstudio_url: String,

    #[arg(long)]
    pub no_log: bool,

    #[arg(long, default_value = "5")]
    pub load_timeout_seconds: u64,
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
    pub fn new(config: Config) -> Self {
        let client = reqwest::Client::new();
        let logger = Logger::new(!config.no_log);

        Self {
            client,
            config,
            logger,
            cancellation_factory: CancellationTokenFactory::new(),
        }
    }

    pub async fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        self.print_startup_banner();

        let addr: SocketAddr = self.config.listen.parse()
            .map_err(|e| format!("Invalid listen address '{}': {}", self.config.listen, e))?;

        let server = Arc::new(self.clone());

        let log = warp::log::custom({
            let server = server.clone();
            move |info: LogInfo| {
                let logger = &server.logger;
                let log_line = format!(
                    "REQUEST: {} {} | RESPONSE: {} | Duration: {:?}",
                    info.method(),
                    info.path(),
                    info.status(),
                    info.elapsed()
                );
                logger.log(&log_line);
            }
        });

        let routes = warp::method()
            .and(warp::path::full())
            .and(warp::body::json().or(warp::any().map(|| Value::Null)).unify())
            .and_then(move |method: warp::http::Method, path: warp::path::FullPath, body: Value| {
                let server = server.clone();
                async move {
                    handle_request_with_cancellation(server, method.to_string(), path.as_str().to_string(), body).await
                }
            })
            .recover(handle_rejection)
            .with(log);

        self.logger.log(&format!("🚀 Server starting on {}", addr));

        warp::serve(routes)
            .run(addr)
            .await;

        Ok(())
    }

    fn print_startup_banner(&self) {
        if self.logger.enabled {
            println!("╭─────────────────────────────────────────────────────────────╮");
            println!("│                  🔄 Ollama ↔ LM Studio Proxy                │");
            println!("│                    with Cancellation Support                │");
            println!("╰─────────────────────────────────────────────────────────────╯");
            println!();
            println!("📡 Listen Address:    {}", self.config.listen);
            println!("🎯 LM Studio URL:     {}", self.config.lmstudio_url);
            println!("📝 Logging:           {}", if self.logger.enabled { "✅ Enabled" } else { "❌ Disabled" });
            println!("⏱️  Load Timeout:     {}s", self.config.load_timeout_seconds);
            println!("🚫 Cancellation:     ✅ Enabled (client disconnect detection)");
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
        self.completed.store(true, Ordering::Relaxed);
    }
}

impl Drop for ConnectionTracker {
    fn drop(&mut self) {
        if !self.completed.load(Ordering::Relaxed) {
            self.token.cancel();
            if std::env::var("RUST_LOG").is_ok() {
                eprintln!("🚫 Client disconnected - cancelling LM Studio request");
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
    server.logger.log(&format!("[CONNECTION] Established: {} {}", method, path));
    server.logger.log(&format!("Received request: {} {}", method, path));

    let cancellation_token = server.cancellation_factory.create_token();
    let connection_tracker = ConnectionTracker::new(cancellation_token.clone());

    let result = match (method.as_str(), path.as_str()) {
        ("GET", "/api/tags") => {
            handlers::handle_ollama_tags_with_cancellation(server.clone(), cancellation_token).await
        },
        ("POST", "/api/chat") => {
            handlers::handle_ollama_chat_with_cancellation(server.clone(), body.clone(), cancellation_token).await
        },
        ("POST", "/api/generate") => {
            handlers::handle_ollama_generate_with_cancellation(server.clone(), body.clone(), cancellation_token).await
        },
        ("POST", "/api/embed") | ("POST", "/api/embeddings") => {
            handlers::handle_ollama_embeddings_with_cancellation(server.clone(), body.clone(), cancellation_token).await
        },
        ("POST", "/api/show") => handlers::handle_ollama_show(body).await,
        ("GET", "/api/ps") => handlers::handle_ollama_ps().await,
        ("GET", "/api/version") => handlers::handle_ollama_version().await,
        ("POST", "/api/create") | ("POST", "/api/pull") | ("POST", "/api/push") |
        ("DELETE", "/api/delete") | ("POST", "/api/copy") =>
            handlers::handle_unsupported(&path).await,
        ("GET", "/v1/models") | ("POST", "/v1/chat/completions") |
        ("POST", "/v1/completions") | ("POST", "/v1/embeddings") => {
            handlers::handle_lmstudio_passthrough_with_cancellation(server.clone(), &method, &path, body.clone(), cancellation_token).await
        },
        _ => Err(ProxyError::not_found(&format!("Unknown endpoint: {} {}", method, path))),
    };

    match result {
        Ok(response) => {
            connection_tracker.mark_completed();
            Ok(response)
        },
        Err(e) if e.is_cancelled() => {
            server.logger.log(&format!("[CONNECTION] Lost: {} {} - Request was cancelled by client disconnection", method, path));
            let error_response = serde_json::json!({
                "error": {
                    "type": "request_cancelled",
                    "message": "Request was cancelled due to client disconnection"
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
        (413, "Payload Too Large".to_string(), false)
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
                "message": message
            }
        }))
    };

    Ok(warp::reply::with_status(json, warp::http::StatusCode::from_u16(code).unwrap_or(warp::http::StatusCode::INTERNAL_SERVER_ERROR)))
}

#[derive(Clone)]
pub struct RequestHandle {
    pub request_id: String,
    pub abort_handle: Option<AbortHandle>,
}

impl RequestHandle {
    pub fn new(request_id: String) -> Self {
        Self {
            request_id,
            abort_handle: None,
        }
    }

    pub fn new_with_abort(abort_handle: AbortHandle) -> Self {
        Self {
            request_id: "unknown".to_string(),
            abort_handle: Some(abort_handle),
        }
    }
}
