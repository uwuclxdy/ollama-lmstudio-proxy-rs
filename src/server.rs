/// src/server.rs - High-performance server with concurrent request support
use clap::Parser;
use moka::future::Cache;
use serde_json::Value;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use warp::log::Info as LogInfo;
use warp::{Filter, Rejection, Reply};

use crate::common::RequestContext;
use crate::constants::*;
use crate::handlers;
use crate::handlers::json_response;
use crate::model::ModelResolver;
// Added
use crate::utils::{
    init_global_logger, is_logging_enabled, log_error, log_info, validate_config, ProxyError,
};

#[derive(Parser, Debug, Clone)]
#[command(name = "ollama-lmstudio-proxy")]
#[command(about = "High-performance proxy server bridging Ollama API and LM Studio")]
pub struct Config {
    #[arg(long, default_value = "0.0.0.0:11434", help = "Server listen address")]
    pub listen: String,

    #[arg(
        long,
        default_value = "http://localhost:1234",
        help = "LM Studio backend URL"
    )]
    pub lmstudio_url: String,

    #[arg(long, help = "Disable logging output")]
    pub no_log: bool,

    #[arg(
        long,
        default_value = "15",
        help = "Model loading wait timeout in seconds (after trigger)"
    )]
    pub load_timeout_seconds: u64,

    #[arg(
        long,
        default_value = "262144",
        help = "Initial buffer size in bytes for SSE message assembly (capacity hint)"
    )]
    pub max_buffer_size: usize,

    #[arg(long, help = "Enable partial chunk recovery for streams")]
    pub enable_chunk_recovery: bool,

    #[arg(
        long,
        default_value = "300", // 5 minutes
        help = "TTL for model resolution cache in seconds"
    )]
    pub model_resolution_cache_ttl_seconds: u64,
}

/// Production-ready proxy server with concurrent request support
#[derive(Clone)]
pub struct ProxyServer {
    pub client: reqwest::Client,
    pub config: Arc<Config>,
    pub model_resolver: Arc<ModelResolver>, // Added
}

/// Wrapper for ollama show handler
async fn handle_ollama_show_rejection_wrapper(body: Value) -> Result<impl Reply, Rejection> {
    handlers::ollama::handle_ollama_show(body)
        .await
        .map_err(warp::reject::custom)
}

/// Wrapper for ollama version handler
async fn handle_ollama_version_rejection_wrapper() -> Result<impl Reply, Rejection> {
    handlers::ollama::handle_ollama_version()
        .await
        .map_err(warp::reject::custom)
}

impl ProxyServer {
    /// Create new proxy server instance
    pub fn new(config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        validate_config(&config)?;

        let runtime_config = RuntimeConfig {
            max_buffer_size: if config.max_buffer_size > 0 {
                config.max_buffer_size
            } else {
                usize::MAX
            },
            max_partial_content_size: usize::MAX,
            string_buffer_size: 2048,
            enable_chunk_recovery: config.enable_chunk_recovery,
        };
        init_runtime_config(runtime_config);
        init_global_logger(!config.no_log);

        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .pool_max_idle_per_host(10)
            .build()?;

        let model_cache: Cache<String, String> = Cache::builder()
            .time_to_live(Duration::from_secs(
                config.model_resolution_cache_ttl_seconds,
            ))
            .build();

        let model_resolver = Arc::new(ModelResolver::new(
            config.lmstudio_url.clone(),
            model_cache,
        ));

        Ok(Self {
            client,
            config: Arc::new(config),
            model_resolver,
        })
    }

    /// Run the proxy server
    pub async fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        self.print_startup_banner();

        let addr: SocketAddr = self
            .config
            .listen
            .parse()
            .map_err(|e| format!("Invalid listen address '{}': {}", self.config.listen, e))?;

        let server_arc = Arc::new(self);

        let log_filter = warp::log::custom({
            let logging_enabled = is_logging_enabled();
            move |info: LogInfo| {
                if logging_enabled {
                    let status_icon = match info.status().as_u16() {
                        200..=299 => LOG_PREFIX_SUCCESS,
                        400..=499 => LOG_PREFIX_WARNING,
                        500..=599 => LOG_PREFIX_ERROR,
                        _ => "❔",
                    };
                    crate::utils::STRING_BUFFER.with(|buf_cell| {
                        let mut buffer = buf_cell.borrow_mut();
                        buffer.clear();
                        use std::fmt::Write;
                        let _ = write!(
                            buffer,
                            "{} {} {} | {} | {}",
                            status_icon,
                            info.method(),
                            info.path(),
                            info.status(),
                            crate::utils::format_duration(info.elapsed())
                        );
                        println!("[{}] {}", chrono::Local::now().format("%H:%M:%S"), buffer);
                    });
                }
            }
        });

        let with_server_state = warp::any().map({
            let server_clone = server_arc.clone();
            move || server_clone.clone()
        });

        let ollama_tags_route = warp::path!("api" / "tags")
            .and(warp::get())
            .and(with_server_state.clone())
            .and_then(|s: Arc<ProxyServer>| async move {
                let context = RequestContext {
                    client: &s.client,
                    lmstudio_url: &s.config.lmstudio_url,
                };
                let token = CancellationToken::new();
                handlers::ollama::handle_ollama_tags(context, token)
                    .await
                    .map_err(warp::reject::custom)
            });

        let ollama_chat_route = warp::path!("api" / "chat")
            .and(warp::post())
            .and(warp::body::json())
            .and(with_server_state.clone())
            .and_then(|body: Value, s: Arc<ProxyServer>| async move {
                let context = RequestContext {
                    client: &s.client,
                    lmstudio_url: &s.config.lmstudio_url,
                };
                let token = CancellationToken::new();
                let config_ref = s.config.as_ref();
                handlers::ollama::handle_ollama_chat(
                    context,
                    s.model_resolver.clone(), // Pass ModelResolver
                    body,
                    token,
                    config_ref,
                )
                    .await
                    .map_err(warp::reject::custom)
            });

        let ollama_generate_route = warp::path!("api" / "generate")
            .and(warp::post())
            .and(warp::body::json())
            .and(with_server_state.clone())
            .and_then(|body: Value, s: Arc<ProxyServer>| async move {
                let context = RequestContext {
                    client: &s.client,
                    lmstudio_url: &s.config.lmstudio_url,
                };
                let token = CancellationToken::new();
                let config_ref = s.config.as_ref();
                handlers::ollama::handle_ollama_generate(
                    context,
                    s.model_resolver.clone(), // Pass ModelResolver
                    body,
                    token,
                    config_ref,
                )
                    .await
                    .map_err(warp::reject::custom)
            });

        let ollama_embeddings_route = warp::path!("api" / "embeddings")
            .or(warp::path!("api" / "embed"))
            .unify()
            .and(warp::post())
            .and(warp::body::json())
            .and(with_server_state.clone())
            .and_then(|body: Value, s: Arc<ProxyServer>| async move {
                let context = RequestContext {
                    client: &s.client,
                    lmstudio_url: &s.config.lmstudio_url,
                };
                let token = CancellationToken::new();
                handlers::ollama::handle_ollama_embeddings(
                    context,
                    s.model_resolver.clone(), // Pass ModelResolver
                    body,
                    token,
                    s.config.as_ref(), // Pass config for load_timeout_seconds
                )
                    .await
                    .map_err(warp::reject::custom)
            });

        let ollama_show_route = warp::path!("api" / "show")
            .and(warp::post())
            .and(warp::body::json())
            .and_then(handle_ollama_show_rejection_wrapper);

        let ollama_ps_route = warp::path!("api" / "ps")
            .and(warp::get())
            .and(with_server_state.clone())
            .and_then(|s: Arc<ProxyServer>| async move {
                let context = RequestContext {
                    client: &s.client,
                    lmstudio_url: &s.config.lmstudio_url,
                };
                let token = CancellationToken::new();
                handlers::ollama::handle_ollama_ps(context, token)
                    .await
                    .map_err(warp::reject::custom)
            });

        let ollama_version_route = warp::path!("api" / "version")
            .and(warp::get())
            .and_then(handle_ollama_version_rejection_wrapper);

        let lmstudio_passthrough_route = warp::path("v1")
            .and(warp::path::tail())
            .and(warp::method())
            .and(
                warp::body::json()
                    .or(warp::any().map(|| Value::Null))
                    .unify(),
            )
            .and(with_server_state.clone())
            .and_then(
                |tail: warp::path::Tail,
                    method: warp::http::Method,
                    body: Value,
                    s: Arc<ProxyServer>| async move {
                    let context = RequestContext {
                        client: &s.client,
                        lmstudio_url: &s.config.lmstudio_url,
                    };
                    let token = CancellationToken::new();
                    let full_path = format!("/v1/{}", tail.as_str());
                    handlers::lmstudio::handle_lmstudio_passthrough(
                        context,
                        s.model_resolver.clone(), // Pass ModelResolver
                        method.as_str(),
                        &full_path,
                        body,
                        token,
                        s.config.load_timeout_seconds,
                    )
                        .await
                        .map_err(warp::reject::custom)
                },
            );

        let health_route = warp::path("health")
            .and(warp::get())
            .and(with_server_state.clone())
            .and_then(|s: Arc<ProxyServer>| async move {
                let context = RequestContext {
                    client: &s.client,
                    lmstudio_url: &s.config.lmstudio_url,
                };
                let token = CancellationToken::new();
                match handlers::ollama::handle_health_check(context, token).await {
                    Ok(status_json) => Ok(json_response(&status_json)),
                    Err(e) => Err(warp::reject::custom(e)),
                }
            });

        let unsupported_ollama_route = warp::path("api")
            .and(warp::path::full())
            .and_then(|path: warp::path::FullPath| async move {
                handlers::ollama::handle_unsupported(path.as_str())
                    .await
                    .map_err(warp::reject::custom)
            });

        let app_routes = ollama_tags_route
            .boxed()
            .or(ollama_chat_route.boxed())
            .or(ollama_generate_route.boxed())
            .or(ollama_embeddings_route.boxed())
            .or(ollama_show_route.boxed())
            .or(ollama_ps_route.boxed())
            .or(ollama_version_route.boxed())
            .or(lmstudio_passthrough_route.boxed())
            .or(health_route.boxed())
            .or(unsupported_ollama_route.boxed());

        let final_routes = app_routes.recover(handle_rejection).with(log_filter);

        log_info("Starting server...");
        warp::serve(final_routes).run(addr).await;
        Ok(())
    }

    /// Print startup banner with configuration info
    fn print_startup_banner(&self) {
        if is_logging_enabled() {
            println!();
            println!("------------------------------------------------------");
            println!("Ollama <-> LM Studio Proxy - Version: {}", crate::VERSION);
            println!("------------------------------------------------------");
            println!(" Listening on: {}", self.config.listen);
            println!(" LM Studio URL: {}", self.config.lmstudio_url);
            println!(
                " Logging: {}",
                if is_logging_enabled() {
                    "Enabled"
                } else {
                    "Disabled"
                }
            );
            println!(
                " Model Load Timeout: {}s",
                self.config.load_timeout_seconds
            );
            println!(
                " Model Resolution Cache TTL: {}s",
                self.config.model_resolution_cache_ttl_seconds
            );
            println!(
                " Initial SSE Buffer: {} bytes",
                self.config.max_buffer_size
            );
            println!(
                " Chunk Recovery: {}",
                if get_runtime_config().enable_chunk_recovery {
                    "Enabled"
                } else {
                    "Disabled"
                }
            );
            println!("------------------------------------------------------");
            println!(" INFO: Proxy forwards all requests and timing to LM Studio backend.");
            println!("------------------------------------------------------");
        }
    }
}

/// Enhanced error handling with proper status codes and JSON response
async fn handle_rejection(err: Rejection) -> Result<impl Reply, Infallible> {
    let code;
    let message;
    let error_type;

    if err.is_not_found() {
        code = warp::http::StatusCode::NOT_FOUND;
        message = "Endpoint not found".to_string();
        error_type = "not_found_error".to_string();
    } else if let Some(proxy_error) = err.find::<ProxyError>() {
        code = warp::http::StatusCode::from_u16(proxy_error.status_code)
            .unwrap_or(warp::http::StatusCode::INTERNAL_SERVER_ERROR);
        message = proxy_error.message.clone();
        error_type = match proxy_error.status_code {
            400 => "bad_request_error".to_string(),
            401 => "authentication_error".to_string(),
            403 => "permission_error".to_string(),
            404 => "not_found_error".to_string(),
            413 => "payload_too_large_error".to_string(),
            429 => "rate_limit_error".to_string(),
            499 => "client_closed_request".to_string(),
            500 => "internal_server_error".to_string(),
            501 => "not_implemented_error".to_string(),
            503 => "service_unavailable_error".to_string(),
            _ => "api_error".to_string(),
        };
    } else if err.find::<warp::reject::MethodNotAllowed>().is_some() {
        code = warp::http::StatusCode::METHOD_NOT_ALLOWED;
        message = "Method Not Allowed".to_string();
        error_type = "method_not_allowed_error".to_string();
    } else if err.find::<warp::reject::PayloadTooLarge>().is_some() {
        code = warp::http::StatusCode::PAYLOAD_TOO_LARGE;
        message = "Payload Too Large (check backend or underlying HTTP server limits)".to_string();
        error_type = "payload_too_large_error".to_string();
    } else if err.find::<warp::reject::UnsupportedMediaType>().is_some() {
        code = warp::http::StatusCode::UNSUPPORTED_MEDIA_TYPE;
        message = "Unsupported Media Type. Expected application/json.".to_string();
        error_type = "unsupported_media_type_error".to_string();
    } else {
        log_error("Unhandled rejection", &format!("{:?}", err));
        code = warp::http::StatusCode::INTERNAL_SERVER_ERROR;
        message = "An unexpected internal error occurred.".to_string();
        error_type = "internal_server_error".to_string();
    }

    let json_error = serde_json::json!({
        "error": {
            "message": message,
            "type": error_type,
            "code": code.as_u16(),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }
    });

    Ok(warp::reply::with_status(
        warp::reply::json(&json_error),
        code,
    ))
}
