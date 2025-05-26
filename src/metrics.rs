/// src/metrics.rs - Comprehensive metrics collection system

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde_json::{json, Value};

/// Thread-safe metrics collector
#[derive(Debug)]
pub struct MetricsCollector {
    enabled: bool,

    // Request metrics
    total_requests: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    cancelled_requests: AtomicU64,

    // Latency tracking
    total_latency_ns: AtomicU64,
    min_latency_ns: AtomicU64,
    max_latency_ns: AtomicU64,

    // Endpoint metrics
    endpoint_stats: Arc<RwLock<HashMap<String, EndpointStats>>>,

    // Streaming metrics
    active_streams: AtomicUsize,
    total_streams: AtomicU64,
    stream_errors: AtomicU64,
    chunks_processed: AtomicU64,

    // Model metrics
    model_stats: Arc<RwLock<HashMap<String, ModelStats>>>,

    // System metrics
    start_time: Instant,
}

#[derive(Debug, Clone, Default)]
struct EndpointStats {
    requests: u64,
    errors: u64,
    total_latency_ns: u64,
    min_latency_ns: u64,
    max_latency_ns: u64,
}

#[derive(Debug, Clone, Default)]
struct ModelStats {
    requests: u64,
    load_attempts: u64,
    load_failures: u64,
    total_tokens: u64,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            cancelled_requests: AtomicU64::new(0),
            total_latency_ns: AtomicU64::new(0),
            min_latency_ns: AtomicU64::new(u64::MAX),
            max_latency_ns: AtomicU64::new(0),
            endpoint_stats: Arc::new(RwLock::new(HashMap::new())),
            active_streams: AtomicUsize::new(0),
            total_streams: AtomicU64::new(0),
            stream_errors: AtomicU64::new(0),
            chunks_processed: AtomicU64::new(0),
            model_stats: Arc::new(RwLock::new(HashMap::new())),
            start_time: Instant::now(),
        }
    }

    /// Record request start
    pub fn record_request_start(&self, endpoint: &str) -> Option<RequestTimer> {
        if !self.enabled {
            return None;
        }

        self.total_requests.fetch_add(1, Ordering::Relaxed);
        Some(RequestTimer {
            start: Instant::now(),
            endpoint: endpoint.to_string(),
            metrics: self,
        })
    }

    /// Record successful request completion
    pub async fn record_request_success(&self, endpoint: &str, duration: Duration) {
        if !self.enabled {
            return;
        }

        self.successful_requests.fetch_add(1, Ordering::Relaxed);
        self.update_latency(duration);
        self.update_endpoint_stats(endpoint, duration, false).await;
    }

    /// Record failed request
    pub async fn record_request_failure(&self, endpoint: &str, duration: Duration) {
        if !self.enabled {
            return;
        }

        self.failed_requests.fetch_add(1, Ordering::Relaxed);
        self.update_latency(duration);
        self.update_endpoint_stats(endpoint, duration, true).await;
    }

    /// Record cancelled request
    pub fn record_request_cancelled(&self) {
        if !self.enabled {
            return;
        }

        self.cancelled_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Record stream start
    pub fn record_stream_start(&self) {
        if !self.enabled {
            return;
        }

        self.active_streams.fetch_add(1, Ordering::Relaxed);
        self.total_streams.fetch_add(1, Ordering::Relaxed);
    }

    /// Record stream end
    pub fn record_stream_end(&self, chunk_count: u64, error: bool) {
        if !self.enabled {
            return;
        }

        self.active_streams.fetch_sub(1, Ordering::Relaxed);
        self.chunks_processed.fetch_add(chunk_count, Ordering::Relaxed);

        if error {
            self.stream_errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record model usage
    pub async fn record_model_usage(&self, model: &str, tokens: u64) {
        if !self.enabled {
            return;
        }

        let mut stats = self.model_stats.write().await;
        let model_stat = stats.entry(model.to_string()).or_default();
        model_stat.requests += 1;
        model_stat.total_tokens += tokens;
    }

    /// Record model load attempt
    pub async fn record_model_load(&self, model: &str, success: bool) {
        if !self.enabled {
            return;
        }

        let mut stats = self.model_stats.write().await;
        let model_stat = stats.entry(model.to_string()).or_default();
        model_stat.load_attempts += 1;

        if !success {
            model_stat.load_failures += 1;
        }
    }

    /// Get comprehensive metrics snapshot
    pub async fn get_metrics(&self) -> Value {
        if !self.enabled {
            return json!({"enabled": false});
        }

        let uptime_secs = self.start_time.elapsed().as_secs();
        let total_reqs = self.total_requests.load(Ordering::Relaxed);
        let successful_reqs = self.successful_requests.load(Ordering::Relaxed);
        let failed_reqs = self.failed_requests.load(Ordering::Relaxed);
        let cancelled_reqs = self.cancelled_requests.load(Ordering::Relaxed);

        let avg_latency_ms = if total_reqs > 0 {
            (self.total_latency_ns.load(Ordering::Relaxed) as f64) / (total_reqs as f64 * 1_000_000.0)
        } else {
            0.0
        };

        let min_latency_ns = self.min_latency_ns.load(Ordering::Relaxed);
        let max_latency_ns = self.max_latency_ns.load(Ordering::Relaxed);

        let endpoint_stats = self.endpoint_stats.read().await;
        let model_stats = self.model_stats.read().await;

        json!({
            "enabled": true,
            "uptime_seconds": uptime_secs,
            "requests": {
                "total": total_reqs,
                "successful": successful_reqs,
                "failed": failed_reqs,
                "cancelled": cancelled_reqs,
                "success_rate": if total_reqs > 0 { (successful_reqs as f64) / (total_reqs as f64) } else { 0.0 },
                "requests_per_second": if uptime_secs > 0 { (total_reqs as f64) / (uptime_secs as f64) } else { 0.0 }
            },
            "latency": {
                "average_ms": avg_latency_ms,
                "min_ms": if min_latency_ns < u64::MAX { (min_latency_ns as f64) / 1_000_000.0 } else { 0.0 },
                "max_ms": (max_latency_ns as f64) / 1_000_000.0
            },
            "streaming": {
                "active_streams": self.active_streams.load(Ordering::Relaxed),
                "total_streams": self.total_streams.load(Ordering::Relaxed),
                "stream_errors": self.stream_errors.load(Ordering::Relaxed),
                "chunks_processed": self.chunks_processed.load(Ordering::Relaxed)
            },
            "endpoints": endpoint_stats.iter().map(|(endpoint, stats)| {
                (endpoint.clone(), json!({
                    "requests": stats.requests,
                    "errors": stats.errors,
                    "error_rate": if stats.requests > 0 { (stats.errors as f64) / (stats.requests as f64) } else { 0.0 },
                    "average_latency_ms": if stats.requests > 0 { (stats.total_latency_ns as f64) / (stats.requests as f64 * 1_000_000.0) } else { 0.0 },
                    "min_latency_ms": if stats.min_latency_ns < u64::MAX { (stats.min_latency_ns as f64) / 1_000_000.0 } else { 0.0 },
                    "max_latency_ms": (stats.max_latency_ns as f64) / 1_000_000.0
                }))
            }).collect::<HashMap<_, _>>(),
            "models": model_stats.iter().map(|(model, stats)| {
                (model.clone(), json!({
                    "requests": stats.requests,
                    "total_tokens": stats.total_tokens,
                    "average_tokens_per_request": if stats.requests > 0 { (stats.total_tokens as f64) / (stats.requests as f64) } else { 0.0 },
                    "load_attempts": stats.load_attempts,
                    "load_failures": stats.load_failures,
                    "load_success_rate": if stats.load_attempts > 0 { ((stats.load_attempts - stats.load_failures) as f64) / (stats.load_attempts as f64) } else { 1.0 }
                }))
            }).collect::<HashMap<_, _>>()
        })
    }

    /// Update latency statistics
    fn update_latency(&self, duration: Duration) {
        let duration_ns = duration.as_nanos() as u64;

        self.total_latency_ns.fetch_add(duration_ns, Ordering::Relaxed);

        // Update min latency atomically
        let mut current_min = self.min_latency_ns.load(Ordering::Relaxed);
        while duration_ns < current_min {
            match self.min_latency_ns.compare_exchange_weak(
                current_min,
                duration_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_min = actual,
            }
        }

        // Update max latency atomically
        let mut current_max = self.max_latency_ns.load(Ordering::Relaxed);
        while duration_ns > current_max {
            match self.max_latency_ns.compare_exchange_weak(
                current_max,
                duration_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }
    }

    /// Update endpoint-specific statistics
    async fn update_endpoint_stats(&self, endpoint: &str, duration: Duration, is_error: bool) {
        let mut stats = self.endpoint_stats.write().await;
        let endpoint_stat = stats.entry(endpoint.to_string()).or_default();

        endpoint_stat.requests += 1;
        if is_error {
            endpoint_stat.errors += 1;
        }

        let duration_ns = duration.as_nanos() as u64;
        endpoint_stat.total_latency_ns += duration_ns;

        if endpoint_stat.min_latency_ns > duration_ns {
            endpoint_stat.min_latency_ns = duration_ns;
        }

        if endpoint_stat.max_latency_ns < duration_ns {
            endpoint_stat.max_latency_ns = duration_ns;
        }
    }
}

/// RAII timer for request duration tracking
pub struct RequestTimer<'a> {
    start: Instant,
    endpoint: String,
    metrics: &'a MetricsCollector,
}

impl<'a> RequestTimer<'a> {
    /// Complete request successfully
    pub async fn complete_success(self) {
        let duration = self.start.elapsed();
        self.metrics.record_request_success(&self.endpoint, duration).await;
    }

    /// Complete request with failure
    pub async fn complete_failure(self) {
        let duration = self.start.elapsed();
        self.metrics.record_request_failure(&self.endpoint, duration).await;
    }

    /// Complete request as canceled
    pub fn complete_cancelled(self) {
        self.metrics.record_request_cancelled();
    }
}

/// Global metrics instance
static GLOBAL_METRICS: std::sync::OnceLock<Arc<MetricsCollector>> = std::sync::OnceLock::new();

/// Initialize global metrics
pub fn init_global_metrics(enabled: bool) {
    let metrics = Arc::new(MetricsCollector::new(enabled));
    GLOBAL_METRICS.set(metrics).ok();
}

/// Get global metrics instance
pub fn get_global_metrics() -> Option<&'static Arc<MetricsCollector>> {
    GLOBAL_METRICS.get()
}
