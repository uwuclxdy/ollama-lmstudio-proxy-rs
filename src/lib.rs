pub mod server;
pub mod utils;
pub mod handlers;
pub mod common;

/// Public re-exports for easy access
pub use server::{Config, ProxyServer};
pub use utils::ProxyError;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");
