pub mod server;
pub mod handlers;
pub mod utils;

/// Public re-exports for easy access
pub use server::{Config, ProxyServer};
pub use utils::ProxyError;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        assert_eq!(NAME, "ollama-lmstudio-proxy");
    }
}
