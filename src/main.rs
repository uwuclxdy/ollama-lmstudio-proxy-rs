/// src/main.rs - Application entry point for the Ollama-LMStudio proxy server.

use clap::Parser;
use ollama_lmstudio_proxy_rust::{Config, ProxyServer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::parse();
    let server = ProxyServer::new(config)?;
    server.run().await?;
    Ok(())
}
