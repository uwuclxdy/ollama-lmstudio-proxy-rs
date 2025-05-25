use clap::Parser;
use ollama_lmstudio_proxy_rust::{Config, ProxyServer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let config = Config::parse();

    // Run the proxy server
    let server = ProxyServer::new(config)?;
    server.run().await?;

    Ok(())
}
