use anyhow::{Context, Result};
use clap::Parser;
use tokio::net::TcpListener;
use tracing::info;
use vkremote_server::{run_server_with_config, ServerConfig};

#[derive(Parser, Debug)]
#[command(
    name = "vkremote-server",
    about = "Cross-platform ash-based Vulkan replay daemon"
)]
struct Cli {
    /// Bind address
    #[arg(short, long, default_value = "0.0.0.0")]
    bind: String,

    /// Listening port
    #[arg(short, long, default_value_t = 50051)]
    port: u16,

    /// Shared authentication token
    #[arg(short, long, default_value = "vkremote-secret")]
    token: String,

    /// Index of physical GPU device to use
    #[arg(long)]
    gpu_index: Option<usize>,

    /// Force running with Mock engine (skips physical Vulkan device)
    #[arg(long, default_value_t = false)]
    mock: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let cli = Cli::parse();

    let token = if cli.token == "vkremote-secret" {
        std::env::var("VKREMOTE_TOKEN").unwrap_or(cli.token)
    } else {
        cli.token
    };

    info!(
        "Starting vkremote-server on {}:{} (token configured: {}, gpu_index: {:?}, mock: {})",
        cli.bind,
        cli.port,
        !token.is_empty(),
        cli.gpu_index,
        cli.mock
    );

    let addr = format!("{}:{}", cli.bind, cli.port);
    let listener = TcpListener::bind(&addr)
        .await
        .with_context(|| format!("Failed to bind TCP listener to {}", addr))?;

    let config = ServerConfig {
        token,
        gpu_index: cli.gpu_index,
        force_mock: cli.mock,
    };

    run_server_with_config(listener, config).await
}
