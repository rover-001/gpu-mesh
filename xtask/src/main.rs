use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use std::fs;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(
    name = "xtask",
    about = "Workflow orchestration and testing tasks for vkremote"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate ICD manifest JSON for Vulkan loader
    InstallIcd {
        /// Build release binary instead of debug
        #[arg(long, default_value_t = false)]
        release: bool,
    },
    /// Run vkremote replay server daemon
    RunServer {
        /// Bind address
        #[arg(short, long, default_value = "0.0.0.0")]
        bind: String,
        /// Listening port
        #[arg(short, long, default_value_t = 50051)]
        port: u16,
        /// Auth token
        #[arg(short, long, default_value = "vkremote-secret")]
        token: String,
        /// Force mock engine (no Vulkan hardware required)
        #[arg(long, default_value_t = false)]
        mock: bool,
    },
    /// Launch server loopback and run probe + demos end-to-end
    RunLoopback {
        /// Listening port for loopback server (0 for ephemeral)
        #[arg(short, long, default_value_t = 55051)]
        port: u16,
        /// Build release mode
        #[arg(long, default_value_t = false)]
        release: bool,
        /// Force mock engine on server
        #[arg(long, default_value_t = false)]
        mock: bool,
    },
}

fn get_workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.parent().unwrap_or(&manifest_dir).to_path_buf()
}

fn get_cdylib_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "vkremote_icd.dll"
    } else if cfg!(target_os = "macos") {
        "libvkremote_icd.dylib"
    } else {
        "libvkremote_icd.so"
    }
}

fn build_icd(root: &Path, release: bool) -> Result<PathBuf> {
    let mut cmd = Command::new("cargo");
    cmd.arg("build").arg("-p").arg("vkremote-icd");
    if release {
        cmd.arg("--release");
    }
    cmd.current_dir(root);
    let status = cmd.status().context("Failed to run cargo build for ICD")?;
    if !status.success() {
        return Err(anyhow!("Failed to build vkremote-icd cdylib"));
    }

    let profile = if release { "release" } else { "debug" };
    let lib_name = get_cdylib_name();
    let lib_path = root.join("target").join(profile).join(lib_name);
    let canonical = fs::canonicalize(&lib_path)
        .with_context(|| format!("Built ICD library not found at {:?}", lib_path))?;
    Ok(canonical)
}

fn install_icd(release: bool) -> Result<PathBuf> {
    let root = get_workspace_root();
    let lib_path = build_icd(&root, release)?;

    let target_dir = root.join("target");
    fs::create_dir_all(&target_dir)?;

    let json_path = target_dir.join("vkremote_icd.json");
    let mut path_str = lib_path.to_string_lossy().to_string();
    if let Some(stripped) = path_str.strip_prefix(r"\\?\") {
        path_str = stripped.to_string();
    }
    let clean_path = path_str.replace('\\', "/");

    let json_content = format!(
        r#"{{
    "file_format_version": "1.0.0",
    "ICD": {{
        "library_path": "{}",
        "api_version": "1.3.0"
    }}
}}
"#,
        clean_path
    );

    fs::write(&json_path, json_content)
        .with_context(|| format!("Failed to write ICD manifest to {:?}", json_path))?;

    let canonical_json = fs::canonicalize(&json_path)?;
    let mut json_str = canonical_json.to_string_lossy().to_string();
    if let Some(stripped) = json_str.strip_prefix(r"\\?\") {
        json_str = stripped.to_string();
    }

    println!("\n=== Vulkan ICD Manifest Generated ===");
    println!("Manifest path: {}", json_str);
    println!("Library path:  {}", clean_path);
    println!("\nTo load vkremote ICD into any Vulkan application, set:");
    if cfg!(target_os = "windows") {
        println!("  # In PowerShell:");
        println!("  $env:VK_DRIVER_FILES = \"{}\"", json_str);
        println!("  $env:VK_ICD_FILENAMES = \"{}\"", json_str);
        println!("  # In CMD:");
        println!("  set VK_DRIVER_FILES={}", json_str);
        println!("  set VK_ICD_FILENAMES={}", json_str);
    } else {
        println!("  export VK_DRIVER_FILES=\"{}\"", json_str);
        println!("  export VK_ICD_FILENAMES=\"{}\"", json_str);
    }
    println!();

    Ok(canonical_json)
}

fn wait_for_server(addr: &str, timeout: Duration) -> Result<()> {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if TcpStream::connect(addr).is_ok() {
            return Ok(());
        }
        sleep(Duration::from_millis(50));
    }
    Err(anyhow!(
        "Server failed to start and bind on {} within {:?}",
        addr,
        timeout
    ))
}

struct ServerGuard(Option<Child>);

impl Drop for ServerGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            println!("Shutting down background server (pid: {})...", child.id());
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::InstallIcd { release } => {
            install_icd(release)?;
        }
        Commands::RunServer {
            bind,
            port,
            token,
            mock,
        } => {
            let root = get_workspace_root();
            let mut cmd = Command::new("cargo");
            cmd.arg("run")
                .arg("-p")
                .arg("vkremote-server")
                .arg("--")
                .arg("--bind")
                .arg(&bind)
                .arg("--port")
                .arg(port.to_string())
                .arg("--token")
                .arg(&token);
            if mock {
                cmd.arg("--mock");
            }
            cmd.current_dir(root);
            let status = cmd.status()?;
            if !status.success() {
                return Err(anyhow!("Server exited with error: {:?}", status.code()));
            }
        }
        Commands::RunLoopback {
            port,
            release,
            mock,
        } => {
            let root = get_workspace_root();

            println!("=== Starting vkremote Loopback Verification ===");

            // 1. Install / generate ICD manifest
            let manifest_path = install_icd(release)?;

            // 2. Build server, probe, and demos
            println!("Building server, probe, and demos...");
            let mut build_cmd = Command::new("cargo");
            build_cmd
                .arg("build")
                .arg("-p")
                .arg("vkremote-server")
                .arg("-p")
                .arg("vkremote-probe")
                .arg("-p")
                .arg("vecadd")
                .arg("-p")
                .arg("matmul")
                .arg("-p")
                .arg("mandelbrot");
            if release {
                build_cmd.arg("--release");
            }
            build_cmd.current_dir(&root);
            let status = build_cmd.status()?;
            if !status.success() {
                return Err(anyhow!("Failed to build workspace members"));
            }

            // 3. Launch server in background
            let token = "loopback-token";
            let profile = if release { "release" } else { "debug" };
            let server_bin =
                root.join("target")
                    .join(profile)
                    .join(if cfg!(target_os = "windows") {
                        "vkremote-server.exe"
                    } else {
                        "vkremote-server"
                    });

            println!(
                "Spawning vkremote-server on 127.0.0.1:{} (mock: {})...",
                port, mock
            );
            let mut srv_cmd = Command::new(&server_bin);
            srv_cmd
                .arg("--bind")
                .arg("127.0.0.1")
                .arg("--port")
                .arg(port.to_string())
                .arg("--token")
                .arg(token)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            if mock {
                srv_cmd.arg("--mock");
            }
            let child = srv_cmd.spawn().context("Failed to spawn vkremote-server")?;
            let _server_guard = ServerGuard(Some(child));

            // 4. Wait for server TCP port
            let addr = format!("127.0.0.1:{}", port);
            wait_for_server(&addr, Duration::from_secs(5))?;
            println!("Server is listening on {}\n", addr);

            let run_client_app = |app_name: &str| -> Result<()> {
                println!("--- Running {} via vkremote ICD ---", app_name);
                let app_bin =
                    root.join("target")
                        .join(profile)
                        .join(if cfg!(target_os = "windows") {
                            format!("{}.exe", app_name)
                        } else {
                            app_name.to_string()
                        });

                let mut cmd = Command::new(&app_bin);
                cmd.env("VK_DRIVER_FILES", &manifest_path)
                    .env("VKREMOTE_SERVER", &addr)
                    .env("VKREMOTE_TOKEN", token)
                    .current_dir(&root);

                let status = cmd
                    .status()
                    .with_context(|| format!("Failed to run app {}", app_name))?;

                if !status.success() {
                    return Err(anyhow!(
                        "App {} failed with exit status {:?}",
                        app_name,
                        status
                    ));
                }
                println!("--- {} PASSED ---\n", app_name);
                Ok(())
            };

            // 5. Run probe
            run_client_app("probe")?;

            // 6. Run vecadd
            run_client_app("vecadd")?;

            // 7. Run matmul
            run_client_app("matmul")?;

            // 8. Run mandelbrot
            run_client_app("mandelbrot")?;

            println!("==================================================");
            println!("[LOOPBACK] ALL INTEGRATION CHECKS PASSED SUCCESSFULLY!");
            println!("==================================================");
        }
    }

    Ok(())
}
