use anyhow::{anyhow, Context, Result};
use rcgen::generate_simple_self_signed;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, SignatureScheme};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::{TlsAcceptor, TlsConnector};

const MAX_CONTROL_FRAME: u32 = 1024 * 1024;
const MODEL_CHUNK_SIZE: usize = 1024 * 1024;
const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(30 * 60);

const DISASTER_BASE_MODEL: &str = "Qwen/Qwen2.5-1.5B-Instruct";
const DISASTER_GGUF_FILENAME: &str = "qwen2.5-1.5b-instruct-q4_k_m.gguf";

#[derive(serde::Deserialize)]
struct ActiveAdapter {
    format_version: u32,
    base_model: String,
    base_gguf_filename: String,
    adapter_file: String,
    adapter_sha256: String,
}

pub const DISASTER_CHAT_TEMPLATE: &str = include_str!("../../../data/disaster_model_chat_template.jinja");

fn is_disaster_model(model: &Path) -> bool {
    model.file_name().and_then(|name| name.to_str()) == Some(DISASTER_GGUF_FILENAME)
        || model.components().any(|part| part.as_os_str().to_string_lossy().starts_with("checkpoints-disaster-1.5b"))
}

pub fn configure_disaster_chat(command: &mut Command, model: &Path) {
    if is_disaster_model(model) {
        command.args(["--jinja", "--chat-template", DISASTER_CHAT_TEMPLATE, "--repeat-penalty", "1.1"]);
    }
}

fn validate_adapter_export(model: &Path, checkpoint_dir: &Path) -> Result<Option<PathBuf>> {
    if model.file_name().and_then(|name| name.to_str()) != Some(DISASTER_GGUF_FILENAME) {
        return Ok(None);
    }
    let manifest_path = checkpoint_dir.join("active_adapter.json");
    if !manifest_path.exists() {
        return Ok(None);
    }
    let manifest: ActiveAdapter = serde_json::from_slice(&std::fs::read(&manifest_path)?)
        .with_context(|| format!("Invalid adapter export {}", manifest_path.display()))?;
    if manifest.format_version != 1
        || manifest.base_model != DISASTER_BASE_MODEL
        || manifest.base_gguf_filename != DISASTER_GGUF_FILENAME
    {
        return Err(anyhow!("Adapter export is incompatible with {}", model.display()));
    }
    let relative = Path::new(&manifest.adapter_file);
    if relative.as_os_str().is_empty()
        || relative.components().any(|part| !matches!(part, std::path::Component::Normal(_)))
    {
        return Err(anyhow!("Adapter export must reference a relative checkpoint file"));
    }
    let checkpoint_root = checkpoint_dir.canonicalize()?;
    let adapter_path = checkpoint_dir.join(relative).canonicalize()?;
    if !adapter_path.starts_with(&checkpoint_root) {
        return Err(anyhow!("Adapter export escapes its checkpoint directory"));
    }
    let mut adapter = std::fs::File::open(&adapter_path)?;
    let mut magic = [0; 4];
    adapter.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(anyhow!("Adapter export is not a GGUF tensor file"));
    }
    let mut digest = Sha256::new();
    digest.update(magic);
    let mut buffer = [0; 65536];
    loop {
        let count = adapter.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    if format!("{:x}", digest.finalize()) != manifest.adapter_sha256.to_ascii_lowercase() {
        return Err(anyhow!("Adapter export checksum mismatch: {}", adapter_path.display()));
    }
    Ok(Some(adapter_path))
}

/// Resolve a validated real LoRA export for the matching base model only.
/// Missing exports use the base model; malformed exports fail rather than loading
/// unverified tensors or presenting a failed training run as an active adapter.
pub fn trained_adapter_for_model(model: &Path) -> Result<Option<PathBuf>> {
    if model.file_name().and_then(|name| name.to_str()) != Some(DISASTER_GGUF_FILENAME) {
        return Ok(None);
    }
    let mut directories = Vec::new();
    if let Some(parent) = model.parent() {
        directories.push(parent.join("checkpoints-disaster-1.5b"));
    }
    let cache = get_cache_dir().join("checkpoints-disaster-1.5b");
    if !directories.contains(&cache) {
        directories.push(cache);
    }
    if cfg!(target_os = "windows") {
        let shared = PathBuf::from("C:\\rgpu\\models\\checkpoints-disaster-1.5b");
        if !directories.contains(&shared) {
            directories.push(shared);
        }
    }
    for directory in directories {
        if let Some(adapter) = validate_adapter_export(model, &directory)? {
            eprintln!("[gpu-mesh] Loading trained LoRA adapter {}", adapter.display());
            return Ok(Some(adapter));
        }
    }
    eprintln!("[gpu-mesh] No validated trained adapter found; using the original Qwen 1.5B base model");
    Ok(None)
}

fn same_local_file(left: &Path, right: &Path) -> bool {
    left == right || match (left.canonicalize(), right.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn worker_configuration_matches(model: &Path, adapter: Option<&Path>, properties: &serde_json::Value, adapters: &serde_json::Value) -> bool {
    let Some(loaded_model) = properties["model_path"].as_str() else { return false };
    if !same_local_file(model, Path::new(loaded_model)) {
        return false;
    }
    if is_disaster_model(model)
        && properties["chat_template"].as_str().map(str::trim) != Some(DISASTER_CHAT_TEMPLATE.trim()) {
        return false;
    }
    let Some(loaded_adapters) = adapters.as_array() else { return false };
    match adapter {
        None => loaded_adapters.is_empty(),
        Some(expected) => loaded_adapters.len() == 1 && loaded_adapters[0]["path"].as_str()
            .map(|loaded| same_local_file(expected, Path::new(loaded))).unwrap_or(false)
            && loaded_adapters[0]["scale"].as_f64() == Some(1.0),
    }
}

async fn worker_matches_requested_model(model: &Path) -> Result<bool> {
    let adapter = trained_adapter_for_model(model)?;
    let model = model.to_path_buf();
    Ok(tokio::task::spawn_blocking(move || {
        let get_json = |path| {
            let body = crate::cluster_engine::http_get_local(18080, path, Duration::from_secs(2)).ok()?;
            serde_json::from_str::<serde_json::Value>(&body).ok()
        };
        match (get_json("/props"), get_json("/lora-adapters")) {
            (Some(properties), Some(adapters)) => worker_configuration_matches(&model, adapter.as_deref(), &properties, &adapters),
            _ => false,
        }
    }).await.unwrap_or(false))
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelManifest {
    pub name: String,
    pub hf_repo: String,
    pub filename: String,
    pub sha256: String,
    pub size: u64,
}

impl ModelManifest {
    pub fn default_qwen_7b() -> Self {
        Self {
            name: "Qwen2.5-Coder-7B-Instruct-Q4_K_M".to_string(),
            hf_repo: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF".to_string(),
            filename: "qwen2.5-coder-7b-instruct-q4_k_m.gguf".to_string(),
            sha256: "509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c".to_string(),
            size: 4683073536,
        }
    }

    pub fn disaster_qwen_1_5b() -> Self {
        Self {
            name: "Qwen 2.5 1.5B (Disaster Model)".to_string(),
            hf_repo: "Qwen/Qwen2.5-1.5B-Instruct-GGUF".to_string(),
            filename: DISASTER_GGUF_FILENAME.to_string(),
            sha256: "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e".to_string(),
            size: 1117320736,
        }
    }

    pub fn qwen_14b_coder() -> Self {
        Self {
            name: "Qwen2.5-Coder-14B-Instruct-Q4_K_M".to_string(),
            hf_repo: "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF".to_string(),
            filename: "qwen2.5-coder-14b-instruct-q4_k_m.gguf".to_string(),
            sha256: "c1e659736d89ac1065fb495330fb824d94001974a4bfa78e7270e43476a8d940".to_string(),
            size: 8988110272,
        }
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
enum WorkerMessage {
    Authenticate { token: String },
    StartManifestSession { manifest: ModelManifest },
    StartSession { size: u64, sha256: [u8; 32] },
    ModelCached,
    NeedModel,
    ModelChunk { size: u32 },
    ModelComplete { sha256: [u8; 32] },
    StartDecode,
    Prompt { prompt: String, n_predict: u32 },
    PromptChunk { done: bool, data: Vec<u8> },
    Error { message: String },
    Goodbye,
}

async fn write_message<S: AsyncWrite + Unpin>(stream: &mut S, msg: &WorkerMessage) -> Result<()> {
    let data = postcard::to_allocvec(msg)?;
    if data.len() > MAX_CONTROL_FRAME as usize {
        return Err(anyhow!("worker control message too large"));
    }
    stream.write_all(&(data.len() as u32).to_le_bytes()).await?;
    stream.write_all(&data).await?;
    stream.flush().await?;
    Ok(())
}

async fn read_message<S: AsyncRead + Unpin>(stream: &mut S) -> Result<WorkerMessage> {
    let mut len = [0; 4];
    stream.read_exact(&mut len).await?;
    let len = u32::from_le_bytes(len);
    if len > MAX_CONTROL_FRAME {
        return Err(anyhow!("worker control message exceeds limit"));
    }
    let mut data = vec![0; len as usize];
    stream.read_exact(&mut data).await?;
    Ok(postcard::from_bytes(&data)?)
}

async fn write_bytes<S: AsyncWrite + Unpin>(stream: &mut S, data: &[u8]) -> Result<()> {
    stream.write_all(data).await?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TrustLevel {
    Trusted,
    Sealed,
    Ephemeral,
}

pub fn get_cache_dir() -> PathBuf {
    if cfg!(target_os = "windows") {
        let program_data =
            std::env::var("ProgramData").unwrap_or_else(|_| "C:\\ProgramData".to_string());
        let dir = PathBuf::from(program_data).join("gpu-mesh").join("models");
        if std::fs::create_dir_all(&dir).is_ok() {
            return dir;
        }
        let fallback = PathBuf::from("C:\\rgpu\\models");
        let _ = std::fs::create_dir_all(&fallback);
        fallback
    } else {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/rover".to_string());
        let dir = PathBuf::from(home)
            .join(".cache")
            .join("gpu-mesh")
            .join("models");
        let _ = std::fs::create_dir_all(&dir);
        dir
    }
}

/// True if `p` looks like a complete copy of a model of `expected` bytes: GGUF
/// magic and a size within 1% of the manifest. Rejects truncated downloads (e.g.
/// a 49-byte stub) while accepting re-uploads of the same quantisation whose
/// metadata differs by a few hundred bytes (seen on peers: 7B +416 B, 14B +512 B).
pub fn is_usable_gguf(p: &Path, expected: u64) -> bool {
    let Ok(meta) = p.metadata() else { return false };
    if !meta.is_file() {
        return false;
    }
    let len = meta.len();
    if expected > 0 && len.abs_diff(expected) > expected / 100 {
        return false;
    }
    if expected == 0 && len < 64 * 1024 * 1024 {
        return false;
    }
    let mut magic = [0u8; 4];
    std::fs::File::open(p)
        .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut magic))
        .map(|_| &magic == b"GGUF")
        .unwrap_or(false)
}

pub fn find_cached_model(size: u64, sha256: &[u8; 32]) -> (PathBuf, bool) {
    let hash = sha256.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let primary = get_cache_dir().join(format!("{hash}.gguf"));
    if primary.exists() && primary.metadata().map(|m| m.len() == size).unwrap_or(false) {
        return (primary, true);
    }

    let mut search_dirs = vec![get_cache_dir()];
    if cfg!(target_os = "windows") {
        search_dirs.push(PathBuf::from("C:\\rgpu\\models"));
        search_dirs.push(std::env::temp_dir());
    } else {
        // The current user's ~/models first, so workers on other machines (e.g.
        // /home/rahul/models) find their local copy offline instead of downloading.
        if let Some(home) = std::env::var_os("HOME") {
            let home = PathBuf::from(home);
            search_dirs.push(home.join("models"));
            search_dirs.push(home);
        }
        search_dirs.push(PathBuf::from("/home/rover/models"));
        search_dirs.push(PathBuf::from("/home/rover"));
    }

    for dir in &search_dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_file() {
                    let fname = p.file_name().unwrap_or_default().to_string_lossy();
                    if fname.contains(&hash) || fname.ends_with(".gguf") {
                        if let Ok(m) = p.metadata() {
                            if m.len() == size {
                                return (p, true);
                            }
                        }
                    }
                }
            }
        }
    }

    // Second pass: same model from a slightly different upload (size within 1%,
    // valid GGUF) so an offline peer uses its local copy instead of a re-transfer.
    for dir in &search_dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                let is_gguf = p.extension().map_or(false, |e| e == "gguf");
                if is_gguf && is_usable_gguf(&p, size) {
                    return (p, true);
                }
            }
        }
    }

    (primary, false)
}

fn cache_file(size: u64, sha256: &[u8; 32]) -> PathBuf {
    find_cached_model(size, sha256).0
}

struct CachedModel {
    path: PathBuf,
}

async fn file_sha256(path: &Path) -> Result<[u8; 32]> {
    let mut file = tokio::fs::File::open(path).await?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; MODEL_CHUNK_SIZE];
    loop {
        let n = file.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().into())
}

struct WorkerSession {
    model: CachedModel,
    server: Child,
}

/// Worker llama-server (port 18080) reports ready, checked without spawning curl.
async fn worker_health_ok(timeout: Duration) -> bool {
    tokio::task::spawn_blocking(move || {
        crate::cluster_engine::http_get_local(18080, "/health", timeout)
            .map(|body| body.contains("\"status\":\"ok\""))
            .unwrap_or(false)
    })
    .await
    .unwrap_or(false)
}

pub async fn resolve_manifest(manifest: &ModelManifest) -> Result<PathBuf> {
    let cache_dir = get_cache_dir();
    let mut search_dirs = vec![cache_dir.clone()];
    if cfg!(target_os = "windows") {
        search_dirs.push(PathBuf::from("C:\\rgpu\\models"));
        search_dirs.push(std::env::temp_dir());
    } else {
        // The current user's ~/models first, so workers on other machines (e.g.
        // /home/rahul/models) find their local copy offline instead of downloading.
        if let Some(home) = std::env::var_os("HOME") {
            let home = PathBuf::from(home);
            search_dirs.push(home.join("models"));
            search_dirs.push(home);
        }
        search_dirs.push(PathBuf::from("/home/rover/models"));
        search_dirs.push(PathBuf::from("/home/rover"));
    }

    // 1. Check local search directories for filename match with matching size
    for dir in &search_dirs {
        let p = dir.join(&manifest.filename);
        if p.exists() {
            if let Ok(meta) = p.metadata() {
                if meta.len() == manifest.size {
                    return Ok(p);
                }
            }
        }
    }
    // 1b. Same filename, valid GGUF, size within 1% (newer upload of the same quant).
    for dir in &search_dirs {
        let p = dir.join(&manifest.filename);
        if is_usable_gguf(&p, manifest.size) {
            return Ok(p);
        }
    }

    // 2. Direct download on PC from Hugging Face if hf_repo is present
    if !manifest.hf_repo.is_empty() {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            manifest.hf_repo, manifest.filename
        );
        let temp_dest = cache_dir.join(format!("{}.tmp", manifest.filename));
        let final_dest = cache_dir.join(&manifest.filename);

        let status = tokio::process::Command::new("curl")
            .args([
                "-L",
                "-C",
                "-",
                "-o",
                temp_dest.to_string_lossy().as_ref(),
                &url,
            ])
            .status()
            .await?;

        if status.success() {
            tokio::fs::rename(temp_dest, &final_dest).await?;
            return Ok(final_dest);
        }
    }

    Err(anyhow!("Model not found in cache and could not be resolved from Hugging Face"))
}

fn spawn_worker_llama(model: &Path, port: u16) -> Result<Child> {
    let bin = crate::cluster_engine::ClusterEngineManager::get_default_bin_path();
    if !Path::new(&bin).exists() {
        return Err(anyhow!("llama-server not found at {bin}"));
    }

    // Terminate any previous llama-server instance to avoid port conflicts
    if cfg!(target_os = "windows") {
        let _ = std::process::Command::new("taskkill")
            .args(["/F", "/IM", "llama-server.exe"])
            .output();
    } else {
        let _ = std::process::Command::new("pkill")
            .args(["-9", "llama-server"])
            .output();
    }

    let mut cmd = Command::new(&bin);
    let mut args = vec![
        "-m".to_string(),
        model.to_string_lossy().to_string(),
        "-ngl".to_string(),
        "99".to_string(),
        "-fa".to_string(),
        "on".to_string(),
        "-lm".to_string(),
        "none".to_string(),
        "--host".to_string(),
        "0.0.0.0".to_string(),
        "--port".to_string(),
        port.to_string(),
        "-c".to_string(),
        "4096".to_string(),
    ];

    if let Some(adapter) = trained_adapter_for_model(model)? {
        args.extend(["--lora".to_string(), adapter.to_string_lossy().to_string()]);
    }

    let webui_path = if cfg!(target_os = "windows") {
        "C:\\rgpu\\tools\\llama-webui".to_string()
    } else {
        crate::cluster_engine::llama_webui_dir().to_string_lossy().to_string()
    };
    if Path::new(&webui_path).exists() {
        args.push("--path".to_string());
        args.push(webui_path);
    }

    configure_disaster_chat(&mut cmd, model);
    cmd.args(&args)
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    if cfg!(target_os = "windows") {
        if let Some(parent) = Path::new(&bin).parent() {
            cmd.current_dir(parent);
            let mut paths = vec![parent.to_string_lossy().to_string()];
            if let Ok(existing) = std::env::var("PATH") {
                paths.push(existing);
            }
            cmd.env("PATH", paths.join(";"));
        }
    }

    #[cfg(unix)]
    cmd.env(
        "LD_LIBRARY_PATH",
        format!(
            "{}:/usr/local/lib/ollama/cuda_v12",
            crate::cluster_engine::llama_bin_dir().display()
        ),
    );
    Ok(cmd.spawn()?)
}

async fn worker_connection<S>(mut stream: S, token: String, idle_timeout: Duration) -> Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let auth = read_message(&mut stream).await?;
    match auth {
        WorkerMessage::Authenticate { token: presented } if presented == token => {}
        WorkerMessage::Authenticate { .. } => {
            write_message(
                &mut stream,
                &WorkerMessage::Error {
                    message: "worker authentication failed".to_string(),
                },
            )
            .await?;
            return Err(anyhow!("worker authentication failed"));
        }
        _ => return Err(anyhow!("worker session must authenticate first")),
    }
    let auth = read_message(&mut stream).await?;
    let path = match auth {
        WorkerMessage::StartManifestSession { manifest } => {
            match resolve_manifest(&manifest).await {
                Ok(p) => {
                    write_message(&mut stream, &WorkerMessage::ModelCached).await?;
                    p
                }
                Err(_) => {
                    write_message(&mut stream, &WorkerMessage::NeedModel).await?;
                    let cache_path = get_cache_dir().join(&manifest.filename);
                    let mut file = tokio::fs::File::create(&cache_path).await?;
                    let mut hasher = Sha256::new();
                    let mut received = 0u64;
                    while received < manifest.size {
                        let msg = tokio::time::timeout(idle_timeout, read_message(&mut stream)).await??;
                        let chunk_size = match msg {
                            WorkerMessage::ModelChunk { size } => size as usize,
                            _ => return Err(anyhow!("expected model chunk")),
                        };
                        if chunk_size == 0 || chunk_size > MODEL_CHUNK_SIZE || received + chunk_size as u64 > manifest.size {
                            return Err(anyhow!("invalid model chunk size"));
                        }
                        let mut chunk = vec![0u8; chunk_size];
                        stream.read_exact(&mut chunk).await?;
                        hasher.update(&chunk);
                        file.write_all(&chunk).await?;
                        received += chunk_size as u64;
                    }
                    file.flush().await?;
                    drop(file);
                    let received_hash = match tokio::time::timeout(idle_timeout, read_message(&mut stream)).await?? {
                        WorkerMessage::ModelComplete { sha256 } => sha256,
                        _ => return Err(anyhow!("expected model completion digest")),
                    };
                    if hasher.finalize().as_slice() != received_hash {
                        let _ = tokio::fs::remove_file(&cache_path).await;
                        return Err(anyhow!("model hash mismatch"));
                    }
                    cache_path
                }
            }
        }
        WorkerMessage::StartSession { size, sha256 } => {
            if size == 0 {
                return Err(anyhow!("empty model is not allowed"));
            }
            let (path, is_found) = find_cached_model(size, &sha256);
            let cached = is_found
                || (sha256 != [0; 32]
                    && tokio::fs::metadata(&path)
                        .await
                        .map(|metadata| metadata.len() == size)
                        .unwrap_or(false)
                    && file_sha256(&path).await.ok() == Some(sha256));

            if cached {
                write_message(&mut stream, &WorkerMessage::ModelCached).await?;
                path
            } else {
                write_message(&mut stream, &WorkerMessage::NeedModel).await?;
                let mut file = tokio::fs::File::create(&path).await?;
                let mut hasher = Sha256::new();
                let mut received = 0u64;
                while received < size {
                    let msg = tokio::time::timeout(idle_timeout, read_message(&mut stream)).await??;
                    let chunk_size = match msg {
                        WorkerMessage::ModelChunk { size } => size as usize,
                        _ => return Err(anyhow!("expected model chunk")),
                    };
                    if chunk_size == 0 || chunk_size > MODEL_CHUNK_SIZE || received + chunk_size as u64 > size {
                        return Err(anyhow!("invalid model chunk size"));
                    }
                    let mut chunk = vec![0u8; chunk_size];
                    stream.read_exact(&mut chunk).await?;
                    hasher.update(&chunk);
                    file.write_all(&chunk).await?;
                    received += chunk_size as u64;
                }
                file.flush().await?;
                drop(file);
                let received_hash = match tokio::time::timeout(idle_timeout, read_message(&mut stream)).await?? {
                    WorkerMessage::ModelComplete { sha256 } => sha256,
                    _ => return Err(anyhow!("expected model completion digest")),
                };
                if hasher.finalize().as_slice() != received_hash {
                    let _ = tokio::fs::remove_file(&path).await;
                    return Err(anyhow!("model hash mismatch"));
                }
                path
            }
        }
        _ => return Err(anyhow!("worker session must start with StartSession or StartManifestSession")),
    };

    // Health alone does not prove this is the requested model or the newly
    // trained adapter. Reusing a stock-model process would silently discard training.
    let already_running = worker_health_ok(Duration::from_secs(1)).await
        && worker_matches_requested_model(&path).await?;

    let mut spawned_child = None;
    if !already_running {
        let child = spawn_worker_llama(&path, 18080)?;
        spawned_child = Some(child);
        let mut ready = false;
        for _ in 0..120 {
            let healthy = worker_health_ok(Duration::from_secs(2)).await;
            if healthy {
                ready = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        if !ready {
            if let Some(mut child) = spawned_child {
                let _ = child.kill();
                let _ = child.wait();
            }
            return Err(anyhow!("worker llama-server did not become ready"));
        }
    }
    write_message(&mut stream, &WorkerMessage::StartDecode).await?;

    loop {
        let msg = match tokio::time::timeout(idle_timeout, read_message(&mut stream)).await {
            Ok(result) => result?,
            Err(_) => break,
        };
        match msg {
            WorkerMessage::Prompt { prompt, n_predict } => {
                let body = serde_json::json!({
                    "prompt": prompt,
                    "n_predict": n_predict,
                    "temperature": 0.7,
                    "stream": false
                });
                let output = tokio::process::Command::new("curl")
                    .args([
                        "-sS",
                        "--max-time",
                        "120",
                        "-X",
                        "POST",
                        "http://127.0.0.1:18080/completion",
                        "-H",
                        "Content-Type: application/json",
                        "-d",
                        &body.to_string(),
                    ])
                    .output()
                    .await
                    .context("worker completion request failed")?;
                if !output.status.success() {
                    write_message(
                        &mut stream,
                        &WorkerMessage::Error {
                            message: String::from_utf8_lossy(&output.stderr).into_owned(),
                        },
                    )
                    .await?;
                    continue;
                }
                write_message(
                    &mut stream,
                    &WorkerMessage::PromptChunk {
                        done: true,
                        data: output.stdout,
                    },
                )
                .await?;
            }
            WorkerMessage::Goodbye => break,
            _ => return Err(anyhow!("unexpected worker message")),
        }
    }
    if let Some(mut child) = spawned_child.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
    if cfg!(target_os = "windows") {
        let _ = std::process::Command::new("taskkill")
            .args(["/F", "/IM", "llama-server.exe"])
            .output();
    } else {
        let _ = std::process::Command::new("pkill")
            .args(["-9", "llama-server"])
            .output();
    }
    Ok(())
}

struct AcceptAnyCertificate;

impl std::fmt::Debug for AcceptAnyCertificate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AcceptAnyCertificate")
    }
}

impl ServerCertVerifier for AcceptAnyCertificate {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> std::result::Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        vec![
            SignatureScheme::ECDSA_NISTP256_SHA256,
            SignatureScheme::ED25519,
        ]
    }
}

fn client_config() -> Result<rustls::ClientConfig> {
    Ok(rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(AcceptAnyCertificate))
        .with_no_client_auth())
}

fn server_config() -> Result<rustls::ServerConfig> {
    let cert = generate_simple_self_signed(vec!["gpu-mesh".to_string()])?;
    let certs = vec![CertificateDer::from(cert.cert.der().to_vec())];
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der()));
    Ok(rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)?)
}

pub fn start_worker_server(port: u16, token: String) -> Result<tokio::sync::oneshot::Sender<()>> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let listener = std::net::TcpListener::bind(("0.0.0.0", port))
        .with_context(|| format!("failed to bind worker port {port}"))?;
    listener.set_nonblocking(true)?;
    let acceptor = TlsAcceptor::from(Arc::new(server_config()?));
    let (stop, mut stop_rx) = tokio::sync::oneshot::channel();
    crate::tokio_rt().spawn(async move {
        let listener = match TcpListener::from_std(listener) {
            Ok(listener) => listener,
            Err(error) => {
                tracing::error!("failed to create worker listener: {error}");
                return;
            }
        };
        loop {
            tokio::select! {
                _ = &mut stop_rx => break,
                result = listener.accept() => {
                    let Ok((stream, _)) = result else { continue };
                    let acceptor = acceptor.clone();
                    let token = token.clone();
                    crate::tokio_rt().spawn(async move {
                        let Ok(tls) = acceptor.accept(stream).await else { return };
                        let _ = worker_connection(tls, token, DEFAULT_IDLE_TIMEOUT).await;
                    });
                }
            }
        }
    });
    Ok(stop)
}

pub async fn start_remote_session(
    endpoint: &str,
    token: &str,
    manifest: &ModelManifest,
) -> Result<RemoteSession> {
    // Bounded connect: an unreachable worker (peer app not running, or a firewall
    // silently dropping SYNs) otherwise blocks for the OS TCP timeout (~2 min)
    // before Start Server falls back to local/RPC mode.
    let stream = tokio::time::timeout(Duration::from_secs(4), TcpStream::connect(endpoint))
        .await
        .map_err(|_| anyhow!("worker {endpoint} did not answer within 4 s"))??;
    let connector = TlsConnector::from(Arc::new(client_config()?));
    let server_name =
        ServerName::try_from("gpu-mesh").map_err(|_| anyhow!("invalid worker name"))?;
    let mut stream = connector.connect(server_name, stream).await?;
    write_message(
        &mut stream,
        &WorkerMessage::Authenticate {
            token: token.to_string(),
        },
    )
    .await?;
    write_message(
        &mut stream,
        &WorkerMessage::StartManifestSession {
            manifest: manifest.clone(),
        },
    )
    .await?;
    let transfer = read_message(&mut stream).await?;
    if matches!(transfer, WorkerMessage::NeedModel) {
        let model_path = crate::cluster_engine::ClusterEngineManager::get_default_model_path();
        if let Ok(mut model) = std::fs::File::open(&model_path) {
            let mut buf = vec![0u8; MODEL_CHUNK_SIZE];
            let mut hasher = Sha256::new();
            loop {
                let n = model.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                hasher.update(&buf[..n]);
                write_message(&mut stream, &WorkerMessage::ModelChunk { size: n as u32 }).await?;
                write_bytes(&mut stream, &buf[..n]).await?;
            }
            let hash: [u8; 32] = hasher.finalize().into();
            write_message(&mut stream, &WorkerMessage::ModelComplete { sha256: hash }).await?;
        } else {
            return Err(anyhow!("Worker requested model transfer but local file is not available"));
        }
    } else if !matches!(transfer, WorkerMessage::ModelCached) {
        return Err(anyhow!("worker rejected model session"));
    }
    match read_message(&mut stream).await? {
        WorkerMessage::StartDecode => Ok(RemoteSession { stream }),
        WorkerMessage::Error { message } => Err(anyhow!(message)),
        _ => Err(anyhow!("worker did not acknowledge model load")),
    }
}

pub struct RemoteSession {
    stream: tokio_rustls::client::TlsStream<TcpStream>,
}

pub fn largest_fitting_worker(
    peers: &[crate::mesh::RemoteGiver],
    model_size_mb: u64,
) -> Option<String> {
    peers
        .iter()
        .filter(|peer| peer.effective_vram_mb() >= model_size_mb.saturating_mul(120) / 100)
        .max_by_key(|peer| peer.effective_vram_mb())
        .map(|peer| {
            let host = peer.endpoint.split(':').next().unwrap_or(&peer.endpoint);
            format!("{host}:50053")
        })
}

impl RemoteSession {
    pub async fn prompt(&mut self, prompt: String, n_predict: u32) -> Result<Vec<u8>> {
        write_message(
            &mut self.stream,
            &WorkerMessage::Prompt { prompt, n_predict },
        )
        .await?;
        match read_message(&mut self.stream).await? {
            WorkerMessage::PromptChunk { data, .. } => Ok(data),
            WorkerMessage::Error { message } => Err(anyhow!(message)),
            _ => Err(anyhow!("worker returned an unexpected prompt response")),
        }
    }

    pub async fn stop(&mut self) -> Result<()> {
        let _ = write_message(&mut self.stream, &WorkerMessage::Goodbye).await;
        Ok(())
    }
}

#[cfg(test)]
mod model_resolution_check {
    /// Prints how each built-in model resolves on this machine (no downloads).
    /// Run: cargo test -p gpu-mesh print_model_resolution -- --ignored --nocapture
    #[test]
    #[ignore]
    fn print_model_resolution() {
        let models = [
            super::ModelManifest::default_qwen_7b(),
            super::ModelManifest::qwen_14b_coder(),
            super::ModelManifest::disaster_qwen_1_5b(),
        ];
        for m in models {
            let ui = crate::cluster_engine::ClusterEngineManager::get_model_path_for_manifest(&m);
            let mut sha = [0u8; 32];
            for (i, b) in sha.iter_mut().enumerate() {
                *b = u8::from_str_radix(&m.sha256[i * 2..i * 2 + 2], 16).unwrap();
            }
            let (worker, found) = super::find_cached_model(m.size, &sha);
            println!(
                "{:<42} ui={} usable={} | worker={} found={}",
                m.filename,
                ui,
                super::is_usable_gguf(std::path::Path::new(&ui), m.size),
                worker.display(),
                found
            );
        }
    }
}

#[cfg(test)]
mod trained_adapter_tests {
    use super::*;

    struct TestExport(PathBuf);

    impl TestExport {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let directory = std::env::temp_dir().join(format!("rgpu-adapter-{}-{}", std::process::id(), NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
            std::fs::create_dir_all(directory.join("epoch-1")).unwrap();
            std::fs::write(directory.join("epoch-1/adapter.gguf"), b"GGUFtest tensor payload").unwrap();
            let export = Self(directory);
            export.write_manifest(DISASTER_BASE_MODEL, "epoch-1/adapter.gguf", &format!("{:x}", Sha256::digest(b"GGUFtest tensor payload")));
            export
        }

        fn write_manifest(&self, base_model: &str, file: &str, checksum: &str) {
            std::fs::write(self.0.join("active_adapter.json"), serde_json::to_vec(&serde_json::json!({
                "format_version": 1,
                "base_model": base_model,
                "base_gguf_filename": DISASTER_GGUF_FILENAME,
                "adapter_file": file,
                "adapter_sha256": checksum
            })).unwrap()).unwrap();
        }
    }

    impl Drop for TestExport {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn loads_matching_verified_export_and_ignores_other_models() {
        let export = TestExport::new();
        let actual = validate_adapter_export(Path::new(DISASTER_GGUF_FILENAME), &export.0).unwrap().unwrap();
        assert_eq!(actual, export.0.join("epoch-1/adapter.gguf").canonicalize().unwrap());
        assert!(validate_adapter_export(Path::new("qwen2.5-coder-7b-instruct-q4_k_m.gguf"), &export.0).unwrap().is_none());
    }

    #[test]
    fn rejects_tampered_adapter_instead_of_loading_unverified_weights() {
        let export = TestExport::new();
        std::fs::write(export.0.join("epoch-1/adapter.gguf"), b"GGUFcorrupted weights").unwrap();
        let error = validate_adapter_export(Path::new(DISASTER_GGUF_FILENAME), &export.0).unwrap_err();
        assert!(error.to_string().contains("checksum mismatch"));
    }

    #[test]
    fn rejects_wrong_base_and_escaping_checkpoint_paths() {
        let export = TestExport::new();
        export.write_manifest("Qwen/Qwen2.5-Coder-7B-Instruct", "epoch-1/adapter.gguf", "");
        assert!(validate_adapter_export(Path::new(DISASTER_GGUF_FILENAME), &export.0).unwrap_err().to_string().contains("incompatible"));
        export.write_manifest(DISASTER_BASE_MODEL, "../outside.gguf", "");
        assert!(validate_adapter_export(Path::new(DISASTER_GGUF_FILENAME), &export.0).unwrap_err().to_string().contains("relative checkpoint"));
    }

    #[test]
    fn running_worker_must_match_model_and_active_adapter() {
        let model = Path::new("/models/qwen.gguf");
        let adapter = Path::new("/models/checkpoints/epoch-1/adapter.gguf");
        let properties = serde_json::json!({"model_path": "/models/qwen.gguf"});
        let loaded = serde_json::json!([{"path": "/models/checkpoints/epoch-1/adapter.gguf", "scale": 1.0}]);
        assert!(worker_configuration_matches(model, Some(adapter), &properties, &loaded));
        assert!(!worker_configuration_matches(model, Some(adapter), &properties, &serde_json::json!([])));
        assert!(!worker_configuration_matches(model, None, &properties, &loaded));
        assert!(!worker_configuration_matches(Path::new("/models/different.gguf"), Some(adapter), &properties, &loaded));
    }
}
