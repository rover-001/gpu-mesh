use anyhow::{anyhow, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingTarget {
    Local,
    Peer,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainTelemetry {
    pub step: usize,
    pub total_steps: usize,
    pub epoch: usize,
    pub total_epochs: usize,
    pub loss: f32,
    pub best_loss: Option<f32>,
    pub lr: f32,
    pub tok_sec: f32,
    pub status: String,
    #[serde(default)]
    pub sample_title: String,
    #[serde(default)]
    pub sample_category: String,
    #[serde(default)]
    pub prompt_preview: String,
    #[serde(default)]
    pub response_preview: String,
}

/// Robust ANSI escape sequence remover that strips both standard ESC[...m
/// sequences and stray [36m / [0m formatting codes when ESC is omitted.
pub fn clean_log_line(s: &str) -> String {
    let mut res = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Standard ANSI CSI: \x1b[ ... [a-zA-Z]
            if chars.peek() == Some(&'[') {
                chars.next();
                while let Some(&next) = chars.peek() {
                    chars.next();
                    if next.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
        } else if c == '[' {
            // Stray bracket sequence like [36m, [0m, [1m, [32;1m
            let mut buf = String::new();
            let mut matched = false;
            while let Some(&next) = chars.peek() {
                if next.is_ascii_digit() || next == ';' {
                    buf.push(chars.next().unwrap());
                } else if next == 'm' {
                    chars.next();
                    matched = true;
                    break;
                } else {
                    break;
                }
            }
            if !matched {
                res.push('[');
                res.push_str(&buf);
            }
        } else {
            res.push(c);
        }
    }
    res
}

fn training_project_root() -> Result<std::path::PathBuf> {
    let mut candidates = Vec::new();
    if let Some(root) = std::env::var_os("RGPU_ROOT") {
        candidates.push(std::path::PathBuf::from(root));
    }
    if let Ok(executable) = std::env::current_exe() {
        candidates.extend(executable.ancestors().skip(1).map(std::path::Path::to_path_buf));
    }
    if let Ok(current) = std::env::current_dir() {
        candidates.push(current);
    }
    candidates.push(std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."));
    candidates.into_iter().find(|root| root.join("scripts/train_qwen.py").is_file())
        .ok_or_else(|| anyhow!("Cannot find scripts/train_qwen.py; set RGPU_ROOT to the checkout"))
}

fn add_training_arguments(command: &mut Command, epochs: usize, lora_r: usize, resume: bool) {
    command.args(["--epochs", &epochs.to_string(), "--lora-r", &lora_r.to_string(), "--export-gguf"]);
    if resume {
        command.arg("--resume");
    }
    command.args(["--model", "Qwen/Qwen2.5-1.5B-Instruct", "--data", "data/disaster_management_verified.jsonl", "--max-length", "384"]);
    let output = if cfg!(target_os = "windows") {
        std::path::PathBuf::from("C:/rgpu/models/checkpoints-disaster-1.5b")
    } else {
        std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/gpu-mesh/models/checkpoints-disaster-1.5b")
    };
    command.arg("--output").arg(output);
    command.env("PYTHONUNBUFFERED", "1");
}

#[derive(Deserialize)]
struct TrainingPeerConfig {
    ssh_target: String,
    root: String,
    platform: String,
    #[serde(default)]
    python: Option<String>,
}

fn training_peer_config(root: &std::path::Path) -> Result<TrainingPeerConfig> {
    if let Ok(target) = std::env::var("RGPU_TRAIN_PEER") {
        return Ok(TrainingPeerConfig {
            ssh_target: target,
            root: std::env::var("RGPU_TRAIN_PEER_ROOT")
                .map_err(|_| anyhow!("Set RGPU_TRAIN_PEER_ROOT for the peer checkout"))?,
            platform: std::env::var("RGPU_TRAIN_PEER_OS").unwrap_or_else(|_| "linux".to_string()),
            python: std::env::var("RGPU_TRAIN_PEER_PYTHON").ok(),
        });
    }
    let config_path = root.join("training-peer.json");
    let config = std::fs::read(&config_path).map_err(|_| anyhow!(
        "Peer training requires training-peer.json (ssh_target, root, platform) or RGPU_TRAIN_PEER and RGPU_TRAIN_PEER_ROOT; it never runs locally as a peer"
    ))?;
    Ok(serde_json::from_slice(&config)?)
}

fn peer_training_command(peer: &TrainingPeerConfig, epochs: usize, lora_r: usize, resume: bool) -> Result<String> {
    if peer.ssh_target.is_empty() || peer.ssh_target.starts_with('-') || peer.ssh_target.chars().any(char::is_whitespace) {
        return Err(anyhow!("Invalid training peer SSH target"));
    }
    if peer.root.trim().is_empty() {
        return Err(anyhow!("Training peer checkout path must not be empty"));
    }
    let arguments = format!("--model Qwen/Qwen2.5-1.5B-Instruct --data data/disaster_management_verified.jsonl --max-length 384 --epochs {epochs} --lora-r {lora_r} --export-gguf{}", if resume { " --resume" } else { "" });
    match peer.platform.as_str() {
        "linux" => {
            let quote = |value: &str| format!("'{}'", value.replace('\'', "'\"'\"'"));
            let python = peer.python.as_deref().unwrap_or(".venv-training/bin/python");
            Ok(format!("cd {} && {} -u scripts/train_qwen.py {arguments} --output ~/.cache/gpu-mesh/models/checkpoints-disaster-1.5b --llama-cpp-dir tools/llama.cpp", quote(&peer.root), quote(python)))
        }
        "windows" => {
            let python = peer.python.as_deref().unwrap_or(r".venv-training\Scripts\python.exe");
            if [&peer.root, python].iter().any(|value| value.chars().any(|c| "\"&|<>^%!\r\n".contains(c))) {
                return Err(anyhow!("Unsupported shell characters in Windows training peer paths"));
            }
            Ok(format!(r#"cd /d "{}" && "{}" -u scripts/train_qwen.py {arguments} --output C:/rgpu/models/checkpoints-disaster-1.5b --llama-cpp-dir tools/llama.cpp"#, peer.root, python))
        }
        _ => Err(anyhow!("Training peer platform must be linux or windows")),
    }
}

pub struct TrainingManager;

impl TrainingManager {
    /// Launches the Disaster Management Qwen 2.5 training process on either
    /// the local node or remote peer, streaming live lines + parsed telemetry.
    pub fn spawn_trainer(
        target: TrainingTarget,
        epochs: usize,
        lora_r: usize,
        resume: bool,
        state_cb: Arc<Mutex<crate::AppState>>,
    ) -> Result<tokio::sync::oneshot::Sender<()>> {
        let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel::<()>();

        let root = training_project_root()?;
        let (mut cmd, target_desc) = match target {
            TrainingTarget::Local => {
                let python = std::env::var_os("RGPU_TRAIN_PYTHON")
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(|| root.join(if cfg!(target_os = "windows") {
                        ".venv-training/Scripts/python.exe"
                    } else {
                        ".venv-training/bin/python"
                    }));
                if !python.is_file() {
                    return Err(anyhow!("Training Python not found at {}. Install scripts/requirements-training.txt into .venv-training or set RGPU_TRAIN_PYTHON.", python.display()));
                }
                let mut command = Command::new(python);
                command.current_dir(&root);
                command.arg(root.join("scripts/train_qwen.py"));
                add_training_arguments(&mut command, epochs, lora_r, resume);
                let converter_root = std::env::var_os("RGPU_LLAMA_CPP_DIR")
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(|| root.join("tools/llama.cpp"));
                command.arg("--llama-cpp-dir").arg(converter_root);
                (command, "Local")
            }
            TrainingTarget::Peer => {
                let peer = training_peer_config(&root)?;
                let mut command = Command::new("ssh");
                command.args(["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]);
                command.arg(&peer.ssh_target);
                command.arg(peer_training_command(&peer, epochs, lora_r, resume)?);
                (command, "Peer (SSH)")
            }
        };

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child: Child = cmd.spawn().map_err(|e| anyhow!("Failed to spawn training process on {target_desc}: {e}"))?;
        let child_pid = child.id();

        {
            let mut s = state_cb.lock();
            s.train_running = true;
            s.train_telemetry = None;
            s.train_logs.push(format!(
                "[TRAINER] Launched training process (PID {}) on {}...",
                child_pid, target_desc
            ));
        }

        let stdout = child.stdout.take().ok_or_else(|| anyhow!("Failed to capture child stdout"))?;
        let stderr = child.stderr.take();

        let state_stdout = Arc::clone(&state_cb);

        // Thread to read stdout, strip ANSI codes, and parse telemetry
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                if let Ok(l) = line {
                    let cleaned = clean_log_line(&l);
                    if cleaned.trim().is_empty() {
                        continue;
                    }
                    if cleaned.starts_with("[TELEMETRY] ") {
                        let json_part = &cleaned["[TELEMETRY] ".len()..];
                        if let Ok(telem) = serde_json::from_str::<TrainTelemetry>(json_part) {
                            let mut s = state_stdout.lock();
                            s.train_telemetry = Some(telem);
                        }
                    } else {
                        let mut s = state_stdout.lock();
                        s.train_logs.push(cleaned);
                        if s.train_logs.len() > 800 {
                            s.train_logs.drain(0..100);
                        }
                    }
                }
            }
        });

        // Thread to read stderr
        if let Some(err_stream) = stderr {
            let state_stderr = Arc::clone(&state_cb);
            std::thread::spawn(move || {
                let reader = BufReader::new(err_stream);
                for line in reader.lines() {
                    if let Ok(l) = line {
                        let cleaned = clean_log_line(&l);
                        if !cleaned.trim().is_empty() {
                            let mut s = state_stderr.lock();
                            s.train_logs.push(format!("[STDERR] {}", cleaned));
                            if s.train_logs.len() > 800 {
                                s.train_logs.drain(0..100);
                            }
                        }
                    }
                }
            });
        }

        // Supervisor thread to watch for completion or stop signal
        let state_sup = Arc::clone(&state_cb);
        std::thread::spawn(move || {
            loop {
                // Check if user requested stop
                if stop_rx.try_recv().is_ok() {
                    let _ = child.kill();
                    let _ = child.wait();
                    let mut s = state_sup.lock();
                    s.train_running = false;
                    s.train_logs.push("[TRAINER] Training cancelled by user.".to_string());
                    break;
                }

                // Check if process finished
                match child.try_wait() {
                    Ok(Some(status)) => {
                        let mut s = state_sup.lock();
                        s.train_running = false;
                        s.train_logs.push(format!(
                            "[TRAINER] Training process finished with exit status: {}",
                            status
                        ));
                        break;
                    }
                    Ok(None) => {
                        std::thread::sleep(std::time::Duration::from_millis(200));
                    }
                    Err(e) => {
                        let mut s = state_sup.lock();
                        s.train_running = false;
                        s.train_logs.push(format!("[TRAINER] Error polling child process: {e}"));
                        break;
                    }
                }
            }
        });

        Ok(stop_tx)
    }
}

#[cfg(test)]
mod training_command_tests {
    use super::*;

    #[test]
    fn peer_training_uses_remote_checkout_and_real_export_resume() {
        let config = TrainingPeerConfig {
            ssh_target: "trainer@peer".to_string(),
            root: "/home/trainer/My Project".to_string(),
            platform: "linux".to_string(),
            python: None,
        };
        let command = peer_training_command(&config, 2, 8, true).unwrap();
        assert!(command.starts_with("cd '/home/trainer/My Project' && '.venv-training/bin/python'"));
        assert!(command.contains("--epochs 2 --lora-r 8 --export-gguf --resume"));
        assert!(!command.contains("windows"));
    }

    #[test]
    fn peer_training_rejects_ssh_option_targets_and_windows_shell_injection() {
        let mut config = TrainingPeerConfig {
            ssh_target: "-oProxyCommand=command".to_string(),
            root: "C:\\rgpu".to_string(),
            platform: "windows".to_string(),
            python: None,
        };
        assert!(peer_training_command(&config, 2, 8, false).is_err());
        config.ssh_target = "trainer@peer".to_string();
        config.root = "C:\\rgpu & echo bad".to_string();
        assert!(peer_training_command(&config, 2, 8, false).is_err());
    }
}
