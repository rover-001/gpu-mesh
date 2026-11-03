use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tokio::net::TcpListener;
use vkremote_server::{run_server_with_config, ServerConfig};

fn ensure_artifacts(workspace_dir: &Path) -> PathBuf {
    let target_dir = workspace_dir.join("target");
    let json_path = target_dir.join("vkremote_icd.json");

    let cdylib_name = if cfg!(target_os = "windows") {
        "vkremote_icd.dll"
    } else if cfg!(target_os = "macos") {
        "libvkremote_icd.dylib"
    } else {
        "libvkremote_icd.so"
    };

    let lib_path = target_dir.join("debug").join(cdylib_name);
    if !lib_path.exists() {
        let status = Command::new("cargo")
            .arg("build")
            .arg("-p")
            .arg("vkremote-icd")
            .arg("-p")
            .arg("vkremote-probe")
            .arg("-p")
            .arg("vecadd")
            .current_dir(workspace_dir)
            .status()
            .expect("Failed to build artifacts");
        assert!(status.success(), "Building required test artifacts failed");
    }

    if !json_path.exists() {
        let abs_lib = fs::canonicalize(&lib_path).unwrap();
        let path_str = abs_lib.to_string_lossy().replace('\\', "\\\\");
        let json_content = format!(
            r#"{{
    "file_format_version": "1.0.0",
    "ICD": {{
        "library_path": "{}",
        "api_version": "1.3.0"
    }}
}}
"#,
            path_str
        );
        fs::write(&json_path, json_content).expect("Failed to write ICD manifest");
    }

    fs::canonicalize(&json_path).expect("Failed to canonicalize manifest path")
}

#[test]
fn test_end_to_end_loopback() {
    let _ = tracing_subscriber::fmt().try_init();
    let (addr_tx, addr_rx) = std::sync::mpsc::channel();
    let token = "integration-token".to_string();
    let token_clone = token.clone();

    let _server_thread = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap().to_string();
            addr_tx.send(addr).unwrap();
            let config = ServerConfig {
                token: token_clone,
                gpu_index: None,
                force_mock: false,
            };
            let _ = run_server_with_config(listener, config).await;
        });
    });

    let addr = addr_rx.recv().unwrap();

    // Ensure server is accepting connections
    let start = std::time::Instant::now();
    while start.elapsed() < std::time::Duration::from_secs(5) {
        if std::net::TcpStream::connect(&addr).is_ok() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    let workspace_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    let manifest_path = ensure_artifacts(&workspace_dir);
    let probe_bin = workspace_dir.join("target").join("debug").join("probe");
    let vecadd_bin = workspace_dir.join("target").join("debug").join("vecadd");

    // 1. Run probe app through ICD over TCP loopback
    let probe_output = Command::new(&probe_bin)
        .env("VK_DRIVER_FILES", &manifest_path)
        .env("VKREMOTE_SERVER", &addr)
        .env("VKREMOTE_TOKEN", &token)
        .output()
        .expect("Failed to execute probe");

    assert!(
        probe_output.status.success(),
        "probe failed: stdout: {}, stderr: {}",
        String::from_utf8_lossy(&probe_output.stdout),
        String::from_utf8_lossy(&probe_output.stderr)
    );
    let probe_stdout = String::from_utf8_lossy(&probe_output.stdout);
    assert!(
        probe_stdout.contains("(Remote)"),
        "probe output must list (Remote) GPU: {}",
        probe_stdout
    );

    // 2. Run vecadd compute demo through ICD over TCP loopback
    let vecadd_output = Command::new(&vecadd_bin)
        .env("VK_DRIVER_FILES", &manifest_path)
        .env("VKREMOTE_SERVER", &addr)
        .env("VKREMOTE_TOKEN", &token)
        .env("RUST_LOG", "trace")
        .output()
        .expect("Failed to execute vecadd");

    assert!(
        vecadd_output.status.success(),
        "vecadd failed: stdout: {}, stderr: {}",
        String::from_utf8_lossy(&vecadd_output.stdout),
        String::from_utf8_lossy(&vecadd_output.stderr)
    );
    let vecadd_stdout = String::from_utf8_lossy(&vecadd_output.stdout);
    assert!(
        vecadd_stdout.contains("[VECADD] SUCCESS"),
        "vecadd must output SUCCESS: {}",
        vecadd_stdout
    );
}
