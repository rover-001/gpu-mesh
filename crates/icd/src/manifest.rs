use std::path::Path;

/// Generate the Khronos Vulkan ICD manifest JSON for Linux (`.so`) or Windows (`.dll`).
pub fn generate_icd_json(lib_path: &Path) -> String {
    let mut path_str = lib_path.to_string_lossy().to_string();
    // Strip Windows canonicalize extended prefix \\?\ if present
    if let Some(stripped) = path_str.strip_prefix(r"\\?\") {
        path_str = stripped.to_string();
    }
    // Forward slashes work cleanly across all Vulkan loaders on both Linux and Windows
    let clean_path = path_str.replace('\\', "/");

    format!(
        r#"{{
    "file_format_version": "1.0.0",
    "ICD": {{
        "library_path": "{}",
        "api_version": "1.3.0"
    }}
}}"#,
        clean_path
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_generate_icd_json_linux() {
        let p = PathBuf::from("/usr/lib/libvkremote_icd.so");
        let json = generate_icd_json(&p);
        assert!(json.contains(r#""file_format_version": "1.0.0""#));
        assert!(json.contains(r#""library_path": "/usr/lib/libvkremote_icd.so""#));
        assert!(json.contains(r#""api_version": "1.3.0""#));
    }

    #[test]
    fn test_generate_icd_json_windows() {
        let p = PathBuf::from(r"C:\Windows\System32\vkremote_icd.dll");
        let json = generate_icd_json(&p);
        assert!(json.contains(r#""file_format_version": "1.0.0""#));
        assert!(json.contains(r#"C:/Windows/System32/vkremote_icd.dll"#));
        assert!(json.contains(r#""api_version": "1.3.0""#));
    }
}
