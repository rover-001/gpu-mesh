fn main() {
    // The ICD exports `vk*` symbols with the same names as libvulkan's. In apps that
    // link the loader globally (e.g. Blender), ELF interposition would bind the ICD's
    // own internal references to the loader's trampolines, so the function pointers we
    // hand back to the loader point at the loader itself (infinite recursion / hang).
    // -Bsymbolic binds the ICD's references to its own definitions.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" || target_os == "android" {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-Bsymbolic");
    }
}
