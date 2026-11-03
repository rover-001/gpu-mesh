// Integration test referencing the loopback end-to-end suite
// Primary execution target is registered under crates/server/tests/end_to_end_loopback.rs
// to automatically participate in `cargo test --workspace`.
#[path = "../crates/server/tests/end_to_end_loopback.rs"]
mod end_to_end_loopback;
