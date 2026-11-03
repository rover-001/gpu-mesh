//! Shared helpers for the opt-in frame profiler (`VKREMOTE_PERF=1`).

use std::fmt::{self, Write};

/// Whether profiling output is enabled for this process (`VKREMOTE_PERF` set and not "0").
pub fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("VKREMOTE_PERF").map_or(false, |v| v != "0"))
}

/// Writer that keeps only the leading identifier and aborts formatting at the
/// first non-identifier character, so a `Debug` impl over megabytes of payload
/// costs a few bytes of work.
struct NameOnly {
    name: String,
}

impl Write for NameOnly {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for c in s.chars() {
            if c.is_ascii_alphanumeric() || c == '_' {
                self.name.push(c);
            } else {
                return Err(fmt::Error);
            }
        }
        Ok(())
    }
}

/// Enum variant name of `value` (e.g. `"QueueSubmit"`), derived from its `Debug` output.
pub fn variant_name<T: fmt::Debug>(value: &T) -> String {
    let mut w = NameOnly {
        name: String::new(),
    };
    let _ = write!(w, "{:?}", value);
    w.name
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    #[allow(dead_code)]
    enum Sample {
        Unit,
        Tuple(Vec<u8>),
        Struct { data: Vec<u8> },
    }

    #[test]
    fn variant_name_stops_at_payload() {
        assert_eq!(variant_name(&Sample::Unit), "Unit");
        assert_eq!(variant_name(&Sample::Tuple(vec![1; 1 << 20])), "Tuple");
        assert_eq!(variant_name(&Sample::Struct { data: vec![] }), "Struct");
    }
}
