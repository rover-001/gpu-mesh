use std::io::{self, Cursor};

pub fn compress(data: &[u8]) -> io::Result<Vec<u8>> {
    zstd::encode_all(Cursor::new(data), 0)
}

pub fn compress_with_level(data: &[u8], level: i32) -> io::Result<Vec<u8>> {
    zstd::encode_all(Cursor::new(data), level)
}

pub fn decompress(data: &[u8], expected_len: usize) -> io::Result<Vec<u8>> {
    let mut decompressed = Vec::with_capacity(expected_len);
    let mut cursor = Cursor::new(data);
    zstd::stream::copy_decode(&mut cursor, &mut decompressed)?;
    Ok(decompressed)
}
