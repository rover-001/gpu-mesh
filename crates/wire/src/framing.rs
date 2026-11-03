use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::io::{self, Read, Write};
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub const MAGIC: &[u8; 8] = b"VKREMOTE";
pub const CURRENT_PROTOCOL_VERSION: u32 = 3;
/// Largest accepted wire message. Host-visible Vulkan allocations are returned as a
/// single `MemoryData` response; Minecraft produces responses larger than 64 MiB.
/// Keep this bounded while allowing the observed 134,234,047-byte readback.
pub const DEFAULT_MAX_FRAME_SIZE: u32 = 256 * 1024 * 1024; // 256 MiB

/// Flag bit in the length prefix indicating payload is zstd-compressed.
pub const COMPRESSION_FLAG: u32 = 0x8000_0000;
/// Minimum message size before considering zstd compression.
pub const COMPRESSION_THRESHOLD: usize = 1024;

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] postcard::Error),

    #[error("Invalid magic header: expected {expected:?}, got {actual:?}")]
    InvalidMagic {
        expected: &'static [u8; 8],
        actual: [u8; 8],
    },

    #[error("Frame too large: size {size} exceeds max allowed {max}")]
    FrameTooLarge { size: u32, max: u32 },

    #[error("Handshake rejected: {0}")]
    HandshakeRejected(String),

    #[error("Protocol version mismatch: client {client_version}, server {server_version}")]
    VersionMismatch {
        client_version: u32,
        server_version: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientHandshake {
    pub version: u32,
    pub token: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerHandshake {
    pub version: u32,
    pub accepted: bool,
    pub message: String,
}

/// Send a length-prefixed postcard message, transparently zstd-compressed if large.
pub async fn write_msg<W, T>(writer: &mut W, msg: &T) -> Result<(), ProtocolError>
where
    W: AsyncWrite + Unpin,
    T: Serialize,
{
    let encoded = postcard::to_allocvec(msg)?;
    let (payload, is_compressed) = if encoded.len() > COMPRESSION_THRESHOLD {
        match zstd::encode_all(&encoded[..], 1) {
            Ok(compressed) if compressed.len() < encoded.len() => (compressed, true),
            _ => (encoded, false),
        }
    } else {
        (encoded, false)
    };

    let len = payload.len() as u32;
    let header = if is_compressed {
        len | COMPRESSION_FLAG
    } else {
        len
    };

    writer.write_all(&header.to_le_bytes()).await?;
    writer.write_all(&payload).await?;
    writer.flush().await?;
    Ok(())
}

/// Read a length-prefixed postcard message with max size limit, transparently decompressing if flagged.
pub async fn read_msg<R, T>(reader: &mut R, max_frame_size: u32) -> Result<T, ProtocolError>
where
    R: AsyncRead + Unpin,
    T: DeserializeOwned,
{
    let mut len_bytes = [0u8; 4];
    reader.read_exact(&mut len_bytes).await?;
    let header = u32::from_le_bytes(len_bytes);
    let is_compressed = (header & COMPRESSION_FLAG) != 0;
    let len = header & !COMPRESSION_FLAG;

    if len > max_frame_size {
        return Err(ProtocolError::FrameTooLarge {
            size: len,
            max: max_frame_size,
        });
    }

    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf).await?;

    let decoded_bytes = if is_compressed {
        zstd::decode_all(&buf[..]).map_err(ProtocolError::Io)?
    } else {
        buf
    };

    let msg: T = match postcard::from_bytes(&decoded_bytes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "[wire async] postcard deserialization failed on {} bytes (compressed: {}): {:?}. Raw bytes: {:?}",
                len,
                is_compressed,
                e,
                &decoded_bytes[..decoded_bytes.len().min(64)]
            );
            return Err(e.into());
        }
    };
    Ok(msg)
}

/// Send a length-prefixed postcard message synchronously, transparently zstd-compressed if large.
/// Returns the total bytes sent (including the 4-byte length prefix).
pub fn write_msg_sync<W, T>(writer: &mut W, msg: &T) -> Result<usize, ProtocolError>
where
    W: Write,
    T: Serialize,
{
    let encoded = postcard::to_allocvec(msg)?;
    let (payload, is_compressed) = if encoded.len() > COMPRESSION_THRESHOLD {
        match zstd::encode_all(&encoded[..], 1) {
            Ok(compressed) if compressed.len() < encoded.len() => (compressed, true),
            _ => (encoded, false),
        }
    } else {
        (encoded, false)
    };

    let len = payload.len() as u32;
    let header = if is_compressed {
        len | COMPRESSION_FLAG
    } else {
        len
    };

    writer.write_all(&header.to_le_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()?;
    Ok(len as usize + 4)
}

/// Read a length-prefixed postcard message with max size limit synchronously.
/// Returns the decoded message and total wire bytes read (including the 4-byte length prefix).
pub fn read_msg_sync_with_len<R, T>(
    reader: &mut R,
    max_frame_size: u32,
) -> Result<(T, usize), ProtocolError>
where
    R: Read,
    T: DeserializeOwned,
{
    let mut len_bytes = [0u8; 4];
    reader.read_exact(&mut len_bytes)?;
    let header = u32::from_le_bytes(len_bytes);
    let is_compressed = (header & COMPRESSION_FLAG) != 0;
    let len = header & !COMPRESSION_FLAG;

    if len > max_frame_size {
        return Err(ProtocolError::FrameTooLarge {
            size: len,
            max: max_frame_size,
        });
    }

    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf)?;

    let decoded_bytes = if is_compressed {
        zstd::decode_all(&buf[..]).map_err(ProtocolError::Io)?
    } else {
        buf
    };

    match postcard::from_bytes(&decoded_bytes) {
        Ok(msg) => Ok((msg, len as usize + 4)),
        Err(e) => {
            eprintln!(
                "[wire] postcard deserialization failed on {} bytes (compressed: {}): {:?}. Raw bytes: {:?}",
                len,
                is_compressed,
                e,
                &decoded_bytes[..decoded_bytes.len().min(64)]
            );
            Err(e.into())
        }
    }
}

/// Read a length-prefixed postcard message with max size limit synchronously
pub fn read_msg_sync<R, T>(reader: &mut R, max_frame_size: u32) -> Result<T, ProtocolError>
where
    R: Read,
    T: DeserializeOwned,
{
    read_msg_sync_with_len(reader, max_frame_size).map(|(msg, _)| msg)
}

/// Perform client-side handshake synchronously
pub fn client_handshake_sync<S>(
    stream: &mut S,
    token: impl Into<String>,
) -> Result<ServerHandshake, ProtocolError>
where
    S: Read + Write,
{
    stream.write_all(MAGIC)?;
    stream.flush()?;

    let mut server_magic = [0u8; 8];
    stream.read_exact(&mut server_magic)?;
    if &server_magic != MAGIC {
        return Err(ProtocolError::InvalidMagic {
            expected: MAGIC,
            actual: server_magic,
        });
    }

    let req = ClientHandshake {
        version: CURRENT_PROTOCOL_VERSION,
        token: token.into(),
    };
    write_msg_sync(stream, &req)?;

    let resp: ServerHandshake = read_msg_sync(stream, DEFAULT_MAX_FRAME_SIZE)?;
    if resp.version != CURRENT_PROTOCOL_VERSION {
        return Err(ProtocolError::VersionMismatch {
            client_version: CURRENT_PROTOCOL_VERSION,
            server_version: resp.version,
        });
    }
    if !resp.accepted {
        return Err(ProtocolError::HandshakeRejected(resp.message));
    }

    Ok(resp)
}

/// Perform client-side handshake
pub async fn client_handshake<S>(
    stream: &mut S,
    token: impl Into<String>,
) -> Result<ServerHandshake, ProtocolError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    // 1. Write Magic
    stream.write_all(MAGIC).await?;
    stream.flush().await?;

    // 2. Read Server Magic
    let mut server_magic = [0u8; 8];
    stream.read_exact(&mut server_magic).await?;
    if &server_magic != MAGIC {
        return Err(ProtocolError::InvalidMagic {
            expected: MAGIC,
            actual: server_magic,
        });
    }

    // 3. Write ClientHandshake
    let req = ClientHandshake {
        version: CURRENT_PROTOCOL_VERSION,
        token: token.into(),
    };
    write_msg(stream, &req).await?;

    // 4. Read ServerHandshake
    let resp: ServerHandshake = read_msg(stream, DEFAULT_MAX_FRAME_SIZE).await?;
    if resp.version != CURRENT_PROTOCOL_VERSION {
        return Err(ProtocolError::VersionMismatch {
            client_version: CURRENT_PROTOCOL_VERSION,
            server_version: resp.version,
        });
    }
    if !resp.accepted {
        return Err(ProtocolError::HandshakeRejected(resp.message));
    }

    Ok(resp)
}

/// Perform server-side handshake
pub async fn server_handshake<S, F>(
    stream: &mut S,
    authenticator: F,
) -> Result<ClientHandshake, ProtocolError>
where
    S: AsyncRead + AsyncWrite + Unpin,
    F: FnOnce(&ClientHandshake) -> (bool, String),
{
    // 1. Read Client Magic
    let mut client_magic = [0u8; 8];
    stream.read_exact(&mut client_magic).await?;
    if &client_magic != MAGIC {
        return Err(ProtocolError::InvalidMagic {
            expected: MAGIC,
            actual: client_magic,
        });
    }

    // 2. Write Server Magic
    stream.write_all(MAGIC).await?;
    stream.flush().await?;

    // 3. Read ClientHandshake
    let client_req: ClientHandshake = read_msg(stream, DEFAULT_MAX_FRAME_SIZE).await?;

    // Check version
    if client_req.version != CURRENT_PROTOCOL_VERSION {
        let resp = ServerHandshake {
            version: CURRENT_PROTOCOL_VERSION,
            accepted: false,
            message: format!(
                "Version mismatch: client {}, server {}",
                client_req.version, CURRENT_PROTOCOL_VERSION
            ),
        };
        write_msg(stream, &resp).await?;
        return Err(ProtocolError::VersionMismatch {
            client_version: client_req.version,
            server_version: CURRENT_PROTOCOL_VERSION,
        });
    }

    // Check token authentication
    let (accepted, message) = authenticator(&client_req);
    let resp = ServerHandshake {
        version: CURRENT_PROTOCOL_VERSION,
        accepted,
        message,
    };
    write_msg(stream, &resp).await?;

    if !accepted {
        return Err(ProtocolError::HandshakeRejected(resp.message));
    }

    Ok(client_req)
}
