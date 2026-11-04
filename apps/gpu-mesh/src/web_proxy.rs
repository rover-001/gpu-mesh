//! Transparent local/peer inference transport. All answers come from llama-server.
use tokio::{net::{TcpListener, TcpStream}, sync::oneshot};

pub fn start(target: String) -> Result<oneshot::Sender<()>, String> {
    let socket = std::net::TcpListener::bind("127.0.0.1:8080")
        .map_err(|e| format!("Web endpoint :8080 unavailable: {e}"))?;
    socket.set_nonblocking(true).map_err(|e| e.to_string())?;
    let (stop, mut cancelled) = oneshot::channel();
    crate::tokio_rt().spawn(async move {
        let Ok(listener) = TcpListener::from_std(socket) else { return };
        loop {
            tokio::select! {
                _ = &mut cancelled => break,
                result = listener.accept() => {
                    let Ok((mut client, _)) = result else { continue };
                    let target = target.clone();
                    tokio::spawn(async move {
                        if let Ok(mut backend) = TcpStream::connect(&target).await {
                            let _ = tokio::io::copy_bidirectional(&mut client, &mut backend).await;
                        }
                    });
                }
            }
        }
    });
    Ok(stop)
}
