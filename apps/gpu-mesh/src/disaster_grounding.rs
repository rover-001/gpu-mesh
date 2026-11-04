//! Source-backed offline answers for the small disaster assistant. Responses are
//! explicitly labelled reference summaries, never presented as learned knowledge.
use serde_json::{json, Value};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::oneshot,
};

const REFERENCES: &str = include_str!("../../../data/disaster_references.json");
const MAX_HEADER: usize = 65536;
const MAX_BODY: usize = 2 * 1024 * 1024;

pub fn applies_to(model: &str) -> bool {
    std::path::Path::new(model)
        .file_name()
        .and_then(|p| p.to_str())
        == Some("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
}

fn reference_answer(request: &Value) -> Option<String> {
    if request["tools"]
        .as_array()
        .map_or(false, |tools| !tools.is_empty())
    {
        return None;
    }
    let message = request["messages"]
        .as_array()?
        .iter()
        .rev()
        .find(|message| message["role"] == "user")?;
    let question = if let Some(text) = message["content"].as_str() {
        text.to_owned()
    } else {
        message["content"]
            .as_array()?
            .iter()
            .filter_map(|part| part["text"].as_str())
            .collect::<Vec<_>>()
            .join(" ")
    };
    let question = question.to_lowercase();
    let words: Vec<_> = question
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();
    let catalog: Value = serde_json::from_str(REFERENCES).ok()?;
    let mut matches = Vec::new();
    for entry in catalog["entries"].as_array()? {
        let relevant = entry["terms"]
            .as_array()?
            .iter()
            .filter_map(Value::as_str)
            .any(|term| {
                if term.contains(' ') {
                    question.contains(term)
                } else {
                    words.contains(&term)
                }
            });
        if relevant {
            matches.push(format!(
                "{}\n\nSource: {}",
                entry["text"].as_str()?,
                entry["source"].as_str()?
            ));
        }
    }
    if matches.is_empty() {
        let disaster = [
            "disaster",
            "tsunami",
            "earthquake",
            "flood",
            "hurricane",
            "cyclone",
            "wildfire",
            "tornado",
            "hazmat",
            "triage",
            "evacuation",
            "emergency",
        ]
        .iter()
        .any(|term| words.contains(term));
        return disaster.then(|| "Offline reference: the bundled sources do not cover this question sufficiently. I cannot provide a grounded answer or invent an operational plan. Consult local emergency authorities and the relevant official response guidance.".to_string());
    }
    Some(format!("Offline reference summary\n\n{}\n\nThese are general source summaries. They do not establish a complete incident plan, local evacuation route, triage protocol, or site-specific hazard calculation. Follow local emergency authorities for the current situation.", matches.join("\n\n")))
}

fn completion(request: &Value, answer: &str) -> (String, String) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let id = format!("chatcmpl-rgpu-{}", now.as_nanos());
    let model = request["model"].as_str().unwrap_or("qwen");
    let created = now.as_secs();
    if request["stream"] == true {
        let chunk = json!({"id":id,"object":"chat.completion.chunk","created":created,"model":model,
            "rgpu_response_mode":"offline_reference", "choices":[{"index":0,"delta":{"role":"assistant","content":answer},"finish_reason":null}]});
        let end = json!({"id":id,"object":"chat.completion.chunk","created":created,"model":model,
            "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]});
        (
            "text/event-stream".into(),
            format!("data: {chunk}\n\ndata: {end}\n\ndata: [DONE]\n\n"),
        )
    } else {
        ("application/json".into(), json!({"id":id,"object":"chat.completion","created":created,"model":model,
            "rgpu_response_mode":"offline_reference", "choices":[{"index":0,"message":{"role":"assistant","content":answer},"finish_reason":"stop"}]}).to_string())
    }
}

async fn connection(mut client: TcpStream, target: String, enabled: bool) -> std::io::Result<()> {
    let mut bytes = Vec::new();
    let header_end;
    loop {
        if let Some(index) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            header_end = index + 4;
            break;
        }
        if bytes.len() >= MAX_HEADER {
            return Ok(());
        }
        let mut buffer = [0; 4096];
        let count = client.read(&mut buffer).await?;
        if count == 0 {
            return Ok(());
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    let header = String::from_utf8_lossy(&bytes[..header_end]).into_owned();
    let mut lines = header.split("\r\n");
    let first = lines.next().unwrap_or("");
    let chat = first.starts_with("POST /v1/chat/completions ")
        || first.starts_with("POST /chat/completions ");
    let mut length = None;
    let mut chunked = false;
    let mut forward = format!("{first}\r\n");
    for line in lines.filter(|line| !line.is_empty()) {
        if let Some((key, value)) = line.split_once(':') {
            if key.eq_ignore_ascii_case("content-length") {
                length = value.trim().parse::<usize>().ok();
            }
            if key.eq_ignore_ascii_case("transfer-encoding") {
                chunked = true;
            }
            if key.eq_ignore_ascii_case("connection") {
                continue;
            }
        }
        forward.push_str(line);
        forward.push_str("\r\n");
    }
    // Close after each response so every request on a browser connection is routed.
    forward.push_str("Connection: close\r\n\r\n");
    if header
        .lines()
        .any(|line| line.to_ascii_lowercase().starts_with("upgrade:"))
    {
        forward = header.clone();
    }
    if enabled && chat && !chunked && length.map_or(false, |n| n <= MAX_BODY) {
        let size = length.unwrap();
        while bytes.len() < header_end + size {
            let mut buffer = [0; 8192];
            let count = client.read(&mut buffer).await?;
            if count == 0 {
                return Ok(());
            }
            bytes.extend_from_slice(&buffer[..count]);
        }
        if let Ok(request) = serde_json::from_slice::<Value>(&bytes[header_end..header_end + size])
        {
            if let Some(answer) = reference_answer(&request) {
                let (kind, body) = completion(&request, &answer);
                let response = format!("HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                client.write_all(response.as_bytes()).await?;
                return Ok(());
            }
        }
    }
    let mut backend = TcpStream::connect(&target).await?;
    backend.write_all(forward.as_bytes()).await?;
    backend.write_all(&bytes[header_end..]).await?;
    tokio::io::copy_bidirectional(&mut client, &mut backend).await?;
    Ok(())
}

pub fn start_proxy(target: String, enabled: bool) -> Result<oneshot::Sender<()>, String> {
    let socket = std::net::TcpListener::bind("127.0.0.1:8080")
        .map_err(|e| format!("Web endpoint :8080 unavailable: {e}"))?;
    socket.set_nonblocking(true).map_err(|e| e.to_string())?;
    let (stop, mut cancelled) = oneshot::channel();
    crate::tokio_rt().spawn(async move {
        let Ok(listener) = TcpListener::from_std(socket) else {
            return;
        };
        loop {
            tokio::select! {
                _ = &mut cancelled => break,
                result = listener.accept() => {
                    let Ok((client, _)) = result else { continue };
                    let target = target.clone();
                    tokio::spawn(async move {
                        let _ = connection(client, target, enabled).await;
                    });
                }
            }
        }
    });
    Ok(stop)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn domain_references_and_abstention_are_distinct_from_model_chat() {
        let ask = |q| reference_answer(&json!({"messages":[{"role":"user","content":q}]}));
        assert!(ask("What does ICS mean?")
            .unwrap()
            .contains("Incident Command System"));
        let quake = ask("Earthquake offshore, incident command protocols for first hour?").unwrap();
        assert!(quake.contains("do not specify a complete first-hour"));
        assert!(!quake.contains("Commander:"));
        assert!(ask("Give a tornado-specific incident plan")
            .unwrap()
            .contains("do not cover"));
        assert!(ask("Write a Python factorial function").is_none());
        assert!(ask("Explain physics").is_none());
    }
    #[test]
    fn openai_stream_includes_content_final_chunk_and_terminator() {
        let (kind, body) = completion(&json!({"stream":true,"model":"test"}), "source answer");
        assert_eq!(kind, "text/event-stream");
        assert!(body.contains("offline_reference"));
        assert!(body.contains("source answer"));
        assert!(body.contains("\"finish_reason\":\"stop\""));
        assert!(body.ends_with("data: [DONE]\n\n"));
    }
    #[test]
    fn json_completion_and_model_scope_preserve_api_contract() {
        assert!(applies_to(
            "C:/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
        ));
        assert!(!applies_to("/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"));
        let (kind, body) = completion(&json!({"model":"qwen"}), "reference text");
        assert_eq!(kind, "application/json");
        let response: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(response["choices"][0]["message"]["role"], "assistant");
        assert_eq!(
            response["choices"][0]["message"]["content"],
            "reference text"
        );
        assert_eq!(response["rgpu_response_mode"], "offline_reference");
    }

    #[tokio::test]
    async fn http_stream_answers_without_contacting_backend() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            connection(socket, "127.0.0.1:1".into(), true)
                .await
                .unwrap();
        });
        let body = json!({"stream":true,"messages":[{"role":"user","content":"What does ICS stand for?"}]}).to_string();
        let mut client = TcpStream::connect(addr).await.unwrap();
        client.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",body.len()).as_bytes()).await.unwrap();
        client.write_all(body.as_bytes()).await.unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).await.unwrap();
        task.await.unwrap();
        assert!(response.starts_with("HTTP/1.1 200"));
        assert!(response.contains("Incident Command System"));
        assert!(response.ends_with("data: [DONE]\n\n"));
    }

    #[tokio::test]
    async fn generic_chat_is_forwarded_to_backend() {
        let backend = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let target = backend.local_addr().unwrap().to_string();
        let body =
            json!({"messages":[{"role":"user","content":"Write a Python factorial function"}]})
                .to_string();
        let expected = body.clone();
        let task = tokio::spawn(async move {
            let (mut socket, _) = backend.accept().await.unwrap();
            let mut bytes = Vec::new();
            loop {
                let mut buffer = [0; 4096];
                let count = socket.read(&mut buffer).await.unwrap();
                bytes.extend_from_slice(&buffer[..count]);
                if String::from_utf8_lossy(&bytes).ends_with(&expected) {
                    break;
                }
                assert_ne!(count, 0);
            }
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nMODEL",
                )
                .await
                .unwrap();
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let proxy = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            connection(socket, target, true).await.unwrap();
        });
        let mut client = TcpStream::connect(addr).await.unwrap();
        client.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}",body.len()).as_bytes()).await.unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).await.unwrap();
        assert!(response.ends_with("MODEL"));
        drop(client);
        task.await.unwrap();
        proxy.await.unwrap();
    }
}
