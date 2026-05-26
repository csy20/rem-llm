use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

use crate::{C_DIM, C_RESET, C_YELLOW, style};

#[derive(Debug, Deserialize)]
pub struct OllamaResponse { pub response: String }

#[derive(Debug, Deserialize)]
pub struct OllamaErrorResponse { pub error: String }

#[derive(Debug, Deserialize)]
pub struct TagsResponse { pub models: Vec<TagModel> }

#[derive(Debug, Deserialize)]
pub struct TagModel { pub name: String }

pub struct OllamaClient {
    pub client: Client,
    pub base_url: String,
    pub model: String,
    pub system_prompt: String,
}

impl OllamaClient {
    pub fn new(base_url: String, model: String, timeout_s: u64, system_prompt: String) -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(timeout_s))
                .build()
                .unwrap_or_else(|_| Client::new()),
            base_url,
            model,
            system_prompt,
        }
    }

    pub fn set_model(&mut self, model: String) { self.model = model; }

    pub async fn list_models(&self) -> Result<Vec<String>> {
        let url = api_url(&self.base_url, "tags");
        let resp = self.client.get(url).send().await?;
        if !resp.status().is_success() {
            return Err(anyhow!("Ollama unreachable at {}", self.base_url));
        }
        let parsed: TagsResponse = resp.json().await.context("invalid tags response")?;
        Ok(parsed.models.into_iter().map(|m| m.name).collect())
    }

    pub async fn healthcheck(&self) -> Result<()> {
        let models = self.list_models().await?;
        if models.is_empty() {
            return Err(anyhow!("Ollama reachable but no models are installed. Pull one with `ollama pull rem-coder:latest`"));
        }
        Ok(())
    }

    pub async fn complete_json(&self, user_prompt: &str) -> Result<crate::ModelReply> {
        let url = api_url(&self.base_url, "generate");
        let final_prompt = format!(
            "{}\n\nUser request:\n{}\n\nReturn JSON only.",
            self.system_prompt, user_prompt
        );
        let payload = json!({
            "model": self.model,
            "prompt": final_prompt,
            "stream": false,
            "options": {
                "num_predict": 512,
                "num_ctx": 2048,
                "num_thread": 4
            },
            "format": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "code": {"type": "string"},
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"}
                            },
                            "required": ["path", "content"]
                        }
                    },
                    "commands": {"type": "array", "items": {"type": "string"}},
                    "checks": {"type": "array", "items": {"type": "string"}},
                    "caution": {"type": "string"}
                },
                "required": ["explanation", "code", "commands", "checks", "caution"]
            }
        });

        let resp = self.client.post(&url).json(&payload).send().await
            .context("failed to call Ollama")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_else(|e| format!("(read error: {})", e));
            let err_msg = serde_json::from_str::<OllamaErrorResponse>(&body)
                .map(|v| v.error).unwrap_or_else(|_| body.clone());
            if status.as_u16() == 404 && err_msg.to_lowercase().contains("model") {
                return Err(anyhow!("Model '{}' not found. Pull it: `ollama pull {}`", self.model, self.model));
            }
            if status.as_u16() == 404 {
                return Err(anyhow!("Endpoint not found (404 at {}). Check --ollama-url", url));
            }
            return Err(anyhow!("Ollama failed: {} — {}", status, err_msg));
        }

        let raw: OllamaResponse = resp.json().await.context("invalid Ollama response")?;
        match serde_json::from_str::<crate::ModelReply>(raw.response.trim()) {
            Ok(parsed) => Ok(parsed),
            Err(e) => {
                eprintln!("  {} JSON parse: {} — falling back", style!(C_YELLOW, "!"), e);
                Ok(crate::ModelReply::fallback(raw.response.trim()))
            }
        }
    }

    pub async fn complete_chat_stream(
        &self,
        user_prompt: &str,
        system_prompt: &str,
        history: &str,
    ) -> Result<String> {
        let url = api_url(&self.base_url, "generate");
        let final_prompt = if history.is_empty() {
            format!("{}\n\nUser: {}\n\nREM:", system_prompt, user_prompt)
        } else {
            format!("{}\n\n{}User: {}\n\nREM:", system_prompt, history, user_prompt)
        };
        let payload = json!({
            "model": self.model,
            "prompt": final_prompt,
            "stream": true,
            "options": {
                "num_predict": 512,
                "num_ctx": 2048,
                "num_thread": 4
            }
        });
        let resp = self.client.post(&url).json(&payload).send().await
            .context("failed to call Ollama")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_else(|e| format!("(read error: {})", e));
            let err_msg = serde_json::from_str::<OllamaErrorResponse>(&body)
                .map(|v| v.error).unwrap_or_else(|_| body.clone());
            if status.as_u16() == 404 && err_msg.to_lowercase().contains("model") {
                return Err(anyhow!("Model '{}' not found. Pull it: `ollama pull {}`", self.model, self.model));
            }
            return Err(anyhow!("Ollama failed: {} — {}", status, err_msg));
        }
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_clone = cancelled.clone();
        let ctrlc_task = tokio::spawn(async move {
            let _ = tokio::signal::ctrl_c().await;
            cancelled_clone.store(true, Ordering::SeqCst);
        });

        let result = self.stream_response(resp, cancelled.clone(), ctrlc_task).await;
        if cancelled.load(Ordering::SeqCst) {
            println!();
        }
        result
    }

    async fn stream_response(
        &self,
        resp: reqwest::Response,
        cancelled: Arc<AtomicBool>,
        ctrlc_task: tokio::task::JoinHandle<()>,
    ) -> Result<String> {
        let mut stream = resp.bytes_stream();
        let mut full = String::new();
        let mut buf = String::new();
        let start = std::time::Instant::now();
        let mut token_count = 0u32;
        'stream_loop: while let Some(chunk) = stream.next().await {
            if cancelled.load(Ordering::SeqCst) {
                ctrlc_task.abort();
                break 'stream_loop;
            }
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    ctrlc_task.abort();
                    return Err(anyhow!("stream read error: {}", e));
                }
            };
            buf.push_str(&String::from_utf8_lossy(&chunk));
            if buf.len() > 32_000 {
                ctrlc_task.abort();
                return Err(anyhow!("response too large ({} bytes buffered)", buf.len()));
            }
            while let Some(pos) = buf.find('\n') {
                let line = buf[..pos].to_string();
                buf = buf[pos + 1..].to_string();
                let trimmed = line.trim();
                if trimmed.is_empty() { continue; }
                if let Ok(obj) = serde_json::from_str::<serde_json::Value>(trimmed) {
                    if let Some(token) = obj["response"].as_str() {
                        if io::stdout().write_all(token.as_bytes()).is_err() {
                            ctrlc_task.abort();
                            break 'stream_loop;
                        }
                        let _ = io::stdout().flush();
                        full.push_str(token);
                        token_count += 1;
                    }
                    if obj["done"].as_bool() == Some(true) {
                        if cancelled.load(Ordering::SeqCst) {
                            ctrlc_task.abort();
                            break 'stream_loop;
                        }
                        let elapsed = start.elapsed();
                        let tps = if elapsed.as_secs_f64() > 0.0 {
                            token_count as f64 / elapsed.as_secs_f64()
                        } else { 0.0 };
                        println!("\n  {} {:.0} tokens/s in {:.1}s",
                            style!(C_DIM, "\u{2502}"), tps, elapsed.as_secs_f64());
                        ctrlc_task.abort();
                        return Ok(full);
                    }
                }
            }
        }
        ctrlc_task.abort();
        if cancelled.load(Ordering::SeqCst) {
            println!("\n  {} stream cancelled — {} tokens received (Ctrl+C again to exit)",
                style!(C_YELLOW, "\u{2502}"), token_count);
        }
        println!();
        Ok(full)
    }
}

pub fn api_url(base_url: &str, endpoint: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let ep = endpoint.trim_start_matches('/');
    if base.ends_with("/api") { format!("{}/{}", base, ep) }
    else { format!("{}/api/{}", base, ep) }
}
