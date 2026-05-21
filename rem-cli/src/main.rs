use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use clap::{Args, Parser, Subcommand};
use regex::Regex;
use reqwest::Client;
use rustyline::DefaultEditor;
use serde::{Deserialize, Serialize};
use serde_json::json;
use futures_util::StreamExt;
use walkdir::WalkDir;

// ── ANSI styling ───────────────────────────────────────────────────────────

const C_RESET: &str = "\x1b[0m";
const C_BOLD: &str = "\x1b[1m";
const C_DIM: &str = "\x1b[2m";
const C_CYAN: &str = "\x1b[36m";
const C_GREEN: &str = "\x1b[32m";
const C_YELLOW: &str = "\x1b[33m";
const C_RED: &str = "\x1b[31m";
const C_MAGENTA: &str = "\x1b[35m";
const C_WHITE_B: &str = "\x1b[1;37m";

macro_rules! style {
    ($color:ident, $($arg:tt)*) => {
        format!("{}{}{}", $color, format!($($arg)*), C_RESET)
    };
}

// ── Config & Prompts ───────────────────────────────────────────────────────

const DEFAULT_SYSTEM_PROMPT: &str = r#"You are REM, a beginner-friendly coding assistant.
Focus on HTML, CSS, and safe terminal basics.

Rules:
- Always return strict JSON with keys: explanation, code, commands, checks, caution.
- Before generating code, ALWAYS ask the user for the file path where the code should be saved. Include this in your explanation.
- commands must be safe for beginners — never suggest sudo, rm -rf, or pipes to bash.
- If a command could be dangerous, flag it in the caution field.
- Keep explanations short and clear.
- Prefer complete, runnable examples.
"#;

const CHAT_SYSTEM_PROMPT: &str = r#"You are REM, a friendly and helpful coding assistant.
You speak conversationally and help beginners with:

- HTML and CSS for building web pages
- Safe terminal commands (never sudo, rm -rf, or dangerous pipes)
- General programming questions

You can search the web. When you need current information, tell the user to type /search <query> and you'll incorporate the results.

Guidelines:
- Be conversational — use markdown for formatting, code fences for code blocks.
- For code generation, always ask for the file path first.
- When the user wants a website, write complete, runnable HTML+CSS (and JS if needed).
- If you're unsure about something, let the user know and suggest a web search.
- Keep responses helpful but concise.
"#;

const BLOCKED_COMMAND_PATTERNS: [&str; 10] = [
    "rm -rf /", "rm -rf", "rm  -rf", "mkfs", "dd if=",
    ":(){:|:&};:", "shutdown", "reboot", "curl ", "sudo ",
];

// ── CLI definition ─────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "rem",
    version,
    about = "REM — Beginner-friendly coding assistant for HTML, CSS, and terminal basics",
    long_about = None,
)]
struct Cli {
    #[arg(long, global = true, help = "Ollama model name")]
    model: Option<String>,
    #[arg(long, global = true, help = "Ollama API URL")]
    ollama_url: Option<String>,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(about = "Ask REM a coding question")]
    Ask(AskArgs),
    #[command(about = "Explain a terminal command safely")]
    Explain(ExplainArgs),
    #[command(about = "Preview a patch for a file")]
    Patch(PatchArgs),
    #[command(about = "Start an interactive chat session")]
    Chat,
}

#[derive(Args, Debug)]
struct AskArgs {
    #[arg(help = "Your coding question")]
    prompt: String,
    #[arg(long, help = "Optional file for context")]
    file: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct ExplainArgs {
    #[arg(help = "Terminal command to explain")]
    command: String,
}

#[derive(Args, Debug)]
struct PatchArgs {
    #[arg(long, help = "Target file to patch")]
    file: PathBuf,
    #[arg(long, help = "Description of changes needed")]
    task: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AppConfig {
    model: String,
    ollama_url: String,
    timeout_s: u64,
    max_context_bytes: usize,
    prompts_dir: Option<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model: "rem-coder:latest".to_string(),
            ollama_url: "http://localhost:11434".to_string(),
            timeout_s: 120,
            max_context_bytes: 16_000,
            prompts_dir: None,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct PartialConfig {
    model: Option<String>,
    ollama_url: Option<String>,
    timeout_s: Option<u64>,
    max_context_bytes: Option<usize>,
    prompts_dir: Option<String>,
}

impl AppConfig {
    fn apply_partial(&mut self, part: PartialConfig) {
        if let Some(v) = part.model { self.model = v; }
        if let Some(v) = part.ollama_url { self.ollama_url = v; }
        if let Some(v) = part.timeout_s { self.timeout_s = v; }
        if let Some(v) = part.max_context_bytes { self.max_context_bytes = v; }
        if let Some(v) = part.prompts_dir { self.prompts_dir = Some(v); }
    }
}

// ── Model reply schema ─────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ModelReply {
    explanation: String,
    code: String,
    commands: Vec<String>,
    checks: Vec<String>,
    caution: String,
}

impl ModelReply {
    fn fallback(raw_text: &str) -> Self {
        let mut commands = Vec::new();
        for line in raw_text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('$') {
                commands.push(trimmed.trim_start_matches('$').trim().to_string());
            } else if looks_like_shell_command(trimmed) {
                commands.push(trimmed.to_string());
            }
        }
        Self {
            explanation: raw_text.trim().to_string(),
            code: extract_code_block(raw_text),
            commands,
            checks: vec!["Verify each step before running.".to_string()],
            caution: "Model returned non-JSON output. Review everything carefully.".to_string(),
        }
    }
}

fn extract_code_block(text: &str) -> String {
    let mut in_fence = false;
    let mut _fence_lang = String::new();
    let mut code_lines: Vec<&str> = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            if in_fence {
                break;
            }
            in_fence = true;
            _fence_lang = trimmed.trim_start_matches("```").to_string();
            continue;
        }
        if in_fence {
            code_lines.push(line);
        }
    }
    if code_lines.is_empty() { String::new() } else { code_lines.join("\n") }
}

// ── Ollama client ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct OllamaResponse { response: String }

#[derive(Debug, Deserialize)]
struct OllamaErrorResponse { error: String }

#[derive(Debug, Deserialize)]
struct TagsResponse { models: Vec<TagModel> }

#[derive(Debug, Deserialize)]
struct TagModel { name: String }

struct OllamaClient {
    client: Client,
    base_url: String,
    model: String,
    system_prompt: String,
}

impl OllamaClient {
    fn new(base_url: String, model: String, timeout_s: u64, system_prompt: String) -> Self {
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

    fn set_model(&mut self, model: String) { self.model = model; }

    async fn list_models(&self) -> Result<Vec<String>> {
        let url = api_url(&self.base_url, "tags");
        let resp = self.client.get(url).send().await?;
        if !resp.status().is_success() {
            return Err(anyhow!("Ollama unreachable at {}", self.base_url));
        }
        let parsed: TagsResponse = resp.json().await.context("invalid tags response")?;
        Ok(parsed.models.into_iter().map(|m| m.name).collect())
    }

    async fn healthcheck(&self) -> Result<()> {
        let models = self.list_models().await?;
        if models.is_empty() {
            return Err(anyhow!("Ollama reachable but no models are installed. Pull one with `ollama pull rem-coder:latest`"));
        }
        Ok(())
    }

    async fn complete_json(&self, user_prompt: &str) -> Result<ModelReply> {
        let url = api_url(&self.base_url, "generate");
        let final_prompt = format!(
            "{}\n\nUser request:\n{}\n\nReturn JSON only.",
            self.system_prompt, user_prompt
        );
        let payload = json!({
            "model": self.model,
            "prompt": final_prompt,
            "stream": false,
            "format": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "code": {"type": "string"},
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
            let body = resp.text().await.unwrap_or_default();
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
        match serde_json::from_str::<ModelReply>(raw.response.trim()) {
            Ok(parsed) => Ok(parsed),
            Err(_) => Ok(ModelReply::fallback(raw.response.trim())),
        }
    }

    async fn complete_chat_stream(&self, user_prompt: &str, system_prompt: &str) -> Result<String> {
        let url = api_url(&self.base_url, "generate");
        let final_prompt = format!("{}\n\nUser: {}\n\nREM:", system_prompt, user_prompt);
        let payload = json!({
            "model": self.model,
            "prompt": final_prompt,
            "stream": true
        });
        let resp = self.client.post(&url).json(&payload).send().await
            .context("failed to call Ollama")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            let err_msg = serde_json::from_str::<OllamaErrorResponse>(&body)
                .map(|v| v.error).unwrap_or_else(|_| body.clone());
            if status.as_u16() == 404 && err_msg.to_lowercase().contains("model") {
                return Err(anyhow!("Model '{}' not found. Pull it: `ollama pull {}`", self.model, self.model));
            }
            return Err(anyhow!("Ollama failed: {} — {}", status, err_msg));
        }
        let mut stream = resp.bytes_stream();
        let mut full = String::new();
        let mut buf = String::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("stream read error")?;
            buf.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(pos) = buf.find('\n') {
                let line = buf[..pos].to_string();
                buf = buf[pos + 1..].to_string();
                let trimmed = line.trim();
                if trimmed.is_empty() { continue; }
                if let Ok(obj) = serde_json::from_str::<serde_json::Value>(trimmed) {
                    if let Some(token) = obj["response"].as_str() {
                        print!("{}", token);
                        let _ = io::stdout().flush();
                        full.push_str(token);
                    }
                    if obj["done"].as_bool() == Some(true) {
                        println!();
                        return Ok(full);
                    }
                }
            }
        }
        println!();
        Ok(full)
    }
}

// ── Web search ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SearchResult {
    title: String,
    snippet: String,
    url: String,
}

async fn perform_web_search(client: &Client, query: &str) -> Result<Vec<SearchResult>> {
    let resp = client
        .get("https://html.duckduckgo.com/html/")
        .query(&[("q", query)])
        .header("User-Agent", "rem-cli/0.1")
        .send()
        .await
        .context("web search request failed")?;
    let html = resp.text().await.context("failed to read search response")?;
    Ok(parse_ddg_html(&html))
}

fn parse_ddg_html(html: &str) -> Vec<SearchResult> {
    let mut results = Vec::new();
    let title_re = Regex::new(r#"class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>"#).unwrap();
    let snippet_re = Regex::new(r#"class="result__snippet"[^>]*>([^<]*(?:<[^/>][^>]*>[^<]*</[^>]+>)*[^<]*)</a>"#).unwrap();
    let mut remaining = html;
    while results.len() < 8 {
        if let Some(cap) = title_re.captures(remaining) {
            let url = cap.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();
            let title = cap.get(2).map(|m| strip_html(m.as_str())).unwrap_or_default();
            let snippet_pos = cap.get(0).map(|m| m.end()).unwrap_or(0);
            let after_title = &remaining[snippet_pos..];
            let snippet = snippet_re.captures(after_title)
                .and_then(|c| c.get(1))
                .map(|m| strip_html(m.as_str()).trim().to_string())
                .unwrap_or_default();
            if !title.is_empty() {
                results.push(SearchResult { title, snippet, url });
            }
            let advance = cap.get(0).map(|m| m.end()).unwrap_or(1);
            if advance >= remaining.len() { break; }
            remaining = &remaining[advance..];
        } else {
            break;
        }
    }
    results
}

fn strip_html(input: &str) -> String {
    let tag_re = Regex::new(r"<[^>]*>").unwrap();
    let amp_re = Regex::new(r"&amp;").unwrap();
    let lt_re = Regex::new(r"&lt;").unwrap();
    let gt_re = Regex::new(r"&gt;").unwrap();
    let quot_re = Regex::new(r"&quot;").unwrap();
    let apos_re = Regex::new(r"&#x27;").unwrap();
    let mut s = tag_re.replace_all(input, "").to_string();
    s = amp_re.replace_all(&s, "&").to_string();
    s = lt_re.replace_all(&s, "<").to_string();
    s = gt_re.replace_all(&s, ">").to_string();
    s = quot_re.replace_all(&s, "\"").to_string();
    s = apos_re.replace_all(&s, "'").to_string();
    s.trim().to_string()
}

fn print_search_results(results: &[SearchResult]) {
    if results.is_empty() {
        println!("{}", style!(C_YELLOW, "  no results found"));
        return;
    }
    println!("{}", style!(C_DIM, "│"));
    for (i, r) in results.iter().enumerate() {
        println!("{} {}", style!(C_CYAN, "│"), style!(C_WHITE_B, "{}. {}", i + 1, r.title));
        println!("{}   {}", style!(C_CYAN, "│"), style!(C_DIM, "{}", r.url));
        if !r.snippet.is_empty() {
            println!("{}   {}", style!(C_CYAN, "│"), r.snippet);
        }
        println!("{}", style!(C_CYAN, "│"));
    }
}

// ── Chat session state ─────────────────────────────────────────────────────

struct ChatSession {
    rl: DefaultEditor,
    last_code: String,
    last_search: Vec<SearchResult>,
    project_dir: Option<PathBuf>,
}

impl ChatSession {
    fn new() -> Result<Self> {
        let rl = DefaultEditor::new().context("failed to start line editor")?;
        Ok(Self { rl, last_code: String::new(), last_search: Vec::new(), project_dir: None })
    }

    fn readline(&mut self, prompt: &str) -> io::Result<String> {
        self.rl.readline(prompt).map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    }

    fn add_history(&mut self, line: &str) {
        let _ = self.rl.add_history_entry(line);
    }

    fn build_search_context(&self) -> String {
        if self.last_search.is_empty() {
            return String::new();
        }
        let mut ctx = String::from("\n\n[Web search results — use these when answering]:\n");
        for (i, r) in self.last_search.iter().enumerate() {
            ctx.push_str(&format!("{}. {}\n   URL: {}\n   {}\n\n", i + 1, r.title, r.url, r.snippet));
        }
        ctx
    }
}

// ── Entry point ────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut cfg = load_config().unwrap_or_default();
    if let Some(m) = cli.model { cfg.model = m; }
    if let Some(url) = cli.ollama_url { cfg.ollama_url = url; }

    let system_prompt = load_system_prompt(cfg.prompts_dir.as_deref());
    let mut client = OllamaClient::new(
        cfg.ollama_url.clone(), cfg.model.clone(), cfg.timeout_s, system_prompt,
    );
    client.healthcheck().await?;
    let models = client.list_models().await?;
    if !models.iter().any(|m| m == &cfg.model) {
        let fallback = models.first().cloned().unwrap_or_else(|| cfg.model.clone());
        eprintln!("{}: model '{}' not found; using '{}'", style!(C_YELLOW, "warning"), cfg.model, fallback);
        client.set_model(fallback);
    }

    match cli.command {
        Commands::Ask(args)    => run_ask(&client, &cfg, args).await,
        Commands::Explain(args) => run_explain(&client, args).await,
        Commands::Patch(args)   => run_patch(&client, &cfg, args).await,
        Commands::Chat          => run_chat(&client).await,
    }
}

fn load_config() -> Result<AppConfig> {
    let mut cfg = AppConfig::default();
    if let Some(home) = dirs::home_dir() {
        let path = home.join(".config/rem-cli/config.toml");
        if path.exists() {
            let text = fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            let partial: PartialConfig = toml::from_str(&text).context("invalid global config")?;
            cfg.apply_partial(partial);
        }
    }
    let local = PathBuf::from(".remcli.toml");
    if local.exists() {
        let text = fs::read_to_string(&local)
            .with_context(|| format!("failed to read {}", local.display()))?;
        let partial: PartialConfig = toml::from_str(&text).context("invalid local config")?;
        cfg.apply_partial(partial);
    }
    Ok(cfg)
}

fn load_system_prompt(custom_prompts_dir: Option<&str>) -> String {
    let mut candidates = Vec::new();
    if let Some(dir) = custom_prompts_dir {
        candidates.push(PathBuf::from(dir).join("system_prompt.txt"));
    }
    candidates.push(PathBuf::from("prompts/system_prompt.txt"));
    for path in candidates {
        if path.exists() {
            if let Ok(text) = fs::read_to_string(path) {
                let trimmed = text.trim();
                if !trimmed.is_empty() { return trimmed.to_string(); }
            }
        }
    }
    DEFAULT_SYSTEM_PROMPT.to_string()
}

// ── Subcommand handlers ────────────────────────────────────────────────────

async fn run_ask(client: &OllamaClient, cfg: &AppConfig, args: AskArgs) -> Result<()> {
    let mut composed = args.prompt;
    if let Some(path) = args.file {
        let ctx = build_context(&path, cfg.max_context_bytes)?;
        composed = format!("{}\n\nFile context:\n{}", composed, ctx);
    }
    print_banner(client);
    print_status("Thinking...");
    let reply = client.complete_json(&composed).await?;
    print_reply(&reply, true);
    Ok(())
}

async fn run_explain(client: &OllamaClient, args: ExplainArgs) -> Result<()> {
    print_banner(client);
    let prompt = format!("Explain this terminal command for a beginner and suggest a safer variant if needed: {}", args.command);
    let reply = client.complete_json(&prompt).await?;
    print_reply(&reply, true);
    Ok(())
}

async fn run_patch(client: &OllamaClient, cfg: &AppConfig, args: PatchArgs) -> Result<()> {
    print_banner(client);
    let existing = fs::read_to_string(&args.file)
        .with_context(|| format!("failed to read {}", args.file.display()))?;
    let dir_ctx = build_context(&args.file, cfg.max_context_bytes)?;
    let prompt = format!(
        "Task: {}\n\nTarget file: {}\n\nCurrent content:\n{}\n\nNearby context:\n{}\n\nReturn updated file content in code.",
        args.task, args.file.display(), existing, dir_ctx
    );
    let reply = client.complete_json(&prompt).await?;
    println!("{}", style!(C_CYAN, "Patch preview for {}", args.file.display()));
    print_reply(&reply, true);
    Ok(())
}

// ── Interactive chat ───────────────────────────────────────────────────────

async fn run_chat(client: &OllamaClient) -> Result<()> {
    let mut session = ChatSession::new()?;
    print_banner(client);
    print_chat_help();

    loop {
        let line = session.readline("rem> ");
        let line = match line {
            Ok(s) => s,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }

        // ── built-in commands ──────────────────────────────────────
        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            println!("  {}", style!(C_DIM, "bye!"));
            break;
        }

        if trimmed.eq_ignore_ascii_case("/help") || trimmed.eq_ignore_ascii_case("help") {
            print_chat_help();
            continue;
        }

        if let Some(tail) = trimmed.strip_prefix("/write ") {
            handle_write(&session, tail);
            continue;
        }
        if let Some(tail) = trimmed.strip_prefix("/save ") {
            handle_write(&session, tail);
            continue;
        }

        if let Some(tail) = trimmed.strip_prefix("/dir ") {
            handle_dir(&mut session, tail);
            continue;
        }

        if let Some(tail) = trimmed.strip_prefix("/search ") {
            handle_search(client, &mut session, tail.trim()).await;
            continue;
        }

        if let Some(_) = trimmed.strip_prefix("/code") {
            if session.last_code.is_empty() {
                println!("  {} No code from last response.", style!(C_YELLOW, "!"));
            } else {
                println!("{}", style!(C_WHITE_B, "── last code ──"));
                println!("{}", session.last_code);
                println!("{}", style!(C_DIM, "───────────────"));
            }
            continue;
        }

        // ── detect creation intent without a path ───────────────────
        let needs_path = has_creation_intent(trimmed) && !has_file_path(trimmed);
        let final_prompt = if needs_path {
            session.add_history(trimmed);
            let path = prompt_for_path(&mut session)?;
            format!("User request: {}\n\nSave file at: {}", trimmed, path)
        } else {
            session.add_history(trimmed);
            if let Some(ref dir) = session.project_dir {
                format!("{}\n\nWorking directory: {}", trimmed, dir.display())
            } else {
                trimmed.to_string()
            }
        };

        // ── call model (streaming) ─────────────────────────────────
        let search_ctx = session.build_search_context();
        let full_prompt = if search_ctx.is_empty() { final_prompt } else { format!("{}{}", final_prompt, search_ctx) };
        print!("{} ", style!(C_CYAN, "│"));
        println!("{}", style!(C_DIM, "── rem ──"));
        match client.complete_chat_stream(&full_prompt, CHAT_SYSTEM_PROMPT).await {
            Ok(text) => {
                let code = extract_code_block(&text);
                if !code.is_empty() {
                    session.last_code = code;
                }
                println!("{}", style!(C_DIM, "│"));
            }
            Err(e) => {
                println!();
                eprintln!("  {} {}", style!(C_RED, "err:"), e);
            }
        }
    }
    Ok(())
}

/// Returns true if the user prompt looks like they want to create/build something.
fn has_creation_intent(input: &str) -> bool {
    let lower = input.to_lowercase();
    let creation_words = ["create", "build", "make", "generate", "write", "scaffold", "set up"];
    creation_words.iter().any(|w| lower.contains(w))
}

/// Returns true if the prompt mentions a file path or directory.
fn has_file_path(input: &str) -> bool {
    let lower = input.to_lowercase();
    lower.contains(".html") || lower.contains(".css") || lower.contains(".js")
        || lower.contains('/') || lower.contains(".ts") || lower.contains("file")
        || lower.contains("path") || lower.contains("directory") || lower.contains("folder")
        || lower.contains("into ") || lower.contains("in ")
}

/// When no file path is in the user's request, ask them where to save.
fn prompt_for_path(session: &mut ChatSession) -> io::Result<String> {
    println!("{}", style!(C_CYAN, "│"));
    println!("{} {}",
        style!(C_MAGENTA, "│  ?"),
        style!(C_WHITE_B, "Where should I create this? (e.g. ./my-site/index.html or ./project/)"));
    println!("{} {}", style!(C_MAGENTA, "│"), style!(C_DIM, "  type '.' for current dir, or /dir <path> to set a project root"));
    println!("{}", style!(C_CYAN, "│"));

    loop {
        let line = session.readline("rem> path: ");
        let line = match line { Ok(s) => s, Err(_) => return Ok(".".to_string()) };
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        session.add_history(trimmed);

        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            return Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled"));
        }

        if let Some(tail) = trimmed.strip_prefix("/dir ") {
            handle_dir(session, tail);
            continue;
        }

        return Ok(trimmed.to_string());
    }
}

fn handle_write(session: &ChatSession, path: &str) {
    if session.last_code.is_empty() {
        println!("  {} No code from last response. Use `/code` to view it.", style!(C_YELLOW, "!"));
        return;
    }
    let file_path = PathBuf::from(path.trim());
    if let Some(parent) = file_path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = fs::create_dir_all(parent);
        }
    }
    match fs::write(&file_path, &session.last_code) {
        Ok(()) => println!("  {} wrote {} ({} bytes)",
            style!(C_GREEN, "✓"), style!(C_WHITE_B, "{}", file_path.display()), session.last_code.len()),
        Err(e) => println!("  {} failed: {}", style!(C_RED, "✗"), e),
    }
}

fn handle_dir(session: &mut ChatSession, path: &str) {
    let dir = PathBuf::from(path.trim());
    if dir.exists() || path.trim() == "." {
        session.project_dir = Some(if path.trim() == "." { std::env::current_dir().unwrap_or_default() } else { dir });
        println!("  {} project root set to {}",
            style!(C_GREEN, "✓"), style!(C_WHITE_B, "{}", session.project_dir.as_ref().unwrap().display()));
    } else {
        println!("  {} directory does not exist", style!(C_YELLOW, "!"));
    }
}

async fn handle_search(client: &OllamaClient, session: &mut ChatSession, query: &str) {
    println!("{} {} searching the web...", style!(C_DIM, "│"), style!(C_CYAN, "🔍"));
    match perform_web_search(&client.client, query).await {
        Ok(results) => {
            if results.is_empty() {
                println!("{} no results found for: {}", style!(C_YELLOW, "│"), query);
            } else {
                println!("{} {} results for: {}", style!(C_DIM, "│"), results.len(), style!(C_WHITE_B, "{}", query));
                print_search_results(&results);
                session.last_search = results;
            }
        }
        Err(e) => {
            println!("{} {}", style!(C_RED, "│  search failed:"), e);
        }
    }
}

fn print_chat_help() {
    println!("{}", style!(C_DIM, "│"));
    println!("{} {}", style!(C_DIM, "│"), style!(C_WHITE_B, "commands:"));
    println!("{}   /help           show this", style!(C_DIM, "│"));
    println!("{}   /write <path>   save last code to file", style!(C_DIM, "│"));
    println!("{}   /save <path>    same as /write", style!(C_DIM, "│"));
    println!("{}   /dir <path>     set project root directory", style!(C_DIM, "│"));
    println!("{}   /search <q>     search the web", style!(C_DIM, "│"));
    println!("{}   /code           show last generated code", style!(C_DIM, "│"));
    println!("{}   exit            quit REM", style!(C_DIM, "│"));
    println!("{}", style!(C_DIM, "│"));
    println!("{} {}", style!(C_DIM, "│"), style!(C_WHITE_B, "tips:"));
    println!("{}   Just describe what you want — I'll ask", style!(C_DIM, "│"));
    println!("{}   for the file path before giving you code.", style!(C_DIM, "│"));
    println!("{}", style!(C_DIM, "│"));
}

// ── Output formatting ──────────────────────────────────────────────────────

fn print_banner(client: &OllamaClient) {
    println!();
    println!("{} {} {}",
        style!(C_CYAN, "╭─"),
        style!(C_BOLD, "REM"),
        style!(C_DIM, "─────────────────────────────"));
    println!("{} model {}{} {}",
        style!(C_CYAN, "│"),
        style!(C_GREEN, "{}", client.model),
        style!(C_DIM, " •"),
        style!(C_DIM, "type /help for commands"));
}

fn print_status(msg: &str) {
    print!("{} ", style!(C_DIM, "│  ...{}", msg));
    let _ = io::stdout().flush();
}

fn print_reply(reply: &ModelReply, newline: bool) {
    if newline {
        println!();
    }
    if !reply.explanation.trim().is_empty() {
        println!("{} {}", style!(C_CYAN, "│"), style!(C_WHITE_B, "{}", reply.explanation));
        println!("{}", style!(C_CYAN, "│"));
    }
    if !reply.code.trim().is_empty() {
        println!("{} {}", style!(C_CYAN, "│  ┌"), style!(C_GREEN, "code:"));
        for code_line in reply.code.lines() {
            println!("{} {}", style!(C_CYAN, "│"), style!(C_GREEN, "  │  {}", code_line));
        }
        println!("{}   {}", style!(C_CYAN, "│  └"), style!(C_DIM, "/write <path> to save"));
        println!("{}", style!(C_CYAN, "│"));
    }
    if !reply.commands.is_empty() {
        println!("{} {}", style!(C_CYAN, "│"), style!(C_MAGENTA, "commands:"));
        for cmd in sanitize_commands(&reply.commands) {
            if is_command_blocked(cmd) {
                println!("{}   {}", style!(C_CYAN, "│"), style!(C_RED, "[blocked] {}", cmd));
            } else {
                println!("{}   {}", style!(C_CYAN, "│"), style!(C_MAGENTA, "  $ {}", cmd));
            }
        }
        println!("{}", style!(C_CYAN, "│"));
    }
    if !reply.checks.is_empty() {
        println!("{} {}", style!(C_CYAN, "│"), style!(C_YELLOW, "checks:"));
        for item in &reply.checks {
            println!("{}   {}", style!(C_CYAN, "│"), style!(C_DIM, "  • {}", item));
        }
        println!("{}", style!(C_CYAN, "│"));
    }
    if !reply.caution.trim().is_empty() {
        println!("{} {}", style!(C_CYAN, "│"), style!(C_RED, "caution: {}", reply.caution));
        println!("{}", style!(C_CYAN, "│"));
    }
    println!("{}", style!(C_CYAN, "╰──────────────────────────────────"));
}

// ── Context builder ────────────────────────────────────────────────────────

fn build_context(target: &Path, max_bytes: usize) -> Result<String> {
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    let mut out = String::from("Directory snapshot:\n");
    for entry in WalkDir::new(parent).max_depth(2) {
        let entry = entry?;
        let p = entry.path();
        let rel = p.strip_prefix(parent).unwrap_or(p);
        if rel.as_os_str().is_empty() { continue; }
        out.push_str(&format!("- {}\n", rel.display()));
        if out.len() > max_bytes { break; }
    }
    if target.exists() {
        let content = fs::read_to_string(target)
            .with_context(|| format!("failed to read {}", target.display()))?;
        out.push_str("\nTarget file:\n");
        out.push_str(&truncate_bytes(&content, max_bytes / 2));
    }
    Ok(truncate_bytes(&out, max_bytes))
}

fn truncate_bytes(s: &str, max: usize) -> String {
    if s.len() <= max { return s.to_string(); }
    let mut end = max;
    while !s.is_char_boundary(end) { end -= 1; }
    format!("{}\n...[truncated]", &s[..end])
}

// ── Utility functions ──────────────────────────────────────────────────────

fn is_command_blocked(cmd: &str) -> bool {
    let lower = cmd.to_lowercase();
    BLOCKED_COMMAND_PATTERNS.iter().any(|p| lower.contains(p))
}

fn looks_like_shell_command(line: &str) -> bool {
    let first = line.split_whitespace().next().unwrap_or_default();
    matches!(first, "ls" | "pwd" | "cd" | "mkdir" | "cp" | "mv" | "touch"
        | "cat" | "echo" | "rm" | "find" | "grep")
}

fn api_url(base_url: &str, endpoint: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let ep = endpoint.trim_start_matches('/');
    if base.ends_with("/api") { format!("{}/{}", base, ep) }
    else { format!("{}/api/{}", base, ep) }
}

fn sanitize_commands(cmds: &[String]) -> Vec<&str> {
    let mut seen = BTreeMap::<String, ()>::new();
    let mut out = Vec::new();
    for cmd in cmds {
        let key = cmd.trim().to_string();
        if key.is_empty() || seen.contains_key(&key) { continue; }
        seen.insert(key.clone(), ());
        out.push(cmd.trim());
    }
    out
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_dangerous_commands() {
        assert!(is_command_blocked("rm -rf /tmp"));
        assert!(is_command_blocked("shutdown now"));
        assert!(!is_command_blocked("ls -la"));
    }

    #[test]
    fn truncates_string() {
        let out = truncate_bytes("abcdef", 3);
        assert!(out.starts_with("abc"));
    }

    #[test]
    fn command_sanitization_dedups() {
        let input = vec![" ls ".to_string(), "ls".to_string(), "".to_string()];
        let out = sanitize_commands(&input);
        assert_eq!(out, vec!["ls"]);
    }

    #[test]
    fn fallback_extracts_commands() {
        let out = ModelReply::fallback("Use:\nmkdir project\ncd project");
        assert!(out.commands.iter().any(|c| c == "mkdir project"));
    }

    #[test]
    fn fallback_extracts_code_block() {
        let out = ModelReply::fallback("Here:\n```html\n<div>hi</div>\n```\ndone");
        assert_eq!(out.code, "<div>hi</div>");
    }

    #[test]
    fn api_url_plain_base() {
        assert_eq!(api_url("http://localhost:11434", "generate"), "http://localhost:11434/api/generate");
    }

    #[test]
    fn api_url_with_api() {
        assert_eq!(api_url("http://localhost:11434/api", "generate"), "http://localhost:11434/api/generate");
    }

    #[test]
    fn detects_creation_intent() {
        assert!(has_creation_intent("create a basic html page"));
        assert!(has_creation_intent("build me a navbar"));
        assert!(has_creation_intent("generate a contact form"));
        assert!(!has_creation_intent("explain ls command"));
    }

    #[test]
    fn detects_file_path() {
        assert!(has_file_path("create page at index.html"));
        assert!(has_file_path("write to ./src/style.css"));
        assert!(has_file_path("add this to file app.js"));
        assert!(!has_file_path("create a basic html page"));
    }

    #[test]
    fn extracts_code_block_from_text() {
        let text = "Sure:\n```css\n.card { padding: 12px; }\n```\nThat's it.";
        assert_eq!(extract_code_block(text), ".card { padding: 12px; }");
    }

    #[test]
    fn extracts_code_no_fence() {
        let text = "Here is some code: .card {}";
        assert_eq!(extract_code_block(text), "");
    }
}
