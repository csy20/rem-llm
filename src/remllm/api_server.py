"""FastAPI server — exposes the rem-llm pipeline as a REST API.

Provides endpoints for:
- Model generation (chat + structured)
- Training pipeline orchestration
- Evaluation and benchmark runs
- Model and dataset management

Start with:
    python -m remllm.api_server --port 8080
    uvicorn remllm.api_server:app --host 0.0.0.0 --port 8080
"""

import datetime
import json
import os
from pathlib import Path
from typing import Any

from remllm.logging import get_logger

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    HTTPException = Exception
    BackgroundTasks = object
    BaseModel = object
    Field = object

log = get_logger(module="api_server")

app: Any = None
try:
    app = FastAPI(
        title="REM-LLM API",
        description="Coding LLM training pipeline — REST API",
        version="0.2.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass


PIPELINE_JOBS: dict[str, dict[str, Any]] = {}


# ── Request / Response models ────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The coding prompt or question")
    model: str = Field(default="rem-coder:latest")
    mode: str = Field(default="chat", description="chat | code | plan")
    max_tokens: int = Field(default=512)
    temperature: float = Field(default=0.7)


class GenerateResponse(BaseModel):
    text: str
    model: str
    tokens: int = 0
    latency_s: float = 0.0


class TrainRequest(BaseModel):
    config_path: str = Field(default="config/config.yaml")
    base_model: str = Field(default="qwen2.5-coder:1.5b")
    trained_model: str = Field(default="rem-coder-trained")


class EvalRequest(BaseModel):
    config_path: str = Field(default="config/config.yaml")
    model: str = Field(..., description="Model to evaluate")
    benchmark: str = Field(
        default="humaneval", description="humaneval | mbpp | quality | executable"
    )
    max_samples: int = Field(default=100)


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: str = ""
    result: dict[str, Any] | None = None


# ── Health ───────────────────────────────────────────────────────────────────


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": "0.2.0",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


# ── Generation ───────────────────────────────────────────────────────────────


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    try:
        import subprocess

        start = datetime.datetime.now(datetime.timezone.utc)
        result = subprocess.run(
            ["ollama", "run", request.model, request.prompt],
            capture_output=True,
            text=True,
            timeout=request.max_tokens + 30,
        )
        elapsed = (datetime.datetime.now(datetime.timezone.utc) - start).total_seconds()

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr.strip())

        text = result.stdout.strip()
        tokens = len(text.split())

        return GenerateResponse(
            text=text,
            model=request.model,
            tokens=tokens,
            latency_s=round(elapsed, 3),
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Model generation timed out")
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Ollama not found. Install via: curl -fsSL https://ollama.com/install.sh | sh",
        )


# ── Training ─────────────────────────────────────────────────────────────────


@app.post("/train", response_model=JobStatus)
async def train(request: TrainRequest, background_tasks: BackgroundTasks) -> JobStatus:
    job_id = datetime.datetime.now(datetime.timezone.utc).strftime(
        "train-%Y%m%d-%H%M%S"
    )

    PIPELINE_JOBS[job_id] = {
        "status": "queued",
        "progress": "Starting training pipeline...",
        "result": None,
    }

    background_tasks.add_task(_run_training_pipeline, job_id, request)
    return JobStatus(job_id=job_id, status="queued", progress="Starting...")


def _run_training_pipeline(job_id: str, request: TrainRequest) -> None:
    try:
        from remllm.cli import cmd_pipeline

        PIPELINE_JOBS[job_id]["status"] = "running"
        PIPELINE_JOBS[job_id]["progress"] = "Pipeline started"

        old_argv = __import__("sys").argv[:]
        __import__("sys").argv = [
            "remllm",
            "pipeline",
            "--config",
            request.config_path,
            "--base-model",
            request.base_model,
            "--trained-model",
            request.trained_model,
        ]

        cmd_pipeline(
            __import__("argparse").Namespace(
                config=request.config_path,
                base_model=request.base_model,
                trained_model=request.trained_model,
            )
        )

        __import__("sys").argv = old_argv

        PIPELINE_JOBS[job_id]["status"] = "completed"
        PIPELINE_JOBS[job_id]["progress"] = "Pipeline completed successfully"
        PIPELINE_JOBS[job_id]["result"] = {"status": "success"}
    except Exception as e:
        PIPELINE_JOBS[job_id]["status"] = "failed"
        PIPELINE_JOBS[job_id]["progress"] = str(e)
        PIPELINE_JOBS[job_id]["result"] = {"error": str(e)}


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    job = PIPELINE_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", ""),
        result=job.get("result"),
    )


@app.get("/jobs")
def list_jobs() -> list[JobStatus]:
    return [
        JobStatus(
            job_id=k,
            status=v["status"],
            progress=v.get("progress", ""),
            result=v.get("result"),
        )
        for k, v in PIPELINE_JOBS.items()
    ]


# ── Evaluation ───────────────────────────────────────────────────────────────


@app.post("/eval", response_model=JobStatus)
async def evaluate(
    request: EvalRequest, background_tasks: BackgroundTasks
) -> JobStatus:
    job_id = datetime.datetime.now(datetime.timezone.utc).strftime("eval-%Y%m%d-%H%M%S")

    PIPELINE_JOBS[job_id] = {
        "status": "queued",
        "progress": f"Starting {request.benchmark} evaluation...",
        "result": None,
    }

    background_tasks.add_task(_run_evaluation, job_id, request)
    return JobStatus(job_id=job_id, status="queued", progress="Starting...")


def _run_evaluation(job_id: str, request: EvalRequest) -> None:
    try:
        from remllm.eval.benchmark_harness import evaluate_on_benchmark
        from remllm.eval.quality import QualityEvaluator

        PIPELINE_JOBS[job_id]["status"] = "running"

        if request.benchmark in ("humaneval", "mbpp"):
            benchmark_path = Path(f"data/benchmarks/{request.benchmark}.jsonl")
            if not benchmark_path.exists():
                PIPELINE_JOBS[job_id]["status"] = "failed"
                PIPELINE_JOBS[job_id]["progress"] = (
                    f"Benchmark file not found: {benchmark_path}"
                )
                return

            def generate(prompt: str) -> str:
                import subprocess

                result = subprocess.run(
                    ["ollama", "run", request.model, prompt],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return result.stdout.strip()

            result = evaluate_on_benchmark(
                request.model,
                request.benchmark,
                benchmark_path,
                generate,
                max_samples=request.max_samples,
            )
            PIPELINE_JOBS[job_id]["result"] = result.to_report()

        elif request.benchmark in ("quality", "executable"):
            from remllm.eval.executable import ExecutableEvaluator
            from remllm.data.loader import load_jsonl
            import yaml

            eval_file = Path("data/eval.jsonl")
            if not eval_file.exists():
                eval_file = Path("data/beginner/eval.jsonl")
            if not eval_file.exists():
                PIPELINE_JOBS[job_id]["status"] = "failed"
                PIPELINE_JOBS[job_id]["progress"] = "No eval data found"
                return

            rows = load_jsonl(eval_file)
            if request.benchmark == "quality":
                evaluator = QualityEvaluator()
            else:
                evaluator = ExecutableEvaluator()

            report = evaluator.evaluate(request.model, rows)
            PIPELINE_JOBS[job_id]["result"] = {
                "model": request.model,
                "benchmark": request.benchmark,
                "rates": report.rates,
            }

        PIPELINE_JOBS[job_id]["status"] = "completed"
        PIPELINE_JOBS[job_id]["progress"] = "Evaluation complete"
    except Exception as e:
        PIPELINE_JOBS[job_id]["status"] = "failed"
        PIPELINE_JOBS[job_id]["progress"] = str(e)
        PIPELINE_JOBS[job_id]["result"] = {"error": str(e)}


# ── Models ───────────────────────────────────────────────────────────────────


@app.get("/models")
def list_models() -> list[dict[str, str]]:
    try:
        import subprocess

        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = result.stdout.strip().split("\n")[1:]
        models = []
        for line in lines:
            parts = line.split()
            if parts:
                models.append({"name": parts[0]})
        return models
    except Exception:
        return []


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="REM-LLM REST API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "remllm.api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
