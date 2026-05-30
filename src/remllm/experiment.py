"""Experiment tracking — unified MLflow/W&B integration for reproducible runs."""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from remllm.logging import get_logger


class ExperimentTracker:
    """Unified experiment tracker that logs to a local JSON store and optionally
    to MLflow or Weights & Biases.

    Usage:
        tracker = ExperimentTracker(run_id="exp-001")
        tracker.log_params({"lr": 1.2e-4, "epochs": 3})
        tracker.log_metrics({"loss": 0.42}, step=100)
        tracker.log_artifact("models/evals/baseline.json")
        tracker.finish()
    """

    def __init__(
        self,
        run_id: str | None = None,
        experiment_dir: str | Path = "models/experiments",
        backend: str | None = None,
        project_name: str = "remllm",
    ):
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.experiment_dir = Path(experiment_dir) / self.run_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend or os.environ.get("REMLLM_TRACKER", "local")
        self.project_name = project_name
        self._log = get_logger(run_id=self.run_id, tracker=self.backend)
        self._metrics_path = self.experiment_dir / "metrics.jsonl"
        self._params_path = self.experiment_dir / "params.json"
        self._artifacts_path = self.experiment_dir / "artifacts.json"
        self._params: dict[str, Any] = {}
        self._artifacts: list[str] = []
        self._git_commit = _get_git_commit()
        self._start_time = datetime.now(timezone.utc).isoformat()

        self._init_backend()

    def _init_backend(self) -> None:
        """Initialize the configured tracking backend."""
        if self.backend == "wandb":
            try:
                import wandb

                wandb.init(project=self.project_name, id=self.run_id, resume="allow")
                self._log.info("wandb_initialized")
            except ImportError:
                self._log.warning("wandb_not_installed", fallback="local")
                self.backend = "local"
            except Exception as e:
                self._log.warning("wandb_init_failed", error=str(e), fallback="local")
                self.backend = "local"

        elif self.backend == "mlflow":
            try:
                import mlflow

                mlflow.set_experiment(self.project_name)
                mlflow.start_run(run_name=self.run_id)
                self._log.info("mlflow_initialized")
            except ImportError:
                self._log.warning("mlflow_not_installed", fallback="local")
                self.backend = "local"
            except Exception as e:
                self._log.warning("mlflow_init_failed", error=str(e), fallback="local")
                self.backend = "local"

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters and configuration."""
        self._params.update(params)
        self._params["git_commit"] = self._git_commit
        self._params["start_time"] = self._start_time
        with self._params_path.open("w") as f:
            json.dump(self._params, f, indent=2)

        if self.backend == "wandb":
            try:
                import wandb

                wandb.config.update(params, allow_val_change=True)
            except Exception:
                pass
        elif self.backend == "mlflow":
            try:
                import mlflow

                mlflow.log_params(params)
            except Exception:
                pass

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        """Log scalar metrics at a given step."""
        entry = {"step": step, "timestamp": datetime.now(timezone.utc).isoformat()}
        entry.update(metrics)

        with self._metrics_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

        if self.backend == "wandb":
            try:
                import wandb

                wandb.log(metrics, step=step)
            except Exception:
                pass
        elif self.backend == "mlflow":
            try:
                import mlflow

                mlflow.log_metrics(metrics, step=step)
            except Exception:
                pass

    def log_artifact(self, path: str | Path) -> None:
        """Track a file artifact (model, report, etc.)."""
        path = Path(path)
        self._artifacts.append(str(path))

        with self._artifacts_path.open("w") as f:
            json.dump(self._artifacts, f, indent=2)

        if self.backend == "wandb" and path.exists():
            try:
                import wandb

                wandb.save(str(path))
            except Exception:
                pass
        elif self.backend == "mlflow" and path.exists():
            try:
                import mlflow

                mlflow.log_artifact(str(path))
            except Exception:
                pass

    def log_dict(self, data: dict[str, Any], name: str) -> None:
        """Save an arbitrary JSON-serializable dict as a file artifact."""
        path = self.experiment_dir / f"{name}.json"
        with path.open("w") as f:
            json.dump(data, f, indent=2)
        self.log_artifact(path)

    def finish(self) -> None:
        """End the experiment run."""
        end_time = datetime.now(timezone.utc).isoformat()
        self._log.info("experiment_finished", end_time=end_time)

        metadata = {
            "run_id": self.run_id,
            "start_time": self._start_time,
            "end_time": end_time,
            "git_commit": self._git_commit,
            "backend": self.backend,
            "params": self._params,
            "num_artifacts": len(self._artifacts),
        }
        meta_path = self.experiment_dir / "metadata.json"
        with meta_path.open("w") as f:
            json.dump(metadata, f, indent=2)

        if self.backend == "wandb":
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass
        elif self.backend == "mlflow":
            try:
                import mlflow

                mlflow.end_run()
            except Exception:
                pass


def _get_git_commit() -> str:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"
