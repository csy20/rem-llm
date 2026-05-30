"""DatasetCatalog — versioned registry of all training datasets."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from remllm.logging import get_logger


class DatasetCatalog:
    """Tracks all domain datasets, their schemas, versions, and provenance.

    Stores metadata in a JSON file (default: data/catalog.json).
    Each dataset entry includes: name, domain, path, sha256, row count,
    schema fields, created/updated timestamps, provenance info.
    """

    def __init__(self, catalog_path: Path | str = "data/catalog.json"):
        self.catalog_path = Path(catalog_path)
        self.entries: dict[str, dict[str, Any]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.catalog_path.exists():
            try:
                with self.catalog_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self.entries = data.get("datasets", {})
            except (json.JSONDecodeError, OSError):
                self.entries = {}

    def register(
        self,
        name: str,
        path: Path | str,
        domain: str = "general",
        schema_fields: list[str] | None = None,
        provenance: dict[str, str] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Register a dataset and return its version hash."""
        log = get_logger(operation="catalog_register", name=name, domain=domain)
        self._ensure_loaded()

        path = Path(path)
        sha = _sha256_file(path) if path.exists() else "missing"
        row_count = _count_rows(path) if path.exists() else 0
        if schema_fields is None:
            schema_fields = _infer_schema(path)

        now = datetime.now(timezone.utc).isoformat()
        version = sha[:12] if sha != "missing" else "unknown"

        entry = {
            "name": name,
            "domain": domain,
            "path": str(path),
            "sha256": sha,
            "version": version,
            "row_count": row_count,
            "schema_fields": schema_fields,
            "created": now,
            "updated": now,
            "provenance": provenance or {},
            "tags": tags or [],
        }

        self.entries[name] = entry
        self._save()
        log.info("dataset_registered", version=version, rows=row_count)
        return version

    def lookup(self, name: str) -> dict[str, Any] | None:
        """Look up a dataset by name."""
        self._ensure_loaded()
        return self.entries.get(name)

    def list_datasets(
        self, domain: str | None = None, tag: str | None = None
    ) -> list[dict[str, Any]]:
        """List datasets, optionally filtered by domain or tag."""
        self._ensure_loaded()
        results = []
        for entry in self.entries.values():
            if domain is not None and entry.get("domain") != domain:
                continue
            if tag is not None and tag not in entry.get("tags", []):
                continue
            results.append(entry)
        return sorted(results, key=lambda e: e["name"])

    def delete(self, name: str) -> bool:
        """Remove a dataset from the catalog."""
        self._ensure_loaded()
        if name in self.entries:
            del self.entries[name]
            self._save()
            return True
        return False

    def check_consistency(self, name: str) -> dict[str, Any]:
        """Check if a dataset's file matches its catalog entry."""
        self._ensure_loaded()
        entry = self.entries.get(name)
        if not entry:
            return {"status": "missing", "name": name}

        path = Path(entry["path"])
        if not path.exists():
            return {"status": "file_missing", "name": name, "expected_path": str(path)}

        current_sha = _sha256_file(path)
        expected_sha = entry.get("sha256", "")
        if current_sha != expected_sha:
            return {
                "status": "modified",
                "name": name,
                "expected_sha256": expected_sha,
                "current_sha256": current_sha,
            }

        return {
            "status": "consistent",
            "name": name,
            "sha256": current_sha,
            "row_count": entry.get("row_count", 0),
        }

    def _save(self) -> None:
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "2.0",
            "updated": datetime.now(timezone.utc).isoformat(),
            "datasets": self.entries,
        }
        with self.catalog_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self.entries)

    def __contains__(self, name: str) -> bool:
        self._ensure_loaded()
        return name in self.entries


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_rows(path: Path) -> int:
    """Count JSONL lines."""
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _infer_schema(path: Path) -> list[str]:
    """Infer schema fields from the first row."""
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    return sorted(row.keys())
    except (json.JSONDecodeError, OSError):
        pass
    return []
