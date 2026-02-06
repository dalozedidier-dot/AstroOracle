from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .config import OracleConfig


def utcnow_iso() -> str:
    return datetime.utcnow().isoformat()


def get_file_hash(path: Path) -> str:
    if not path.exists():
        return "no_file"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def log_event(cfg: OracleConfig, event: Dict[str, Any]) -> None:
    payload = {
        **event,
        "timestamp": utcnow_iso(),
        "model_version": get_file_hash(cfg.model_path),
        "candidates_hash": get_file_hash(cfg.candidates_path),
    }
    cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
