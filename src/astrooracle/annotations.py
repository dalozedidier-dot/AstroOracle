from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import OracleConfig


def append_annotations(cfg: OracleConfig, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    cfg.annot_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not cfg.annot_path.exists()
    df_new.to_csv(cfg.annot_path, mode="a", header=write_header, index=False)


def _parse_embedding_cell(x: Any) -> Optional[np.ndarray]:
    if pd.isna(x):
        return None
    if isinstance(x, str) and x.strip():
        try:
            return np.array(json.loads(x), dtype=float)
        except Exception:
            return None
    return None


def read_annotations(cfg: OracleConfig) -> pd.DataFrame:
    if not cfg.annot_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(cfg.annot_path)
    if "embedding" in df.columns and not df.empty:
        df["embedding_vec"] = df["embedding"].map(_parse_embedding_cell)
    return df


def count_labels(cfg: OracleConfig) -> int:
    if not cfg.annot_path.exists():
        return 0
    with cfg.annot_path.open("r", encoding="utf-8") as f:
        n = sum(1 for _ in f)
    return max(n - 1, 0)


def get_retrain_cursor(cfg: OracleConfig) -> int:
    cursor_path = cfg.annot_path.with_suffix(".cursor.json")
    if not cursor_path.exists():
        return 0
    try:
        return int(json.loads(cursor_path.read_text(encoding="utf-8")).get("rows", 0))
    except Exception:
        return 0


def set_retrain_cursor(cfg: OracleConfig, rows: int) -> None:
    cursor_path = cfg.annot_path.with_suffix(".cursor.json")
    cursor_path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
