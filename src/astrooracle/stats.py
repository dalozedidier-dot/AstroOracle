from __future__ import annotations

from typing import Dict

import pandas as pd

from .config import OracleConfig

try:
    import duckdb  # type: ignore[import-not-found]

    _HAS_DUCKDB = True
except Exception:  # pragma: no cover
    duckdb = None  # type: ignore[assignment]
    _HAS_DUCKDB = False


def annotation_stats(cfg: OracleConfig) -> Dict[str, object]:
    if not cfg.annot_path.exists():
        return {"n": 0}

    df = pd.read_csv(cfg.annot_path)
    out: Dict[str, object] = {
        "n": int(len(df)),
        "label_counts": df["label"].value_counts(dropna=False).to_dict()
        if "label" in df.columns
        else {},
    }
    return out


def _log_stats_pandas(cfg: OracleConfig) -> Dict[str, object]:
    try:
        df = pd.read_json(cfg.log_path, lines=True)
    except ValueError:
        # Empty file or invalid JSON.
        return {"n": 0, "by_event": []}

    if df.empty or "event" not in df.columns:
        return {"n": int(len(df)), "by_event": []}

    counts = df["event"].value_counts(dropna=False)
    by_event = [{"event": str(ev), "n": int(n)} for ev, n in counts.items()]
    return {"n": int(len(df)), "by_event": by_event}


def log_stats(cfg: OracleConfig) -> Dict[str, object]:
    if not cfg.log_path.exists():
        return {"n": 0}

    if not _HAS_DUCKDB:
        return _log_stats_pandas(cfg)

    try:
        con = duckdb.connect(database=":memory:")
        con.execute("CREATE TABLE logs AS SELECT * FROM read_json_auto(?)", [str(cfg.log_path)])
        n = int(con.execute("SELECT COUNT(*) FROM logs").fetchone()[0])
        by_event = con.execute(
            "SELECT event, COUNT(*) AS n FROM logs GROUP BY event ORDER BY n DESC"
        ).fetchall()
        return {"n": n, "by_event": [{"event": e, "n": int(k)} for e, k in by_event]}
    except Exception:
        return _log_stats_pandas(cfg)
