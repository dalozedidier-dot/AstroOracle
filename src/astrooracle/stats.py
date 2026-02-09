from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .config import OracleConfig

try:
    import duckdb
except Exception:  # pragma: no cover
    duckdb = None  # type: ignore


def annotation_stats(cfg: OracleConfig) -> Dict[str, Any]:
    if not cfg.annot_path.exists():
        return {"n": 0}

    df = pd.read_csv(cfg.annot_path)
    out: Dict[str, Any] = {"n": int(len(df))}
    if "label" in df.columns:
        out["label_counts"] = df["label"].value_counts(dropna=False).to_dict()
    else:
        out["label_counts"] = {}
    return out


def log_stats(cfg: OracleConfig) -> Dict[str, Any]:
    if not cfg.log_path.exists():
        return {"n": 0}

    if duckdb is not None:
        con = duckdb.connect(database=":memory:")
        con.execute("CREATE TABLE logs AS SELECT * FROM read_json_auto(?)", [str(cfg.log_path)])
        n = int(con.execute("SELECT COUNT(*) FROM logs").fetchone()[0])
        by_event = con.execute(
            "SELECT event, COUNT(*) AS n FROM logs GROUP BY event ORDER BY n DESC"
        ).fetchall()
        return {"n": n, "by_event": [{"event": str(e), "n": int(k)} for e, k in by_event]}

    # Fallback: ndjson via pandas
    try:
        df = pd.read_json(cfg.log_path, lines=True)
    except Exception:
        return {"n": 0, "by_event": []}

    if "event" in df.columns:
        vc = df["event"].value_counts(dropna=False)
        by_event = [{"event": str(ev), "n": int(cnt)} for ev, cnt in vc.items()]
    else:
        by_event = []
    return {"n": int(len(df)), "by_event": by_event}
