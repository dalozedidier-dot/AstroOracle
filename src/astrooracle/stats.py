from __future__ import annotations

from typing import Dict

import pandas as pd

try:
    # duckdb est un extra optionnel (.[stats]) : ne doit pas casser l'import du package
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None  # type: ignore[assignment]

from .config import OracleConfig


def annotation_stats(cfg: OracleConfig) -> Dict[str, object]:
    if not cfg.annot_path.exists():
        return {"n": 0}
    df = pd.read_csv(cfg.annot_path)
    out = {
        "n": int(len(df)),
        "label_counts": df["label"].value_counts(dropna=False).to_dict() if "label" in df.columns else {},
    }
    return out


def _log_stats_fallback(cfg: OracleConfig) -> Dict[str, object]:
    # JSON Lines / NDJSON
    df = pd.read_json(cfg.log_path, lines=True)
    n = int(len(df))
    if "event" not in df.columns:
        return {"n": n, "by_event": []}

    vc = df["event"].value_counts(dropna=False)
    by_event = []
    for event, k in vc.items():
        if pd.isna(event):
            e_out = None
        else:
            e_out = str(event)
        by_event.append({"event": e_out, "n": int(k)})
    return {"n": n, "by_event": by_event}


def log_stats(cfg: OracleConfig) -> Dict[str, object]:
    if not cfg.log_path.exists():
        return {"n": 0}

    if duckdb is None:
        return _log_stats_fallback(cfg)

    con = duckdb.connect(database=":memory:")
    con.execute("CREATE TABLE logs AS SELECT * FROM read_json_auto(?)", [str(cfg.log_path)])
    n = int(con.execute("SELECT COUNT(*) FROM logs").fetchone()[0])
    by_event = con.execute("SELECT event, COUNT(*) AS n FROM logs GROUP BY event ORDER BY n DESC").fetchall()
    return {"n": n, "by_event": [{"event": e, "n": int(k)} for e, k in by_event]}
