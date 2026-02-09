from __future__ import annotations

from typing import Dict

import pandas as pd

try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None  # type: ignore

from .config import OracleConfig


def annotation_stats(cfg: OracleConfig) -> Dict[str, object]:
    if not cfg.annot_path.exists():
        return {"n": 0}
    df = pd.read_csv(cfg.annot_path)
    out = {
        "n": int(len(df)),
        "label_counts": (
            df["label"].value_counts(dropna=False).to_dict() if "label" in df.columns else {}
        ),
    }
    return out


def log_stats(cfg: OracleConfig) -> Dict[str, object]:
    if not cfg.log_path.exists():
        return {"n": 0, "by_event": []}

    if duckdb is None:
        try:
            df = pd.read_json(cfg.log_path, lines=True)
        except Exception:
            df = pd.DataFrame()

        if df.empty:
            return {"n": 0, "by_event": []}

        if "event" in df.columns:
            vc = df["event"].value_counts(dropna=False)
            by_event = [{"event": str(e), "n": int(k)} for e, k in vc.items()]
        else:
            by_event = []

        return {"n": int(len(df)), "by_event": by_event}

    con = duckdb.connect(database=":memory:")
    con.execute("CREATE TABLE logs AS SELECT * FROM read_json_auto(?)", [str(cfg.log_path)])
    n = int(con.execute("SELECT COUNT(*) FROM logs").fetchone()[0])
    by_event = con.execute(
        "SELECT event, COUNT(*) AS n FROM logs GROUP BY event ORDER BY n DESC"
    ).fetchall()
    return {"n": n, "by_event": [{"event": e, "n": int(k)} for e, k in by_event]}
