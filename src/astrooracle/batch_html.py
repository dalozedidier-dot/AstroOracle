from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .config import OracleConfig
from .core import fetch_cutouts


def _slug(s: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(s))


def _save_png(array2d: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array2d, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        norm = np.zeros_like(arr, dtype=np.uint8)
    else:
        lo = np.percentile(finite, 1)
        hi = np.percentile(finite, 99)
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((arr - lo) / (hi - lo), 0, 1)
        norm = (norm * 255).astype(np.uint8)
    Image.fromarray(norm).save(path)


def generate_batch_html(cfg: OracleConfig, candidates_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cut_dir = out_dir / "cutouts"
    cut_dir.mkdir(exist_ok=True)

    recs: list[dict[str, Any]] = []
    for _, r in candidates_df.iterrows():
        ra = float(r["ra"])
        dec = float(r["dec"])
        cid = r["id"]
        cutouts = fetch_cutouts(ra, dec, cfg)

        surveys = []
        for survey, data in cutouts:
            surveys.append(survey)
            png = cut_dir / f"{_slug(cid)}_{_slug(survey)}.png"
            _save_png(data, png)

        recs.append(
            {
                "id": str(cid),
                "ra": ra,
                "dec": dec,
                "score": float(r["anomaly_score"]),
                "surveys": surveys,
            }
        )

    (out_dir / "candidates.json").write_text(json.dumps(recs, indent=2), encoding="utf-8")

    template = Path(__file__).resolve().parents[2] / "templates" / "batch_index.html"
    (out_dir / "index.html").write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
