from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from PIL import Image

from .config import OracleConfig
from .core import fetch_cutouts


def _slug(s: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(s))


def _save_png(array2d: np.ndarray, path: Path) -> None:
    """Save a 2D array as an 8-bit PNG with robust contrast."""

    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(array2d, dtype=float)
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        norm = np.zeros_like(arr, dtype=np.uint8)
    else:
        lo = float(np.percentile(finite, 1))
        hi = float(np.percentile(finite, 99))
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        norm = (norm * 255.0).astype(np.uint8)

    Image.fromarray(norm).save(path)


def generate_batch_html(cfg: OracleConfig, candidates_df: pd.DataFrame, out_dir: Path) -> None:
    """Generate a fully static HTML report.

    Output layout:
      - index.html (static UI)
      - candidates.json (data for the UI)
      - cutouts/*.png
      - viz3d_globe.html (best effort)

    The UI stores annotations locally (localStorage). Use the export buttons to
    download JSON or CSV.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    cut_dir = out_dir / "cutouts"
    cut_dir.mkdir(exist_ok=True)

    recs: List[Dict[str, Any]] = []

    for _, r in candidates_df.iterrows():
        ra = float(r["ra"])
        dec = float(r["dec"])
        cid = r["id"]

        cutouts = fetch_cutouts(ra, dec, cfg)

        surveys: List[str] = []
        cutout_map: Dict[str, str] = {}

        for survey, data in cutouts:
            surveys.append(str(survey))
            fname = f"{_slug(cid)}_{_slug(survey)}.png"
            png_path = cut_dir / fname
            _save_png(data, png_path)
            cutout_map[str(survey)] = f"cutouts/{fname}"

        recs.append(
            {
                "id": str(cid),
                "ra": ra,
                "dec": dec,
                "score": float(r["anomaly_score"]),
                "surveys": surveys,
                "cutouts": cutout_map,
            }
        )

    (out_dir / "candidates.json").write_text(
        json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Static UI page
    template = Path(__file__).resolve().parents[2] / "templates" / "batch_index.html"
    (out_dir / "index.html").write_text(template.read_text(encoding="utf-8"), encoding="utf-8")

    # Best-effort 3D HTML (optional)
    try:
        from .viz3d import build_viz3d_figure, write_viz3d_html

        fig = build_viz3d_figure(
            candidates_df,
            mode="globe",
            max_points=5000,
            cutouts_dir=cut_dir,
            embed_cutouts=False,
            title="AstroOracle batch (globe)",
        )
        write_viz3d_html(fig, out_dir / "viz3d_globe.html", cdn=True)
    except Exception:
        # Plotly may be missing in minimal installs. The report still works.
        pass
