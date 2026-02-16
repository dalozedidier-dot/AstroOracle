from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .config import OracleConfig
from .core import fetch_cutouts


def _slug(s: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(s))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _write_candidates_csv(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "ra", "dec", "score", "label"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "id": r.get("id", ""),
                    "ra": r.get("ra", ""),
                    "dec": r.get("dec", ""),
                    "score": r.get("score", ""),
                    "label": "",
                }
            )


def _collect_files(out_dir: Path) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(out_dir).as_posix()
        # Skip huge or irrelevant caches if any appear
        if rel.startswith(".") or rel.endswith(".pyc"):
            continue
        files.append(
            {
                "path": rel,
                "bytes": int(p.stat().st_size),
                "sha256": _sha256_file(p),
            }
        )
    return files


def generate_batch_html(cfg: OracleConfig, candidates_df: pd.DataFrame, out_dir: Path) -> None:
    """Generate a fully static HTML report.

    Output layout (minimum):
    - index.html (static UI)
    - candidates.json (data for the UI)

    Optional:
    - cutouts/*.png (if cutouts are generated and saved)
    - report_meta.json (metadata for the report)
    - manifest.json (file list + sha256 for auditing)
    - viz3d_globe.html / viz3d_scatter.html (best effort, requires plotly)
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    cut_dir = out_dir / "cutouts"
    cut_dir.mkdir(exist_ok=True)

    # Normalize required columns
    required = {"id", "ra", "dec", "anomaly_score"}
    missing = required - set(candidates_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Build candidate records and optionally cutouts
    recs: List[Dict[str, Any]] = []
    surveys_seen: set[str] = set()

    for _, r in candidates_df.iterrows():
        ra = float(r["ra"])
        dec = float(r["dec"])
        cid = str(r["id"])
        score = float(r["anomaly_score"])

        cutouts = fetch_cutouts(ra, dec, cfg)

        surveys: List[str] = []
        cutout_map: Dict[str, str] = {}

        for survey, data in cutouts:
            survey_s = str(survey)
            surveys.append(survey_s)
            surveys_seen.add(survey_s)

            fname = f"{_slug(cid)}_{_slug(survey_s)}.png"
            png_path = cut_dir / fname
            try:
                _save_png(data, png_path)
                cutout_map[survey_s] = f"cutouts/{fname}"
            except Exception:
                # If PNG saving fails for any reason, keep the record without breaking the report
                continue

        recs.append(
            {
                "id": cid,
                "ra": ra,
                "dec": dec,
                "score": score,
                "anomaly_score": score,
                "surveys": surveys,
                "cutouts": cutout_map,
            }
        )

    (out_dir / "candidates.json").write_text(
        json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_candidates_csv(out_dir / "candidates.csv", recs)

    report_meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_candidates": int(len(recs)),
        "surveys": sorted(surveys_seen),
        "offline": bool(getattr(cfg, "offline", False)),
        "cutouts_dir": "cutouts/",
    }
    (out_dir / "report_meta.json").write_text(
        json.dumps(report_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Static UI page
    template = Path(__file__).resolve().parents[2] / "templates" / "batch_index.html"
    (out_dir / "index.html").write_text(
        template.read_text(encoding="utf-8"), encoding="utf-8"
    )

    # Best-effort 3D HTML exports
    try:
        from .viz3d import build_viz3d_figure, write_viz3d_html

        fig_globe = build_viz3d_figure(
            candidates_df,
            mode="globe",
            max_points=5000,
            cutouts_dir=cut_dir,
            embed_cutouts=False,
            title="AstroOracle batch (globe)",
        )
        write_viz3d_html(fig_globe, out_dir / "viz3d_globe.html", cdn=True)

        fig_scatter = build_viz3d_figure(
            candidates_df,
            mode="scatter",
            max_points=5000,
            cutouts_dir=cut_dir,
            embed_cutouts=False,
            title="AstroOracle batch (scatter)",
        )
        write_viz3d_html(fig_scatter, out_dir / "viz3d_scatter.html", cdn=True)
    except Exception:
        # Plotly may be missing. The report still works.
        pass

    # Manifest (audit)
    try:
        manifest = {
            "version": 1,
            "generated_at": report_meta["generated_at"],
            "files": _collect_files(out_dir),
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass
