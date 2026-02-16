from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _guess_type(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    if ext in {"html", "htm"}:
        return "html"
    if ext in {"json", "jsonl"}:
        return "json"
    if ext in {"csv", "tsv"}:
        return "csv"
    if ext in {"png", "jpg", "jpeg", "webp", "gif"}:
        return "image"
    return "file"


def _build_manifest(
    cfg: OracleConfig,
    out_dir: Path,
    *,
    candidates_count: int,
    cutouts_count: int,
    surveys: List[str],
) -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []

    # Prefer a curated list first.
    curated = [
        ("index.html", "Index (rapport)"),
        ("candidates.json", "Candidates JSON"),
        ("candidates.csv", "Candidates CSV"),
        ("report_meta.json", "Report meta"),
        ("viz3d_globe.html", "Viz3D globe"),
        ("viz3d_scatter.html", "Viz3D scatter"),
    ]
    seen = set()
    for rel, title in curated:
        p = out_dir / rel
        if p.exists() and p.is_file():
            files.append(
                {
                    "path": rel,
                    "title": title,
                    "type": _guess_type(p),
                    "bytes": int(p.stat().st_size),
                    "sha256": _sha256(p),
                }
            )
            seen.add(rel)

    # Then add any other small top-level files.
    for p in sorted(out_dir.glob("*")):
        if not p.is_file():
            continue
        rel = p.name
        if rel in seen:
            continue
        # Avoid listing a huge amount of derived assets if you later add them.
        if p.stat().st_size > 25_000_000:
            continue
        files.append(
            {
                "path": rel,
                "title": rel,
                "type": _guess_type(p),
                "bytes": int(p.stat().st_size),
                "sha256": _sha256(p),
            }
        )

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "offline": bool(cfg.offline),
        "surveys": list(surveys),
        "candidates_count": int(candidates_count),
        "cutouts_count": int(cutouts_count),
        "viz3d_globe": "viz3d_globe.html",
        "files": files,
        "note": "manifest généré par AstroOracle batch_html",
    }


def generate_batch_html(cfg: OracleConfig, candidates_df: pd.DataFrame, out_dir: Path) -> None:
    """Generate a fully static HTML report.

    Output layout:
    - index.html (static UI)
    - manifest.json (report file index)
    - report_meta.json (small summary)
    - candidates.json (data for the UI)
    - candidates.csv (portable table)
    - cutouts/*.png
    - viz3d_globe.html (best effort)

    The UI stores annotations locally (localStorage). Use the export buttons to download JSON or CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cut_dir = out_dir / "cutouts"
    cut_dir.mkdir(exist_ok=True)

    surveys = list(cfg.surveys)
    recs: List[Dict[str, Any]] = []

    cutouts_total = 0
    for _, r in candidates_df.iterrows():
        ra = float(r["ra"])
        dec = float(r["dec"])
        cid = r["id"]
        cutouts = fetch_cutouts(ra, dec, cfg)

        surveys_seen: List[str] = []
        cutout_map: Dict[str, str] = {}
        for survey, data in cutouts:
            surveys_seen.append(str(survey))
            fname = f"{_slug(cid)}_{_slug(survey)}.png"
            png_path = cut_dir / fname
            _save_png(data, png_path)
            cutout_map[str(survey)] = f"cutouts/{fname}"
            cutouts_total += 1

        recs.append(
            {
                "id": str(cid),
                "ra": ra,
                "dec": dec,
                "score": float(r["anomaly_score"]),
                "surveys": surveys_seen,
                "cutouts": cutout_map,
            }
        )

    (out_dir / "candidates.json").write_text(
        json.dumps(recs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Portable table (no cutout paths).
    pd.DataFrame(
        [
            {
                "id": rec["id"],
                "ra": rec["ra"],
                "dec": rec["dec"],
                "anomaly_score": rec["score"],
            }
            for rec in recs
        ]
    ).to_csv(out_dir / "candidates.csv", index=False)

    # Minimal meta for humans and for UI badges.
    meta = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidates_count": len(recs),
        "offline": bool(cfg.offline),
        "surveys": surveys,
        "pixels": int(cfg.pixels),
        "cutout_radius_arcmin": float(cfg.cutout_radius_arcmin),
        "n_query": int(cfg.n_query),
        "ranking": {
            "strategy": cfg.ranking.strategy,
            "diversity": cfg.ranking.diversity,
            "w_anomaly": float(cfg.ranking.w_anomaly),
            "w_acq": float(cfg.ranking.w_acq),
            "w_div": float(cfg.ranking.w_div),
            "w_prior": float(cfg.ranking.w_prior),
            "acq_temperature": float(cfg.ranking.acq_temperature),
        },
        "model_path": str(cfg.model_path),
    }
    (out_dir / "report_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Static UI page.
    template = Path(__file__).resolve().parents[2] / "templates" / "batch_index.html"
    (out_dir / "index.html").write_text(
        template.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # Best-effort 3D HTML (optional).
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
        # Plotly may be missing in minimal installs.
        # The report still works.
        pass

    # Optional scatter page if plotly is available.
    try:
        from .viz3d import build_viz3d_figure, write_viz3d_html

        fig2 = build_viz3d_figure(
            candidates_df,
            mode="scatter",
            max_points=5000,
            cutouts_dir=cut_dir,
            embed_cutouts=False,
            title="AstroOracle batch (scatter)",
        )
        write_viz3d_html(fig2, out_dir / "viz3d_scatter.html", cdn=True)
    except Exception:
        pass

    manifest = _build_manifest(
        cfg,
        out_dir,
        candidates_count=len(recs),
        cutouts_count=cutouts_total,
        surveys=surveys,
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
