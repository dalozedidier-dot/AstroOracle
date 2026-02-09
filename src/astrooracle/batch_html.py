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
from .crossmatch import crossmatch_all
from .image_features import aggregate_candidate_features, compute_cutout_features


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

    cache_path = out_dir / "crossmatch_cache.json"
    cache: Dict[str, Any] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    recs: List[Dict[str, Any]] = []
    for _, r in candidates_df.iterrows():
        ra = float(r["ra"])
        dec = float(r["dec"])
        cid = r["id"]
        cid_s = str(cid)

        cutouts = fetch_cutouts(ra, dec, cfg)
        surveys = []
        by_survey = {}
        for survey, data in cutouts:
            surveys.append(survey)
            png = cut_dir / f"{_slug(cid)}_{_slug(survey)}.png"
            _save_png(data, png)
            by_survey[str(survey)] = compute_cutout_features(data)

        feats = aggregate_candidate_features(by_survey)

        # Crossmatch (cached)
        if cid_s in cache:
            cm = cache[cid_s]
        else:
            cm = crossmatch_all(ra, dec, radius_arcsec=5.0, neighbor_limit=25)
            cache[cid_s] = cm

        # Make match flags numeric-friendly
        if "gaia_match" in cm:
            cm["gaia_match"] = int(bool(cm["gaia_match"]))
        if "simbad_match" in cm:
            cm["simbad_match"] = int(bool(cm["simbad_match"]))

        recs.append(
            {
                "id": cid_s,
                "ra": ra,
                "dec": dec,
                "score": float(r.get("rank_score", r.get("anomaly_score", float("nan")))),
                "anomaly_score": float(r.get("anomaly_score", float("nan"))),
                "surveys": surveys,
                "features": feats,
                **cm,
            }
        )

    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")

    # Keep candidates.json for audit/export, but also embed for offline use.
    (out_dir / "candidates.json").write_text(json.dumps(recs, indent=2), encoding="utf-8")

    template = Path(__file__).resolve().parents[2] / "templates" / "batch_index.html"
    html = template.read_text(encoding="utf-8")

    embedded = json.dumps(
        {
            "candidates": recs,
            "pixels": int(cfg.pixels),
            "radius_arcsec": float(cfg.cutout_radius_arcmin) * 60.0,
        },
        ensure_ascii=False,
    )
    marker = "<!--__EMBEDDED_DATA__-->"
    if marker in html:
        html = html.replace(marker, f'<script type="application/json" id="embedded-data">{embedded}</script>')
    else:
        html = html.replace("</body>", f'<script type="application/json" id="embedded-data">{embedded}</script></body>')

    (out_dir / "index.html").write_text(html, encoding="utf-8")
