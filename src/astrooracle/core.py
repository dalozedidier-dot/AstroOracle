from __future__ import annotations

from typing import List, Tuple

import os
import hashlib

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.skyview import SkyView
from sklearn.metrics import pairwise_distances

from .config import OracleConfig
from .annotations import read_annotations

REQUIRED_CAND_COLS = {"id", "ra", "dec", "anomaly_score"}


def load_candidates(cfg: OracleConfig) -> pd.DataFrame:
    if not cfg.candidates_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(cfg.candidates_path)
    missing = REQUIRED_CAND_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in candidates: {missing}")
    return df


def select_candidates(df: pd.DataFrame, cfg: OracleConfig) -> pd.DataFrame:
    df = df.copy()
    median_score = float(df["anomaly_score"].median())
    df["unc"] = np.abs(df["anomaly_score"] - median_score)

    diversity = np.ones(len(df), dtype=float)
    if "embedding" in df.columns and cfg.annot_path.exists():
        annot = read_annotations(cfg)
        if not annot.empty and "embedding_vec" in annot.columns:
            ann_vecs = [v for v in annot["embedding_vec"].tolist() if isinstance(v, np.ndarray)]
            if ann_vecs:
                emb_all = np.stack(df["embedding"].values)
                emb_ann = np.stack(ann_vecs)
                dists = pairwise_distances(emb_all, emb_ann, metric="cosine")
                diversity = dists.min(axis=1)
                denom = (diversity.max() - diversity.min()) + 1e-8
                diversity = (diversity - diversity.min()) / denom

    unc_norm = df["unc"].to_numpy(dtype=float)
    unc_norm = unc_norm / (float(unc_norm.max()) + 1e-8)

    df["query_score"] = 0.6 * unc_norm + 0.4 * diversity
    return df.nlargest(cfg.n_query, "query_score").reset_index(drop=True)


def fetch_cutouts(ra_deg: float, dec_deg: float, cfg: OracleConfig) -> List[Tuple[str, np.ndarray]]:
    offline = bool(getattr(cfg, "offline", False)) or os.environ.get("ASTROORACLE_OFFLINE", "") in {"1", "true", "yes"}
    results: List[Tuple[str, np.ndarray]] = []

    if offline:
        # Deterministic synthetic cutouts for CI/offline demos.
        # Generates a smooth Gaussian blob + noise, seeded by (ra, dec, survey).
        for survey in cfg.surveys:
            key = f"{ra_deg:.6f}|{dec_deg:.6f}|{survey}"
            seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)

            n = int(cfg.pixels)
            yy, xx = np.mgrid[0:n, 0:n]
            cx = rng.uniform(0.35 * n, 0.65 * n)
            cy = rng.uniform(0.35 * n, 0.65 * n)
            sx = rng.uniform(0.06 * n, 0.12 * n)
            sy = rng.uniform(0.06 * n, 0.12 * n)

            blob = np.exp(-(((xx - cx) ** 2) / (2 * sx**2) + ((yy - cy) ** 2) / (2 * sy**2)))
            noise = rng.normal(0, 0.05, size=(n, n))
            bg = rng.normal(0, 0.01, size=(n, n))
            img = (0.8 * blob + noise + bg).astype(float)
            results.append((survey, img))
        return results

    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    for survey in cfg.surveys:
        try:
            images = SkyView.get_images(
                position=coord,
                survey=[survey],
                pixels=cfg.pixels,
                radius=cfg.cutout_radius,
            )
            if not images:
                continue
            data = images[0][0].data
            if data is None:
                continue
            results.append((survey, data))
        except Exception:
            continue
    return results
