from __future__ import annotations

from typing import List, Tuple

import os
import hashlib

import numpy as np
import pandas as pd

from .config import OracleConfig
from .model_io import load_model
from .ranking import rank_candidates, select_batch

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
    model = load_model(cfg.model_path)
    ranked, _ = rank_candidates(df, cfg, model=model)
    return select_batch(ranked, cfg, k=cfg.n_query)


def fetch_cutouts(ra_deg: float, dec_deg: float, cfg: OracleConfig) -> List[Tuple[str, np.ndarray]]:
    offline = bool(getattr(cfg, "offline", False)) or os.environ.get("ASTROORACLE_OFFLINE", "") in {"1", "true", "yes"}
    results: List[Tuple[str, np.ndarray]] = []

    if offline:
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

    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.skyview import SkyView

    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    for survey in cfg.surveys:
        try:
            images = SkyView.get_images(
                position=coord,
                survey=[survey],
                pixels=cfg.pixels,
                radius=cfg.cutout_radius_arcmin * u.arcmin,
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
