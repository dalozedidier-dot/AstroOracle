from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


NUMERIC_PRIOR_COLS = [
    "mag",
    "snr",
    "ruwe",
    "anomaly_score",
    "feat_snr_max",
    "feat_snr_min",
    "feat_circularity_min",
    "feat_spike_max",
    "feat_color_dss2red_minus_2massj",
    "gaia_match",
    "simbad_match",
    "nearest_gaia_dist_arcsec",
    "gaia_parallax_mas",
    "gaia_pmra_masyr",
    "gaia_pmdec_masyr",
    "gaia_gmag",
]


def _safe_numeric(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    out = []
    for c in cols:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        else:
            v = np.full(len(df), np.nan, dtype=float)
        out.append(v)
    X = np.stack(out, axis=1)
    # Replace nan with column median
    for j in range(X.shape[1]):
        col = X[:, j]
        med = np.nanmedian(col) if np.isfinite(col).any() else 0.0
        col = np.where(np.isfinite(col), col, med)
        X[:, j] = col
    return X

def build_feature_matrix(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, List[str], Optional[np.ndarray]]:
    prior = _safe_numeric(df, NUMERIC_PRIOR_COLS)
    names = [f"prior_{c}" for c in NUMERIC_PRIOR_COLS]

    emb: Optional[np.ndarray] = None
    if "embedding" in df.columns:
        try:
            emb = np.stack(
                df["embedding"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy()
            )
        except Exception:
            emb = None

    if emb is None:
        return prior, names, None

    X = np.concatenate([prior, emb], axis=1)
    names = names + [f"emb_{i}" for i in range(emb.shape[1])]
    return X, names, emb
