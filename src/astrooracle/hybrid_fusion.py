from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .ml.features import build_feature_matrix


@dataclass(frozen=True)
class HybridFusionResult:
    fused_score: np.ndarray
    components: Dict[str, np.ndarray]
    meta: Dict[str, float]


def _normalize01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = float(np.nanmin(x)) if x.size else 0.0
    hi = float(np.nanmax(x)) if x.size else 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < eps:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo + eps)


def compute_hybrid_fused_scores(
    df: pd.DataFrame,
    *,
    w_base: float = 0.50,
    w_iforest: float = 0.30,
    w_lof: float = 0.20,
    seed: int = 7,
) -> HybridFusionResult:
    """Fuse upstream anomaly score with ML outlier detectors.

    - Base: normalized anomaly_score from upstream
    - IsolationForest: anomaly score from feature matrix
    - LocalOutlierFactor: novelty-like score using negative_outlier_factor_

    Output score is normalized to [0,1].

    This does not change the model-based acquisition component. It is meant
    to improve the *anomaly* axis (signal) in heterogeneous pipelines.
    """

    if df.empty:
        return HybridFusionResult(
            fused_score=np.array([], dtype=float),
            components={"base": np.array([], dtype=float), "iforest": np.array([], dtype=float), "lof": np.array([], dtype=float)},
            meta={"w_base": float(w_base), "w_iforest": float(w_iforest), "w_lof": float(w_lof)},
        )

    base = pd.to_numeric(df.get("anomaly_score"), errors="coerce").to_numpy(float)
    base = _normalize01(base)

    X, _, _ = build_feature_matrix(df)

    # IsolationForest
    try:
        from sklearn.ensemble import IsolationForest

        iforest = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=int(seed),
        )
        iforest.fit(X)
        # Higher = more anomalous.
        if_s = -iforest.score_samples(X)
        if_s = _normalize01(if_s)
    except Exception:
        if_s = np.zeros(len(df), dtype=float)

    # LocalOutlierFactor
    try:
        from sklearn.neighbors import LocalOutlierFactor

        n_neighbors = min(35, max(5, len(df) // 10))
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        lof.fit(X)
        # negative_outlier_factor_: lower is more anomalous.
        lof_s = -lof.negative_outlier_factor_
        lof_s = _normalize01(lof_s)
    except Exception:
        lof_s = np.zeros(len(df), dtype=float)

    wsum = float(w_base + w_iforest + w_lof)
    if wsum <= 0:
        wsum = 1.0

    fused = (w_base * base + w_iforest * if_s + w_lof * lof_s) / wsum
    fused = _normalize01(fused)

    return HybridFusionResult(
        fused_score=fused,
        components={"base": base, "iforest": if_s, "lof": lof_s},
        meta={"w_base": float(w_base), "w_iforest": float(w_iforest), "w_lof": float(w_lof), "seed": float(seed)},
    )


def apply_hybrid_mode(df: pd.DataFrame, *, overwrite_anomaly_score: bool = True) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Return a copy of df with an extra fused score and optionally override anomaly_score."""

    res = compute_hybrid_fused_scores(df)
    out = df.copy()
    out["hybrid_fused_score"] = res.fused_score
    if overwrite_anomaly_score:
        out["anomaly_score"] = res.fused_score
    meta = {"hybrid_w_base": float(res.meta["w_base"]), "hybrid_w_iforest": float(res.meta["w_iforest"]), "hybrid_w_lof": float(res.meta["w_lof"])}
    return out, meta
