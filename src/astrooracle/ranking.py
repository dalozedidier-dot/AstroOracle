from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
except Exception:  # pragma: no cover
    IsolationForest = None  # type: ignore[assignment]


from .acquisition import acquire, AcquisitionResult
from .config import OracleConfig
from .diversity import dpp_greedy, kcenter_greedy
from .ml.features import build_feature_matrix
from .ml.model import EnsembleModel


def _normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < eps:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo + eps)


def prior_score(df: pd.DataFrame) -> np.ndarray:
    s = np.zeros(len(df), dtype=float)

    if "snr" in df.columns:
        snr = pd.to_numeric(df["snr"], errors="coerce")
        snr = snr.fillna(float(snr.median()) if len(snr) else 0.0).to_numpy(float)
        s += _normalize(snr)
    if "ruwe" in df.columns:
        ruwe = pd.to_numeric(df["ruwe"], errors="coerce")
        ruwe = ruwe.fillna(float(ruwe.median()) if len(ruwe) else 1.0).to_numpy(float)
        s -= _normalize(ruwe)
    if "mag" in df.columns:
        mag = pd.to_numeric(df["mag"], errors="coerce")
        mag = mag.fillna(float(mag.median()) if len(mag) else 20.0).to_numpy(float)
        s += _normalize(-mag)

    return _normalize(s)



def artifact_score(df: pd.DataFrame) -> np.ndarray:
    # Higher = more artefact-like (heuristic)
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=float)

    def col(name: str, default: float = 0.0) -> np.ndarray:
        if name not in df.columns:
            return np.full(n, default, dtype=float)
        v = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        med = float(np.nanmedian(v)) if np.isfinite(v).any() else default
        return np.where(np.isfinite(v), v, med)

    spike = col("feat_spike_max", 1.0)
    circ = col("feat_circularity_min", 1.0)
    snr = col("feat_snr_max", 0.0)

    raw = _normalize(spike) + _normalize(1.0 - np.clip(circ, 0.0, 1.0)) + _normalize(-snr)
    return _normalize(raw)


def known_score(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=float)

    def flag(name: str) -> np.ndarray:
        if name not in df.columns:
            return np.zeros(n, dtype=float)
        v = pd.to_numeric(df[name], errors="coerce").fillna(0).to_numpy(dtype=float)
        return (v > 0).astype(float)

    raw = np.maximum(flag("gaia_match"), flag("simbad_match"))
    return _normalize(raw)


def iforest_score(X: np.ndarray, seed: int = 7) -> np.ndarray:
    if IsolationForest is None:
        return np.zeros(X.shape[0], dtype=float)
    try:
        iso = IsolationForest(n_estimators=200, random_state=seed, contamination="auto")
        iso.fit(X)
        # decision_function: higher = more normal; invert
        s = -iso.decision_function(X)
        return _normalize(s)
    except Exception:
        return np.zeros(X.shape[0], dtype=float)

def _heuristic_acquisition(df: pd.DataFrame) -> AcquisitionResult:
    # Offline / no-model acquisition: prioritize around the median anomaly score
    med = float(pd.to_numeric(df["anomaly_score"], errors="coerce").median())
    unc = np.abs(df["anomaly_score"].to_numpy(float) - med)
    return AcquisitionResult(score=unc, components={"median_distance": unc})


def rank_candidates(
    df: pd.DataFrame,
    cfg: OracleConfig,
    model: Optional[EnsembleModel] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = df.copy()
    n = len(df)
    if n == 0:
        return df, {"n": 0}

    anomaly = _normalize(df["anomaly_score"].to_numpy(float))

    X, _, emb = build_feature_matrix(df)
    probs = None
    probs_mc = None

    if model is not None:
        probs_mc = model.predict_proba_mc(X)
        probs = probs_mc.mean(axis=0)
        df["p_max"] = probs.max(axis=1)
        df["y_hat"] = probs.argmax(axis=1)

    if model is None:
        acq_res = _heuristic_acquisition(df)
    else:
        acq_res = acquire(
            probs=probs,
            embeddings=emb,
            strategy=cfg.ranking.strategy,
            probs_mc=probs_mc,
        )

    acq_norm = _normalize(acq_res.score)
    pr = prior_score(df)

    div_proxy = np.ones(n, dtype=float)
    if emb is not None:
        div_proxy = _normalize(np.linalg.norm(emb - emb.mean(axis=0, keepdims=True), axis=1))

    df["score_anomaly"] = anomaly
    df["score_acq"] = acq_norm
    df["score_prior"] = pr
    df["score_div_proxy"] = div_proxy

    w = cfg.ranking

    art = artifact_score(df)
    known = known_score(df)
    if w.w_iforest > 0:
        s_if = iforest_score(X)
    else:
        s_if = np.zeros(n, dtype=float)

    df["score_artifact"] = art
    df["score_known"] = known
    df["score_iforest"] = s_if

    df["rank_score"] = (
        w.w_anomaly * df["score_anomaly"]
        + w.w_acq * df["score_acq"]
        + w.w_prior * df["score_prior"]
        + w.w_div * df["score_div_proxy"]
        + w.w_iforest * df["score_iforest"]
        - w.w_artifact * df["score_artifact"]
        - w.w_known * df["score_known"]
    )

    metrics = {"n": int(n), "strategy": cfg.ranking.strategy, "diversity": cfg.ranking.diversity}
    return df.sort_values("rank_score", ascending=False).reset_index(drop=True), metrics


def select_batch(ranked: pd.DataFrame, cfg: OracleConfig, k: int) -> pd.DataFrame:
    if ranked.empty:
        return ranked
    if k <= 0:
        return ranked.head(0)

    if cfg.ranking.diversity.lower() in {"none", "off"} or "embedding" not in ranked.columns:
        return ranked.head(k).reset_index(drop=True)

    try:
        emb = np.stack(ranked["embedding"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy())
    except Exception:
        return ranked.head(k).reset_index(drop=True)

    pool = min(len(ranked), max(k * 5, k))
    ranked_pool = ranked.head(pool).reset_index(drop=True)
    emb_pool = emb[:pool]

    if cfg.ranking.diversity.lower() in {"kcenter", "coreset", "core-set"}:
        idx = kcenter_greedy(emb_pool, k)
    elif cfg.ranking.diversity.lower() in {"dpp"}:
        idx = dpp_greedy(emb_pool, k)
    else:
        idx = list(range(min(k, len(ranked_pool))))

    return ranked_pool.iloc[idx].reset_index(drop=True)
