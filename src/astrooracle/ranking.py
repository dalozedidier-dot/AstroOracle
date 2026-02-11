from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

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
            probs=probs, embeddings=emb, strategy=cfg.ranking.strategy, probs_mc=probs_mc
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
    df["rank_score"] = (
        w.w_anomaly * df["score_anomaly"]
        + w.w_acq * df["score_acq"]
        + w.w_prior * df["score_prior"]
        + w.w_div * df["score_div_proxy"]
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
