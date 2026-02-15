from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ChaosMetrics:
    lyapunov_proxy: float
    rqa_recurrence_rate: float
    rqa_determinism: float
    rqa_entropy: float
    score: float
    meta: Dict[str, float]


def _zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(x)
    return (x - mu) / (sd + eps)


def takens_embedding(series: np.ndarray, *, emb_dim: int = 3, emb_lag: int = 1) -> np.ndarray:
    """Takens embedding for a 1D time series.

    Returns an array of shape (n_vectors, emb_dim).
    """

    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < (emb_dim - 1) * emb_lag + 2:
        return np.empty((0, emb_dim), dtype=float)

    n_vec = x.size - (emb_dim - 1) * emb_lag
    out = np.empty((n_vec, emb_dim), dtype=float)
    for j in range(emb_dim):
        out[:, j] = x[j * emb_lag : j * emb_lag + n_vec]
    return out


def lyapunov_proxy(series: np.ndarray, *, emb_dim: int = 3, emb_lag: int = 1) -> float:
    """A lightweight Lyapunov proxy.

    Heuristic: in embedded space, pick for each point its nearest neighbor,
    then track the average log-divergence after one time step.

    This is not a strict Lyapunov exponent estimator. It is a cheap proxy
    suitable for ranking candidates.
    """

    emb = takens_embedding(series, emb_dim=emb_dim, emb_lag=emb_lag)
    if emb.shape[0] < 10:
        return 0.0

    # Standardize dimensions to avoid scale bias.
    emb = _zscore(emb)

    # Pairwise distances, excluding self.
    # O(n^2) but n is usually small for candidate time series.
    n = emb.shape[0]
    d = np.linalg.norm(emb[:, None, :] - emb[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = np.argmin(d, axis=1)

    # One-step divergence in embedded space.
    # Align i -> i+1 and nn(i) -> nn(i)+1 where possible.
    valid = (np.arange(n) + 1 < n) & (nn + 1 < n)
    if not np.any(valid):
        return 0.0

    i0 = np.where(valid)[0]
    i1 = i0 + 1
    j0 = nn[i0]
    j1 = j0 + 1

    d0 = np.linalg.norm(emb[i0] - emb[j0], axis=1)
    d1 = np.linalg.norm(emb[i1] - emb[j1], axis=1)

    eps = 1e-12
    ratio = (d1 + eps) / (d0 + eps)
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if ratio.size == 0:
        return 0.0

    return float(np.mean(np.log(ratio)))


def rqa_light(series: np.ndarray, *, emb_dim: int = 3, emb_lag: int = 1, eps_quantile: float = 0.10) -> Tuple[float, float, float]:
    """Very small RQA-like summary: (RR, DET, ENT).

    - RR: recurrence rate
    - DET: fraction of recurrence points that belong to diagonal lines of length >= 2
    - ENT: entropy of diagonal line lengths (>=2)

    This is a simplified proxy, not a full RQA implementation.
    """

    emb = takens_embedding(series, emb_dim=emb_dim, emb_lag=emb_lag)
    if emb.shape[0] < 10:
        return 0.0, 0.0, 0.0

    emb = _zscore(emb)
    n = emb.shape[0]
    d = np.linalg.norm(emb[:, None, :] - emb[None, :, :], axis=2)
    # Exclude the main diagonal.
    np.fill_diagonal(d, np.inf)

    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return 0.0, 0.0, 0.0

    eps = float(np.quantile(finite, eps_quantile))
    if not np.isfinite(eps) or eps <= 0:
        return 0.0, 0.0, 0.0

    R = (d <= eps).astype(np.uint8)

    rr = float(R.sum() / (n * (n - 1)))

    # Diagonal lines lengths (excluding main diagonal). We count lines in the upper triangle only.
    line_lengths = []
    for k in range(1, n):
        diag = np.diagonal(R, offset=k)
        if diag.size == 0:
            continue
        run = 0
        for v in diag:
            if v:
                run += 1
            else:
                if run >= 2:
                    line_lengths.append(run)
                run = 0
        if run >= 2:
            line_lengths.append(run)

    if not line_lengths:
        return rr, 0.0, 0.0

    total_points_in_lines = float(sum(line_lengths))
    det = total_points_in_lines / float(R.sum() + 1e-12)

    # Entropy over lengths.
    uniq, cnt = np.unique(np.array(line_lengths, dtype=int), return_counts=True)
    p = cnt / cnt.sum()
    ent = float(-(p * np.log(p + 1e-12)).sum())

    return rr, float(det), ent


def compute_chaos_metrics(
    series: np.ndarray,
    *,
    emb_dim: int = 3,
    emb_lag: int = 1,
    eps_quantile: float = 0.10,
) -> ChaosMetrics:
    """Compute chaos-style proxies and aggregate them into a single score."""

    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return ChaosMetrics(
            lyapunov_proxy=0.0,
            rqa_recurrence_rate=0.0,
            rqa_determinism=0.0,
            rqa_entropy=0.0,
            score=0.0,
            meta={"n": float(x.size)},
        )

    lyap = lyapunov_proxy(x, emb_dim=emb_dim, emb_lag=emb_lag)
    rr, det, ent = rqa_light(x, emb_dim=emb_dim, emb_lag=emb_lag, eps_quantile=eps_quantile)

    # Aggregate a bounded score in [0,1].
    # Rationale: high positive lyap, moderate-to-high det, and non-zero entropy can indicate non-trivial dynamics.
    lyap_sig = 1.0 / (1.0 + math.exp(-2.0 * float(lyap)))
    det_sig = float(np.clip(det, 0.0, 1.0))
    rr_sig = float(np.clip(rr / 0.10, 0.0, 1.0))  # normalize around 10% recurrence.
    ent_sig = float(np.clip(ent / 2.0, 0.0, 1.0))

    score = float(np.clip(0.45 * lyap_sig + 0.25 * det_sig + 0.15 * ent_sig + 0.15 * rr_sig, 0.0, 1.0))

    return ChaosMetrics(
        lyapunov_proxy=float(lyap),
        rqa_recurrence_rate=float(rr),
        rqa_determinism=float(det),
        rqa_entropy=float(ent),
        score=score,
        meta={
            "n": float(x.size),
            "emb_dim": float(emb_dim),
            "emb_lag": float(emb_lag),
            "eps_quantile": float(eps_quantile),
        },
    )


def maybe_parse_series(value: object) -> Optional[np.ndarray]:
    """Parse a series from list/np array or JSON-ish string."""

    if value is None:
        return None

    if isinstance(value, np.ndarray):
        return value.astype(float)

    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value, dtype=float)
        except Exception:
            return None

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # Very small permissive JSON list parser.
        if s.startswith("[") and s.endswith("]"):
            try:
                import json

                arr = json.loads(s)
                return np.asarray(arr, dtype=float)
            except Exception:
                return None

        # CSV-ish numbers.
        try:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if len(parts) >= 5:
                return np.asarray([float(p) for p in parts], dtype=float)
        except Exception:
            return None

    return None
