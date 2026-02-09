from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class AcquisitionResult:
    score: np.ndarray
    components: Dict[str, np.ndarray]


def entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def margin(p: np.ndarray) -> np.ndarray:
    # Small margin = more uncertain
    p_sorted = np.sort(p, axis=1)[:, ::-1]
    return 1.0 - (p_sorted[:, 0] - p_sorted[:, 1])


def bald_mc(probs_mc: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # probs_mc: [T, N, C]
    p_bar = probs_mc.mean(axis=0)
    h_bar = entropy(p_bar, eps=eps)
    h_each = np.array([entropy(probs_mc[t], eps=eps) for t in range(probs_mc.shape[0])]).mean(axis=0)
    return h_bar - h_each


def badge_score(emb: np.ndarray, p: np.ndarray) -> np.ndarray:
    # BADGE: gradient embeddings for softmax, approximated for multi-class.
    # g_i = emb_i * (p_i - e_yhat) where yhat is argmax.
    yhat = p.argmax(axis=1)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(yhat)), yhat] = 1.0
    g = emb[:, None, :] * (p - onehot)[:, :, None]  # [N, C, D]
    g = g.reshape(len(emb), -1)
    return np.linalg.norm(g, axis=1)


def acquire(
    *,
    probs: Optional[np.ndarray],
    embeddings: Optional[np.ndarray],
    strategy: str,
    probs_mc: Optional[np.ndarray] = None,
) -> AcquisitionResult:
    strategy = strategy.lower()
    if probs is None and strategy in {"entropy", "margin"}:
        raise ValueError("probs is required for entropy/margin acquisition.")
    if strategy == "entropy":
        s = entropy(probs)
        return AcquisitionResult(score=s, components={"entropy": s})
    if strategy == "margin":
        s = margin(probs)
        return AcquisitionResult(score=s, components={"margin": s})
    if strategy == "bald":
        if probs_mc is None:
            raise ValueError("probs_mc is required for BALD.")
        s = bald_mc(probs_mc)
        return AcquisitionResult(score=s, components={"bald": s})
    if strategy == "badge":
        if embeddings is None or probs is None:
            raise ValueError("embeddings and probs are required for BADGE.")
        s = badge_score(embeddings, probs)
        return AcquisitionResult(score=s, components={"badge": s})
    raise ValueError(f"Unknown strategy: {strategy}")
