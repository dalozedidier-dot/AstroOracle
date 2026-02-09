from __future__ import annotations

from typing import List, Optional

import numpy as np


def _cosine_dist(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    return 1.0 - (a_n @ b_n.T)


def min_dist_to_set(x: np.ndarray, selected: np.ndarray) -> np.ndarray:
    if len(selected) == 0:
        return np.ones(len(x), dtype=float)
    d = _cosine_dist(x, selected)
    return d.min(axis=1)


def kcenter_greedy(emb: np.ndarray, k: int, seed_idx: Optional[int] = None) -> List[int]:
    n = emb.shape[0]
    if n == 0 or k <= 0:
        return []
    if seed_idx is None:
        seed_idx = int(np.argmax(np.linalg.norm(emb, axis=1)))
    selected = [seed_idx]
    min_d = min_dist_to_set(emb, emb[[seed_idx]])
    for _ in range(1, min(k, n)):
        idx = int(np.argmax(min_d))
        selected.append(idx)
        min_d = np.minimum(min_d, min_dist_to_set(emb, emb[[idx]]))
    return selected


def dpp_greedy(emb: np.ndarray, k: int, gamma: float = 1.0, eps: float = 1e-12) -> List[int]:
    # Greedy MAP for an RBF kernel DPP on cosine distance.
    n = emb.shape[0]
    if n == 0 or k <= 0:
        return []
    dist = _cosine_dist(emb, emb, eps=eps)
    K = np.exp(-gamma * dist**2)
    selected: List[int] = []
    diag = np.diag(K).copy()
    for _ in range(min(k, n)):
        i = int(np.argmax(diag))
        if diag[i] <= 0:
            break
        selected.append(i)
        # Schur complement update
        ki = K[:, i].copy()
        denom = K[i, i] + eps
        K = K - np.outer(ki, ki) / denom
        diag = np.diag(K).copy()
    return selected
