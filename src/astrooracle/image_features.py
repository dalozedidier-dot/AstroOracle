from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class CutoutFeatures:
    snr: float
    circularity: float
    bg_skew: float
    bg_kurtosis: float
    spike_score: float
    ap_mag: float


def _nan_to_num(x: float, default: float = 0.0) -> float:
    try:
        if np.isfinite(x):
            return float(x)
    except Exception:
        pass
    return float(default)


def _background_region(arr: np.ndarray, frac: float = 0.12) -> np.ndarray:
    n = arr.shape[0]
    w = max(1, int(round(frac * n)))
    # Border region: union of 4 strips
    top = arr[:w, :]
    bot = arr[-w:, :]
    left = arr[:, :w]
    right = arr[:, -w:]
    return np.concatenate([top.ravel(), bot.ravel(), left.ravel(), right.ravel()])


def _moments_circularity(arr: np.ndarray, bg: float) -> float:
    a = np.asarray(arr, dtype=float)
    w = a - float(bg)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0:
        return 0.0

    yy, xx = np.mgrid[0 : a.shape[0], 0 : a.shape[1]]
    cx = float((w * xx).sum() / s)
    cy = float((w * yy).sum() / s)

    dx = xx - cx
    dy = yy - cy
    mxx = float((w * dx * dx).sum() / s)
    myy = float((w * dy * dy).sum() / s)
    mxy = float((w * dx * dy).sum() / s)

    cov = np.array([[mxx, mxy], [mxy, myy]], dtype=float)
    try:
        evals = np.linalg.eigvalsh(cov)
    except Exception:
        return 0.0
    lam1 = float(np.max(evals))
    lam2 = float(np.min(evals))
    if lam1 <= 0:
        return 0.0
    # 1.0 = round, 0.0 = very elongated
    return float(np.clip(lam2 / lam1, 0.0, 1.0))


def _skew_kurtosis(x: np.ndarray) -> Tuple[float, float]:
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 8:
        return 0.0, 0.0
    mu = float(v.mean())
    s = float(v.std(ddof=0))
    if s <= 0:
        return 0.0, 0.0
    z = (v - mu) / s
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4) - 3.0)
    return skew, kurt


def _spike_score(arr: np.ndarray, bg: float) -> float:
    a = np.asarray(arr, dtype=float)
    n = a.shape[0]
    c = n // 2
    w = np.where(np.isfinite(a - bg), a - bg, 0.0)
    w = np.clip(w, 0.0, None)

    band = max(1, n // 40)  # ~2.5%
    hor = float(w[c - band : c + band + 1, :].sum())
    ver = float(w[:, c - band : c + band + 1].sum())
    diag = float(np.diag(w).sum())
    adiag = float(np.diag(np.fliplr(w)).sum())

    vec = np.array([hor, ver, diag, adiag], dtype=float)
    med = float(np.median(vec))
    if med <= 0:
        return 0.0
    return float(np.max(vec) / (med + 1e-12))


def _aperture_mag(arr: np.ndarray, bg: float) -> float:
    a = np.asarray(arr, dtype=float)
    n = a.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((xx - (n - 1) / 2) ** 2 + (yy - (n - 1) / 2) ** 2)
    r0 = 0.12 * n
    mask = r <= r0
    flux = float(np.where(np.isfinite(a), a - bg, 0.0)[mask].sum())
    flux = max(flux, 1e-12)
    return float(-2.5 * np.log10(flux))


def compute_cutout_features(arr: np.ndarray) -> CutoutFeatures:
    a = np.asarray(arr, dtype=float)
    bg = _background_region(a)
    bg_med = float(np.nanmedian(bg)) if np.isfinite(bg).any() else 0.0
    bg_std = float(np.nanstd(bg)) if np.isfinite(bg).any() else 1.0
    if bg_std <= 0:
        bg_std = 1.0

    peak = float(np.nanmax(a)) if np.isfinite(a).any() else bg_med
    snr = (peak - bg_med) / bg_std

    skew, kurt = _skew_kurtosis(bg)
    circ = _moments_circularity(a, bg_med)
    spike = _spike_score(a, bg_med)
    ap_mag = _aperture_mag(a, bg_med)

    return CutoutFeatures(
        snr=_nan_to_num(snr, 0.0),
        circularity=_nan_to_num(circ, 0.0),
        bg_skew=_nan_to_num(skew, 0.0),
        bg_kurtosis=_nan_to_num(kurt, 0.0),
        spike_score=_nan_to_num(spike, 0.0),
        ap_mag=_nan_to_num(ap_mag, 0.0),
    )


def aggregate_candidate_features(by_survey: Dict[str, CutoutFeatures]) -> Dict[str, float]:
    if not by_survey:
        return {}

    snr = np.array([f.snr for f in by_survey.values()], dtype=float)
    circ = np.array([f.circularity for f in by_survey.values()], dtype=float)
    spike = np.array([f.spike_score for f in by_survey.values()], dtype=float)

    out: Dict[str, float] = {
        "feat_snr_max": float(np.nanmax(snr)),
        "feat_snr_min": float(np.nanmin(snr)),
        "feat_circularity_min": float(np.nanmin(circ)),
        "feat_spike_max": float(np.nanmax(spike)),
    }

    # Pseudo color if DSS2 Red and 2MASS J present
    if "DSS2 Red" in by_survey and "2MASS J" in by_survey:
        out["feat_color_dss2red_minus_2massj"] = float(by_survey["DSS2 Red"].ap_mag - by_survey["2MASS J"].ap_mag)

    return out
