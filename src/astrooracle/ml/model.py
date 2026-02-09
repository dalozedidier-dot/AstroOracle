from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class EnsembleModel:
    classes_: List[str]
    pipelines: List[Pipeline]

    def predict_proba_mc(self, X: np.ndarray) -> np.ndarray:
        probs = []
        for p in self.pipelines:
            probs.append(p.predict_proba(X))
        return np.stack(probs, axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba_mc(X).mean(axis=0)

    @property
    def model_version(self) -> str:
        return f"ensemble_logreg_calib_v2|m={len(self.pipelines)}"


def _make_logreg() -> LogisticRegression:
    params = inspect.signature(LogisticRegression).parameters
    kwargs = {"max_iter": 2000}
    if "multi_class" in params:
        kwargs["multi_class"] = "auto"
    return LogisticRegression(**kwargs)


def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    classes: List[str],
    n_models: int = 5,
    seed: int = 7,
) -> EnsembleModel:
    rng = np.random.default_rng(seed)
    pipes: List[Pipeline] = []
    n = X.shape[0]
    all_labels = np.unique(y)

    per_class_idx = {}
    for lab in all_labels:
        per_class_idx[int(lab)] = int(np.where(y == lab)[0][0])

    for _ in range(n_models):
        idx = rng.integers(0, n, size=n)

        present = set(int(v) for v in np.unique(y[idx]))
        missing = [lab for lab in all_labels if int(lab) not in present]
        if missing:
            extra = np.array([per_class_idx[int(lab)] for lab in missing], dtype=int)
            idx = np.concatenate([idx, extra])

        Xm, ym = X[idx], y[idx]

        base = _make_logreg()

        unique, counts = np.unique(ym, return_counts=True)
        min_count = int(counts.min()) if len(counts) else 0
        cv = 3 if min_count >= 3 else (2 if min_count >= 2 else None)

        if cv is None:
            clf = base
        else:
            clf = CalibratedClassifierCV(base, method="sigmoid", cv=cv)

        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(Xm, ym)
        pipes.append(pipe)

    return EnsembleModel(classes_=classes, pipelines=pipes)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (conf >= lo) & (conf < hi)
        else:
            mask = (conf >= lo) & (conf <= hi)

        if not np.any(mask):
            continue
        ece += np.abs(acc[mask].mean() - conf[mask].mean()) * (mask.mean())
    return float(ece)
