from __future__ import annotations

import json
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from .annotations import read_annotations
from .config import OracleConfig
from .ml.features import build_feature_matrix
from .ml.model import expected_calibration_error, train_ensemble
from .model_io import save_model


LABEL_TO_INT = {
    "real_anomaly": 0,
    "artefact": 1,
    "known": 2,
    "new_type": 3,
    "unsure": 4,
}


def _merge_candidates_and_labels(cand: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    if cand.empty or ann.empty:
        return pd.DataFrame()
    ann = ann.copy()
    ann["id"] = ann["id"].astype(str)
    cand = cand.copy()
    cand["id"] = cand["id"].astype(str)
    return cand.merge(ann[["id", "label"]], on="id", how="inner")


def train_from_files(cfg: OracleConfig) -> Dict[str, object]:
    if not cfg.candidates_path.exists() or not cfg.annot_path.exists():
        raise FileNotFoundError("Need candidates.parquet and annotations.csv to train.")

    cand = pd.read_parquet(cfg.candidates_path)
    ann = read_annotations(cfg)
    df = _merge_candidates_and_labels(cand, ann)
    if df.empty:
        raise ValueError("No overlap between candidates and annotations for training.")

    df = df[df["label"].isin(LABEL_TO_INT)]
    y = df["label"].map(LABEL_TO_INT).to_numpy(int)
    classes = [k for k, _ in sorted(LABEL_TO_INT.items(), key=lambda kv: kv[1])]

    X, feat_names, _ = build_feature_matrix(df)

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=7,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    model = train_ensemble(Xtr, ytr, classes=classes, n_models=5, seed=7)
    probs = model.predict_proba(Xte)
    ece = expected_calibration_error(probs, yte)

    yhat = probs.argmax(axis=1)
    report = classification_report(yte, yhat, output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, yhat).tolist()

    save_model(model, cfg.model_path)

    metrics = {
        "model_version": model.model_version,
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "ece": float(ece),
        "classes": classes,
        "classification_report": report,
        "confusion_matrix": cm,
        "features": feat_names[:32],  # keep short
    }
    metrics_path = cfg.model_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
