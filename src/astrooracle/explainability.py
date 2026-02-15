from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Explanation:
    candidate_id: str
    method: str
    score: float
    top_features: List[Tuple[str, float]]
    prompt: str
    meta: Dict[str, Any]


def _robust_z(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    if not np.isfinite(mad) or mad < eps:
        return np.zeros_like(x)
    return (x - med) / (1.4826 * mad + eps)


def _collect_numeric_features(df: pd.DataFrame) -> List[str]:
    skip = {"id", "label", "label_pred", "comment", "annotated_at", "embedding"}
    cols = []
    for c in df.columns:
        if c in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    # Prefer known priors first.
    priors = [c for c in ["anomaly_score", "mag", "snr", "ruwe", "rank_score", "uncertainty", "p_max"] if c in cols]
    rest = [c for c in cols if c not in priors]
    return priors + rest


def explain_candidate_zscores(
    df_all: pd.DataFrame,
    row: pd.Series,
    *,
    score_field: str = "rank_score",
    top_k: int = 8,
) -> Explanation:
    """Explain a candidate using robust z-score contributions across features.

    This is dependency-free, stable, and shareable.
    """

    feats = _collect_numeric_features(df_all)
    contribs: List[Tuple[str, float]] = []

    for c in feats:
        x = pd.to_numeric(df_all[c], errors="coerce").to_numpy(float)
        z = _robust_z(x)
        val = float(pd.to_numeric(row.get(c, np.nan), errors="coerce"))
        # Find the index of this row value position by matching id.
        # We avoid relying on index alignment.
        # Approximate: compute z of val with same median/mad.
        med = float(np.nanmedian(x))
        mad = float(np.nanmedian(np.abs(x - med)))
        denom = 1.4826 * mad + 1e-12
        z_val = 0.0 if not np.isfinite(val) or not np.isfinite(denom) else (val - med) / denom
        contribs.append((c, float(z_val)))

    # Sort by absolute contribution.
    contribs = sorted(contribs, key=lambda t: abs(t[1]), reverse=True)[: int(top_k)]

    cid = str(row.get("id"))
    score = float(pd.to_numeric(row.get(score_field, row.get("anomaly_score", 0.0)), errors="coerce"))

    prompt = build_llm_prompt(row=row, top_features=contribs, score=score)

    return Explanation(
        candidate_id=cid,
        method="robust_zscores",
        score=score,
        top_features=contribs,
        prompt=prompt,
        meta={"score_field": score_field, "top_k": float(top_k)},
    )


def build_llm_prompt(*, row: pd.Series, top_features: List[Tuple[str, float]], score: float) -> str:
    """Create a shareable prompt for an LLM, without calling any model."""

    ra = float(pd.to_numeric(row.get("ra", np.nan), errors="coerce"))
    dec = float(pd.to_numeric(row.get("dec", np.nan), errors="coerce"))
    anom = float(pd.to_numeric(row.get("anomaly_score", np.nan), errors="coerce"))

    lines = []
    lines.append("Tu es un assistant scientifique. Analyse cette anomalie astronomique.")
    lines.append("Contexte: outil de triage actif (anomaly triage).")
    lines.append("")
    lines.append(f"Candidat: id={row.get('id')} RA={ra:.5f} deg Dec={dec:.5f} deg")
    lines.append(f"Scores: anomaly_score={anom:.4f} rank_score={score:.4f}")
    lines.append("")
    lines.append("Signaux numériques (z-scores robustes, valeurs positives = au-dessus de la médiane):")
    for name, z in top_features:
        lines.append(f"- {name}: {z:+.3f}")
    lines.append("")
    lines.append("Tâche:")
    lines.append("1) Propose une hypothèse scientifique plausible.")
    lines.append("2) Liste les 3 tests ou vérifications (catalogues, crossmatch, inspection cutouts).")
    lines.append("3) Dis si c'est probablement: artefact, connu, nouveau type, ou réel.")

    return "\n".join(lines)


def explain_top_n(
    ranked_df: pd.DataFrame,
    *,
    n: int = 10,
    score_field: str = "rank_score",
    top_k_features: int = 8,
) -> List[Explanation]:
    if ranked_df.empty:
        return []

    n = min(int(n), len(ranked_df))
    out: List[Explanation] = []
    for _i in range(n):
        row = ranked_df.iloc[_i]
        out.append(explain_candidate_zscores(ranked_df, row, score_field=score_field, top_k=top_k_features))
    return out


def write_explanations_jsonl(explanations: List[Explanation], path) -> None:
    import pathlib

    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for e in explanations:
            f.write(
                json.dumps(
                    {
                        "id": e.candidate_id,
                        "method": e.method,
                        "score": e.score,
                        "top_features": e.top_features,
                        "prompt": e.prompt,
                        "meta": e.meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
