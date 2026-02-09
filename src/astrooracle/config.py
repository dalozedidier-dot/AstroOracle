from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class RankingConfig:
    strategy: str
    diversity: str
    w_anomaly: float
    w_acq: float
    w_div: float
    w_prior: float
    w_artifact: float
    w_known: float
    w_iforest: float
    acq_temperature: float

    @staticmethod
    def default() -> "RankingConfig":
        return RankingConfig(
            strategy="entropy",
            diversity="kcenter",
            w_anomaly=0.35,
            w_acq=0.35,
            w_div=0.20,
            w_prior=0.10,
            w_artifact=0.0,
            w_known=0.0,
            w_iforest=0.0,
            acq_temperature=1.0,
        )


@dataclass(frozen=True)
class OracleConfig:
    candidates_path: Path
    annot_path: Path
    log_path: Path
    retrain_script: Path
    model_path: Path
    min_new_labels_for_retrain: int
    check_interval_s: int

    surveys: List[str]
    cutout_radius_arcmin: float
    pixels: int

    n_query: int
    no_gui: bool
    save_cutouts_dir: Optional[Path]
    offline: bool

    ranking: RankingConfig

    @staticmethod
    def default() -> "OracleConfig":
        return OracleConfig(
            candidates_path=Path("candidates.parquet"),
            annot_path=Path("annotations.csv"),
            log_path=Path("oracle_log.jsonl"),
            retrain_script=Path("scripts/retrain_model.py"),
            model_path=Path("best_model.pkl"),
            min_new_labels_for_retrain=10,
            check_interval_s=300,
            surveys=["DSS2 Red", "2MASS J"],
            cutout_radius_arcmin=0.15,
            pixels=400,
            n_query=6,
            no_gui=False,
            save_cutouts_dir=None,
            offline=False,
            ranking=RankingConfig.default(),
        )
