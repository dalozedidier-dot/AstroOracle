from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from astropy import units as u


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
    cutout_radius: u.Quantity
    pixels: int
    n_query: int
    no_gui: bool
    save_cutouts_dir: Optional[Path]
    offline: bool

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
            cutout_radius=0.15 * u.arcmin,
            pixels=400,
            n_query=6,
            no_gui=False,
            save_cutouts_dir=None,
            offline=False,
        )
