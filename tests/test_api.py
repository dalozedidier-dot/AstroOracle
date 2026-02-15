from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from astrooracle.api import create_app
from astrooracle.config import OracleConfig, RankingConfig


def test_api_smoke(tmp_path: Path):
    # Minimal candidates
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "ra": [10.0, 11.0, 12.0],
            "dec": [-5.0, -5.5, -6.0],
            "anomaly_score": [0.1, 0.9, 0.3],
            "mag": [18.1, 17.5, 19.0],
            "snr": [10.0, 30.0, 8.0],
            "ruwe": [1.1, 1.2, 1.0],
            "embedding": [[0.0] * 8, [0.1] * 8, [0.2] * 8],
        }
    )
    cand_path = tmp_path / "candidates.parquet"
    df.to_parquet(cand_path, index=False)

    cfg = OracleConfig(
        candidates_path=cand_path,
        annot_path=tmp_path / "annotations.csv",
        log_path=tmp_path / "oracle_log.jsonl",
        retrain_script=tmp_path / "retrain_model.py",
        model_path=tmp_path / "missing_model.pkl",
        min_new_labels_for_retrain=10,
        check_interval_s=300,
        surveys=["DSS2 Red", "2MASS J"],
        cutout_radius_arcmin=2.0,
        pixels=64,
        n_query=2,
        no_gui=True,
        save_cutouts_dir=None,
        offline=True,
        ranking=RankingConfig(),
    )

    app = create_app(cfg)
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    r = client.get("/api/top?n=2")
    assert r.status_code == 200
    data = r.json()
    assert len(data["top"]) == 2

    r = client.get("/api/cutouts?ra=10&dec=-5")
    assert r.status_code == 200
    assert "cutouts" in r.json()

    # Run endpoint without upload.
    r = client.post("/run", data={"k": "2", "mode": "vanilla"})
    assert r.status_code == 200
    assert len(r.json()["batch"]) == 2
