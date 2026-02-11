import numpy as np
import pandas as pd

from astrooracle.config import OracleConfig, RankingConfig
from astrooracle.ranking import rank_candidates, select_batch
from astrooracle.ml.model import train_ensemble
from astrooracle.ml.features import build_feature_matrix


def test_rank_and_select_runs():
    cfg = OracleConfig.default()
    cfg = OracleConfig(**{**cfg.__dict__, "ranking": RankingConfig.default()})

    df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "ra": [1.0, 2.0, 3.0, 4.0],
            "dec": [0.1, 0.2, 0.3, 0.4],
            "anomaly_score": [0.9, 0.2, 0.5, 0.7],
            "mag": [18.0, 20.0, 19.0, 17.5],
            "snr": [10.0, 5.0, 7.0, 12.0],
            "embedding": [np.ones(8), np.zeros(8), np.arange(8), np.arange(8)[::-1]],
            "label": ["real_anomaly", "artefact", "known", "new_type"],
        }
    )

    # train a tiny model
    X, _, _ = build_feature_matrix(df)
    y = np.array([0, 1, 2, 3])
    model = train_ensemble(
        X, y, classes=["real_anomaly", "artefact", "known", "new_type", "unsure"], n_models=2
    )

    ranked, meta = rank_candidates(df.drop(columns=["label"]), cfg, model=model)
    assert len(ranked) == 4
    assert "rank_score" in ranked.columns
    batch = select_batch(ranked, cfg, k=2)
    assert len(batch) == 2
