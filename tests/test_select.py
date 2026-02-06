import numpy as np
import pandas as pd
from astrooracle.config import OracleConfig
from astrooracle.core import select_candidates

def test_select_candidates_basic():
    cfg = OracleConfig.default()
    cfg = OracleConfig(**{**cfg.__dict__, "n_query": 5})
    df = pd.DataFrame({
        "id": [f"a{i}" for i in range(10)],
        "ra": np.linspace(0, 1, 10),
        "dec": np.linspace(0, 1, 10),
        "anomaly_score": np.linspace(-2, 2, 10),
    })
    top = select_candidates(df, cfg)
    assert len(top) == 5
    assert "query_score" in top.columns
