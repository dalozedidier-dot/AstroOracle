from __future__ import annotations

import numpy as np
import pandas as pd

from astrooracle.graph_anomaly import graph_anomaly


def test_graph_anomaly_small():
    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame(
        {
            "id": [f"c{i}" for i in range(n)],
            "ra": rng.uniform(0, 360, size=n),
            "dec": rng.uniform(-60, 60, size=n),
            "anomaly_score": rng.uniform(0, 1, size=n),
        }
    )
    res = graph_anomaly(df, k=5, max_nodes=100)
    assert res.n_nodes > 0
    assert res.n_edges > 0
    assert not res.node_metrics.empty
