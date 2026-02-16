from __future__ import annotations

import numpy as np
import pandas as pd

from astrooracle.hybrid_fusion import apply_hybrid_mode


def test_apply_hybrid_mode_overwrites_anomaly_score():
    rng = np.random.default_rng(1)
    n = 100
    df = pd.DataFrame(
        {
            "id": [f"x{i}" for i in range(n)],
            "ra": rng.uniform(0, 360, size=n),
            "dec": rng.uniform(-80, 80, size=n),
            "anomaly_score": rng.uniform(0, 1, size=n),
            "mag": rng.normal(18, 1, size=n),
            "snr": rng.uniform(5, 50, size=n),
            "ruwe": rng.uniform(0.8, 2.0, size=n),
            "embedding": [rng.normal(size=8).tolist() for _ in range(n)],
        }
    )

    out, meta = apply_hybrid_mode(df, overwrite_anomaly_score=True)
    assert "hybrid_fused_score" in out.columns
    assert not np.allclose(
        out["anomaly_score"].to_numpy(float), df["anomaly_score"].to_numpy(float)
    )
    assert "hybrid_w_base" in meta
