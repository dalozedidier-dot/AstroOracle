#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    out = Path("candidates.parquet")
    n = 50
    df = pd.DataFrame(
        {
            "id": [f"cand_{i:04d}" for i in range(n)],
            "ra": np.random.uniform(0, 360, size=n),
            "dec": np.random.uniform(-30, 30, size=n),
            "anomaly_score": np.random.normal(0, 1, size=n),
        }
    )
    df["embedding"] = [np.random.normal(0, 1, size=16).astype(float) for _ in range(n)]
    df.to_parquet(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
