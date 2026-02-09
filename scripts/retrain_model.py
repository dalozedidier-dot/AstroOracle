#!/usr/bin/env python3
from __future__ import annotations

from astrooracle.config import OracleConfig
from astrooracle.train import train_from_files


def main() -> None:
    cfg = OracleConfig.default()
    metrics = train_from_files(cfg)
    print("Model trained.")
    print(f"Saved: {cfg.model_path}")
    print(f"Metrics: {cfg.model_path.with_suffix(".metrics.json")}")
    print(f"ECE: {metrics.get("ece")}")


if __name__ == "__main__":
    main()
