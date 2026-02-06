#!/usr/bin/env python3
# Stub retrain script.
# Replace with your real training pipeline.
# Expected side-effect: write/update best_model.pkl (or adjust --model-path).

from pathlib import Path
import pickle

MODEL_PATH = Path("best_model.pkl")

def main():
    model = {"model": "stub", "note": "replace scripts/retrain_model.py with your real retrain."}
    MODEL_PATH.write_bytes(pickle.dumps(model))
    print(f"Wrote stub model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
