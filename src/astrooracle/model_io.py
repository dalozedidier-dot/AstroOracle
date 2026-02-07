from __future__ import annotations

from pathlib import Path
import pickle
from typing import Optional

from .ml.model import EnsembleModel


def load_model(path: Path) -> Optional[EnsembleModel]:
    if not path.exists():
        return None
    try:
        obj = pickle.loads(path.read_bytes())
        if isinstance(obj, EnsembleModel):
            return obj
    except Exception:
        return None
    return None


def save_model(model: EnsembleModel, path: Path) -> None:
    path.write_bytes(pickle.dumps(model))
