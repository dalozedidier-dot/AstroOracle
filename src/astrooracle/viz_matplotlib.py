from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from .config import OracleConfig

try:
    from astropy.visualization import ImageNormalize, ZScaleInterval
except Exception:  # pragma: no cover
    ImageNormalize = None  # type: ignore
    ZScaleInterval = None  # type: ignore


def _fallback_norm(data: np.ndarray) -> Normalize:
    arr = np.asarray(data, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0, clip=True)
    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return Normalize(vmin=lo, vmax=hi, clip=True)


def render_cutouts_matplotlib(
    cutouts: List[Tuple[str, np.ndarray]],
    title: str,
    cfg: OracleConfig,
    save_path: Optional[Path] = None,
) -> None:
    if not cutouts:
        print("No cutouts available.")
        return

    n = len(cutouts)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (survey, data) in zip(axes, cutouts):
        if ImageNormalize is not None and ZScaleInterval is not None:
            norm = ImageNormalize(data, interval=ZScaleInterval())
        else:
            norm = _fallback_norm(data)
        ax.imshow(data, cmap="gray", norm=norm, origin="lower")
        ax.set_title(survey, fontsize=9)
        ax.axis("off")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    if not cfg.no_gui:
        plt.show(block=False)
        plt.pause(0.25)

    plt.close(fig)
