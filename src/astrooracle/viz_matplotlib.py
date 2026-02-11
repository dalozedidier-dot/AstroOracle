from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from astropy.visualization import ImageNormalize, ZScaleInterval
except Exception:  # pragma: no cover
    ImageNormalize = None  # type: ignore[assignment]
    ZScaleInterval = None  # type: ignore[assignment]

from .config import OracleConfig


def _imshow_fallback(ax, data: np.ndarray) -> None:
    arr = np.asarray(data, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo = float(np.percentile(finite, 1))
        hi = float(np.percentile(finite, 99))
        if hi <= lo:
            hi = lo + 1.0
    ax.imshow(arr, cmap="gray", vmin=lo, vmax=hi, origin="lower")


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

    for ax, (survey, data) in zip(axes, cutouts, strict=False):
        if ImageNormalize is not None and ZScaleInterval is not None:
            norm = ImageNormalize(data, interval=ZScaleInterval())
            ax.imshow(data, cmap="gray", norm=norm, origin="lower")
        else:
            _imshow_fallback(ax, data)

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
