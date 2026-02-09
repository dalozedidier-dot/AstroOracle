from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .config import OracleConfig

try:
    from astropy.visualization import ImageNormalize, ZScaleInterval  # type: ignore
except Exception:  # pragma: no cover
    ImageNormalize = None
    ZScaleInterval = None


def _fallback_norm(data: np.ndarray) -> mcolors.Normalize:
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return mcolors.Normalize(vmin=0.0, vmax=1.0)
    vmin, vmax = np.percentile(finite, [1, 99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin == vmax:
            vmax = vmin + 1e-9
    return mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))


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
