from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from .config import OracleConfig

try:
    from astropy.visualization import ImageNormalize, ZScaleInterval

    _HAS_ASTROPY = True
except Exception:  # pragma: no cover
    ImageNormalize = None  # type: ignore[assignment]
    ZScaleInterval = None  # type: ignore[assignment]
    _HAS_ASTROPY = False


def _fallback_norm(data: np.ndarray) -> Normalize | None:
    finite = np.asarray(data[np.isfinite(data)], dtype=float)
    if finite.size == 0:
        return None

    try:
        vmin, vmax = np.nanpercentile(finite, [1, 99])
    except Exception:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0

    return Normalize(vmin=vmin, vmax=vmax, clip=True)


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
        if _HAS_ASTROPY:
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
    plt.close(fig)
