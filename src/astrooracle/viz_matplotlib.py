from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    # astropy est un extra optionnel (.[astro]) : ne doit pas casser l'import du package
    from astropy.visualization import ZScaleInterval, ImageNormalize
except Exception:  # pragma: no cover
    ZScaleInterval = None  # type: ignore[assignment]
    ImageNormalize = None  # type: ignore[assignment]

from .config import OracleConfig


def _fallback_norm(data: np.ndarray) -> Optional[mcolors.Normalize]:
    """Normalisation robuste sans dépendance astropy.

    - utilise des percentiles (1, 99) pour limiter l'impact des outliers
    - retourne None si les bornes sont invalides (NaN, inf, bornes égales)
    """
    try:
        lo, hi = np.nanpercentile(data, [1, 99])
    except Exception:
        return None
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    if float(lo) == float(hi):
        return None
    return mcolors.Normalize(vmin=float(lo), vmax=float(hi), clip=True)


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
