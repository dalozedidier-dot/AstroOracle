from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ImageNormalize, ZScaleInterval

from .config import OracleConfig


def render_cutouts_matplotlib(
    cutouts: list[tuple[str, np.ndarray]],
    title: str,
    cfg: OracleConfig,
    save_path: Path | None = None,
) -> None:
    if not cutouts:
        print("No cutouts available.")
        return

    n = len(cutouts)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (survey, data) in zip(axes, cutouts, strict=False):
        norm = ImageNormalize(data, interval=ZScaleInterval())
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
