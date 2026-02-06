from __future__ import annotations

import numpy as np

try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None  # type: ignore


def render_cutouts_plotly(cutouts: list[tuple[str, np.ndarray]], title: str) -> None:
    if px is None:
        raise RuntimeError("Plotly not installed. Install extras: pip install -e '.[plotly]'")
    for survey, data in cutouts:
        arr = np.asarray(data, dtype=float)
        fig = px.imshow(arr, origin="lower", aspect="equal", title=f"{title} | {survey}")
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.show()
