from __future__ import annotations

import base64
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_COLS = {"id", "ra", "dec", "anomaly_score"}


def _slug(s: object) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(s))


def load_candidates_table(path: Path) -> pd.DataFrame:
    """Load a candidates table from parquet or csv.

    Required columns: id, ra, dec, anomaly_score.
    """
    if not path.exists():
        return pd.DataFrame()

    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".tsv"}:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix} (expected parquet/csv/tsv)")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in candidates: {sorted(missing)}")

    return df


def _sample_df(df: pd.DataFrame, max_points: int, seed: int = 7) -> pd.DataFrame:
    if max_points <= 0 or len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=seed).reset_index(drop=True)


def _encode_png_b64(path: Path) -> str:
    raw = path.read_bytes()
    return base64.b64encode(raw).decode("ascii")


def _find_cutout_for_row(
    cutouts_dir: Path,
    candidate_id: object,
    survey: Optional[str] = None,
) -> Optional[Path]:
    if not cutouts_dir.exists():
        return None

    sid = _slug(candidate_id)
    if survey is not None:
        p = cutouts_dir / f"{sid}_{_slug(survey)}.png"
        return p if p.exists() else None

    # Fallback: pick the first cutout matching this id prefix.
    hits = sorted(cutouts_dir.glob(f"{sid}_*.png"))
    return hits[0] if hits else None


def build_viz3d_figure(
    df: pd.DataFrame,
    *,
    mode: str = "scatter",
    max_points: int = 5000,
    color: Optional[str] = None,
    cutouts_dir: Optional[Path] = None,
    cutout_survey: Optional[str] = None,
    embed_cutouts: bool = False,
    title: Optional[str] = None,
):
    """Build a Plotly 3D figure for anomaly triage.

    Modes:
      - scatter: x=RA, y=Dec, z=anomaly_score
      - globe: points on a unit celestial sphere (RA/Dec -> xyz), optional sphere surface.
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Plotly not installed") from e

    df = _sample_df(df, int(max_points))

    color_col = color
    if color_col is None:
        if "label_pred" in df.columns:
            color_col = "label_pred"
        elif "uncertainty" in df.columns:
            color_col = "uncertainty"
        else:
            color_col = "anomaly_score"

    hover_cols = ["id", "ra", "dec", "anomaly_score"]
    for c in ("rank_score", "label_pred", "uncertainty"):
        if c in df.columns and c not in hover_cols:
            hover_cols.append(c)

    df2 = df.copy()

    img_html = None
    if cutouts_dir is not None:
        cutouts_dir = Path(cutouts_dir)
        paths = [
            _find_cutout_for_row(cutouts_dir, cid, survey=cutout_survey)
            for cid in df2["id"].tolist()
        ]
        if embed_cutouts:
            htmls = []
            for p in paths:
                if p is None:
                    htmls.append("")
                    continue
                b64 = _encode_png_b64(p)
                htmls.append(f'<br><img src="data:image/png;base64,{b64}" style="width:180px;" />')
            img_html = htmls
        else:
            # Use relative paths inside hover: works when HTML is next to the cutouts/ folder.
            rels = []
            for p in paths:
                if p is None:
                    rels.append("")
                    continue
                rel = p.as_posix()
                rels.append(f'<br><img src="{rel}" style="width:180px;" />')
            img_html = rels

    if img_html is not None:
        df2["_cutout_img"] = img_html

    if mode.lower() == "scatter":
        fig = px.scatter_3d(
            df2,
            x="ra",
            y="dec",
            z="anomaly_score",
            color=color_col if color_col in df2.columns else None,
            hover_data=hover_cols,
            title=title or "3D Anomaly Candidates (RA/Dec/Score)",
        )
        fig.update_layout(
            scene=dict(
                xaxis_title="RA (deg)",
                yaxis_title="Dec (deg)",
                zaxis_title="Anomaly Score",
            )
        )

        if "_cutout_img" in df2.columns:
            # Add a custom hovertemplate that appends the image HTML.
            fig.update_traces(
                customdata=np.stack(
                    [
                        df2["id"].astype(str).to_numpy(),
                        df2["ra"].to_numpy(float),
                        df2["dec"].to_numpy(float),
                        df2["anomaly_score"].to_numpy(float),
                        df2["_cutout_img"].astype(str).to_numpy(),
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "ID: %{customdata[0]}<br>"
                    "RA: %{customdata[1]:.5f}<br>"
                    "Dec: %{customdata[2]:.5f}<br>"
                    "Score: %{customdata[3]:.4f}"
                    "%{customdata[4]}<extra></extra>"
                ),
            )
        return fig

    if mode.lower() == "globe":
        ra = np.deg2rad(df2["ra"].to_numpy(float))
        dec = np.deg2rad(df2["dec"].to_numpy(float))
        df2["_x"] = np.cos(dec) * np.cos(ra)
        df2["_y"] = np.cos(dec) * np.sin(ra)
        df2["_z"] = np.sin(dec)

        fig = px.scatter_3d(
            df2,
            x="_x",
            y="_y",
            z="_z",
            color=color_col if color_col in df2.columns else None,
            hover_data=hover_cols,
            title=title or "Celestial Globe (RA/Dec projected on a sphere)",
        )

        # Add an (optional) faint sphere surface for context.
        u = np.linspace(0, 2 * math.pi, 60)
        v = np.linspace(0, math.pi, 30)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))
        sphere = go.Surface(x=xs, y=ys, z=zs, opacity=0.08, showscale=False, hoverinfo="skip")
        fig.add_trace(sphere)
        # Put the sphere first so points stay on top.
        fig.data = (fig.data[-1],) + fig.data[:-1]

        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
            )
        )

        if "_cutout_img" in df2.columns:
            fig.update_traces(
                selector=dict(type="scatter3d"),
                customdata=np.stack(
                    [
                        df2["id"].astype(str).to_numpy(),
                        df2["ra"].to_numpy(float),
                        df2["dec"].to_numpy(float),
                        df2["anomaly_score"].to_numpy(float),
                        df2["_cutout_img"].astype(str).to_numpy(),
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "ID: %{customdata[0]}<br>"
                    "RA: %{customdata[1]:.5f}<br>"
                    "Dec: %{customdata[2]:.5f}<br>"
                    "Score: %{customdata[3]:.4f}"
                    "%{customdata[4]}<extra></extra>"
                ),
            )

        return fig

    raise ValueError(f"Unknown mode: {mode} (expected: scatter|globe)")


def write_viz3d_html(fig, output: Path, *, cdn: bool = False) -> None:
    """Write a standalone HTML file."""
    include = "cdn" if cdn else True
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs=include, full_html=True)
