from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


def _read_candidates(path: str) -> pd.DataFrame:
    p = str(path)
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    if p.endswith(".csv"):
        return pd.read_csv(p)
    raise ValueError("Unsupported candidates format. Expected .parquet or .csv")


def _first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _safe_read_bytes(path: Path) -> Optional[bytes]:
    try:
        return path.read_bytes()
    except Exception:
        return None


def _guess_cutout_pairs(
    cutouts_dir: Path,
    cand_id: str,
) -> Tuple[Optional[Path], Optional[Path], List[Path]]:
    """Return (normal_path, anomalous_path, all_matches).

    Heuristic: pick first two matches as (normal, anomalous).
    """
    if not cutouts_dir.exists():
        return None, None, []

    pat = re.compile(rf"(?i)(^|[^0-9a-z]){re.escape(str(cand_id))}([^0-9a-z]|$)")
    matches: List[Path] = []
    for p in sorted(cutouts_dir.glob("*.png")):
        if pat.search(p.name):
            matches.append(p)

    normal = matches[0] if len(matches) >= 1 else None
    anomalous = matches[1] if len(matches) >= 2 else None
    return normal, anomalous, matches


def _img_widget_from_path(path: Optional[Path], width: int = 256, height: int = 256):
    import ipywidgets as widgets

    if path is None:
        return widgets.HTML(
            f"<div style='width:{width}px;height:{height}px;"
            "display:flex;align-items:center;justify-content:center;"
            "border:1px solid #ddd;border-radius:8px;'>"
            "<span style='color:#666;'>No image</span></div>"
        )

    raw = _safe_read_bytes(path)
    if raw is None:
        return widgets.HTML(
            f"<div style='width:{width}px;height:{height}px;"
            "display:flex;align-items:center;justify-content:center;"
            "border:1px solid #ddd;border-radius:8px;'>"
            "<span style='color:#666;'>Unreadable</span></div>"
        )

    return widgets.Image(
        value=raw,
        format="png",
        layout=widgets.Layout(width=f"{width}px", height=f"{height}px"),
    )


def _build_meta_html(row: pd.Series, cols: Sequence[str]) -> str:
    items = []
    for c in cols:
        if c not in row.index:
            continue
        v = row[c]
        if isinstance(v, float):
            items.append(f"<tr><td><b>{c}</b></td><td>{v:.6g}</td></tr>")
        else:
            items.append(f"<tr><td><b>{c}</b></td><td>{v}</td></tr>")
    return (
        "<div style='border:1px solid #eee;border-radius:10px;padding:10px;'>"
        "<table style='border-collapse:collapse;width:100%;'>"
        + "".join(items)
        + "</table></div>"
    )


def _make_uncertainty_figure(df_top: pd.DataFrame, id_col: str, u_col: str):
    import plotly.graph_objects as go

    x = [str(v) for v in df_top[id_col].tolist()]
    y = [_as_float(v) for v in df_top[u_col].tolist()]

    fig = go.FigureWidget(data=[go.Bar(x=x, y=y)])
    fig.update_layout(
        title="Uncertainty (Top candidates)",
        xaxis_title="Candidate ID",
        yaxis_title="Uncertainty",
        height=320,
        margin=dict(l=30, r=20, t=50, b=40),
    )
    return fig


def _highlight_bar(fig, selected_x: str) -> None:
    xs = list(fig.data[0].x)
    opacity = [1.0 if str(x) == str(selected_x) else 0.35 for x in xs]
    fig.data[0].marker.opacity = opacity


@dataclass
class TriageWidgetConfig:
    candidates_path: str
    cutouts_dir: str
    top_k: int = 100
    image_size: int = 256
    review_save_path: str = "triage_review.json"


class TriageWidget:
    def __init__(self, cfg: TriageWidgetConfig):
        import ipywidgets as widgets

        self.cfg = cfg
        self.df = _read_candidates(cfg.candidates_path).copy()

        self.id_col = _first_existing_col(
            self.df,
            ["id", "source_id", "candidate_id", "obj_id"],
        )
        if self.id_col is None:
            raise ValueError(
                "No id column found. Expected one of: id, source_id, candidate_id, obj_id"
            )

        self.rank_col = _first_existing_col(
            self.df,
            ["rank", "rank_score", "score", "anomaly_score"],
        )
        if self.rank_col is None:
            self.rank_col = self.id_col

        self.u_col = _first_existing_col(
            self.df,
            ["uncertainty", "uncertainty_score", "entropy", "prob_margin"],
        )
        if self.u_col is None:
            num_cols = [c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c])]
            self.u_col = num_cols[0] if num_cols else self.rank_col

        if "rank" in self.rank_col.lower() and "score" not in self.rank_col.lower():
            self.df_sorted = self.df.sort_values(self.rank_col, ascending=True).reset_index(
                drop=True
            )
        else:
            self.df_sorted = self.df.sort_values(self.rank_col, ascending=False).reset_index(
                drop=True
            )

        self.df_top = self.df_sorted.head(int(cfg.top_k)).copy()
        self.cutouts_dir = Path(cfg.cutouts_dir)

        self.select = widgets.Select(
            options=[str(v) for v in self.df_top[self.id_col].tolist()],
            description="Candidate",
            rows=min(14, len(self.df_top)),
            layout=widgets.Layout(width="320px"),
        )

        self.btn_prev = widgets.Button(description="Prev", layout=widgets.Layout(width="80px"))
        self.btn_next = widgets.Button(description="Next", layout=widgets.Layout(width="80px"))
        self.btn_save = widgets.Button(description="Save review", layout=widgets.Layout(width="140px"))

        self.label_ok = widgets.ToggleButton(description="OK", value=False)
        self.label_anom = widgets.ToggleButton(description="Anomalous", value=False)
        self.label_skip = widgets.ToggleButton(description="Skip", value=False)

        self.img_left = _img_widget_from_path(None, width=cfg.image_size, height=cfg.image_size)
        self.img_right = _img_widget_from_path(None, width=cfg.image_size, height=cfg.image_size)

        self.meta = widgets.HTML(value="")
        self.msg = widgets.HTML(value="")

        self.fig = _make_uncertainty_figure(self.df_top, self.id_col, self.u_col)

        self._review: Dict[str, Dict[str, Any]] = self._load_review(cfg.review_save_path)

        self.ui = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.select,
                        widgets.VBox([widgets.HBox([self.btn_prev, self.btn_next, self.btn_save])]),
                    ]
                ),
                widgets.HBox(
                    [
                        widgets.VBox(
                            [widgets.HTML("<b>Cutout A (normal/reference)</b>"), self.img_left]
                        ),
                        widgets.VBox(
                            [widgets.HTML("<b>Cutout B (anomalous/contrast)</b>"), self.img_right]
                        ),
                        widgets.VBox(
                            [
                                widgets.HTML("<b>Labels</b>"),
                                widgets.HBox([self.label_ok, self.label_anom, self.label_skip]),
                                widgets.HTML("<b>Metadata</b>"),
                                self.meta,
                            ],
                            layout=widgets.Layout(width="420px"),
                        ),
                    ]
                ),
                self.fig,
                self.msg,
            ]
        )

        self._wire_events()

        if len(self.select.options) > 0:
            self.select.value = self.select.options[0]
            self._render_selected(self.select.value)

    def display(self) -> None:
        from IPython.display import display

        display(self.ui)

    def _wire_events(self) -> None:
        self.select.observe(self._on_select_change, names="value")
        self.btn_prev.on_click(self._on_prev)
        self.btn_next.on_click(self._on_next)
        self.btn_save.on_click(self._on_save)

        self.label_ok.observe(self._on_label_change, names="value")
        self.label_anom.observe(self._on_label_change, names="value")
        self.label_skip.observe(self._on_label_change, names="value")

    def _on_select_change(self, change) -> None:
        v = change["new"]
        if v is None:
            return
        self._render_selected(str(v))

    def _on_prev(self, _btn) -> None:
        opts = list(self.select.options)
        if not opts:
            return
        cur = str(self.select.value)
        i = opts.index(cur)
        if i > 0:
            self.select.value = opts[i - 1]

    def _on_next(self, _btn) -> None:
        opts = list(self.select.options)
        if not opts:
            return
        cur = str(self.select.value)
        i = opts.index(cur)
        if i < len(opts) - 1:
            self.select.value = opts[i + 1]

    def _on_label_change(self, _change) -> None:
        cand_id = str(self.select.value)
        if cand_id not in self._review:
            self._review[cand_id] = {}

        if self.label_ok.value:
            self.label_anom.value = False
            self.label_skip.value = False
            label = "ok"
        elif self.label_anom.value:
            self.label_ok.value = False
            self.label_skip.value = False
            label = "anomalous"
        elif self.label_skip.value:
            self.label_ok.value = False
            self.label_anom.value = False
            label = "skip"
        else:
            label = "unset"

        self._review[cand_id]["label"] = label

    def _on_save(self, _btn) -> None:
        out = Path(self.cfg.review_save_path)
        out.write_text(
            json.dumps(self._review, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.msg.value = f"<div style='color:#1a7f37;'><b>Saved</b> to {out}</div>"

    def _render_selected(self, cand_id: str) -> None:
        row = self.df_top[self.df_top[self.id_col].astype(str) == str(cand_id)]
        if row.empty:
            return
        r = row.iloc[0]

        normal, anomalous, matches = _guess_cutout_pairs(self.cutouts_dir, cand_id)

        self.img_left = _img_widget_from_path(
            normal, width=self.cfg.image_size, height=self.cfg.image_size
        )
        self.img_right = _img_widget_from_path(
            anomalous, width=self.cfg.image_size, height=self.cfg.image_size
        )

        img_hbox = self.ui.children[1]
        left_vbox = img_hbox.children[0]
        right_vbox = img_hbox.children[1]
        left_vbox.children = (left_vbox.children[0], self.img_left)
        right_vbox.children = (right_vbox.children[0], self.img_right)

        meta_cols = [self.id_col, self.rank_col, self.u_col]
        for c in ["ra", "dec", "label_pred", "prob", "score", "anomaly_score"]:
            if c in self.df_top.columns and c not in meta_cols:
                meta_cols.append(c)

        self.meta.value = _build_meta_html(r, meta_cols)

        label = self._review.get(str(cand_id), {}).get("label", "unset")
        self.label_ok.value = label == "ok"
        self.label_anom.value = label == "anomalous"
        self.label_skip.value = label == "skip"

        _highlight_bar(self.fig, str(cand_id))

        extra = ""
        if matches:
            extra = "<br/>".join([m.name for m in matches[:8]])
            if len(matches) > 8:
                extra += "<br/>..."

        self.msg.value = (
            f"<div style='color:#555;'>"
            f"Selected: <b>{cand_id}</b> (cutouts found: {len(matches)})"
            f"{('<br/><b>Matches</b><br/>' + extra) if extra else ''}"
            f"</div>"
        )

    @staticmethod
    def _load_review(path: str) -> Dict[str, Dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
