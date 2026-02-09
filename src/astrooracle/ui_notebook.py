from __future__ import annotations

import json
from typing import Callable, Any, Dict, List

import numpy as np
import pandas as pd

try:
    import ipywidgets as w
    from IPython.display import display, clear_output
except Exception:  # pragma: no cover
    w = None  # type: ignore
    display = None  # type: ignore
    clear_output = None  # type: ignore

from .config import OracleConfig

LABELS = [
    ("Réel", "real_anomaly"),
    ("Artefact", "artefact"),
    ("Connu", "known"),
    ("Nouveau type", "new_type"),
    ("Skip", "skip"),
]


class OracleUI:
    def __init__(
        self,
        cfg: OracleConfig,
        load_candidates_fn: Callable[[OracleConfig], pd.DataFrame],
        select_candidates_fn: Callable[[pd.DataFrame, OracleConfig], pd.DataFrame],
        fetch_cutouts_fn: Callable[[float, float, OracleConfig], Any],
        render_fn: Callable[[Any, str], None],
        append_annotations_fn: Callable[[OracleConfig, List[Dict[str, Any]]], None],
    ):
        if w is None:
            raise RuntimeError(
                "ipywidgets not installed. Install extras: pip install -e '.[notebook]'"
            )

        self.cfg = cfg
        self.load_candidates = load_candidates_fn
        self.select_candidates = select_candidates_fn
        self.fetch_cutouts = fetch_cutouts_fn
        self.render = render_fn
        self.append_annotations = append_annotations_fn

        self.batch = pd.DataFrame()
        self.idx = 0

        self.title = w.HTML("")
        self.comment = w.Textarea(
            description="Commentaire",
            layout=w.Layout(width="100%", height="80px"),
        )
        self.next_btn = w.Button(description="Next")
        self.reload_btn = w.Button(description="Reload candidates", button_style="info")
        self.out = w.Output()

        self.label_buttons = []
        for txt, lab in LABELS:
            b = w.Button(description=txt)
            b.on_click(lambda _, lab=lab: self.on_label(lab))
            self.label_buttons.append(b)

        self.next_btn.on_click(lambda _: self.on_next())
        self.reload_btn.on_click(lambda _: self.on_reload())

        controls = w.HBox(self.label_buttons + [self.next_btn, self.reload_btn])
        display(w.VBox([self.title, self.comment, controls, self.out]))

        self.on_reload()

    def on_reload(self):
        df = self.load_candidates(self.cfg)
        if df.empty:
            self.batch = pd.DataFrame()
            self.idx = 0
            with self.out:
                clear_output()
                print("Aucun candidat.")
            return
        self.batch = self.select_candidates(df, self.cfg)
        self.idx = 0
        self.show_current()

    def show_current(self):
        if self.batch.empty or self.idx >= len(self.batch):
            with self.out:
                clear_output()
                print("Batch terminé.")
            return

        row = self.batch.iloc[self.idx]
        title = (
            f"ID: {row['id']} | score: {float(row['anomaly_score']):.4f} | "
            f"RA: {float(row['ra']):.5f} Dec: {float(row['dec']):.5f}"
        )
        self.title.value = f"<b>{title}</b>"

        cutouts = self.fetch_cutouts(float(row["ra"]), float(row["dec"]), self.cfg)
        with self.out:
            clear_output()
            self.render(cutouts, title)

    def on_label(self, label: str):
        if label == "skip":
            self.on_next()
            return

        row = self.batch.iloc[self.idx]
        entry: Dict[str, Any] = {
            "id": row["id"],
            "ra": float(row["ra"]),
            "dec": float(row["dec"]),
            "anomaly_score": float(row["anomaly_score"]),
            "label": label,
            "comment": self.comment.value.strip(),
            "annotated_at": pd.Timestamp.utcnow().isoformat(),
        }
        if "embedding" in self.batch.columns and row.get("embedding", None) is not None:
            entry["embedding"] = json.dumps(np.asarray(row["embedding"], dtype=float).tolist())

        self.append_annotations(self.cfg, [entry])
        self.comment.value = ""
        self.on_next()

    def on_next(self):
        self.idx += 1
        self.show_current()
