from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import OracleConfig
from .core import load_candidates, select_candidates, fetch_cutouts
from .viz_matplotlib import render_cutouts_matplotlib
from .annotations import append_annotations, count_labels, get_retrain_cursor, set_retrain_cursor
from .logging_utils import log_event
from .watch import watch_candidates
from .batch_html import generate_batch_html


def _make_cfg(args: argparse.Namespace) -> OracleConfig:
    cfg0 = OracleConfig.default()
    save_dir = Path(args.save_cutouts) if args.save_cutouts else None
    cutout_radius = cfg0.cutout_radius if args.cutout_radius_arcmin is None else (float(args.cutout_radius_arcmin) * cfg0.cutout_radius.unit)
    return OracleConfig(
        candidates_path=Path(args.candidates),
        annot_path=Path(args.annotations),
        log_path=Path(args.log),
        retrain_script=Path(args.retrain_script),
        model_path=Path(args.model_path),
        min_new_labels_for_retrain=int(args.min_new_labels),
        check_interval_s=int(args.interval),
        surveys=list(args.survey),
        cutout_radius=cutout_radius,
        pixels=int(args.pixels),
        n_query=int(args.n_query),
        no_gui=bool(args.no_gui),
        save_cutouts_dir=save_dir,
        offline=bool(args.offline),
    )


def maybe_retrain(cfg: OracleConfig) -> None:
    total = count_labels(cfg)
    cursor = get_retrain_cursor(cfg)
    new_since = total - cursor
    if new_since >= cfg.min_new_labels_for_retrain:
        print(f"Retrain triggered: {new_since} new labels.")
        log_event(cfg, {"event": "retrain_triggered", "new_labels": new_since, "total_labels": total})
        try:
            subprocess.run(["python", str(cfg.retrain_script)], check=True)
            set_retrain_cursor(cfg, total)
            log_event(cfg, {"event": "retrain_success"})
        except Exception as e:
            log_event(cfg, {"event": "retrain_failed", "error": str(e)})
            print(f"Retrain failed: {e}")


def annotate_batch(cfg: OracleConfig, top: pd.DataFrame) -> None:
    label_map = {"r": "real_anomaly", "a": "artefact", "c": "known", "j": "new_type"}

    new_rows = []
    has_emb = "embedding" in top.columns

    for _, row in top.iterrows():
        title = (
            f"ID: {row['id']} | Score: {float(row['anomaly_score']):.4f} | "
            f"RA: {float(row['ra']):.5f} Dec: {float(row['dec']):.5f}"
        )
        print("\n" + "=" * 60)
        print(title)

        cutouts = fetch_cutouts(float(row["ra"]), float(row["dec"]), cfg)

        save_path = None
        if cfg.save_cutouts_dir is not None:
            safe_id = str(row["id"]).replace("/", "_")
            save_path = cfg.save_cutouts_dir / f"{safe_id}.png"

        render_cutouts_matplotlib(cutouts, title, cfg, save_path=save_path)

        while True:
            choice = input("Label ? [r]éel [a]rtefact [c]onnu [j]junk/nouveau [s]kip -> ").strip().lower()
            if choice in {"r", "a", "c", "j", "s"}:
                break

        if choice == "s":
            continue

        comment = input("Commentaire (Enter pour skip): ").strip()
        entry = {
            "id": row["id"],
            "ra": float(row["ra"]),
            "dec": float(row["dec"]),
            "anomaly_score": float(row["anomaly_score"]),
            "label": label_map[choice],
            "comment": comment,
            "annotated_at": pd.Timestamp.utcnow().isoformat(),
        }

        if has_emb and row.get("embedding", None) is not None:
            entry["embedding"] = json.dumps(np.asarray(row["embedding"], dtype=float).tolist())

        new_rows.append(entry)

    if new_rows:
        append_annotations(cfg, new_rows)
        log_event(cfg, {"event": "new_annotations", "count": len(new_rows)})
        maybe_retrain(cfg)


def cmd_run(args: argparse.Namespace) -> None:
    cfg = _make_cfg(args)
    log_event(cfg, {"event": "oracle_started", "mode": "poll"})
    print(f"AstroOracle running. candidates={cfg.candidates_path} interval={cfg.check_interval_s}s")

    while True:
        try:
            df = load_candidates(cfg)
            if not df.empty:
                top = select_candidates(df, cfg)
                if not top.empty:
                    annotate_batch(cfg, top)
        except KeyboardInterrupt:
            log_event(cfg, {"event": "oracle_stopped"})
            print("Stopped.")
            break
        except Exception as e:
            log_event(cfg, {"event": "error", "msg": str(e)})
            print(f"Error: {e}")

        time.sleep(cfg.check_interval_s)


def cmd_watch(args: argparse.Namespace) -> None:
    cfg = _make_cfg(args)
    log_event(cfg, {"event": "oracle_started", "mode": "watch"})
    print(f"AstroOracle watching {cfg.candidates_path}")

    def _on_change():
        try:
            df = load_candidates(cfg)
            if not df.empty:
                top = select_candidates(df, cfg)
                if not top.empty:
                    annotate_batch(cfg, top)
        except Exception as e:
            log_event(cfg, {"event": "error", "msg": str(e)})
            print(f"Error: {e}")

    watch_candidates(cfg.candidates_path, _on_change)


def cmd_batch_html(args: argparse.Namespace) -> None:
    cfg = _make_cfg(args)
    out_dir = Path(args.out_dir)
    df = load_candidates(cfg)
    if df.empty:
        print("No candidates.")
        return

    cfg2 = OracleConfig(**{**cfg.__dict__, "n_query": int(args.n_query)})
    top = select_candidates(df, cfg2)
    if top.empty:
        print("No selected candidates.")
        return

    generate_batch_html(cfg2, top, out_dir)
    log_event(cfg2, {"event": "batch_html_generated", "out_dir": str(out_dir), "count": len(top)})
    print(f"Batch HTML written to: {out_dir}/index.html")


def main() -> None:
    p = argparse.ArgumentParser(prog="astrooracle")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--candidates", default="candidates.parquet")
        sp.add_argument("--annotations", default="annotations.csv")
        sp.add_argument("--log", default="oracle_log.jsonl")
        sp.add_argument("--retrain-script", default="scripts/retrain_model.py")
        sp.add_argument("--model-path", default="best_model.pkl")
        sp.add_argument("--min-new-labels", type=int, default=10)
        sp.add_argument("--interval", type=int, default=300)
        sp.add_argument("--n-query", type=int, default=6)
        sp.add_argument("--pixels", type=int, default=400)
        sp.add_argument("--cutout-radius-arcmin", type=float, default=None)
        sp.add_argument("--survey", action="append", default=["DSS2 Red", "2MASS J"])
        sp.add_argument("--no-gui", action="store_true")
        sp.add_argument("--save-cutouts", default=None)
        sp.add_argument("--offline", action="store_true", help="Use synthetic cutouts (no network).")

    sp_run = sub.add_parser("run", help="Poll candidates file and annotate in CLI.")
    add_common(sp_run)
    sp_run.set_defaults(fn=cmd_run)

    sp_watch = sub.add_parser("watch", help="Watch candidates file changes (watchdog).")
    add_common(sp_watch)
    sp_watch.set_defaults(fn=cmd_watch)

    sp_html = sub.add_parser("batch-html", help="Generate static HTML batch annotator.")
    add_common(sp_html)
    sp_html.add_argument("--out-dir", required=True)
    sp_html.set_defaults(fn=cmd_batch_html)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
