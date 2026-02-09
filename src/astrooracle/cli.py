from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import OracleConfig, RankingConfig
from .core import load_candidates, select_candidates, fetch_cutouts
from .viz_matplotlib import render_cutouts_matplotlib
from .annotations import append_annotations, count_labels, get_retrain_cursor, set_retrain_cursor
from .logging_utils import log_event
from .watch import watch_candidates
from .batch_html import generate_batch_html
from .train import train_from_files
from .stats import annotation_stats, log_stats


def _make_cfg(args: argparse.Namespace) -> OracleConfig:
    cfg0 = OracleConfig.default()
    save_dir = Path(args.save_cutouts) if args.save_cutouts else None
    cutout_radius = cfg0.cutout_radius_arcmin if args.cutout_radius_arcmin is None else float(args.cutout_radius_arcmin)

    ranking = RankingConfig(
        strategy=str(args.acq),
        diversity=str(args.diversity),
        w_anomaly=float(args.w_anomaly),
        w_acq=float(args.w_acq),
        w_div=float(args.w_div),
        w_prior=float(args.w_prior),
        acq_temperature=float(args.acq_temp),
    )

    return OracleConfig(
        candidates_path=Path(args.candidates),
        annot_path=Path(args.annotations),
        log_path=Path(args.log),
        retrain_script=Path(args.retrain_script),
        model_path=Path(args.model_path),
        min_new_labels_for_retrain=int(args.min_new_labels),
        check_interval_s=int(args.interval),
        surveys=list(args.survey),
        cutout_radius_arcmin=cutout_radius,
        pixels=int(args.pixels),
        n_query=int(args.n_query),
        no_gui=bool(args.no_gui),
        save_cutouts_dir=save_dir,
        offline=bool(args.offline),
        ranking=ranking,
    )


def maybe_retrain(cfg: OracleConfig, session_id: str | None = None, annotator_id: str | None = None) -> None:
    total = count_labels(cfg)
    cursor = get_retrain_cursor(cfg)
    new_since = total - cursor
    if new_since >= cfg.min_new_labels_for_retrain:
        print(f"Retrain triggered: {new_since} new labels.")
        log_event(cfg, {"event": "retrain_triggered", "new_labels": new_since, "total_labels": total}, session_id=session_id, annotator_id=annotator_id)
        try:
            subprocess.run(["python", str(cfg.retrain_script)], check=True)
            set_retrain_cursor(cfg, total)
            log_event(cfg, {"event": "retrain_success"}, session_id=session_id, annotator_id=annotator_id)
        except Exception as e:
            log_event(cfg, {"event": "retrain_failed", "error": str(e)}, session_id=session_id, annotator_id=annotator_id)
            print(f"Retrain failed: {e}")


def annotate_batch(cfg: OracleConfig, top: pd.DataFrame, session_id: str | None = None, annotator_id: str | None = None) -> None:
    label_map = {"r": "real_anomaly", "a": "artefact", "c": "known", "j": "new_type", "u": "unsure"}

    new_rows = []
    has_emb = "embedding" in top.columns

    for _, row in top.iterrows():
        title = (
            f"ID: {row['id']} | Rank: {float(row.get('rank_score', float('nan'))):.4f} | "
            f"Anom: {float(row['anomaly_score']):.4f} | "
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
            choice = input("Label ? [r]éel [a]rtefact [c]onnu [j]nouveau type [u]incertain [s]kip -> ").strip().lower()
            if choice in {"r", "a", "c", "j", "u", "s"}:
                break

        if choice == "s":
            continue

        comment = input("Commentaire (Enter pour skip): ").strip()
        entry = {
            "id": str(row["id"]),
            "ra": float(row["ra"]),
            "dec": float(row["dec"]),
            "anomaly_score": float(row["anomaly_score"]),
            "label": label_map[choice],
            "comment": comment,
            "annotated_at": pd.Timestamp.utcnow().isoformat(),
            "annotator_id": annotator_id,
            "acquisition": cfg.ranking.strategy,
            "diversity": cfg.ranking.diversity,
            "model_version": str(cfg.model_path),
        }

        if has_emb and row.get("embedding", None) is not None:
            entry["embedding"] = json.dumps(np.asarray(row["embedding"], dtype=float).tolist())

        new_rows.append(entry)

    if new_rows:
        append_annotations(cfg, new_rows)
        log_event(cfg, {"event": "new_annotations", "count": len(new_rows)}, session_id=session_id, annotator_id=annotator_id)
        maybe_retrain(cfg, session_id=session_id, annotator_id=annotator_id)


def cmd_run(args: argparse.Namespace) -> None:
    cfg = _make_cfg(args)
    session_id = args.session_id
    annotator_id = args.annotator_id
    log_event(cfg, {"event": "oracle_started", "mode": "poll"}, session_id=session_id, annotator_id=annotator_id)
    print(f"AstroOracle running. candidates={cfg.candidates_path} interval={cfg.check_interval_s}s")

    while True:
        try:
            df = load_candidates(cfg)
            if not df.empty:
                top = select_candidates(df, cfg)
                if not top.empty:
                    annotate_batch(cfg, top, session_id=session_id, annotator_id=annotator_id)
        except KeyboardInterrupt:
            log_event(cfg, {"event": "oracle_stopped"}, session_id=session_id, annotator_id=annotator_id)
            print("Stopped.")
            break
        except Exception as e:
            log_event(cfg, {"event": "error", "msg": str(e)}, session_id=session_id, annotator_id=annotator_id)
            print(f"Error: {e}")

        time.sleep(cfg.check_interval_s)


def cmd_watch(args: argparse.Namespace) -> None:
    cfg = _make_cfg(args)
    session_id = args.session_id
    annotator_id = args.annotator_id
    log_event(cfg, {"event": "oracle_started", "mode": "watch"}, session_id=session_id, annotator_id=annotator_id)
    print(f"AstroOracle watching {cfg.candidates_path}")

    def _on_change():
        try:
            df = load_candidates(cfg)
            if not df.empty:
                top = select_candidates(df, cfg)
                if not top.empty:
                    annotate_batch(cfg, top, session_id=session_id, annotator_id=annotator_id)
        except Exception as e:
            log_event(cfg, {"event": "error", "msg": str(e)}, session_id=session_id, annotator_id=annotator_id)
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


def cmd_train(args: argparse.Namespace) -> None:
    cfg = _make_cfg(args)
    metrics = train_from_files(cfg)
    log_event(cfg, {"event": "train", "metrics": {"ece": metrics.get("ece"), "n_train": metrics.get("n_train")}})
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def cmd_stats(args: argparse.Namespace) -> None:
    cfg = _make_cfg(args)
    a = annotation_stats(cfg)
    l = log_stats(cfg)
    print(json.dumps({"annotations": a, "logs": l}, indent=2, ensure_ascii=False))


def cmd_serve(args: argparse.Namespace) -> None:
    cfg = _make_cfg(args)
    from .api import create_app
    import uvicorn

    app = create_app(cfg)
    uvicorn.run(app, host=str(args.host), port=int(args.port))


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

        sp.add_argument("--session-id", default=None)
        sp.add_argument("--annotator-id", default=None)

        sp.add_argument("--acq", default="entropy", choices=["entropy", "margin", "bald", "badge"])
        sp.add_argument("--diversity", default="kcenter", choices=["kcenter", "dpp", "none"])
        sp.add_argument("--w-anomaly", type=float, default=0.35)
        sp.add_argument("--w-acq", type=float, default=0.35)
        sp.add_argument("--w-div", type=float, default=0.20)
        sp.add_argument("--w-prior", type=float, default=0.10)
        sp.add_argument("--acq-temp", type=float, default=1.0)

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

    sp_train = sub.add_parser("train", help="Train/update the ensemble model from annotations.")
    add_common(sp_train)
    sp_train.set_defaults(fn=cmd_train)

    sp_stats = sub.add_parser("stats", help="Quick stats from annotations and logs.")
    add_common(sp_stats)
    sp_stats.set_defaults(fn=cmd_stats)

    sp_serve = sub.add_parser("serve", help="Run the FastAPI server.")
    add_common(sp_serve)
    sp_serve.add_argument("--host", default="127.0.0.1")
    sp_serve.add_argument("--port", type=int, default=8000)
    sp_serve.set_defaults(fn=cmd_serve)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
