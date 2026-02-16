from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .annotations import (
    append_annotations,
    count_labels,
    get_retrain_cursor,
    set_retrain_cursor,
)
from .chaos_metrics import compute_chaos_metrics, maybe_parse_series
from .config import OracleConfig, RankingConfig
from .core import fetch_cutouts, load_candidates
from .explainability import explain_top_n, write_explanations_jsonl
from .graph_anomaly import plot_graph_context
from .hybrid_fusion import apply_hybrid_mode
from .logging_utils import log_event
from .stats import annotation_stats, log_stats
from .train import train_from_files
from .viz3d import build_viz3d_figure, load_candidates_table, write_viz3d_html
from .viz_matplotlib import render_cutouts_matplotlib
from .watch import watch_candidates


def _make_cfg(args: argparse.Namespace) -> OracleConfig:
    cfg0 = OracleConfig.default()

    save_dir = Path(args.save_cutouts) if args.save_cutouts else None
    cutout_radius = (
        cfg0.cutout_radius_arcmin
        if args.cutout_radius_arcmin is None
        else float(args.cutout_radius_arcmin)
    )

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
        cutout_radius_arcmin=float(cutout_radius),
        pixels=int(args.pixels),
        n_query=int(args.n_query),
        no_gui=bool(args.no_gui),
        save_cutouts_dir=save_dir,
        offline=bool(args.offline),
        ranking=ranking,
    )


def maybe_retrain(
    cfg: OracleConfig, *, session_id: str | None = None, annotator_id: str | None = None
) -> None:
    total = count_labels(cfg)
    cursor = get_retrain_cursor(cfg)
    new_since = total - cursor

    if new_since < cfg.min_new_labels_for_retrain:
        return

    print(f"Retrain triggered: {new_since} new labels.")
    log_event(
        cfg,
        {"event": "retrain_triggered", "new_labels": new_since, "total_labels": total},
        session_id=session_id,
        annotator_id=annotator_id,
    )

    try:
        subprocess.run(["python", str(cfg.retrain_script)], check=True)
        set_retrain_cursor(cfg, total)
        log_event(
            cfg, {"event": "retrain_success"}, session_id=session_id, annotator_id=annotator_id
        )
    except Exception as e:
        log_event(
            cfg,
            {"event": "retrain_failed", "error": str(e)},
            session_id=session_id,
            annotator_id=annotator_id,
        )
        print(f"Retrain failed: {e}")


def annotate_batch(
    cfg: OracleConfig,
    top: pd.DataFrame,
    *,
    session_id: str | None = None,
    annotator_id: str | None = None,
) -> None:
    label_map = {
        "r": "real_anomaly",
        "a": "artefact",
        "c": "known",
        "j": "new_type",
        "u": "unsure",
    }

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
            choice = (
                input("Label ? [r]éel [a]rtefact [c]onnu [j]nouveau type [u]incertain [s]kip -> ")
                .strip()
                .lower()
            )
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
        log_event(
            cfg,
            {"event": "new_annotations", "count": len(new_rows)},
            session_id=session_id,
            annotator_id=annotator_id,
        )
        maybe_retrain(cfg, session_id=session_id, annotator_id=annotator_id)


def _select_candidates_with_mode(df: pd.DataFrame, cfg: OracleConfig, *, mode: str) -> pd.DataFrame:
    from .ranking import rank_candidates, select_batch
    from .model_io import load_model

    df2 = df
    if mode.lower() == "hybrid":
        df2, _meta = apply_hybrid_mode(df2, overwrite_anomaly_score=True)

    model = load_model(cfg.model_path)
    ranked, _ = rank_candidates(df2, cfg, model=model)
    selected = select_batch(ranked, cfg, k=cfg.n_query)
    return selected.reset_index(drop=True)


def cmd_run(args: argparse.Namespace) -> None:
    cfg = _make_cfg(args)
    session_id = args.session_id
    annotator_id = args.annotator_id

    log_event(
        cfg,
        {"event": "oracle_started", "mode": "poll"},
        session_id=session_id,
        annotator_id=annotator_id,
    )
    print(
        f"AstroOracle running.\ncandidates={cfg.candidates_path} interval={cfg.check_interval_s}s"
    )

    while True:
        try:
            df = load_candidates(cfg)
            if not df.empty:
                top = _select_candidates_with_mode(df, cfg, mode=str(args.mode))
                if not top.empty:
                    annotate_batch(cfg, top, session_id=session_id, annotator_id=annotator_id)
        except KeyboardInterrupt:
            log_event(
                cfg, {"event": "oracle_stopped"}, session_id=session_id, annotator_id=annotator_id
            )
            print("Stopped.")
            break
        except Exception as e:
            log_event(
                cfg,
                {"event": "error", "msg": str(e)},
                session_id=session_id,
                annotator_id=annotator_id,
            )
            print(f"Error: {e}")

        time.sleep(cfg.check_interval_s)


def cmd_watch(args: argparse.Namespace) -> None:
    cfg = _make_cfg(args)
    session_id = args.session_id
    annotator_id = args.annotator_id

    log_event(
        cfg,
        {"event": "oracle_started", "mode": "watch"},
        session_id=session_id,
        annotator_id=annotator_id,
    )
    print(f"AstroOracle watching {cfg.candidates_path}")

    def _on_change() -> None:
        try:
            df = load_candidates(cfg)
            if not df.empty:
                top = _select_candidates_with_mode(df, cfg, mode=str(args.mode))
                if not top.empty:
                    annotate_batch(cfg, top, session_id=session_id, annotator_id=annotator_id)
        except Exception as e:
            log_event(
                cfg,
                {"event": "error", "msg": str(e)},
                session_id=session_id,
                annotator_id=annotator_id,
            )
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

    # Optional hybrid mode affects anomaly_score axis.
    df2 = df
    if str(args.mode).lower() == "hybrid":
        df2, _ = apply_hybrid_mode(df2, overwrite_anomaly_score=True)

    from .model_io import load_model
    from .ranking import rank_candidates, select_batch

    model = load_model(cfg2.model_path)
    ranked, _meta = rank_candidates(df2, cfg2, model=model)
    top = select_batch(ranked, cfg2, k=cfg2.n_query)

    if top.empty:
        print("No selected candidates.")
        return

    from .batch_html import generate_batch_html

    generate_batch_html(cfg2, top, out_dir)
    log_event(cfg2, {"event": "batch_html_generated", "out_dir": str(out_dir), "count": len(top)})
    print(f"Batch HTML written to: {out_dir}/index.html")


def cmd_train(args: argparse.Namespace) -> None:
    cfg = _make_cfg(args)
    metrics = train_from_files(cfg)
    log_event(
        cfg,
        {
            "event": "train",
            "metrics": {"ece": metrics.get("ece"), "n_train": metrics.get("n_train")},
        },
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def cmd_stats(args: argparse.Namespace) -> None:
    cfg = _make_cfg(args)
    a = annotation_stats(cfg)
    logs = log_stats(cfg)
    print(json.dumps({"annotations": a, "logs": logs}, indent=2, ensure_ascii=False))


def cmd_viz3d(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    out_path = Path(args.output)

    df = load_candidates_table(in_path)
    if df.empty:
        print("No candidates.")
        return

    fig = build_viz3d_figure(
        df,
        mode=str(args.mode),
        max_points=int(args.max_points),
        color=args.color,
        cutouts_dir=Path(args.cutouts_dir) if args.cutouts_dir else None,
        cutout_survey=args.cutout_survey,
        embed_cutouts=bool(args.embed_cutouts),
        title=args.title,
    )

    write_viz3d_html(fig, out_path, cdn=bool(args.cdn))
    print(f"viz3d written: {out_path}")


def cmd_serve(args: argparse.Namespace) -> None:
    cfg = _make_cfg(args)

    from .api import create_app

    import uvicorn

    app = create_app(cfg)
    uvicorn.run(app, host=str(args.host), port=int(args.port))


def cmd_graph_anomaly(args: argparse.Namespace) -> None:
    df = load_candidates_table(Path(args.input))
    if df.empty:
        print("No candidates.")
        return

    fig = plot_graph_context(df, k=int(args.k), max_nodes=int(args.max_nodes))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.write_html(out)
    except Exception:
        # Fallback: write JSON.
        out = out.with_suffix(out.suffix + ".json")
        out.write_text(fig.to_json(), encoding="utf-8")

    print(f"graph anomaly written: {out}")


def cmd_explain_top(args: argparse.Namespace) -> None:
    cfg = _make_cfg(args)
    df = load_candidates(cfg)
    if df.empty:
        print("No candidates.")
        return

    from .model_io import load_model
    from .ranking import rank_candidates

    model = load_model(cfg.model_path)
    ranked, meta = rank_candidates(df, cfg, model=model)

    expl = explain_top_n(ranked, n=int(args.n))
    write_explanations_jsonl(expl, Path(args.output))

    log_event(
        cfg, {"event": "explain_top", "n": int(args.n), "out": str(args.output), "meta": meta}
    )
    print(f"explanations written: {args.output}")


def cmd_gaia_cone(args: argparse.Namespace) -> None:
    from .gaia_ingest import gaia_cone_search, write_gaia_table

    res = gaia_cone_search(
        ra_deg=float(args.ra),
        dec_deg=float(args.dec),
        radius_arcmin=float(args.radius_arcmin),
        max_rows=int(args.max_rows),
    )
    write_gaia_table(res.df, args.output)
    print(f"Gaia rows: {res.n_rows} -> {args.output}")


def cmd_gaia_adql(args: argparse.Namespace) -> None:
    from .gaia_ingest import gaia_adql_query, write_gaia_table

    res = gaia_adql_query(adql=str(args.adql), max_rows=int(args.max_rows))
    write_gaia_table(res.df, args.output)
    print(f"Gaia rows: {res.n_rows} -> {args.output}")


def cmd_chaos_score(args: argparse.Namespace) -> None:
    df = load_candidates_table(Path(args.input))
    if df.empty:
        print("No candidates.")
        return

    col = str(args.series_col)
    if col not in df.columns:
        raise SystemExit(f"Missing series column: {col}")

    rows = []
    for _, r in df.iterrows():
        s = maybe_parse_series(r[col])
        if s is None or len(s) < 10:
            continue
        m = compute_chaos_metrics(s, emb_dim=int(args.emb_dim), emb_lag=int(args.emb_lag))
        rows.append(
            {
                "id": str(r.get("id")),
                "chaos_score": m.score,
                "lyapunov_proxy": m.lyapunov_proxy,
                "rqa_rr": m.rqa_recurrence_rate,
                "rqa_det": m.rqa_determinism,
                "rqa_entropy": m.rqa_entropy,
                "n": int(m.meta.get("n", 0)),
            }
        )

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"chaos metrics written: {args.output} ({len(out)} rows)")


def main() -> None:
    p = argparse.ArgumentParser(prog="astrooracle")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
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
        sp.add_argument(
            "--offline", action="store_true", help="Use synthetic cutouts (no network)."
        )
        sp.add_argument("--session-id", default=None)
        sp.add_argument("--annotator-id", default=None)
        sp.add_argument("--acq", default="entropy", choices=["entropy", "margin", "bald", "badge"])
        sp.add_argument("--diversity", default="kcenter", choices=["kcenter", "dpp", "none"])
        sp.add_argument("--w-anomaly", type=float, default=0.35)
        sp.add_argument("--w-acq", type=float, default=0.35)
        sp.add_argument("--w-div", type=float, default=0.20)
        sp.add_argument("--w-prior", type=float, default=0.10)
        sp.add_argument("--acq-temp", type=float, default=1.0)
        sp.add_argument(
            "--mode", default="vanilla", choices=["vanilla", "hybrid"], help="Optional scoring mode"
        )

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

    sp_viz = sub.add_parser("viz3d", help="Generate an interactive Plotly 3D visualization (HTML).")
    sp_viz.add_argument("--input", default="candidates.parquet")
    sp_viz.add_argument("--output", default="viz3d.html")
    sp_viz.add_argument("--mode", default="scatter", choices=["scatter", "globe"])
    sp_viz.add_argument(
        "--color", default=None, help="Column name to color by (e.g. label_pred, uncertainty)."
    )
    sp_viz.add_argument("--max-points", type=int, default=5000)
    sp_viz.add_argument(
        "--cutouts-dir",
        default=None,
        help="Folder containing cutout PNGs (e.g. batch_out/cutouts).",
    )
    sp_viz.add_argument("--cutout-survey", default="DSS2 Red")
    sp_viz.add_argument(
        "--embed-cutouts", action="store_true", help="Embed cutouts as base64 in hover."
    )
    sp_viz.add_argument(
        "--cdn", action="store_true", help="Use Plotly CDN instead of embedding JS."
    )
    sp_viz.add_argument("--title", default=None)
    sp_viz.set_defaults(fn=cmd_viz3d)

    sp_serve = sub.add_parser("serve", help="Run the FastAPI server.")
    add_common(sp_serve)
    sp_serve.add_argument("--host", default="127.0.0.1")
    sp_serve.add_argument("--port", type=int, default=8000)
    sp_serve.set_defaults(fn=cmd_serve)

    sp_g = sub.add_parser(
        "graph-anomaly", help="Graph kNN anomaly context on the celestial sphere (HTML)."
    )
    sp_g.add_argument("--input", default="candidates.parquet")
    sp_g.add_argument("--output", default="graph_anomaly.html")
    sp_g.add_argument("--k", type=int, default=10)
    sp_g.add_argument("--max-nodes", type=int, default=2000)
    sp_g.set_defaults(fn=cmd_graph_anomaly)

    sp_e = sub.add_parser(
        "explain-top", help="Generate shareable explanations JSONL for the top ranked candidates."
    )
    add_common(sp_e)
    sp_e.add_argument("--n", type=int, default=10)
    sp_e.add_argument("--output", default="explanations.jsonl")
    sp_e.set_defaults(fn=cmd_explain_top)

    sp_gc = sub.add_parser("gaia-cone", help="Cone search Gaia DR3 via astroquery.")
    sp_gc.add_argument("--ra", type=float, required=True)
    sp_gc.add_argument("--dec", type=float, required=True)
    sp_gc.add_argument("--radius-arcmin", type=float, default=5.0)
    sp_gc.add_argument("--max-rows", type=int, default=2000)
    sp_gc.add_argument("--output", default="gaia_cone.parquet")
    sp_gc.set_defaults(fn=cmd_gaia_cone)

    sp_ga = sub.add_parser(
        "gaia-adql", help="Run an ADQL SELECT query against Gaia via astroquery."
    )
    sp_ga.add_argument("--adql", type=str, required=True)
    sp_ga.add_argument("--max-rows", type=int, default=20000)
    sp_ga.add_argument("--output", default="gaia_query.parquet")
    sp_ga.set_defaults(fn=cmd_gaia_adql)

    sp_c = sub.add_parser(
        "chaos-score", help="Compute chaos-style metrics for a time series column."
    )
    sp_c.add_argument("--input", default="candidates.parquet")
    sp_c.add_argument("--series-col", default="timeseries")
    sp_c.add_argument("--emb-dim", type=int, default=3)
    sp_c.add_argument("--emb-lag", type=int, default=1)
    sp_c.add_argument("--output", default="chaos_metrics.csv")
    sp_c.set_defaults(fn=cmd_chaos_score)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
