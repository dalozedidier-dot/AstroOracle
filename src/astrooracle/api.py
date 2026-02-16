from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .annotations import append_annotations, read_annotations
from .chaos_metrics import compute_chaos_metrics, maybe_parse_series
from .config import OracleConfig
from .core import fetch_cutouts
from .explainability import explain_top_n
from .graph_anomaly import graph_anomaly, plot_graph_context
from .hybrid_fusion import apply_hybrid_mode
from .logging_utils import log_event
from .model_io import load_model
from .ranking import rank_candidates, select_batch
from .schemas import AnnotationRecord, CandidateRecord
from .utils_images import png_b64_data_uri
from .viz3d import build_viz3d_figure, load_candidates_table


class RankRequest(BaseModel):
    candidates: List[CandidateRecord]
    k: int = 16


class RankResponse(BaseModel):
    ranked_ids: List[str]
    batch: List[CandidateRecord]
    meta: dict


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _templates() -> Jinja2Templates:
    """Resolve templates directory.

    Preference order:
    1) repo root templates/ (developer mode)
    2) packaged astrooracle/_templates (installed mode)
    """

    tmpl_dir = _project_root() / "templates"
    if tmpl_dir.exists():
        return Jinja2Templates(directory=str(tmpl_dir))

    try:
        from importlib.resources import files  # type: ignore

        pkg_dir = files("astrooracle").joinpath("_templates")
        return Jinja2Templates(directory=str(pkg_dir))
    except Exception:
        return Jinja2Templates(directory=str(tmpl_dir))


def _load_candidates_any(path: Path) -> pd.DataFrame:
    try:
        return load_candidates_table(path)
    except Exception:
        return pd.DataFrame()


def _top_candidates(cfg: OracleConfig, n: int = 5) -> List[CandidateRecord]:
    df = _load_candidates_any(cfg.candidates_path)
    if df.empty:
        return []
    df = df.sort_values("anomaly_score", ascending=False).head(int(n)).reset_index(drop=True)
    out = []
    for row in df.to_dict(orient="records"):
        # CandidateRecord is strict about fields; allow extra by filtering.
        base = {
            "id": str(row.get("id")),
            "ra": float(row.get("ra")),
            "dec": float(row.get("dec")),
            "anomaly_score": float(row.get("anomaly_score")),
        }
        for k in ("timestamp", "survey", "mag", "snr", "ruwe"):
            if k in row and row[k] is not None:
                base[k] = row[k]
        if "embedding" in row and row["embedding"] is not None:
            try:
                base["embedding"] = list(np.asarray(row["embedding"], dtype=float))
            except Exception:
                pass
        out.append(CandidateRecord(**base))
    return out


def _read_upload_table(upload: UploadFile) -> pd.DataFrame:
    name = (upload.filename or "").lower()
    raw = upload.file.read()
    if not raw:
        return pd.DataFrame()

    from io import BytesIO

    bio = BytesIO(raw)
    if name.endswith((".parquet", ".pq")):
        return pd.read_parquet(bio)
    if name.endswith((".csv", ".tsv")):
        sep = "," if name.endswith(".csv") else "\t"
        return pd.read_csv(bio, sep=sep)

    raise HTTPException(status_code=400, detail="Unsupported upload format. Use CSV/TSV/Parquet.")


def create_app(cfg: Optional[OracleConfig] = None) -> FastAPI:
    cfg = cfg or OracleConfig.default()
    app = FastAPI(title="AstroOracle API", version="0.3.0")
    templates = _templates()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        top = _top_candidates(cfg, n=5)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "top_candidates": top,
                "project": {"name": "AstroOracle", "version": app.version},
            },
        )

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request, max_rows: int = 5000):
        ann = read_annotations(cfg)
        ann_preview = (
            ann.tail(int(min(len(ann), max_rows))).to_dict(orient="records")
            if not ann.empty
            else []
        )
        counts = (
            ann["label"].value_counts().to_dict()
            if (not ann.empty and "label" in ann.columns)
            else {}
        )
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "annotation_counts": counts,
                "annotations_preview": ann_preview,
                "paths": {
                    "candidates": str(cfg.candidates_path),
                    "annotations": str(cfg.annot_path),
                },
            },
        )

    @app.get("/api/top")
    def api_top(n: int = 5):
        top = _top_candidates(cfg, n=int(n))
        return {"top": [c.model_dump() for c in top]}

    @app.get("/api/annotations")
    def api_annotations(limit: int = 5000):
        ann = read_annotations(cfg)
        if ann.empty:
            return {"rows": [], "count": 0}
        out = ann.tail(int(limit)).to_dict(orient="records")
        return {"rows": out, "count": int(len(ann))}

    @app.get("/api/export")
    def api_export(format: str = "csv"):
        ann = read_annotations(cfg)
        if ann.empty:
            raise HTTPException(status_code=404, detail="No annotations yet.")

        if format.lower() == "csv":
            return HTMLResponse(content=ann.to_csv(index=False), media_type="text/csv")
        if format.lower() == "json":
            return ann.to_dict(orient="records")

        raise HTTPException(status_code=400, detail="format must be csv or json")

    @app.get("/api/cutouts")
    def api_cutouts(
        ra: float = Query(..., description="RA in degrees"),
        dec: float = Query(..., description="Dec in degrees"),
    ):
        try:
            cutouts = fetch_cutouts(float(ra), float(dec), cfg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

        imgs = []
        for survey, data in cutouts:
            imgs.append({"survey": survey, "png": png_b64_data_uri(data)})

        return {"ra": float(ra), "dec": float(dec), "cutouts": imgs}

    @app.post("/rank", response_model=RankResponse)
    def rank(req: RankRequest):
        if not req.candidates:
            return RankResponse(ranked_ids=[], batch=[], meta={"n": 0})

        df = pd.DataFrame([c.model_dump() for c in req.candidates])
        model = load_model(cfg.model_path)
        ranked, meta = rank_candidates(df, cfg, model=model)
        batch = select_batch(ranked, cfg, k=req.k)
        out = [CandidateRecord(**row) for row in batch.to_dict(orient="records")]
        log_event(cfg, {"event": "api_rank", "count": len(out)})
        return RankResponse(
            ranked_ids=ranked["id"].astype(str).tolist(),
            batch=out,
            meta=meta,
        )

    @app.post("/run", response_model=RankResponse)
    async def run_oracle(
        request: Request,
        ra: Optional[float] = Form(None),
        dec: Optional[float] = Form(None),
        k: int = Form(16),
        mode: str = Form("vanilla"),
        candidates: Optional[UploadFile] = File(None),
    ):
        """CLI-like endpoint: upload a table or use the server candidates file, then rank + select."""

        if candidates is not None:
            df = _read_upload_table(candidates)
        else:
            df = _load_candidates_any(cfg.candidates_path)

        if df.empty:
            return RankResponse(ranked_ids=[], batch=[], meta={"n": 0, "msg": "No candidates"})

        # Optional focus around RA/Dec: keep candidates within a radius (deg) if provided.
        if ra is not None and dec is not None:
            try:
                ra0 = float(ra)
                dec0 = float(dec)
                # quick spherical approx in degrees.
                dra = (pd.to_numeric(df["ra"], errors="coerce").to_numpy(float) - ra0) * np.cos(
                    np.deg2rad(dec0)
                )
                ddec = pd.to_numeric(df["dec"], errors="coerce").to_numpy(float) - dec0
                dist = np.sqrt(dra * dra + ddec * ddec)
                df = df.loc[dist <= 2.0].copy()  # 2 deg default window
            except Exception:
                pass

        meta_extra = {}
        if mode.lower() == "hybrid":
            df, m = apply_hybrid_mode(df, overwrite_anomaly_score=True)
            meta_extra.update(m)

        model = load_model(cfg.model_path)
        ranked, meta = rank_candidates(df, cfg, model=model)
        meta = {**meta, **meta_extra, "mode": mode}
        batch = select_batch(ranked, cfg, k=int(k))
        out = [CandidateRecord(**row) for row in batch.to_dict(orient="records")]
        log_event(cfg, {"event": "api_run", "count": len(out), "mode": mode})
        return RankResponse(ranked_ids=ranked["id"].astype(str).tolist(), batch=out, meta=meta)

    @app.post("/annotate")
    def annotate(rows: List[AnnotationRecord]):
        if not rows:
            return {"ok": True, "count": 0}
        append_annotations(cfg, [r.model_dump() for r in rows])
        log_event(cfg, {"event": "api_annotate", "count": len(rows)})
        return {"ok": True, "count": len(rows)}

    @app.get("/viz3d")
    def viz3d(
        mode: str = "scatter",
        max_points: int = 5000,
        color: Optional[str] = None,
    ):
        """Return a Plotly figure JSON for interactive 3D triage."""

        df = _load_candidates_any(cfg.candidates_path)
        if df.empty:
            return {"data": [], "layout": {"title": {"text": "No candidates"}}}

        fig = build_viz3d_figure(df, mode=mode, max_points=max_points, color=color)
        return json.loads(fig.to_json())

    @app.get("/graph_anomaly")
    def graph_anomaly_endpoint(
        k: int = 10,
        max_nodes: int = 2000,
        bridges_top: int = 50,
    ):
        df = _load_candidates_any(cfg.candidates_path)
        if df.empty:
            return {"ok": True, "n": 0}

        res = graph_anomaly(df, k=int(k), max_nodes=int(max_nodes), bridges_top=int(bridges_top))
        # Also provide a Plotly preview.
        try:
            fig = plot_graph_context(df, k=int(k), max_nodes=int(max_nodes))
            fig_json = json.loads(fig.to_json())
        except Exception:
            fig_json = None

        return {
            "ok": True,
            "meta": res.meta,
            "n_nodes": res.n_nodes,
            "n_edges": res.n_edges,
            "node_metrics": res.node_metrics.to_dict(orient="records"),
            "edge_bridges": res.edge_bridges.to_dict(orient="records"),
            "fig": fig_json,
        }

    @app.get("/explain")
    def explain_endpoint(n: int = 10):
        df = _load_candidates_any(cfg.candidates_path)
        if df.empty:
            return {"rows": [], "n": 0}

        model = load_model(cfg.model_path)
        ranked, meta = rank_candidates(df, cfg, model=model)
        expl = explain_top_n(ranked, n=int(n))
        return {
            "meta": meta,
            "rows": [
                {
                    "id": e.candidate_id,
                    "method": e.method,
                    "score": e.score,
                    "top_features": e.top_features,
                    "prompt": e.prompt,
                    "meta": e.meta,
                }
                for e in expl
            ],
            "n": int(len(expl)),
        }

    @app.get("/chaos")
    def chaos_endpoint(candidate_id: Optional[str] = None):
        df = _load_candidates_any(cfg.candidates_path)
        if df.empty:
            return {"ok": True, "score": 0.0, "msg": "No candidates"}

        row = None
        if candidate_id is not None:
            hit = df.loc[df["id"].astype(str) == str(candidate_id)]
            if not hit.empty:
                row = hit.iloc[0]

        if row is None:
            row = df.iloc[0]

        # Best-effort time series extraction.
        for key in ("timeseries", "ts_values", "flux_series", "series"):
            if key in row:
                s = maybe_parse_series(row[key])
                if s is not None and len(s) >= 10:
                    m = compute_chaos_metrics(s)
                    return {
                        "ok": True,
                        "id": str(row.get("id")),
                        "metrics": {
                            "score": m.score,
                            "lyapunov_proxy": m.lyapunov_proxy,
                            "rqa_rr": m.rqa_recurrence_rate,
                            "rqa_det": m.rqa_determinism,
                            "rqa_entropy": m.rqa_entropy,
                            "meta": m.meta,
                        },
                    }

        return {
            "ok": True,
            "id": str(row.get("id")),
            "metrics": None,
            "msg": "No time series column found (expected timeseries/ts_values/flux_series).",
        }

    @app.get("/gaia/cone")
    def gaia_cone(
        ra: float,
        dec: float,
        radius_arcmin: float = 5.0,
        max_rows: int = 2000,
    ):
        try:
            from .gaia_ingest import gaia_cone_search

            res = gaia_cone_search(
                ra_deg=float(ra),
                dec_deg=float(dec),
                radius_arcmin=float(radius_arcmin),
                max_rows=int(max_rows),
            )
            return {"n": res.n_rows, "meta": res.meta, "rows": res.df.to_dict(orient="records")}
        except RuntimeError as e:
            raise HTTPException(status_code=501, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.get("/gaia/adql")
    def gaia_adql(adql: str = Query(..., min_length=8), max_rows: int = 20000):
        try:
            from .gaia_ingest import gaia_adql_query

            res = gaia_adql_query(adql=adql, max_rows=int(max_rows))
            return {"n": res.n_rows, "meta": res.meta, "rows": res.df.to_dict(orient="records")}
        except RuntimeError as e:
            raise HTTPException(status_code=501, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    return app


# Backwards-compatible ASGI app (used by tests and by `uvicorn astrooracle.api:app`).
app = create_app()
