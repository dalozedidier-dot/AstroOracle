from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel

from .annotations import append_annotations
from .config import OracleConfig
from .core import fetch_cutouts
from .logging_utils import log_event
from .model_io import load_model
from .ranking import rank_candidates, select_batch
from .schemas import AnnotationRecord, CandidateRecord


class RankRequest(BaseModel):
    candidates: List[CandidateRecord]
    k: int = 16


class RankResponse(BaseModel):
    ranked_ids: List[str]
    batch: List[CandidateRecord]
    meta: Dict[str, Any]


def _resolve_templates_dir() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        # repo root templates/ (dev mode)
        here.parents[2] / "templates",
        # packaged templates/ (if you later decide to ship them)
        here.parent / "templates",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _synthetic_candidates(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0.0, 360.0, size=n)
    dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
    score = np.clip(rng.normal(0.5, 0.18, size=n), 0.0, 1.0)
    df = pd.DataFrame(
        {
            "id": [f"demo_{i:04d}" for i in range(n)],
            "ra": ra,
            "dec": dec,
            "anomaly_score": score,
        }
    )
    return df


def _load_candidates_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    suf = path.suffix.lower()
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suf in {".csv", ".tsv"}:
        sep = "," if suf == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported input format: {path.suffix} (expected parquet/csv/tsv)")


def _array_to_png_b64(arr2d: np.ndarray) -> str:
    arr = np.asarray(arr2d, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        norm = np.zeros_like(arr, dtype=np.uint8)
    else:
        lo = float(np.percentile(finite, 1))
        hi = float(np.percentile(finite, 99))
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        norm = (norm * 255).astype(np.uint8)

    img = Image.fromarray(norm)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _df_to_candidate_dicts(df: pd.DataFrame, k: int) -> List[Dict[str, Any]]:
    cols = ["id", "ra", "dec", "anomaly_score"]
    extra = [c for c in ("rank_score", "uncertainty", "label_pred") if c in df.columns]
    out_cols = cols + extra

    out: List[Dict[str, Any]] = []
    for _, r in df.head(k).iterrows():
        d: Dict[str, Any] = {}
        for c in out_cols:
            v = r.get(c, None)
            if isinstance(v, (np.floating, np.integer)):
                v = v.item()
            d[c] = v
        d["id"] = str(d["id"])
        d["ra"] = float(d["ra"])
        d["dec"] = float(d["dec"])
        d["anomaly_score"] = float(d["anomaly_score"])
        out.append(d)
    return out


def create_app(cfg: Optional[OracleConfig] = None) -> FastAPI:
    cfg = cfg or OracleConfig.default()

    app = FastAPI(title="AstroOracle API", version="0.2.0")
    templates_dir = _resolve_templates_dir()
    templates = Jinja2Templates(directory=str(templates_dir))

    def get_top_candidates(k: int = 5) -> List[Dict[str, Any]]:
        df = _load_candidates_any(cfg.candidates_path)
        if df.empty:
            df = _synthetic_candidates()

        model = load_model(cfg.model_path)
        ranked, _meta = rank_candidates(df, cfg, model=model)
        batch = select_batch(ranked, cfg, k=k)
        return _df_to_candidate_dicts(batch, k=k)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        top = get_top_candidates(k=5)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "top_candidates": top,
                "top_candidates_json": json.dumps(top, ensure_ascii=False),
            },
        )

    @app.get("/api/top")
    def api_top(k: int = 5) -> Dict[str, Any]:
        top = get_top_candidates(k=int(k))
        return {"top": top, "k": len(top)}

    @app.get("/api/cutouts")
    def api_cutouts(ra: float, dec: float, mode: str = "standard") -> Dict[str, Any]:
        # `mode` is currently a UI toggle. It is passed through so you can later
        # branch to non-linear time-series metrics without breaking the API.
        _ = str(mode)

        cutouts = fetch_cutouts(float(ra), float(dec), cfg)
        out = [
            {"survey": survey, "png_b64": _array_to_png_b64(arr)} for (survey, arr) in cutouts
        ]
        return {"ra": float(ra), "dec": float(dec), "cutouts": out}

    @app.post("/run")
    async def run_oracle(
        ra: Optional[float] = Form(default=None),
        dec: Optional[float] = Form(default=None),
        k: int = Form(default=5),
        mode: str = Form(default="standard"),
        candidates_file: Optional[UploadFile] = File(default=None),
    ) -> Dict[str, Any]:
        df: pd.DataFrame
        if candidates_file is not None:
            raw = await candidates_file.read()
            name = candidates_file.filename or "candidates.csv"
            suf = Path(name).suffix.lower()
            try:
                if suf in {".parquet", ".pq"}:
                    df = pd.read_parquet(io.BytesIO(raw))
                elif suf in {".csv", ".tsv"}:
                    sep = "," if suf == ".csv" else "\t"
                    df = pd.read_csv(io.BytesIO(raw), sep=sep)
                else:
                    raise ValueError(f"Unsupported upload format: {suf} (expected parquet/csv/tsv)")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Cannot parse upload: {e}") from e
        else:
            df = _load_candidates_any(cfg.candidates_path)
            if df.empty:
                df = _synthetic_candidates()

        required = {"id", "ra", "dec", "anomaly_score"}
        missing = required - set(df.columns)
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {sorted(missing)}")

        if ra is not None and dec is not None:
            # Ensure that a manual (ra, dec) point can be previewed even if absent from the file.
            exists = ((df["ra"].astype(float) - float(ra)).abs() < 1e-9) & (
                (df["dec"].astype(float) - float(dec)).abs() < 1e-9
            )
            if not bool(exists.any()):
                med = float(pd.to_numeric(df["anomaly_score"], errors="coerce").median())
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            [
                                {
                                    "id": "manual_point",
                                    "ra": float(ra),
                                    "dec": float(dec),
                                    "anomaly_score": med,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

        model = load_model(cfg.model_path)
        ranked, meta = rank_candidates(df, cfg, model=model)
        batch_df = select_batch(ranked, cfg, k=int(k))
        batch = _df_to_candidate_dicts(batch_df, k=int(k))

        log_event(cfg, {"event": "api_run", "count": len(batch), "mode": str(mode)})

        return {
            "mode": str(mode),
            "meta": meta,
            "batch": batch,
            "ranked_ids": ranked["id"].astype(str).tolist(),
        }

    @app.post("/rank", response_model=RankResponse)
    def rank(req: RankRequest) -> RankResponse:
        if not req.candidates:
            return RankResponse(ranked_ids=[], batch=[], meta={"n": 0})

        df = pd.DataFrame([c.model_dump() for c in req.candidates])
        model = load_model(cfg.model_path)
        ranked, meta = rank_candidates(df, cfg, model=model)
        batch = select_batch(ranked, cfg, k=req.k)
        out = [CandidateRecord(**row) for row in batch.to_dict(orient="records")]
        log_event(cfg, {"event": "api_rank", "count": len(out)})
        return RankResponse(ranked_ids=ranked["id"].astype(str).tolist(), batch=out, meta=meta)

    @app.post("/annotate")
    def annotate(rows: List[AnnotationRecord]) -> Dict[str, Any]:
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
    ) -> Dict[str, Any]:
        '''Return a Plotly figure JSON for interactive 3D triage.

        This endpoint is optional: it requires the `plotly` extra.
        '''
        try:
            from .viz3d import build_viz3d_figure, load_candidates_table
        except Exception as e:  # pragma: no cover
            raise HTTPException(
                status_code=501,
                detail=(
                    f"Plotly viz3d not available: {e}. "
                    "Install extras: pip install -e '.[plotly]'"
                ),
            ) from e

        df = load_candidates_table(cfg.candidates_path)
        if df.empty:
            return {"data": [], "layout": {"title": {"text": "No candidates"}}}

        fig = build_viz3d_figure(df, mode=mode, max_points=max_points, color=color)
        return json.loads(fig.to_json())

    return app

# Backwards-compatible ASGI app (used by tests and by `uvicorn astrooracle.api:app`).
app = create_app()
