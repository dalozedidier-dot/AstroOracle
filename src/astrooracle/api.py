from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import OracleConfig
from .model_io import load_model
from .ranking import rank_candidates, select_batch
from .schemas import AnnotationRecord, CandidateRecord
from .annotations import append_annotations
from .logging_utils import log_event


class RankRequest(BaseModel):
    candidates: List[CandidateRecord]
    k: int = 16


class RankResponse(BaseModel):
    ranked_ids: List[str]
    batch: List[CandidateRecord]
    meta: dict


def create_app(cfg: Optional[OracleConfig] = None) -> FastAPI:
    cfg = cfg or OracleConfig.default()
    app = FastAPI(title="AstroOracle API", version="0.2.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

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

    @app.post("/annotate")
    def annotate(rows: List[AnnotationRecord]):
        if not rows:
            return {"ok": True, "count": 0}
        append_annotations(cfg, [r.model_dump() for r in rows])
        log_event(cfg, {"event": "api_annotate", "count": len(rows)})
        return {"ok": True, "count": len(rows)}

    return app
