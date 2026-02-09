from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


LabelType = Literal[
    "real_anomaly",
    "artefact",
    "known",
    "new_type",
    "unsure",
]


class CandidateRecord(BaseModel):
    id: str = Field(..., description="Stable candidate identifier.")
    ra: float = Field(..., description="Right ascension in degrees.")
    dec: float = Field(..., description="Declination in degrees.")
    anomaly_score: float = Field(
        ...,
        description="Upstream anomaly score (higher = more anomalous).",
    )

    # Optional context
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp for candidate epoch.")
    survey: Optional[str] = Field(default=None, description="Upstream survey tag.")
    mag: Optional[float] = Field(default=None, description="Apparent magnitude if available.")
    snr: Optional[float] = Field(default=None, description="Signal-to-noise ratio if available.")
    ruwe: Optional[float] = Field(default=None, description="Gaia RUWE if available.")

    # Model inputs
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding.")
    features: Optional[Dict[str, float]] = Field(
        default=None,
        description="Structured numeric features.",
    )
    meta: Optional[Dict[str, Any]] = Field(default=None, description="Free-form metadata.")


class AnnotationRecord(BaseModel):
    id: str
    label: LabelType
    comment: str = ""
    annotated_at: str
    annotator_id: Optional[str] = None

    # Snapshots for traceability
    model_version: Optional[str] = None
    acquisition: Optional[str] = None
    acquisition_version: Optional[str] = None
    cutout_hash: Optional[str] = None

    # Optional payload
    ra: Optional[float] = None
    dec: Optional[float] = None
    anomaly_score: Optional[float] = None
    embedding: Optional[List[float]] = None
    features: Optional[Dict[str, float]] = None
    meta: Optional[Dict[str, Any]] = None


class OracleEvent(BaseModel):
    event: str
    timestamp: str
    session_id: Optional[str] = None
    annotator_id: Optional[str] = None
    model_version: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
