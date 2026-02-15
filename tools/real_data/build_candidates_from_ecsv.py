#!/usr/bin/env python3
"""
Build an AstroOracle candidates.parquet from a (real) Gaia-derived CSV/ECSV table.

AstroOracle expects a Parquet file with (at minimum):
- id (string)
- ra (float degrees)
- dec (float degrees)
- anomaly_score (float)

This script supports two coordinate modes:
- pseudo (default): deterministic pseudo RA/Dec derived from source_id (offline, no network)
- gaia: query Gaia DR3 via astroquery to fetch real ra/dec for a subset of source_id (needs network)

It also builds a small deterministic embedding vector to make diversity selection meaningful.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BuildMeta:
    input_path: str
    n_in: int
    n_out: int
    coords_mode: str
    score_column: str
    gaia_matched: int
    gaia_requested: int


def _read_table(path: Path) -> pd.DataFrame:
    # Supports:
    # - plain CSV (possibly gz)
    # - ECSV (Astropy) style with leading "#" metadata lines
    return pd.read_csv(path, compression="infer", comment="#")


def _pick_score_column(df: pd.DataFrame) -> str:
    # Prefer variability/spread columns if available.
    preferred = [
        "mad_mag_g_fov",
        "std_dev_mag_g_fov",
        "range_mag_g_fov",
        "mad_mag_gfov",
        "std_dev_mag_gfov",
        "range_mag_gfov",
        "mad_mag",
        "std_mag",
        "range_mag",
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    for key in preferred:
        if key in cols_lower:
            return cols_lower[key]

    # Fallback: first numeric column (excluding obvious IDs).
    for c in df.columns:
        cl = c.lower()
        if cl in {"solution_id", "source_id"}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            return c

    raise SystemExit("No numeric column found to build anomaly_score.")


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad <= 1e-12:
        # Fallback to IQR-like scale
        q1 = np.nanpercentile(x, 25)
        q3 = np.nanpercentile(x, 75)
        scale = (q3 - q1) / 1.349 if np.isfinite(q3 - q1) and (q3 - q1) > 1e-12 else 1.0
    else:
        scale = 1.4826 * mad
    z = (x - med) / float(scale)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z


def _pseudo_coords_from_source_id(source_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic mapping from source_id (uint64) -> (ra, dec) in degrees.
    Not physical, but stable and offline.
    """
    s = np.asarray(source_ids, dtype=np.uint64)
    ra = (s % np.uint64(3600000)).astype(np.float64) / 10000.0  # 0..360
    dec = ((s // np.uint64(3600000)) % np.uint64(1800000)).astype(np.float64) / 10000.0 - 90.0  # -90..90
    return ra, dec


def _embedding_from_source_id(source_ids: np.ndarray, dim: int = 16) -> list[np.ndarray]:
    """
    Deterministic pseudo-embedding vector from source_id.
    """
    s = np.asarray(source_ids, dtype=np.uint64)
    # xorshift-ish mixing
    x = s ^ (s >> np.uint64(12))
    x = x ^ (x << np.uint64(25))
    x = x ^ (x >> np.uint64(27))
    # Build dim floats in [-1, 1]
    out: list[np.ndarray] = []
    for v in x:
        # create a small PRNG stream seeded by v (downcast to 32 bits for numpy)
        seed = int(v % np.uint64(2**32 - 1))
        rng = np.random.default_rng(seed)
        out.append(rng.normal(0, 1, size=dim).astype(float))
    return out


def _chunked(seq: Iterable[int], size: int) -> Iterable[list[int]]:
    buf: list[int] = []
    for x in seq:
        buf.append(int(x))
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def _gaia_fetch_coords(source_ids: list[int], *, max_rows: int, chunk: int = 200) -> pd.DataFrame:
    """
    Fetch real Gaia DR3 coords for the given source_id list using astroquery.
    Returns a DataFrame with columns: source_id, ra, dec.
    """
    try:
        from astroquery.gaia import Gaia  # type: ignore
    except Exception as e:
        raise SystemExit(
            "astroquery is required for --coords gaia. Install with: pip install -e '.[astro]'"
        ) from e

    ids = source_ids[: max_rows if max_rows > 0 else len(source_ids)]
    frames = []
    for part in _chunked(ids, chunk):
        # ADQL IN list
        in_list = ",".join(str(i) for i in part)
        query = (
            "SELECT source_id, ra, dec "
            "FROM gaiadr3.gaia_source "
            f"WHERE source_id IN ({in_list})"
        )
        job = Gaia.launch_job_async(query, dump_to_file=False)
        tbl = job.get_results().to_pandas()
        frames.append(tbl)
    if not frames:
        return pd.DataFrame(columns=["source_id", "ra", "dec"])
    res = pd.concat(frames, ignore_index=True)
    res = res.drop_duplicates(subset=["source_id"]).reset_index(drop=True)
    return res


def build_candidates(
    df: pd.DataFrame,
    *,
    coords_mode: str,
    score_col: str,
    max_rows: int,
    strict_gaia: bool,
) -> tuple[pd.DataFrame, BuildMeta]:
    if "source_id" not in df.columns:
        raise SystemExit("Input table must contain a 'source_id' column.")

    if max_rows > 0:
        df = df.head(max_rows).copy()
    else:
        df = df.copy()

    n_in = int(len(df))
    source_ids = pd.to_numeric(df["source_id"], errors="coerce").fillna(0).astype("uint64").to_numpy()

    ra: Optional[np.ndarray] = None
    dec: Optional[np.ndarray] = None
    gaia_matched = 0
    gaia_requested = 0

    if coords_mode == "gaia":
        ids = [int(x) for x in source_ids.tolist() if int(x) != 0]
        gaia_requested = len(ids[: max_rows if max_rows > 0 else len(ids)])
        coords = _gaia_fetch_coords(ids, max_rows=(max_rows if max_rows > 0 else len(ids)))
        if coords.empty:
            if strict_gaia:
                raise SystemExit("Gaia query returned no rows (strict mode).")
            ra, dec = _pseudo_coords_from_source_id(source_ids)
        else:
            # Merge, keep original order
            m = pd.DataFrame({"source_id": source_ids.astype("uint64")})
            m2 = m.merge(coords, on="source_id", how="left")
            missing = m2["ra"].isna() | m2["dec"].isna()
            gaia_matched = int((~missing).sum())
            if strict_gaia and gaia_matched < max(1, int(0.8 * len(m2))):
                raise SystemExit(
                    f"Gaia coords matched {gaia_matched}/{len(m2)} (strict mode requires >=80%)."
                )
            # Fill missing with pseudo
            ra_p, dec_p = _pseudo_coords_from_source_id(source_ids)
            ra = pd.to_numeric(m2["ra"], errors="coerce").to_numpy(float)
            dec = pd.to_numeric(m2["dec"], errors="coerce").to_numpy(float)
            ra = np.where(np.isfinite(ra), ra, ra_p)
            dec = np.where(np.isfinite(dec), dec, dec_p)

    else:
        ra, dec = _pseudo_coords_from_source_id(source_ids)

    x = pd.to_numeric(df[score_col], errors="coerce").to_numpy(float)
    z = _robust_z(x)

    out = pd.DataFrame(
        {
            "id": df["source_id"].astype(str),
            "ra": ra.astype(float),
            "dec": dec.astype(float),
            "anomaly_score": z.astype(float),
        }
    )
    out["embedding"] = _embedding_from_source_id(source_ids, dim=16)

    meta = BuildMeta(
        input_path="(in-memory)",
        n_in=n_in,
        n_out=int(len(out)),
        coords_mode=coords_mode,
        score_column=score_col,
        gaia_matched=int(gaia_matched),
        gaia_requested=int(gaia_requested),
    )
    return out, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV/ECSV path (optionally .gz)")
    ap.add_argument("--output", required=True, help="Output Parquet path")
    ap.add_argument(
        "--coords",
        choices=["pseudo", "gaia"],
        default="pseudo",
        help="Coordinate mode. pseudo is offline and deterministic. gaia queries Gaia DR3.",
    )
    ap.add_argument("--max-rows", type=int, default=2000, help="Max rows to keep from input.")
    ap.add_argument(
        "--score-col",
        default=None,
        help="Column to transform into anomaly_score. Default: auto-pick.",
    )
    ap.add_argument(
        "--strict-gaia",
        action="store_true",
        help="Fail if Gaia coords coverage is too low (only for --coords gaia).",
    )
    ap.add_argument(
        "--meta-json",
        default=None,
        help="Optional JSON output describing the build.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = _read_table(in_path)
    score_col = str(args.score_col) if args.score_col else _pick_score_column(df)

    cand, meta = build_candidates(
        df,
        coords_mode=str(args.coords),
        score_col=score_col,
        max_rows=int(args.max_rows),
        strict_gaia=bool(args.strict_gaia),
    )

    cand.to_parquet(out_path, index=False)

    meta2 = BuildMeta(
        input_path=str(in_path),
        n_in=meta.n_in,
        n_out=meta.n_out,
        coords_mode=meta.coords_mode,
        score_column=meta.score_column,
        gaia_matched=meta.gaia_matched,
        gaia_requested=meta.gaia_requested,
    )
    print(json.dumps(meta2.__dict__, indent=2, ensure_ascii=False))

    if args.meta_json:
        Path(args.meta_json).write_text(json.dumps(meta2.__dict__, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
