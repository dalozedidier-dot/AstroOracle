from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd


def _pseudo_radec_from_source_id(source_id: int) -> tuple[float, float]:
    # Deterministic pseudo coordinates in degrees derived from the Gaia source_id.
    # This makes the offline workflow fully reproducible without network access.
    h = hashlib.sha256(str(int(source_id)).encode("utf-8")).digest()
    u0 = int.from_bytes(h[:8], "big") / 2**64
    u1 = int.from_bytes(h[8:16], "big") / 2**64
    ra = 360.0 * u0
    dec = 180.0 * u1 - 90.0
    return float(ra), float(dec)


def _read_ecsv_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        return pd.read_csv(f, comment="#")


def _gaia_fetch_radec(source_ids: Iterable[int]) -> pd.DataFrame:
    # Requires astroquery. We keep this function isolated to avoid importing astroquery by default.
    try:
        from astroquery.gaia import Gaia  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("astroquery is required for mode=gaia (install extra 'astro')") from e

    ids = list(dict.fromkeys(int(x) for x in source_ids))
    if not ids:
        return pd.DataFrame(columns=["source_id", "ra", "dec"])

    # Chunk to avoid overly long queries.
    rows: list[pd.DataFrame] = []
    chunk = 200
    for i in range(0, len(ids), chunk):
        sub = ids[i : i + chunk]
        # DR3 table name.
        q = (
            "SELECT source_id, ra, dec "
            "FROM gaiadr3.gaia_source "
            f"WHERE source_id IN ({','.join(str(x) for x in sub)})"
        )
        job = Gaia.launch_job_async(q)
        tbl = job.get_results()
        rows.append(tbl.to_pandas())
    out = pd.concat(rows, ignore_index=True)
    out["source_id"] = pd.to_numeric(out["source_id"], errors="coerce").astype("Int64")
    return out.dropna(subset=["source_id", "ra", "dec"])


def build_candidates(
    *,
    input_path: Path,
    output_path: Path,
    mode: str,
    n: int,
    seed: int,
) -> None:
    df = _read_ecsv_gz(input_path)
    if df.empty:
        raise RuntimeError(f"No rows in {input_path}")

    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    # Canonical ID column expected by AstroOracle.
    df["id"] = df["source_id"].astype(str)

    if mode == "gaia":
        radec = _gaia_fetch_radec(df["source_id"].astype(int).tolist())
        df = df.merge(radec, on="source_id", how="left")
    elif mode == "pseudo":
        ra_dec = df["source_id"].astype(int).map(_pseudo_radec_from_source_id)
        df["ra"] = [x[0] for x in ra_dec]
        df["dec"] = [x[1] for x in ra_dec]
    else:
        raise ValueError("mode must be 'pseudo' or 'gaia'")

    # Provide a default anomaly_score if upstream score is not present.
    if "anomaly_score" not in df.columns:
        # Prefer uncertainty-like score: high when classification confidence is low.
        if "vari_best_class_score" in df.columns:
            s = pd.to_numeric(df["vari_best_class_score"], errors="coerce").fillna(0.0)
            df["anomaly_score"] = (1.0 - s).clip(0.0, 1.0)
        elif "classprob_dsc_combmod_galaxy" in df.columns:
            s = pd.to_numeric(df["classprob_dsc_combmod_galaxy"], errors="coerce").fillna(0.0)
            df["anomaly_score"] = (1.0 - s).clip(0.0, 1.0)
        else:
            df["anomaly_score"] = 0.5

    # Minimal column set expected by most pipeline pieces.
    keep_first = ["id", "source_id", "ra", "dec", "anomaly_score"]
    cols = keep_first + [c for c in df.columns if c not in keep_first]
    df = df[cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def main() -> int:
    p = argparse.ArgumentParser(description="Build AstroOracle candidates.parquet from Gaia ECSV sample.")
    p.add_argument("--input", type=Path, required=True, help="Input .csv.gz (ECSV) file")
    p.add_argument("--output", type=Path, required=True, help="Output parquet file")
    p.add_argument("--mode", choices=["pseudo", "gaia"], default="pseudo")
    p.add_argument("--n", type=int, default=500, help="Number of rows to sample")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    build_candidates(input_path=args.input, output_path=args.output, mode=args.mode, n=args.n, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
