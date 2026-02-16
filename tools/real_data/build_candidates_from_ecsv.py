from __future__ import annotations

import argparse
import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BuildResult:
    df: pd.DataFrame
    n_in: int
    n_out: int


def _read_table(path: Path) -> pd.DataFrame:
    """Read ECSV/CSV/TSV or gzipped variants.

    Many Gaia exports are ECSV, where metadata lines start with '#'.
    """
    suf = "".join(path.suffixes).lower()
    is_gz = suf.endswith(".gz")
    raw_suffix = suf[:-3] if is_gz else suf

    if is_gz:
        with gzip.open(path, "rt", errors="replace") as f:
            return _read_table_stream(f, raw_suffix)
    with path.open("rt", errors="replace") as f:
        return _read_table_stream(f, raw_suffix)


def _read_table_stream(f, raw_suffix: str) -> pd.DataFrame:
    if raw_suffix.endswith(".tsv"):
        return pd.read_csv(f, sep="\t", comment="#")
    # default csv or ecsv-in-csv
    return pd.read_csv(f, sep=",", comment="#")


def _pseudo_radec_from_source_id(source_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic pseudo RA/Dec for offline demos.

    It is not physical, but it is stable across runs and does not require network.
    """
    sid = source_id.astype(np.int64)
    h = (sid ^ (sid >> 33) ^ (sid << 11)) & 0xFFFFFFFFFFFFFFFF
    ra = (h % 3600000) / 10000.0  # 0..360
    u = ((h // 3600000) % 2000000) / 1000000.0  # 0..2
    u = np.clip(u - 1.0, -1.0, 1.0)
    dec = np.degrees(np.arcsin(u))
    return ra.astype(float), dec.astype(float)


def _normalize_score(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    lo = float(x.quantile(0.01))
    hi = float(x.quantile(0.99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    y = (x - lo) / (hi - lo)
    y = y.clip(0.0, 1.0).fillna(0.0)
    return y.astype(float)


def _score_from_kind(df: pd.DataFrame, kind: str) -> pd.Series:
    k = kind.lower().strip()
    if k == "galaxy_candidates":
        if "vari_best_class_score" in df.columns:
            return _normalize_score(df["vari_best_class_score"])
        if "classprob_dsc_combmod_quasar" in df.columns:
            return _normalize_score(df["classprob_dsc_combmod_quasar"])
        return _normalize_score(df.select_dtypes(include=["number"]).sum(axis=1))
    if k == "vari_summary":
        for col in [
            "stetson_mag_g_fov",
            "range_mag_g_fov",
            "std_dev_over_rms_err_mag_g_fov",
            "iqr_mag_g_fov",
        ]:
            if col in df.columns:
                return _normalize_score(df[col])
        return _normalize_score(df.select_dtypes(include=["number"]).sum(axis=1))
    if k == "galaxy_catalogue_name":
        # very sparse table, build a stable score from catalogue_id
        if "catalogue_id" in df.columns:
            return _normalize_score(df["catalogue_id"])
        return pd.Series(np.zeros(len(df)), dtype=float)
    # fallback
    return _normalize_score(df.select_dtypes(include=["number"]).sum(axis=1))


def _get_source_id(df: pd.DataFrame) -> pd.Series:
    if "source_id" in df.columns:
        return pd.to_numeric(df["source_id"], errors="coerce")
    if "id" in df.columns:
        return pd.to_numeric(df["id"], errors="coerce")
    raise ValueError("No source identifier column found (expected source_id or id).")


def _gaia_coords_for_source_ids(source_ids: list[int], max_rows: int) -> pd.DataFrame:
    """Fetch RA/Dec from Gaia for a list of source_ids.

    Requires astroquery. This is optional and used only in 'gaia' mode.
    """
    try:
        from astroquery.gaia import Gaia  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "astroquery is required for --mode gaia. Install extras: pip install -e '.[astro]'"
        ) from e

    if not source_ids:
        return pd.DataFrame(columns=["source_id", "ra", "dec"])

    # Gaia ADQL 'IN' has practical size limits, so chunk.
    out_frames: list[pd.DataFrame] = []
    chunk = 500
    for i in range(0, min(len(source_ids), max_rows), chunk):
        part = source_ids[i : i + chunk]
        ids = ",".join(str(int(x)) for x in part)
        adql = (
            "SELECT source_id, ra, dec " "FROM gaiadr3.gaia_source " f"WHERE source_id IN ({ids})"
        )
        job = Gaia.launch_job_async(adql)
        tab = job.get_results().to_pandas()
        out_frames.append(tab)
    if not out_frames:
        return pd.DataFrame(columns=["source_id", "ra", "dec"])
    return pd.concat(out_frames, ignore_index=True)


def build_candidates(
    input_path: Path, kind: str, mode: str, limit: int, gaia_max_rows: int
) -> BuildResult:
    df0 = _read_table(input_path)
    n_in = len(df0)
    if limit > 0:
        df0 = df0.head(int(limit)).copy()

    sid = _get_source_id(df0).fillna(-1).astype(np.int64)
    score = _score_from_kind(df0, kind)

    if mode == "gaia":
        unique_ids = [int(x) for x in pd.unique(sid) if int(x) > 0]
        coords = _gaia_coords_for_source_ids(unique_ids, max_rows=int(gaia_max_rows))
        merged = df0.copy()
        merged["source_id"] = sid.values
        merged = merged.merge(coords, on="source_id", how="left")
        ra = pd.to_numeric(merged["ra"], errors="coerce")
        dec = pd.to_numeric(merged["dec"], errors="coerce")
        # fallback if Gaia returned nothing for some ids
        ra_f, dec_f = _pseudo_radec_from_source_id(sid.values)
        ra = ra.fillna(pd.Series(ra_f))
        dec = dec.fillna(pd.Series(dec_f))
    else:
        ra_f, dec_f = _pseudo_radec_from_source_id(sid.values)
        ra = pd.Series(ra_f)
        dec = pd.Series(dec_f)

    out = pd.DataFrame(
        {
            "id": sid.astype(str),
            "ra": ra.astype(float),
            "dec": dec.astype(float),
            "anomaly_score": score.astype(float),
            "source_id": sid.astype(np.int64),
        }
    )

    return BuildResult(df=out, n_in=n_in, n_out=len(out))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--kind",
        required=True,
        choices=["galaxy_candidates", "vari_summary", "galaxy_catalogue_name"],
    )
    p.add_argument("--mode", default="pseudo", choices=["pseudo", "gaia"])
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--gaia-max-rows", type=int, default=2000)
    args = p.parse_args()

    res = build_candidates(
        input_path=Path(args.input),
        kind=str(args.kind),
        mode=str(args.mode),
        limit=int(args.limit),
        gaia_max_rows=int(args.gaia_max_rows),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.df.to_parquet(out_path, index=False)

    print(f"Built candidates: in={res.n_in} out={res.n_out} mode={args.mode} kind={args.kind}")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
