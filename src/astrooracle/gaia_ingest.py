from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass(frozen=True)
class GaiaQueryResult:
    n_rows: int
    meta: Dict[str, Any]
    df: pd.DataFrame


def _validate_radec(ra_deg: float, dec_deg: float) -> None:
    if not (0.0 <= ra_deg < 360.0):
        raise ValueError(f"RA must be in [0, 360). Got {ra_deg}")
    if not (-90.0 <= dec_deg <= 90.0):
        raise ValueError(f"Dec must be in [-90, 90]. Got {dec_deg}")


def gaia_cone_search(
    *,
    ra_deg: float,
    dec_deg: float,
    radius_arcmin: float = 5.0,
    columns: Optional[str] = None,
    max_rows: int = 2000,
) -> GaiaQueryResult:
    """Cone search against Gaia archive via astroquery.

    This function requires the optional dependency: astroquery.

    Notes:
    - The default query uses Gaia DR3 public table gaia_source.
    - You can override the selected columns using ADQL select list.
    """

    _validate_radec(float(ra_deg), float(dec_deg))

    try:
        from astroquery.gaia import Gaia  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("astroquery is required: pip install -e '.[astro]' ") from e

    r = float(radius_arcmin)
    if not math.isfinite(r) or r <= 0:
        raise ValueError("radius_arcmin must be positive")

    select_cols = (
        columns
        if columns
        else "source_id, ra, dec, phot_g_mean_mag, bp_rp, ruwe, parallax, pmra, pmdec"
    )

    # ADQL cone via DISTANCE on ICRS coordinates.
    adql = f"""
    SELECT {select_cols}
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
      POINT('ICRS', ra, dec),
      CIRCLE('ICRS', {float(ra_deg)}, {float(dec_deg)}, {r/60.0})
    )
    """.strip()

    job = Gaia.launch_job_async(adql, dump_to_file=False)
    tab = job.get_results()
    df = tab.to_pandas()

    if max_rows and len(df) > int(max_rows):
        df = df.head(int(max_rows)).copy()

    return GaiaQueryResult(
        n_rows=int(len(df)),
        meta={
            "ra_deg": float(ra_deg),
            "dec_deg": float(dec_deg),
            "radius_arcmin": float(radius_arcmin),
            "max_rows": int(max_rows),
        },
        df=df,
    )


def gaia_adql_query(*, adql: str, max_rows: int = 20000) -> GaiaQueryResult:
    try:
        from astroquery.gaia import Gaia  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("astroquery is required: pip install -e '.[astro]' ") from e

    if not adql or not adql.strip().lower().startswith("select"):
        raise ValueError("ADQL must be a SELECT query")

    job = Gaia.launch_job_async(adql, dump_to_file=False)
    tab = job.get_results()
    df = tab.to_pandas()

    if max_rows and len(df) > int(max_rows):
        df = df.head(int(max_rows)).copy()

    return GaiaQueryResult(n_rows=int(len(df)), meta={"max_rows": int(max_rows)}, df=df)


def write_gaia_table(df: pd.DataFrame, path: str) -> None:
    """Write Gaia results as parquet or csv, based on extension."""

    import pathlib

    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suf = p.suffix.lower()
    if suf in {".parquet", ".pq"}:
        df.to_parquet(p, index=False)
    elif suf in {".csv"}:
        df.to_csv(p, index=False)
    else:
        raise ValueError("Unsupported output format. Use .parquet or .csv")
