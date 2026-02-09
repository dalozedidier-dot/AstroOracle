from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .config import OracleConfig
from .core import fetch_cutouts
from .crossmatch import crossmatch_all
from .image_features import aggregate_candidate_features, compute_cutout_features


def augment_row(
    cfg: OracleConfig,
    ra: float,
    dec: float,
    do_cutouts: bool = True,
    do_crossmatch: bool = True,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if do_cutouts:
        cutouts = fetch_cutouts(ra, dec, cfg)
        by_survey = {}
        for survey, data in cutouts:
            by_survey[str(survey)] = compute_cutout_features(data)
        agg = aggregate_candidate_features(by_survey)
        out.update(agg)

    if do_crossmatch:
        out.update(crossmatch_all(ra, dec, radius_arcsec=5.0, neighbor_limit=25))

        # Make sure match flags are numeric-friendly for ML
        if "gaia_match" in out:
            out["gaia_match"] = int(bool(out["gaia_match"]))
        if "simbad_match" in out:
            out["simbad_match"] = int(bool(out["simbad_match"]))

        # Avoid embedding neighbor lists in parquet by default
        if "gaia_neighbors" in out:
            out.pop("gaia_neighbors", None)

    return out


def augment_candidates_file(
    cfg: OracleConfig,
    input_path: Path,
    output_path: Path,
    do_cutouts: bool = True,
    do_crossmatch: bool = True,
) -> None:
    df = pd.read_parquet(input_path)
    if df.empty:
        df.to_parquet(output_path, index=False)
        return

    rows = []
    for _, r in df.iterrows():
        ra = float(r["ra"])
        dec = float(r["dec"])
        rows.append(augment_row(cfg, ra, dec, do_cutouts=do_cutouts, do_crossmatch=do_crossmatch))

    extra = pd.DataFrame(rows)
    out_df = pd.concat([df.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
