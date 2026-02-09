from __future__ import annotations

from typing import Any, Dict, List

import os


def _offline() -> bool:
    return os.environ.get("ASTROORACLE_OFFLINE", "") in {"1", "true", "yes"}


def _gaia_link(source_id: Any) -> str:
    return f"https://gea.esac.esa.int/archive/?q=source_id={source_id}"


def _simbad_link(main_id: str) -> str:
    return f"https://simbad.u-strasbg.fr/simbad/sim-id?Ident={main_id}"


def crossmatch_gaia(
    ra_deg: float,
    dec_deg: float,
    radius_arcsec: float = 5.0,
    neighbor_limit: int = 25,
) -> Dict[str, Any]:
    if _offline():
        return {"gaia_match": False, "gaia_neighbors": []}

    try:
        from astropy import units as u  # type: ignore
        from astropy.coordinates import SkyCoord  # type: ignore
        from astroquery.gaia import Gaia  # type: ignore
    except Exception:
        return {"gaia_match": False, "gaia_neighbors": []}

    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    Gaia.ROW_LIMIT = int(max(5, neighbor_limit))
    try:
        job = Gaia.cone_search_async(coord, radius=radius_arcsec * u.arcsec)
        tbl = job.get_results()
    except Exception:
        return {"gaia_match": False, "gaia_neighbors": []}

    if tbl is None or len(tbl) == 0:
        return {"gaia_match": False, "gaia_neighbors": []}

    # Compute separations and offsets
    try:
        src_coord = SkyCoord(ra=tbl["ra"], dec=tbl["dec"], unit="deg", frame="icrs")
        sep = coord.separation(src_coord).arcsec
        dlon, dlat = coord.spherical_offsets_to(src_coord)
        dx = dlon.to_value(u.arcsec)
        dy = dlat.to_value(u.arcsec)
    except Exception:
        return {"gaia_match": False, "gaia_neighbors": []}

    order = sep.argsort()
    neighbors: List[Dict[str, Any]] = []
    for i in order[:neighbor_limit]:
        row = tbl[int(i)]
        neighbors.append(
            {
                "source_id": str(row.get("source_id")),
                "dist_arcsec": float(sep[int(i)]),
                "dx_arcsec": float(dx[int(i)]),
                "dy_arcsec": float(dy[int(i)]),
                "gmag": (
                    float(row.get("phot_g_mean_mag"))
                    if row.get("phot_g_mean_mag") is not None
                    else None
                ),
            }
        )

    nearest = neighbors[0]
    out: Dict[str, Any] = {
        "gaia_match": True,
        "nearest_gaia_dist_arcsec": float(nearest["dist_arcsec"]),
        "gaia_source_id": str(nearest["source_id"]),
        "gaia_link": _gaia_link(nearest["source_id"]),
        "gaia_neighbors": neighbors,
    }

    # Optional scalar fields for nearest row
    row0 = tbl[int(order[0])]
    for k_src, k_out in [
        ("parallax", "gaia_parallax_mas"),
        ("pmra", "gaia_pmra_masyr"),
        ("pmdec", "gaia_pmdec_masyr"),
        ("phot_g_mean_mag", "gaia_gmag"),
    ]:
        try:
            v = row0.get(k_src)
        except Exception:
            v = None
        if v is None:
            out[k_out] = None
        else:
            try:
                out[k_out] = float(v)
            except Exception:
                out[k_out] = None

    return out


def crossmatch_simbad(
    ra_deg: float,
    dec_deg: float,
    radius_arcsec: float = 5.0,
) -> Dict[str, Any]:
    if _offline():
        return {"simbad_match": False}

    try:
        from astropy import units as u  # type: ignore
        from astropy.coordinates import SkyCoord  # type: ignore
        from astroquery.simbad import Simbad  # type: ignore
    except Exception:
        return {"simbad_match": False}

    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    s = Simbad()
    try:
        s.add_votable_fields("otype")
    except Exception:
        pass

    try:
        res = s.query_region(coord, radius=radius_arcsec * u.arcsec)
    except Exception:
        return {"simbad_match": False}

    if res is None or len(res) == 0:
        return {"simbad_match": False}

    # pick nearest by separation if possible
    try:
        src_coord = SkyCoord(ra=res["RA"], dec=res["DEC"], unit=(u.hourangle, u.deg), frame="icrs")
        sep = coord.separation(src_coord).arcsec
        i = int(sep.argmin())
        dist = float(sep[i])
    except Exception:
        i = 0
        dist = None

    main_id = res["MAIN_ID"][i]
    if isinstance(main_id, bytes):
        main_id = main_id.decode("utf-8", errors="ignore")
    main_id = str(main_id).strip()

    otype = None
    if "OTYPE" in res.colnames:
        otype = res["OTYPE"][i]
        if isinstance(otype, bytes):
            otype = otype.decode("utf-8", errors="ignore")
        otype = str(otype).strip()

    out: Dict[str, Any] = {
        "simbad_match": True,
        "simbad_main_id": main_id,
        "simbad_type": otype,
        "simbad_link": _simbad_link(main_id),
        "nearest_simbad_dist_arcsec": dist,
    }
    return out


def crossmatch_all(
    ra_deg: float,
    dec_deg: float,
    radius_arcsec: float = 5.0,
    neighbor_limit: int = 25,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out.update(
        crossmatch_gaia(
            ra_deg,
            dec_deg,
            radius_arcsec=radius_arcsec,
            neighbor_limit=neighbor_limit,
        )
    )
    out.update(crossmatch_simbad(ra_deg, dec_deg, radius_arcsec=radius_arcsec))
    return out
