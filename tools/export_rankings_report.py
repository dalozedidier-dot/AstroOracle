from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import re
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd


def read_candidates(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("Unsupported candidates format (expected .parquet or .csv)")


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_cutouts_for_id(cutouts_dir: Path, cand_id: str) -> List[Path]:
    if not cutouts_dir.exists():
        return []
    cid = str(cand_id)
    out: List[Path] = []
    for p in sorted(cutouts_dir.glob("*.png")):
        if cid in p.name:
            out.append(p)
    return out


def img_tag(path: Path, embed: bool, width: int = 120) -> str:
    if embed:
        raw = path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        return (
            f"<img src='data:image/png;base64,{b64}' width='{width}' "
            "style='border-radius:8px;border:1px solid #eee;'/>"
        )
    return (
        f"<img src='{html.escape(path.as_posix())}' width='{width}' "
        "style='border-radius:8px;border:1px solid #eee;'/>"
    )


def build_html(
    df: pd.DataFrame,
    cutouts_dir: Path,
    id_col: str,
    rank_col: str,
    u_col: Optional[str],
    top_k: int,
    embed_images: bool,
) -> str:
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    if "rank" in rank_col.lower() and "score" not in rank_col.lower():
        df_s = df.sort_values(rank_col, ascending=True).head(top_k)
    else:
        df_s = df.sort_values(rank_col, ascending=False).head(top_k)

    rows: List[str] = []
    for _, r in df_s.iterrows():
        cid = str(r[id_col])
        rank_v = r.get(rank_col, "")
        u_v = r.get(u_col, "") if u_col else ""

        cutouts = find_cutouts_for_id(cutouts_dir, cid)[:3]
        if cutouts:
            thumbs = " ".join([img_tag(p, embed_images, width=110) for p in cutouts])
        else:
            thumbs = "<span style='color:#777;'>No cutouts</span>"

        ra = r.get("ra", "")
        dec = r.get("dec", "")
        pred = r.get("label_pred", "")

        candidate_cell = (
            "<td style='padding:10px;vertical-align:top;'>"
            f"<b>{html.escape(cid)}</b><br/>"
            "<span style='color:#666;font-size:12px;'>"
            f"RA {html.escape(str(ra))} Dec {html.escape(str(dec))}"
            "</span><br/>"
            "<span style='color:#666;font-size:12px;'>"
            f"pred {html.escape(str(pred))}"
            "</span>"
            "</td>"
        )

        row_html = (
            "<tr>"
            f"{candidate_cell}"
            "<td style='padding:10px;vertical-align:top;'>"
            f"{html.escape(str(rank_v))}"
            "</td>"
            "<td style='padding:10px;vertical-align:top;'>"
            f"{html.escape(str(u_v))}"
            "</td>"
            f"<td style='padding:10px;vertical-align:top;'>{thumbs}</td>"
            "</tr>"
        )
        rows.append(row_html)

    css = (
        "body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, "
        "Arial, "
        "sans-serif; margin: 24px; }"
        "h1 { margin: 0 0 6px 0; font-size: 22px; }"
        ".sub { color: #666; margin-bottom: 18px; }"
        "table { width: 100%; border-collapse: collapse; }"
        "th { text-align: left; font-size: 12px; color: #555; "
        "border-bottom: 1px solid #eee; "
        "padding: 8px 10px; }"
        "tr { border-bottom: 1px solid #f2f2f2; }"
        ".badge { display:inline-block; padding:2px 8px; "
        "border:1px solid #ddd; "
        "border-radius: 999px; font-size: 12px; color:#444; }"
    )

    u_head = "<th>Uncertainty</th>" if u_col else "<th></th>"

    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'/>"
        f"<style>{css}</style></head><body>"
        "<h1>AstroOracle Rankings</h1>"
        f"<div class='sub'>Generated {html.escape(now)} "
        f"<span class='badge'>top_k={top_k}</span> "
        f"<span class='badge'>embed_images={str(embed_images).lower()}</span></div>"
        "<table><thead><tr>"
        "<th>Candidate</th>"
        f"<th>{html.escape(rank_col)}</th>"
        f"{u_head}"
        "<th>Cutouts</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>"
    )
    return html_doc


def write_pdf_reportlab(html_text: str, out_pdf: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    text = re.sub(r"<[^>]+>", "", html_text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    c = canvas.Canvas(str(out_pdf), pagesize=letter)
    _w, h = letter
    x = 40
    y = h - 40
    c.setFont("Helvetica", 10)

    for ln in lines:
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = h - 40
        c.drawString(x, y, ln[:120])
        y -= 12

    c.save()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates.parquet or .csv",
    )
    ap.add_argument(
        "--cutouts-dir",
        required=True,
        help="Directory containing cutout PNGs",
    )
    ap.add_argument("--out", required=True, help="Output file (.html or .pdf)")
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--embed-images", action="store_true")
    ap.add_argument("--rank-col", default="", help="Optional override rank column name")
    ap.add_argument("--id-col", default="", help="Optional override id column name")
    ap.add_argument(
        "--uncertainty-col",
        default="",
        help="Optional override uncertainty column name",
    )
    ap.add_argument("--pdf-engine", choices=["reportlab"], default="reportlab")
    args = ap.parse_args()

    df = read_candidates(args.candidates)

    id_col = args.id_col or first_existing_col(
        df,
        ["id", "source_id", "candidate_id", "obj_id"],
    )
    if not id_col:
        raise SystemExit("No id column found. Provide --id-col.")

    rank_col = (
        args.rank_col
        or first_existing_col(df, ["rank", "rank_score", "score", "anomaly_score"])
        or id_col
    )

    u_col = args.uncertainty_col or first_existing_col(
        df, ["uncertainty", "uncertainty_score", "entropy", "prob_margin"]
    )

    cutouts_dir = Path(args.cutouts_dir)
    out = Path(args.out)

    html_doc = build_html(
        df=df,
        cutouts_dir=cutouts_dir,
        id_col=id_col,
        rank_col=rank_col,
        u_col=u_col,
        top_k=int(args.top_k),
        embed_images=bool(args.embed_images),
    )

    if out.suffix.lower() == ".html":
        out.write_text(html_doc, encoding="utf-8")
        return 0

    if out.suffix.lower() == ".pdf":
        if args.pdf_engine == "reportlab":
            write_pdf_reportlab(html_doc, out)
            return 0

    raise SystemExit("Unsupported output format. Use .html or .pdf")


if __name__ == "__main__":
    raise SystemExit(main())
