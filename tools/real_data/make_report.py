from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to an ECSV/CSV(.gz) file.")
    p.add_argument("--kind", required=True, choices=["galaxy_candidates", "vari_summary", "galaxy_catalogue_name"])
    p.add_argument("--out-dir", required=True, help="Output folder for batch HTML.")
    p.add_argument("--mode", default="pseudo", choices=["pseudo", "gaia"])
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--gaia-max-rows", type=int, default=2000)
    args = p.parse_args()

    root = Path(".").resolve()
    candidates = root / "candidates.parquet"

    cmd_build = [
        "python",
        "tools/real_data/build_candidates_from_ecsv.py",
        "--input",
        str(args.input),
        "--out",
        str(candidates),
        "--kind",
        str(args.kind),
        "--mode",
        str(args.mode),
        "--limit",
        str(int(args.limit)),
        "--gaia-max-rows",
        str(int(args.gaia_max_rows)),
    ]
    subprocess.check_call(cmd_build)

    cmd_html = [
        "astrooracle",
        "batch-html",
        "--out-dir",
        str(args.out_dir),
        "--offline",
        "--candidates",
        str(candidates),
    ]
    subprocess.check_call(cmd_html)

    print(f"Report written: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
