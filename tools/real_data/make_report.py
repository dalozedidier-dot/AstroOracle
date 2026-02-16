from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


HTML_TEMPLATE = """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>{title}</title>
    <link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css\" rel=\"stylesheet\">
    <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
    <style>
      body {{ padding-top: 2rem; padding-bottom: 2rem; }}
      .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; }}
      #plot {{ height: 520px; }}
      .small-muted {{ color: #666; font-size: 0.9rem; }}
      .table-wrap {{ max-height: 520px; overflow: auto; }}
    </style>
  </head>
  <body>
    <div class=\"container\">
      <div class=\"d-flex align-items-center justify-content-between mb-3\">
        <div>
          <h1 class=\"h3 mb-1\">{title}</h1>
          <div class=\"small-muted\">{subtitle}</div>
        </div>
        <div class=\"mono\"><a href=\"{repo_url}\">{repo_url}</a></div>
      </div>

      <div class=\"row g-3\">
        <div class=\"col-lg-7\">
          <div class=\"card shadow-sm\">
            <div class=\"card-body\">
              <h2 class=\"h6\">Sky scatter (RA, Dec)</h2>
              <div id=\"plot\"></div>
              <div class=\"small-muted mt-2\">
                Color shows anomaly_score. This is a demo report produced in CI.
              </div>
            </div>
          </div>
        </div>

        <div class=\"col-lg-5\">
          <div class=\"card shadow-sm\">
            <div class=\"card-body\">
              <div class=\"d-flex justify-content-between align-items-center\">
                <h2 class=\"h6 mb-0\">Top candidates</h2>
                <a class=\"btn btn-sm btn-outline-secondary\" href=\"candidates_top.json\">JSON</a>
              </div>
              <div class=\"table-wrap mt-2\">
                <table class=\"table table-sm table-striped\">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th class=\"text-end\">RA</th>
                      <th class=\"text-end\">Dec</th>
                      <th class=\"text-end\">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows}
                  </tbody>
                </table>
              </div>
              <div class=\"small-muted\">
                Files: report.json, report.md, candidates.parquet, index.html
              </div>
            </div>
          </div>
        </div>
      </div>

      <hr class=\"my-4\">

      <h2 class=\"h6\">Run metadata</h2>
      <pre class=\"mono bg-light p-3 rounded\">{meta_json}</pre>
    </div>

    <script>
      const points = {points_json};

      const trace = {{
        x: points.map(p => p.ra),
        y: points.map(p => p.dec),
        mode: "markers",
        type: "scattergl",
        marker: {{
          size: 7,
          color: points.map(p => p.anomaly_score),
          colorscale: "Viridis",
          showscale: true
        }},
        text: points.map(p => `id=${{p.id}}<br>score=${{p.anomaly_score.toFixed(4)}}`),
        hoverinfo: "text"
      }};

      const layout = {{
        margin: {{l: 45, r: 15, t: 15, b: 40}},
        xaxis: {{title: "RA (deg)"}},
        yaxis: {{title: "Dec (deg)"}}
      }};

      Plotly.newPlot("plot", [trace], layout, {{displayModeBar: true}});
    </script>
  </body>
</html>
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--title", type=str, default="AstroOracle real data report")
    p.add_argument("--repo-url", type=str, default="https://github.com/dalozedidier-dot/AstroOracle")
    p.add_argument("--tag", type=str, default="realdata")
    args = p.parse_args()

    df = pd.read_parquet(args.candidates)
    if df.empty:
        raise SystemExit("candidates parquet is empty")

    for col in ["id", "ra", "dec", "anomaly_score"]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")

    df = df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    top = df.head(int(min(args.top, len(df)))).copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    points = [
        {
            "id": str(r["id"]),
            "ra": float(r["ra"]),
            "dec": float(r["dec"]),
            "anomaly_score": float(r["anomaly_score"]),
        }
        for r in top.to_dict(orient="records")
    ]

    rows_html = "\n".join(
        f"<tr><td class=\"mono\">{p['id']}</td><td class=\"text-end\">{p['ra']:.5f}</td><td class=\"text-end\">{p['dec']:.5f}</td><td class=\"text-end\">{p['anomaly_score']:.5f}</td></tr>"
        for p in points[: min(20, len(points))]
    )

    meta = {
        "tag": args.tag,
        "n_total": int(len(df)),
        "n_top": int(len(top)),
        "columns": list(df.columns),
    }

    (args.out_dir / "candidates_top.json").write_text(json.dumps(points, indent=2), encoding="utf-8")
    (args.out_dir / "report.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (args.out_dir / "report.md").write_text(
        "\n".join(
            [
                f"# {args.title}",
                "",
                f"Total rows: {meta['n_total']}",
                f"Top rows: {meta['n_top']}",
                "",
                "Top 10 ids:",
                "",
                "\n".join([f"- {p['id']} score={p['anomaly_score']:.5f}" for p in points[:10]]),
                "",
            ]
        ),
        encoding="utf-8",
    )

    html_out = HTML_TEMPLATE.format(
        title=args.title,
        subtitle=f"Generated from candidates.parquet, tag={args.tag}",
        repo_url=args.repo_url,
        rows=rows_html,
        meta_json=json.dumps(meta, indent=2),
        points_json=json.dumps(points),
    )
    (args.out_dir / "index.html").write_text(html_out, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
