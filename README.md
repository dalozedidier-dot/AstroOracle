# AstroOracle

Active learning oracle for astronomical anomaly triage.

- Fetch SkyView cutouts (or synthetic offline cutouts)
- Rank candidates via anomaly score + acquisition + diversity
- Label in CLI or in a small web UI
- Retrain hook when enough new labels are collected

## Badges

- CI: GitHub Actions (see `.github/workflows/ci.yml`)
- Release: GitHub Actions tags `v*` (see `.github/workflows/release.yml`)

## Install

```bash
pip install -e ".[dev,api,plotly,graph]"
```

Optional extras:

```bash
pip install -e ".[astro,explain,docs]"
```

## Quickstart

### 1) Web landing page (FastAPI)

```bash
astrooracle serve --candidates candidates.parquet --annotations annotations.csv
# then open http://127.0.0.1:8000/
```

Landing page features:

- `GET /` landing page with Plotly 3D preview and a CLI-like `/run` form
- `GET /dashboard` annotations dashboard + export
- `GET /viz3d` Plotly JSON
- `GET /graph_anomaly` graph context (kNN on sphere)
- `GET /explain` lightweight explanations + shareable prompts

### 2) CLI triage (interactive)

```bash
astrooracle run --candidates candidates.parquet --annotations annotations.csv
```

### 3) Generate a static batch annotator (HTML)

```bash
astrooracle batch-html --out-dir batch_out --n-query 12
# open batch_out/index.html
```

## New features included in this patch

- Landing page `/` + dashboard `/dashboard` (Bootstrap + Plotly)
- Toggle "Mode Chaos" (served via `/chaos`, requires a `timeseries` column)
- Graph anomaly context (`astrooracle graph-anomaly` or `/graph_anomaly`)
- Explainability JSONL (`astrooracle explain-top` or `/explain`)
- Gaia ingestion (`astrooracle gaia-cone` and `astrooracle gaia-adql`, requires `.[astro]`)
- Hybrid anomaly fusion (`--mode hybrid` in CLI and landing page)

## Citation

See `CITATION.cff`.

## License

MIT.
