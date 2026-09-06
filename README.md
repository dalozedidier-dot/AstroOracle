# AstroOracle

[![CI](https://github.com/dalozedidier-dot/AstroOracle/actions/workflows/ci.yml/badge.svg)](https://github.com/dalozedidier-dot/AstroOracle/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Active-learning oracle for astronomical anomaly triage.**

AstroOracle is a Python package for programmatic, reproducible review of sky candidates:
fetch (or synthesize) cutouts, rank objects with anomaly + acquisition + diversity scores,
label them in a CLI or small web UI, then retrain when enough new labels exist.

This project is about **astronomy** (catalogs, cutouts, anomaly scores), not astrology.
It is also unrelated to [uiucsn/Astro-ORACLE](https://github.com/uiucsn/Astro-ORACLE),
a hierarchical classifier for the LSST alert stream.

## What it does

- Fetch SkyView cutouts, or use deterministic synthetic cutouts in `--offline` mode
- Rank candidates from `anomaly_score` plus acquisition and diversity terms
- Label in an interactive CLI or a FastAPI UI
- Trigger a retrain hook once a label threshold is reached
- Optional Gaia ingest, graph context, hybrid fusion, and explainability export

## Install

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,api,plotly,graph]"
```

Optional extras:

```bash
pip install -e ".[astro,explain,docs]"
```

| Extra | Purpose |
| --- | --- |
| `dev` | pytest, ruff, black, coverage |
| `api` | FastAPI + uvicorn UI |
| `plotly` | 3D / dashboard figures |
| `graph` | kNN graph context |
| `astro` | astropy / astroquery / Gaia |
| `explain` | LIME explanations |
| `docs` | Sphinx |

Offline mode does not need the `astro` extra. Set `ASTROORACLE_OFFLINE=1` or pass `--offline`.

## Quickstart

### 1) Web UI

```bash
astrooracle serve --candidates candidates.parquet --annotations annotations.csv --offline
# open http://127.0.0.1:8000/
```

Useful routes:

- `GET /` landing page, Plotly preview, `/run` form
- `GET /dashboard` annotation dashboard + export
- `GET /viz3d` Plotly JSON
- `GET /graph_anomaly` spherical kNN context
- `GET /explain` lightweight explanations
- `GET /chaos` chaos metrics (needs a `timeseries` column)

### 2) CLI triage

```bash
astrooracle run --candidates candidates.parquet --annotations annotations.csv --offline
```

### 3) Static batch annotator

```bash
astrooracle batch-html --out-dir batch_out --n-query 12 --offline
# open batch_out/index.html
```

## Candidate table

A candidates file (Parquet by default) must include at least:

- `id`
- `ra`
- `dec`
- `anomaly_score`

Optional columns used when present: `snr`, `ruwe`, `mag`, `embedding`, `timeseries`.

## Features

- Landing page + dashboard (Bootstrap + Plotly)
- Graph anomaly context (`astrooracle graph-anomaly` or `/graph_anomaly`)
- Explainability JSONL (`astrooracle explain-top` or `/explain`)
- Gaia helpers (`astrooracle gaia-cone`, `astrooracle gaia-adql`, requires `.[astro]`)
- Hybrid anomaly fusion (`--mode hybrid`)
- Chaos-mode toggle (`/chaos`) when timeseries data is available

## Development

```bash
pip install -e ".[dev,api,plotly,graph]"
pre-commit install
ruff check .
pytest -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
