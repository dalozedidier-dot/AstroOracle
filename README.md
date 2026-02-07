# AstroOracle

AstroOracle is an active learning “oracle” for astronomical anomaly triage.

- Reads candidate detections from a Parquet file.
- Ranks them by uncertainty + optional embedding diversity.
- Pulls multi-survey cutouts from SkyView.
- Annotates via CLI (matplotlib) or Jupyter (ipywidgets + Plotly).
- Logs events in append-only JSONL.
- Optionally triggers retraining.

## Install (editable)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[plotly,watch,notebook]"
```

## Quickstart (CLI)

Generate sample candidates:

```bash
python examples/sample_candidates.py
```

Run oracle:

```bash
astrooracle run --candidates candidates.parquet --interval 300 --n-query 6
```

Headless + PNG export:

```bash
astrooracle run --no-gui --save-cutouts cutouts/ --candidates candidates.parquet
```

## Jupyter UI

```bash
jupyter lab
```

Open `examples/oracle_notebook.ipynb`.

## Watch mode (watchdog)

```bash
astrooracle watch --candidates candidates.parquet
```

## Batch HTML export

```bash
astrooracle batch-html --candidates candidates.parquet --out-dir batch_out/ --n-query 60
```

Open `batch_out/index.html` and export annotations from the page.

## License

MIT


## Demo visuals (offline)

Generate sample candidates and create an HTML batch annotator + PNG cutouts without network access:

```bash
python examples/sample_candidates.py
astrooracle batch-html --candidates candidates.parquet --out-dir demo_out --n-query 12 --offline
```

Open `demo_out/index.html`.


## Nouveautés v0.2.0

- Ranking multicritère (anomaly score + acquisition + prior + diversité)
- Acquisition: entropy, margin, BALD (via ensemble), BADGE
- Diversité batch: k-center (core-set) et DPP greedy
- Entraînement: modèle ensemble logistic regression calibré + métriques (ECE)
- API: FastAPI (endpoints /rank, /annotate)
- Stats: DuckDB sur logs JSONL

### Commandes

```bash
astrooracle train --candidates candidates.parquet --annotations annotations.csv
astrooracle batch-html --out-dir out_html
astrooracle serve --host 127.0.0.1 --port 8000
astrooracle stats
```
