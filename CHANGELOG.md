# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Repository hygiene: single advertised version (`0.3.0`), clearer README, tighter `.gitignore`.
- `__version__` now reads the installed package metadata when available.

## [0.3.0] - 2026-02-15

### Added
- FastAPI landing page and annotations dashboard (Bootstrap + Plotly).
- Static batch HTML annotator (`astrooracle batch-html`).
- Graph anomaly context (kNN on the sphere).
- Lightweight explainability export (`explain-top` / `/explain`).
- Gaia ingestion helpers (`gaia-cone`, `gaia-adql`) behind the `[astro]` extra.
- Hybrid anomaly fusion (`--mode hybrid`).
- Optional "Mode Chaos" endpoint when a `timeseries` column is present.

### Changed
- Packaging extras split into `dev`, `api`, `plotly`, `graph`, `astro`, `explain`, and `docs`.

## [0.1.0]

### Added
- First installable version: candidates, ranking, annotation loop, and retrain hook.
