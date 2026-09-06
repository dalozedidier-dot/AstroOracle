# Contributing

## Development setup

```bash
python -m pip install -U pip
pip install -e ".[dev,api]"
pre-commit install
```

## Running checks

```bash
ruff check .
pytest -q --cov=astrooracle --cov-report=term-missing
```

Optional local format:

```bash
ruff format .
```

`black` remains available via the `dev` extra for existing hooks, but new work
should prefer Ruff.

## Pull requests

- Keep PRs focused and small.
- Add tests for bug fixes and new features.
- Avoid changing public APIs without a clear migration path.
- Do not commit `__pycache__/`, `.pytest_cache/`, `.coverage`, or root `*.patch` files.
