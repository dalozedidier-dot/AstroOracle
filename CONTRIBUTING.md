# Contributing

## Development setup

```bash
python -m pip install -U pip
pip install -e ".[dev]"
pre-commit install
```

## Running checks

```bash
ruff check .
black .
pytest
mypy src
```

## Pull requests
- Keep PRs focused and small.
- Add tests for bug fixes and new features.
- Avoid changing public APIs without a clear migration path.
