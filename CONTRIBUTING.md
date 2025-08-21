# Contributing

Thank you for contributing! This repo uses:

- Typer CLI as the primary interface
- Ruff for lint/format
- Pytest for tests (fast tests only in CI)

## Setup

```
pip install -e '.[dev]'
pre-commit install
```

## Development

- Run fast tests:
```
pytest -q tests -k 'not slow'
```
- Lint/format:
```
ruff check . && ruff format --check .
```

## PRs

- Include a short description and screenshots/logs for UX changes
- Keep edits focused; avoid unrelated formatting churn
- Green CI required
