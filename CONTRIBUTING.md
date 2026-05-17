# Contributing to ROBIN

Thanks for your interest in contributing. This project welcomes bug reports,
feature requests, and pull requests.

## Development setup

```bash
git clone https://github.com/Saher-Elsayed/ROBIN
cd ROBIN
pip install -e ".[dev,notebooks]"
pre-commit install
```

## Running the test suite

```bash
make test                 # unit tests + coverage
make lint                 # black + ruff + mypy
make format               # auto-format
```

## Pull request checklist

- [ ] Unit tests added or updated.
- [ ] `make test` and `make lint` pass locally.
- [ ] Public API changes are reflected in `docs/`.
- [ ] If a change affects published numbers in `paper/robin.tex`, the figure-
      regeneration scripts (`scripts/generate_figures.py`,
      `scripts/generate_tables.py`) are re-run and the changes match the paper.

## Branching

- `main` is the integration branch.
- Feature branches: `feat/<short-description>`.
- Bugfix branches: `fix/<short-description>`.

## Code style

- Python 3.10+, type hints everywhere.
- `black` with line length 100, `ruff` for linting, `mypy` for type checks.
- Docstrings in NumPy style.

## Reporting security issues

See `SECURITY.md`.
