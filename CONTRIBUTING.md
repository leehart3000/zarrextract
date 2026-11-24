# Contributing to zarrextract

## Purpose

zarrextract extracts points, time series and bounding boxes from Zarr
stores into Apache Arrow tables. It is a lighter alternative to
`xarray.open_zarr(...).sel(...)` for the common case where you only want a
small piece of a large store.

## Stack

- Python 3.12 or newer (tested locally on 3.13)
- Poetry 2 for dependency management and builds, with the PEP 621
  `[project]` table in `pyproject.toml`
- Runtime dependencies: `zarr` (3.x), `pyarrow`, `numpy`
- Development tools: Pytest, Mypy (strict) and Ruff (lint and format)

There is no pandas, xarray or Dask dependency. Ask before adding, removing
or upgrading any dependency.

## Conventions

- Source code lives in `src/zarrextract/`, tests in `tests/`.
- The package is typed (`py.typed`). All code must pass `mypy` in strict
  mode.
- Public names are exported from `zarrextract/__init__.py`. Modules that
  start with an underscore are private.
- Errors derive from `ZarrExtractError`.
- Tests run offline. They build small Zarr stores in `tmp_path` and never
  read remote data.
- Keep changes small and focused, and update the tests with the code.
- Commit messages have a short summary line in the imperative mood, with an
  optional body.

## Commands

Run all commands from the repository root.

Install the package and the development tools:

```bash
poetry install
```

Lint:

```bash
poetry run ruff check .
```

Check formatting (use `poetry run ruff format .` to apply it):

```bash
poetry run ruff format --check .
```

Type-check:

```bash
poetry run mypy
```

Run the tests:

```bash
poetry run pytest
```

Build the source distribution and wheel into `dist/`:

```bash
poetry build
```

All of lint, format check, type check and tests must pass before a change
is submitted.
