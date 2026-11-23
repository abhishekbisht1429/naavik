# Development and Testing

[← Docs](README.md) · [Project README](../README.md)

This page is for contributors working on Naavik itself.

## Testing

Install the package in editable mode with the `vision` extra used by the
broader test suite, then run the tests:

```bash
python -m pip install -e '.[vision]'
python -m pytest
```

To focus on one area, pass a test directory or file:

```bash
python -m pytest src/tests/engine -q
```

With `pytest-xdist` installed, run files in parallel with:

```bash
python -m pytest -n 8 --dist loadfile
```

The GitHub Actions workflow runs tests on Python 3.12 and 3.13. The suite
covers components, sessions, engine execution, distributed training,
checkpoints, the launcher (local, SSH, and Slurm hosts), run records and reporting, data,
configuration files, documentation examples, and output handling.

## Project layout

```text
naavik/
├── docs/                   # Guides, concepts, and reference
├── src/
│   ├── naavik/             # Main Python package
│   │   ├── cli/            # Naavik command
│   │   ├── components/     # Component model and built-ins
│   │   ├── engine/         # Worker execution and supervision
│   │   ├── session/        # Training and analysis sessions
│   │   ├── launcher/       # Mission construction and execution
│   │   ├── hpc/            # Slurm scheduler support
│   │   ├── runs/           # Run registry
│   │   ├── reporting/      # Run comparison
│   │   ├── data/           # Data utilities
│   │   └── config_files.py
│   ├── training_framework/ # Compatibility alias for the earlier package name
│   └── tests/              # Unit and integration tests
├── .github/workflows/      # CI configuration
├── README.md
└── pyproject.toml
```
