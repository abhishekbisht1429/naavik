"""The framework's distribution name, `naavik`, and its earlier one,
`training-framework`.

An environment may have the framework installed under either name, and a
requirements file may pin either (`naavik @ git+...@<sha>`,
`training-framework==0.5.0`). Everything that asks which framework is
installed, or which one a requirements file pins, asks here, so both names
are read the same way everywhere.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

#: The distribution's name.
DISTRIBUTION = "naavik"
#: Names it was published under before.
EARLIER_DISTRIBUTIONS = ("training-framework",)
#: Every name the framework may be installed or pinned under, newest first.
NAMES = (DISTRIBUTION, *EARLIER_DISTRIBUTIONS)
#: How messages name it.
DESCRIBED = f"{DISTRIBUTION} (or its earlier name {EARLIER_DISTRIBUTIONS[0]})"


def installed_version() -> str | None:
    """The installed framework's version, by either name (the current one
    first); None when neither is installed."""
    for name in NAMES:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def distribution_of(package_dir: Path) -> importlib.metadata.Distribution | None:
    """The installed distribution, by either name, whose files are the
    package at `package_dir`. None when no installation is: an older
    install beside a source checkout says nothing about the checkout."""
    for name in NAMES:
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for package in ("naavik", "training_framework"):
            located = Path(distribution.locate_file(f"{package}/__init__.py")).resolve()
            if located.parent == package_dir:
                return distribution
    return None


def framework_pins(requirements) -> list:
    """The pins of the framework, under either name, in `requirements`
    (`ParsedRequirements`)."""
    return [pin for name in NAMES for pin in requirements.pins_named(name)]


def is_framework_project(name) -> bool:
    """Whether a `pyproject.toml`'s `project.name` is the framework's."""
    return isinstance(name, str) and name.strip().lower().replace("_", "-") in NAMES
