"""`naavik run CONFIG`: one config as a one-task batch, built from the
project file, so a quick try is staged, checked, recorded, and followed
like any batch.

The manifest is made here as data (never written into the repository):
the project file's keys through `project:`, `training.config` = CONFIG,
and one group `run` with one experiment named after the config file, whose
overrides are the `--override`s. Resources are the project's
`run_defaults.resources` with `--gpus` / `--walltime` on top.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from naavik.launcher.errors import HpcError

GROUP = "run"


def parse_override(text: str) -> dict:
    """`PATH=VALUE` (a path in the session) or `component:PATH=VALUE`
    (inside that component's entry wherever it is listed); VALUE is YAML."""
    key, separator, value = text.partition("=")
    if not separator or not key.strip():
        raise HpcError(f"--override {text!r} must be PATH=VALUE or component:PATH=VALUE")
    component, colon, path = key.strip().partition(":")
    item = {"path": path, "component": component} if colon else {"path": component}
    if not item["path"] or (colon and not component):
        raise HpcError(f"--override {text!r} must be PATH=VALUE or component:PATH=VALUE")
    try:
        item["value"] = yaml.safe_load(value)
    except yaml.YAMLError as error:
        raise HpcError(f"--override {text!r}: the value is not valid YAML ({error})") from None
    return item


def experiment_name(config: str) -> str:
    """A name from the config file's (`vit-attnpool.yaml` -> `vit-attnpool`)."""
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(config).stem).strip("-")
    return name or "config"


def run_manifest(*, config: str, project: str, run_defaults, overrides: list[str],
                 note: str | None, gpus: int | None, walltime: str | None) -> dict:
    """The mission file of `naavik run`, in the current names
    (repository-relative `config` and `project`)."""
    resources = dict(run_defaults.resources)
    if gpus is not None:
        resources["gpus"] = gpus
    if walltime is not None:
        resources["walltime"] = walltime
    job = {"per_experiment": resources,
           "recipes": [{"name": experiment_name(config),
                        "overrides": [parse_override(item) for item in overrides]}]}
    if run_defaults.max_parallel:
        job["max_parallel_experiments"] = run_defaults.max_parallel
    manifest = {"format_version": 1, "name": experiment_name(config), "project_file": project,
                "base": {"config_file": config}, "jobs": {GROUP: job}}
    if note:
        manifest["note"] = note
    return manifest


def manifest_bytes(manifest: dict) -> bytes:
    return yaml.safe_dump(manifest, sort_keys=False).encode()
