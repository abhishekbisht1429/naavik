"""The launcher / hpc split: target-neutral code in `training_framework.launcher`,
only Slurm in `training_framework.hpc`, and every earlier module path still
importing as the same module."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import training_framework.hpc as hpc
import training_framework.launcher as launcher

SRC = Path(launcher.__file__).resolve().parents[2]      # .../src


def _launcher_modules() -> list[str]:
    return sorted(
        info.name for info in pkgutil.walk_packages(launcher.__path__, "training_framework.launcher.")
        if not info.name.endswith("__main__")
    )


def _run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *argv], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(SRC)}, check=False)


@pytest.mark.parametrize("new", _launcher_modules())
def test_every_earlier_module_path_is_the_moved_module(new):
    old = new.replace("training_framework.launcher.", "training_framework.hpc.", 1)
    assert importlib.import_module(old) is importlib.import_module(new)


@pytest.mark.parametrize("old, new", [
    ("training_framework.hpc.scheduler.slurm", "training_framework.hpc.slurm"),
    ("training_framework.hpc.scheduler.slurm_submitter", "training_framework.hpc.slurm_submitter"),
])
def test_the_slurm_modules_import_under_their_earlier_paths(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)


def test_a_patch_through_the_earlier_path_reaches_the_moved_module(monkeypatch):
    from training_framework.hpc.services import submit as old_submit
    from training_framework.launcher.services import submit as new_submit

    monkeypatch.setattr(old_submit, "submit", "patched")
    assert new_submit.submit == "patched"


def test_the_earlier_cli_module_is_the_launcher_cli():
    from training_framework.hpc import __main__ as old_cli
    from training_framework.launcher import __main__ as new_cli

    assert old_cli is new_cli


def test_an_earlier_path_warns_that_it_is_deprecated():
    result = _run("-W", "error::DeprecationWarning", "-c",
                  "import training_framework.hpc.compile.compiler")
    assert result.returncode != 0
    assert ("training_framework.hpc.compile moved to training_framework.launcher.compile"
            in result.stderr)


def test_a_path_that_never_existed_still_fails():
    # Probing (as tools and optional-import code do) must not be told it exists.
    assert importlib.util.find_spec("training_framework.hpc.services.no_such_module") is None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("training_framework.hpc.services.no_such_module")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("training_framework.hpc.no_such_module")


def test_the_hpc_package_holds_only_slurm():
    names = {info.name for info in pkgutil.iter_modules(hpc.__path__)}
    assert names == {"__main__", "slurm", "slurm_submitter", "task"}


def test_the_launcher_names_the_hpc_package_only_at_its_composition_point():
    """Dependency direction: hpc uses the launcher, the launcher only looks
    the Slurm bundle up (`scheduler/__init__.py`) and keeps the earlier
    `schema.SlurmSettings` name importable (`config/schema.py`, lazily)."""
    users = set()
    for name in _launcher_modules():
        tree = ast.parse(inspect.getsource(importlib.import_module(name)))
        for node in ast.walk(tree):
            module = (node.module if isinstance(node, ast.ImportFrom) else
                      node.names[0].name if isinstance(node, ast.Import) else None)
            if module and module.startswith("training_framework.hpc"):
                users.add(name)
    assert users == {"training_framework.launcher.scheduler",
                     "training_framework.launcher.config.schema"}


@pytest.mark.parametrize("package", ["training_framework.hpc", "training_framework.launcher"])
def test_the_command_line_runs_under_both_names(package):
    result = _run("-m", package, "--help")
    assert result.returncode == 0, result.stderr
    assert "validate" in result.stdout and "submit" in result.stdout


def test_the_remote_task_entry_point_keeps_its_protocol_name():
    """Rendered job scripts and environments built from older framework
    commits know only `training_framework.hpc.task`, so it is the one entry
    point, whatever package implements it."""
    from training_framework.hpc import task
    from training_framework.launcher.remote import task as implementation

    assert task.main is implementation.main
    assert importlib.util.find_spec("training_framework.launcher.task") is None
    result = _run("-m", "training_framework.hpc.task", "--help")
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
