"""The launcher / hpc split: target-neutral code in `naavik.launcher`,
only Slurm in `naavik.hpc`, and every earlier module path still
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

import naavik.hpc as hpc
import naavik.launcher as launcher

SRC = Path(launcher.__file__).resolve().parents[2]      # .../src


def _launcher_modules() -> list[str]:
    return sorted(
        info.name for info in pkgutil.walk_packages(launcher.__path__, "naavik.launcher.")
        if not info.name.endswith("__main__")
    )


def _run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *argv], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(SRC)}, check=False)


#: The top-level names `naavik.hpc` had before the split; a module
#: added to the launcher since (e.g. `evaluate`) never had an earlier path.
EARLIER_TOP_LEVEL = {"compile", "config", "errors", "framework", "names", "ports", "remote",
                     "scheduler", "services"}


@pytest.mark.parametrize("new", [name for name in _launcher_modules()
                                 if name.split(".")[2] in EARLIER_TOP_LEVEL])
def test_every_earlier_module_path_is_the_moved_module(new):
    old = new.replace("naavik.launcher.", "naavik.hpc.", 1)
    assert importlib.import_module(old) is importlib.import_module(new)


def test_a_module_added_since_the_split_has_no_earlier_path():
    assert importlib.util.find_spec("naavik.hpc.evaluate") is None


@pytest.mark.parametrize("old, new", [
    ("naavik.hpc.scheduler.slurm", "naavik.hpc.slurm"),
    ("naavik.hpc.scheduler.slurm_submitter", "naavik.hpc.slurm_submitter"),
])
def test_the_slurm_modules_import_under_their_earlier_paths(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)


def test_a_patch_through_the_earlier_path_reaches_the_moved_module(monkeypatch):
    from naavik.hpc.services import submit as old_submit
    from naavik.launcher.services import submit as new_submit

    monkeypatch.setattr(old_submit, "submit", "patched")
    assert new_submit.submit == "patched"


def test_the_earlier_cli_module_is_the_launcher_cli():
    from naavik.hpc import __main__ as old_cli
    from naavik.launcher import __main__ as new_cli

    assert old_cli is new_cli


def test_an_earlier_path_warns_that_it_is_deprecated():
    result = _run("-W", "error::DeprecationWarning", "-c",
                  "import naavik.hpc.compile.compiler")
    assert result.returncode != 0
    assert ("naavik.hpc.compile moved to naavik.launcher.compile"
            in result.stderr)


def test_a_path_that_never_existed_still_fails():
    # Probing (as tools and optional-import code do) must not be told it exists.
    assert importlib.util.find_spec("naavik.hpc.services.no_such_module") is None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("naavik.hpc.services.no_such_module")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("naavik.hpc.no_such_module")


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
            if module and module.startswith("naavik.hpc"):
                users.add(name)
    assert users == {"naavik.launcher.scheduler",
                     "naavik.launcher.config.schema"}


@pytest.mark.parametrize("package", ["naavik.hpc", "naavik.launcher"])
def test_the_command_line_runs_under_both_names(package):
    result = _run("-m", package, "--help")
    assert result.returncode == 0, result.stderr
    assert "validate" in result.stdout and "submit" in result.stdout


def test_the_remote_task_entry_point_keeps_its_protocol_name():
    """Rendered job scripts and environments built from older framework
    commits know only `training_framework.hpc.task`, so it is the one entry
    point, whatever package implements it."""
    from naavik.hpc import task
    from naavik.launcher.remote import task as implementation

    assert task.main is implementation.main
    assert importlib.util.find_spec("naavik.launcher.task") is None
    result = _run("-m", "training_framework.hpc.task", "--help")
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
