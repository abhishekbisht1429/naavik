"""The framework's distribution is `naavik`; `training-framework`, its
earlier name, is read too: requirements may pin either (once), a checkout
may be named either, and an environment may have either installed."""

from __future__ import annotations

import copy
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from naavik import distribution
from tests.hpc.batch_repo import MANIFEST, PROFILE, compile_repo, make_repo
from naavik.launcher.errors import PlanError


def compiled_with(tmp_path, requirements, framework="dependency"):
    manifest = copy.deepcopy(MANIFEST)
    manifest["code"] = {"framework": framework, "paths": ["."]}
    return compile_repo(make_repo(tmp_path, manifest=manifest, requirements=requirements))


@pytest.mark.parametrize("name", ["naavik", "training-framework", "training_framework"])
def test_either_name_pins_the_framework(tmp_path, name):
    plan = compiled_with(tmp_path, f"torch==2.13.0\n{name}==0.5.0\n").plan
    assert plan.preflight["framework"] == {"location": "environment", "distribution": "0.5.0"}


def test_both_names_pinned_is_refused(tmp_path):
    with pytest.raises(PlanError, match="pin both naavik and training-framework: they are the same "
                                        "framework under its current and earlier name; keep one"):
        compiled_with(tmp_path, "naavik==0.5.0\ntraining-framework==0.5.0\n")


def test_no_pin_names_both(tmp_path):
    with pytest.raises(PlanError, match=r"must pin naavik \(or its earlier name training-framework\) "
                                        "exactly once"):
        compiled_with(tmp_path, "torch==2.13.0\n")


@pytest.mark.parametrize("name", ["naavik", "training-framework"])
def test_a_checkout_mission_refuses_a_pin_naming_it(tmp_path, name):
    with pytest.raises(PlanError, match=f"must not contain {name} \\(it comes from the checkout\\)"):
        compiled_with(tmp_path, f"{name}==0.5.0\n", framework="checkout")


@pytest.mark.parametrize("name, framework", [("naavik", True), ("training-framework", True),
                                             ("Training_Framework", True), ("my-project", False),
                                             (None, False)])
def test_a_checkout_named_either_is_the_framework(name, framework):
    assert distribution.is_framework_project(name) is framework


def test_the_installed_version_prefers_the_current_name(monkeypatch):
    installed = {"naavik": "0.6.0", "training-framework": "0.3.4"}

    def version(name):
        if name not in installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed[name]

    monkeypatch.setattr(importlib.metadata, "version", version)
    assert distribution.installed_version() == "0.6.0"
    del installed["naavik"]
    assert distribution.installed_version() == "0.3.4"
    installed.clear()
    assert distribution.installed_version() is None


def test_the_distribution_of_the_imported_code_is_the_one_whose_files_it_is(monkeypatch, tmp_path):
    here, elsewhere = tmp_path / "site", tmp_path / "old-site"
    (here / "naavik").mkdir(parents=True)
    (elsewhere / "training_framework").mkdir(parents=True)
    found = {"naavik": SimpleNamespace(version="0.6.0", locate_file=lambda p: here / p),
             "training-framework": SimpleNamespace(version="0.3.4", locate_file=lambda p: elsewhere / p)}

    def lookup(name):
        if name not in found:
            raise importlib.metadata.PackageNotFoundError(name)
        return found[name]

    monkeypatch.setattr(importlib.metadata, "distribution", lookup)
    assert distribution.distribution_of(here / "naavik").version == "0.6.0"
    # An older install beside the imported code says nothing about it.
    assert distribution.distribution_of(tmp_path / "checkout" / "naavik") is None


def test_an_environment_built_from_a_naavik_pin_passes_preflight(tmp_path):
    from tests.hpc.remote.test_bootstrap import record, run_bootstrap
    from tests.hpc.stub_framework import FILES, build_wheel, pinned_line
    from naavik.launcher.compile.render import plan_files
    from naavik.launcher.ports.local import LocalFilesystem, LocalSourceStager
    from naavik.launcher.remote.task import interpreter_facts

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    requirements = pinned_line("naavik", build_wheel(wheels, "naavik", "0.5.0", FILES))
    root = make_repo(tmp_path, requirements=requirements,
                     extra_files={"my_project/components.py": "COMPONENTS = True\n"})
    profile = copy.deepcopy(PROFILE)
    cluster = profile["profiles"]["cluster"]
    cluster["roots"] = {key: str(tmp_path / "cluster" / key) for key in ("batches", "envs", "data")}
    cluster["env"]["site_python_identity"] = interpreter_facts()
    cluster["hooks"] = {}
    plan = compile_repo(root, profile_data=profile).plan
    filesystem = LocalFilesystem()
    filesystem.makedirs(plan.batch_dir)
    LocalSourceStager(root).stage(plan.sha, f"{plan.batch_dir}/source")
    for relative, content in plan_files(plan, {}).items():
        filesystem.write(f"{plan.batch_dir}/{relative}", content.encode())
    result = run_bootstrap(plan)
    assert result.returncode == 0, result.stderr
    assert record(plan)["status"] == "built"
    assert Path(plan.environment["location"]).is_dir()


def test_the_remote_task_reports_the_installed_version_by_either_name(monkeypatch):
    from naavik.launcher.remote.task import framework_facts

    monkeypatch.setattr(importlib.metadata, "version", lambda name: {"naavik": "0.6.0"}.get(name) or (
        _ for _ in ()).throw(importlib.metadata.PackageNotFoundError(name)))
    assert framework_facts()["distribution"] == "0.6.0"
