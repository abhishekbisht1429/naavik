"""`validate`, `render`, and `launch` check every task config with the batch's
own code, in a separate process on the batch's exported tree."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

import naavik
from tests.hpc.batch_repo import PROFILE, make_repo
from tests.hpc.test_end_to_end import COMPONENTS, environment, session  # noqa: F401 (fixture)
from tests.launcher.test_host_end_to_end import here, launch, record, wait_for  # noqa: F401
from naavik.cli import main as naavik_cli
from naavik.launcher.ports.source_tree import source_tree

SRC = Path(naavik.__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def code_check_on(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAINING_FRAMEWORK_CODE_CHECK", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True).stdout.strip()


def commit(root, message="change"):
    git(root, "add", "-A")
    git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", message)


def project(tmp_path, *configs, requirements=None, code=None, extra=None):
    """A repository with the integration components and one experiment per
    config (files `configs/<name>.yaml`)."""
    experiments, files = [], {"my_project/components.py": COMPONENTS.read_text()}
    for index, config in enumerate(configs):
        files[f"configs/c{index}.yaml"] = yaml.safe_dump({"sessions": [config]})
        experiments.append({"name": f"c{index}"})
    manifest = {
        "version": 1, "name": "check", "code": code or {"framework": "dependency", "paths": ["."]},
        "training": {"config": "configs/c0.yaml"},
        "groups": {"cpu": {"resources": {"walltime": "00:10:00", "cpus": 2, "memory": "4G"},
                           "experiments": experiments[:1]}},
    }
    if requirements is not None:
        manifest["environment"] = {"requirements": "requirements/hpc.txt"}
    files.update(extra or {})
    return make_repo(tmp_path, manifest=manifest, config={"sessions": [configs[0]]},
                     requirements=requirements, extra_files=files)


def validate(root, targets):
    return naavik_cli(["validate", str(root / "batch.yaml"), "--targets", str(targets)])


# What it finds ---------------------------------------------------------------------------------


def test_a_sound_batch_passes_the_check(tmp_path, here, capsys):
    assert validate(project(tmp_path, session()), here) == 0
    output = capsys.readouterr().out
    assert "checking 1 experiment config(s) with the code of" in output and "ok: 1 experiment(s)" in output


def test_a_misspelt_component_is_refused_naming_the_task(tmp_path, here, capsys):
    config = session()
    config["integration_resultz"] = {}
    assert validate(project(tmp_path, config), here) == 2
    err = capsys.readouterr().err
    assert "cpu/c0/000: " in err and "'integration_resultz'" in err


def test_a_bad_schema_value_is_refused_naming_the_component(tmp_path, here, capsys):
    config = dict(session(), tensorboard={"serve": False}, tensorboard_summaries={"call_every": 0})
    assert validate(project(tmp_path, config), here) == 2
    err = capsys.readouterr().err
    assert "cpu/c0/000: tensorboard_summaries: " in err and "call_every" in err


def test_render_and_launch_refuse_before_anything_is_staged(tmp_path, here, capsys):
    config = session()
    config["integration_resultz"] = {}
    root = project(tmp_path, config)
    assert naavik_cli(["render", str(root / "batch.yaml"), "--targets", str(here),
                   "--output", str(tmp_path / "preview")]) == 2
    assert not (tmp_path / "preview").exists()
    assert naavik_cli(["launch", str(root / "batch.yaml"), "--targets", str(here), "--yes"]) == 2
    assert not (tmp_path / "here" / "batches").exists()


def test_what_cannot_be_checked_is_said_once(tmp_path, here, capsys):
    config = dict(session(), import_components={"model": {"checkpoint": str(tmp_path / "missing")}})
    assert validate(project(tmp_path, config), here) == 0
    notes = [line for line in capsys.readouterr().out.splitlines() if line.startswith("note: ")]
    assert notes == ["note: cpu/c0/000: not checked: import_components restores components "
                     "from checkpoints, so its plan is only known when the session is built"]


# Which code ------------------------------------------------------------------------------------


def test_the_check_runs_the_batchs_commit_not_the_working_tree(tmp_path, here, capsys):
    # Committed broken, fixed in the working tree only: a snapshot batch runs
    # (and is checked with) the fix; a commit batch would run HEAD.
    config = session()
    root = project(tmp_path, config)
    components = root / "my_project" / "components.py"
    components.write_text(components.read_text().replace('@resource("integration_dataset"',
                                                          '@resource("integration_datasett"'))
    commit(root, "a broken name")
    assert validate(root, here) == 2                                   # HEAD is broken
    capsys.readouterr()
    components.write_text(COMPONENTS.read_text())                      # fixed, uncommitted
    targets = yaml.safe_load(here.read_text())
    targets["profiles"]["cluster"]["worktree"] = "snapshot"
    here.write_text(yaml.safe_dump(targets))
    assert validate(root, here) == 0                                   # the snapshot is fixed


def test_a_commit_is_exported_once_and_reused(tmp_path, monkeypatch):
    root = make_repo(tmp_path / "r")
    sha = git(root, "rev-parse", "HEAD")
    first = source_tree(root, sha)
    assert (first / "batch.yaml").exists() and first.parent.name == "trees"
    marker = first / "marker"
    marker.write_text("kept")
    assert source_tree(root, sha) == first and marker.exists()         # not exported again
    assert not [path for path in first.parent.iterdir() if path.name.startswith(".")]


# Which framework -------------------------------------------------------------------------------


def _managed(profile_path: Path, tmp_path) -> Path:
    """A remote-style profile with a managed environment (the pin matters)."""
    data = copy.deepcopy(PROFILE)
    data["profiles"]["cluster"]["roots"] = {k: str(tmp_path / "c" / k) for k in ("batches", "envs", "data")}
    profile_path.write_text(yaml.safe_dump(data))
    return profile_path


def test_a_git_pin_must_be_the_framework_that_checks(tmp_path, capsys):
    pin = "training-framework @ git+https://example.org/tf.git@" + "a" * 40 + "\ntorch==2.13.0\n"
    root = project(tmp_path, session(), requirements=pin)
    assert validate(root, _managed(tmp_path / "targets.yaml", tmp_path)) == 2
    err = capsys.readouterr().err
    assert f"the host runs training-framework at commit {'a' * 12}" in err
    assert "install the pinned framework here" in err


def test_a_version_pin_accepts_a_checkout_of_that_version_with_a_note(tmp_path, capsys):
    version = __import__("tomllib").loads((SRC.parent / "pyproject.toml").read_text())["project"]["version"]
    root = project(tmp_path, session(), requirements=f"training-framework=={version}\ntorch==2.13.0\n")
    assert validate(root, _managed(tmp_path / "targets.yaml", tmp_path)) == 0
    # The note when the check ran on a checkout of that version (as from a
    # clone); none when that version is installed (as after `pip install .`).
    from naavik.runs.facts import framework_facts

    installed = framework_facts()["version"] == version
    assert ("checked with a naavik checkout" in capsys.readouterr().out) is not installed
    other = project(tmp_path / "other", session(), requirements="training-framework==0.1.0\ntorch==2.13.0\n")
    assert validate(other, _managed(tmp_path / "targets2.yaml", tmp_path)) == 2
    assert "the host runs training-framework 0.1.0" in capsys.readouterr().err


def test_checkout_mode_checks_with_the_framework_in_the_tree(tmp_path, here, capsys):
    framework = {f"src/naavik/{path.relative_to(SRC / 'naavik')}": path.read_text()
                 for path in (SRC / "naavik").rglob("*.py")}
    config = session()
    root = project(tmp_path, config, code={"framework": "checkout", "paths": ["src", "."]},
                   extra=framework)
    assert validate(root, here) == 0, capsys.readouterr().err
    tree = source_tree(root, git(root, "rev-parse", "HEAD"))
    assert (tree / "src" / "naavik" / "launcher" / "evaluate.py").exists()


def test_checkout_mode_without_the_framework_in_its_paths_is_refused(tmp_path, here, capsys):
    root = project(tmp_path, session(), code={"framework": "checkout", "paths": ["."]})
    assert validate(root, here) == 2
    err = capsys.readouterr().err
    # Whatever framework the interpreter finds, it is not the batch's own.
    assert "not the mission's own checkout" in err and "is it in code.python_paths?" in err


# Skipping, never silently --------------------------------------------------------------------------


def test_a_skipped_check_is_warned_about_and_recorded(tmp_path, here, monkeypatch, capsys):
    monkeypatch.setenv("TRAINING_FRAMEWORK_CODE_CHECK", "0")
    config = session()
    config["integration_resultz"] = {}                 # would be refused
    batch_dir, receipt = launch(project(tmp_path, session()), here, capsys)
    assert receipt["code_check"] == "skipped"
    assert naavik_cli(["validate", str(project(tmp_path / "x", config) / "batch.yaml"),
                   "--targets", str(here)]) == 0
    assert "warning: the experiment configs were not checked (TRAINING_FRAMEWORK_CODE_CHECK=0)" in \
        capsys.readouterr().err


def test_a_checked_launch_is_recorded_as_checked(tmp_path, here, capsys):
    batch_dir, receipt = launch(project(tmp_path, session()), here, capsys)
    assert receipt["code_check"] == "done"
    wait_for(lambda: record(batch_dir, "cpu/c0/000"), "the task's record")


def test_the_check_follows_source_ref_not_head(tmp_path, here, capsys):
    # The batch runs `source.ref`, a commit whose components are broken; HEAD
    # (clean, same manifest) has them fixed. Checking HEAD would pass.
    root = project(tmp_path, session())
    data = yaml.safe_load((root / "batch.yaml").read_text())
    data["source"] = {"strategy": "bundle", "ref": "broken"}
    (root / "batch.yaml").write_text(yaml.safe_dump(data))
    components = root / "my_project" / "components.py"
    components.write_text(components.read_text().replace('@resource("integration_dataset"',
                                                          '@resource("integration_datasett"'))
    commit(root, "broken")
    git(root, "tag", "broken")
    components.write_text(COMPONENTS.read_text())
    commit(root, "fixed")
    capsys.readouterr()
    assert validate(root, here) == 2
    assert "'integration_dataset'" in capsys.readouterr().err


def test_a_note_common_to_many_tasks_is_said_once(tmp_path, here, capsys):
    config = dict(session(), import_components={"model": {"checkpoint": str(tmp_path / "missing")}})
    root = project(tmp_path, config)
    data = yaml.safe_load((root / "batch.yaml").read_text())
    data["groups"]["cpu"]["experiments"][0]["matrix"] = [
        {"path": "session_config.rng_seed", "values": [1, 2, 3]}]
    (root / "batch.yaml").write_text(yaml.safe_dump(data))
    commit(root, "three tasks")
    assert validate(root, here) == 0
    notes = [line for line in capsys.readouterr().out.splitlines() if line.startswith("note: ")]
    assert len(notes) == 1 and notes[0].startswith("note: cpu/c0/000: not checked:")
