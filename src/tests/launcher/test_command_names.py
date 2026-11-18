"""One command, `naavik` (alias `navik`): `launch` for `submit`, `--target`
for `--profile`, `targets.yaml` for `hpc.yaml`, and `summarize` from the
reporting program. The earlier names keep working."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

import training_framework
from tests.hpc.batch_repo import PROFILE, make_repo
from training_framework.cli import main as naavik
from training_framework.hpc.__main__ import main as hpc_main

SRC = Path(training_framework.__file__).resolve().parents[1]
PYPROJECT = SRC.parent / "pyproject.toml"


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    home = tmp_path / "config"
    (home / "training-framework").mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return home / "training-framework"


def _targets(**extra):
    data = copy.deepcopy(PROFILE)
    data["profiles"].update(extra)
    return data


def _validate(root, *options):
    return naavik(["validate", str(root / "batch.yaml"), *options])


def test_both_command_names_are_installed_and_run_the_cli():
    scripts = tomllib.loads(PYPROJECT.read_text())["project"]["scripts"]
    assert scripts == {"naavik": "training_framework.cli:main",
                       "navik": "training_framework.cli:main"}
    result = subprocess.run([sys.executable, "-m", "training_framework.cli", "--help"],
                            capture_output=True, text=True, check=False,
                            env={**os.environ, "PYTHONPATH": str(SRC)})
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage: naavik ")


def test_launch_and_submit_are_one_command(capsys):
    with pytest.raises(SystemExit):
        naavik(["launch", "--help"])
    launch = capsys.readouterr().out
    with pytest.raises(SystemExit):
        naavik(["submit", "--help"])
    submit = capsys.readouterr().out
    assert launch.startswith("usage: naavik launch ")
    assert launch.split("\n", 1)[1] == submit.split("\n", 1)[1]


def test_launch_is_listed_and_submit_kept_as_its_alias(capsys):
    with pytest.raises(SystemExit):
        naavik(["--help"])
    assert "launch (submit)" in capsys.readouterr().out


def test_launch_and_submit_fail_the_same_way(tmp_path, capsys):
    missing = tmp_path / "missing.yaml"
    results = []
    for name in ("launch", "submit"):
        code = naavik([name, "batch.yaml", "--targets", str(missing)])
        results.append((code, capsys.readouterr().err))
    assert results[0] == results[1]
    assert results[0][0] == 2


@pytest.mark.parametrize("name", ["launch", "submit"])
def test_launch_and_submit_both_reach_the_launch(tmp_path, monkeypatch, capsys, name):
    # Launching first checks for ssh and rsync; with neither on PATH that is
    # where it stops, which shows the command reached the launch itself.
    root = make_repo(tmp_path)
    targets = tmp_path / "targets.yaml"
    targets.write_text(yaml.safe_dump(PROFILE))
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert naavik([name, str(root / "batch.yaml"), "--targets", str(targets)]) == 2
    assert "ssh and rsync not found on this machine's PATH" in capsys.readouterr().err


def test_target_and_profile_select_the_same_entry(tmp_path, capsys):
    root = make_repo(tmp_path)
    targets = tmp_path / "targets.yaml"
    other = copy.deepcopy(PROFILE["profiles"]["cluster"])
    other["roots"] = {"batches": "/other/batches", "envs": "/other/envs", "data": "/other/data"}
    targets.write_text(yaml.safe_dump(_targets(other=other)))

    outputs = []
    for options in (["--targets", str(targets), "--target", "other"],
                    ["--profiles", str(targets), "--profile", "other"]):
        assert _validate(root, *options) == 0
        outputs.append(capsys.readouterr().out)
    assert outputs[0] == outputs[1]
    assert "ok: 4 task(s)" in outputs[0]


def test_the_target_file_is_read_from_targets_yaml(tmp_path, config_home, capsys):
    root = make_repo(tmp_path)
    (config_home / "targets.yaml").write_text(yaml.safe_dump(PROFILE))
    assert _validate(root) == 0
    assert "hpc.yaml" not in capsys.readouterr().err


def test_an_earlier_hpc_yaml_is_still_read_with_a_note(tmp_path, config_home, capsys):
    root = make_repo(tmp_path)
    (config_home / "hpc.yaml").write_text(yaml.safe_dump(PROFILE))
    assert _validate(root) == 0
    err = capsys.readouterr().err
    assert f"reading hosts from {config_home / 'hpc.yaml'}" in err
    assert f"rename it to {config_home / 'hosts.yaml'}" in err


def test_targets_yaml_wins_and_the_ignored_hpc_yaml_is_named(tmp_path, config_home, capsys):
    root = make_repo(tmp_path)
    (config_home / "targets.yaml").write_text(yaml.safe_dump(PROFILE))
    broken = {"default": "cluster", "profiles": {"cluster": {"nonsense": 1}}}
    (config_home / "hpc.yaml").write_text(yaml.safe_dump(broken))
    assert _validate(root) == 0
    assert "ignored" in capsys.readouterr().err


def test_no_target_file_names_the_file_to_create(tmp_path, config_home, capsys):
    root = make_repo(tmp_path)
    assert _validate(root) == 2
    assert str(config_home / "hosts.yaml") in capsys.readouterr().err


def test_a_target_file_may_name_its_entries_targets(tmp_path, capsys):
    root = make_repo(tmp_path)
    data = {"default": "cluster", "targets": PROFILE["profiles"]}
    path = tmp_path / "targets.yaml"
    path.write_text(yaml.safe_dump(data))
    assert _validate(root, "--targets", str(path)) == 0


def test_a_target_file_with_both_entry_keys_is_refused(tmp_path, capsys):
    root = make_repo(tmp_path)
    data = {"default": "cluster", "targets": PROFILE["profiles"], "profiles": PROFILE["profiles"]}
    path = tmp_path / "targets.yaml"
    path.write_text(yaml.safe_dump(data))
    assert _validate(root, "--targets", str(path)) == 2
    err = capsys.readouterr().err
    assert "targets" in err and "profiles" in err


def test_summarize_is_a_naavik_command(capsys):
    with pytest.raises(SystemExit):
        naavik(["summarize", "--help"])
    assert capsys.readouterr().out.startswith("usage: naavik summarize ")


def test_the_earlier_entry_points_keep_their_names(capsys):
    with pytest.raises(SystemExit):
        hpc_main(["--help"])
    assert capsys.readouterr().out.startswith("usage: python -m training_framework.hpc ")
    from training_framework.reporting.__main__ import main as reporting_main

    with pytest.raises(SystemExit):
        reporting_main(["summarize", "--help"])
    assert capsys.readouterr().out.startswith(
        "usage: python -m training_framework.reporting summarize ")
