"""The commands and flags in their current names (missions, experiments,
jobs, recipes, hosts), and the earlier names from dev still accepted."""

from __future__ import annotations

import json

import pytest
import yaml

from tests.hpc.batch_repo import make_repo
from tests.launcher.test_hosts_file import CURRENT as HOSTS
from training_framework.cli import main as naavik


def receipt(batch_id):
    return {"batch_id": batch_id, "batch_dir": f"/m/{batch_id}", "status": "submitted",
            "nodes": {}, "profile": "cluster", "created": "2026-11-18T09:00:00+00:00",
            "task_count": 3}


@pytest.fixture
def receipts(tmp_path):
    directory = tmp_path / "receipts"
    directory.mkdir()
    (directory / "arch-1.json").write_text(json.dumps(receipt("arch-1")))
    return directory


def test_missions_lists_missions_and_batches_is_its_earlier_name(receipts, capsys):
    assert naavik(["missions", "--receipts", str(receipts)]) == 0
    current = capsys.readouterr().out
    assert current.splitlines()[0].split() == ["mission", "launched", "host", "experiments", "submit"]
    assert naavik(["batches", "--receipts", str(receipts)]) == 0
    assert capsys.readouterr().out == current


@pytest.mark.parametrize("command", [["show-site-interpreter", "--help"], ["probe-python", "--help"],
                                     ["launch", "--help"], ["submit", "--help"]])
def test_current_and_earlier_command_names(command, capsys):
    with pytest.raises(SystemExit) as stopped:
        naavik(command)
    assert stopped.value.code == 0


@pytest.mark.parametrize("flags", [["--host", "cluster", "--hosts-file"],
                                   ["--target", "cluster", "--targets"],
                                   ["--profile", "cluster", "--profiles"]])
def test_the_host_flags_and_their_earlier_names(tmp_path, flags, capsys):
    hosts = tmp_path / "hosts.yaml"
    hosts.write_text(yaml.safe_dump(HOSTS))
    root = make_repo(tmp_path)
    assert naavik(["validate", str(root / "batch.yaml"), *flags, str(hosts)]) == 0
    assert "ok: 4 experiment(s) in 1 job(s)" in capsys.readouterr().out


def test_job_and_recipe_choose_part_of_a_mission(tmp_path, capsys):
    hosts = tmp_path / "hosts.yaml"
    hosts.write_text(yaml.safe_dump(HOSTS))
    root = make_repo(tmp_path)
    for flags in (["--job", "probes"], ["--recipe", "probes/frozen"]):
        assert naavik(["validate", str(root / "batch.yaml"), "--hosts-file", str(hosts), *flags]) == 0
        assert "ok: 4 experiment(s)" in capsys.readouterr().out
    assert naavik(["validate", str(root / "batch.yaml"), "--hosts-file", str(hosts),
                   "--job", "nope"]) == 2
    assert "--job nope: the mission file has no such job" in capsys.readouterr().err
    assert naavik(["validate", str(root / "batch.yaml"), "--hosts-file", str(hosts),
                   "--recipe", "probes/nope"]) == 2
    assert "job 'probes' has no recipe 'nope'" in capsys.readouterr().err


def test_flags_are_spelled_out_not_abbreviated(tmp_path, capsys):
    # `--target` must not silently be taken for `--targets` (or a typo for another flag).
    with pytest.raises(SystemExit) as stopped:
        naavik(["validate", "missions/arch.yaml", "--hosts", "x.yaml"])
    assert stopped.value.code == 2 and "unrecognized arguments: --hosts" in capsys.readouterr().err


def test_group_was_only_on_this_branch_and_is_gone(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        naavik(["validate", "missions/arch.yaml", "--group", "probes"])
    assert stopped.value.code == 2
    assert "unrecognized arguments: --group" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--experiment", "--task"])
def test_one_experiment_is_named_with_experiment_or_its_earlier_name_task(receipts, flag, capsys):
    assert naavik(["cancel", str(receipts / "arch-1.json"), flag, "g/e/000", "--yes"]) == 2
    # The receipt never reached a host: the experiment is named in the refusal
    # path, not rejected by the parser.
    assert "never reached its host" in capsys.readouterr().err
