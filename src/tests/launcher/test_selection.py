"""`--group G` / `--experiment G/E`: only part of a manifest becomes tasks,
named and numbered as the whole manifest names them, so parts launched on
different targets read like one batch."""

from __future__ import annotations

import copy
import json

import pytest
import yaml

from tests.hpc.batch_repo import CLOCK, MANIFEST, PROFILE, TRAINING_CONFIG, FakeFacts, make_repo
from training_framework.launcher.compile.compiler import Selection, compile_plan
from training_framework.launcher.compile.render import plan_files
from training_framework.launcher.config.schema import parse_profiles
from training_framework.launcher.errors import PlanError
from training_framework.launcher.ports.repository import GitRepository


def two_groups(tmp_path):
    manifest = copy.deepcopy(MANIFEST)
    probes = manifest["groups"]["probes"]
    probes["experiments"].append({"name": "unfrozen", "matrix": [
        {"path": "session_config.rng_seed", "values": [1, 2, 3]}]})
    manifest["groups"]["long"] = {
        "resources": {"walltime": "08:00:00", "gpus": 1, "cpus": 8, "memory": "64G"},
        "experiments": [{"name": "full", "config": "configs/other.yaml"}]}
    return make_repo(tmp_path, manifest=manifest,
                     extra_files={"configs/other.yaml": yaml.safe_dump(TRAINING_CONFIG)})


def compile_selected(root, groups=(), experiments=()):
    return compile_plan(manifest_path="batch.yaml", repository=GitRepository(root),
                        profile=parse_profiles(PROFILE, None, "profiles.yaml"),
                        facts=FakeFacts(root), clock=CLOCK,
                        selection=Selection(tuple(groups), tuple(experiments)))


def names(compiled):
    return [task.name for task in compiled.plan.tasks]


def test_a_group_alone(tmp_path):
    compiled = compile_selected(two_groups(tmp_path), groups=["long"])
    assert names(compiled) == ["long/full/000"]
    assert [node.id for node in compiled.plan.nodes if node.kind == "task_set"] == ["group-long"]


def test_an_experiment_keeps_its_name_and_numbers(tmp_path):
    compiled = compile_selected(two_groups(tmp_path), experiments=["probes/unfrozen"])
    assert names(compiled) == ["probes/unfrozen/000", "probes/unfrozen/001", "probes/unfrozen/002"]
    assert [task.array_index for task in compiled.plan.tasks] == [0, 1, 2]


def test_a_group_and_one_of_its_experiments_is_the_group(tmp_path):
    compiled = compile_selected(two_groups(tmp_path), groups=["probes"],
                                experiments=["probes/unfrozen"])
    assert len(names(compiled)) == 7


def test_the_selection_is_in_the_plan_and_the_whole_manifest_is_staged(tmp_path):
    compiled = compile_selected(two_groups(tmp_path), groups=["long"],
                                experiments=["probes/frozen"])
    files = plan_files(compiled.plan, compiled.manifest.raw)
    assert json.loads(files["plan.json"])["selection"] == {"groups": ["long"],
                                                           "experiments": ["probes/frozen"]}
    assert set(yaml.safe_load(files["experiments.yaml"])["groups"]) == {"probes", "long"}


def test_without_a_selection_the_plan_says_nothing(tmp_path):
    compiled = compile_selected(two_groups(tmp_path))
    assert "selection" not in json.loads(plan_files(compiled.plan, compiled.manifest.raw)["plan.json"])
    assert len(names(compiled)) == 8


def test_config_files_are_recorded_as_for_the_whole_manifest(tmp_path):
    # Only `long` sets `config:`; a part without it still records its file,
    # so both parts compare in one summary.
    compiled = compile_selected(two_groups(tmp_path), experiments=["probes/frozen"])
    assert {task.config_file for task in compiled.plan.tasks} == {"configs/train.yaml"}


@pytest.mark.parametrize("groups, experiments, message", [
    (["nope"], [], r"--group nope: the experiment file has no such group \(it has: long, probes\)"),
    ([], ["probes"], "--experiment probes: name it GROUP/EXPERIMENT"),
    ([], ["nope/full"], "--experiment nope/full: the experiment file has no group 'nope'"),
    ([], ["probes/nope"], r"group 'probes' has no experiment 'nope' \(it has: frozen, unfrozen\)"),
])
def test_an_unknown_name_is_refused(tmp_path, groups, experiments, message):
    with pytest.raises(PlanError, match=message):
        compile_selected(two_groups(tmp_path), groups, experiments)


# End to end on this machine -------------------------------------------------------------------

from tests.hpc.test_end_to_end import environment, session  # noqa: E402,F401 (fixture)
from tests.launcher.test_host_end_to_end import here, record, wait_for  # noqa: E402,F401
from training_framework.cli import main as naavik  # noqa: E402


def test_a_launched_part_is_followed_like_any_batch(tmp_path, here, capsys):
    from pathlib import Path

    from tests.launcher.test_code_check import project

    root = project(tmp_path / "p", session())
    manifest = yaml.safe_load((root / "batch.yaml").read_text())
    manifest["groups"]["cpu"]["experiments"].append({"name": "other", "overrides": [
        {"path": "session_config.rng_seed", "value": 5}]})
    (root / "batch.yaml").write_text(yaml.safe_dump(manifest))
    import subprocess

    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "two"],
                   cwd=root, check=True)
    assert naavik(["launch", str(root / "batch.yaml"), "--targets", str(here), "--yes",
                   "--experiment", "cpu/other"]) == 0
    output = capsys.readouterr().out
    batch_dir = Path(output.split("submitted: ")[1].splitlines()[0])
    receipt = json.loads((batch_dir / "receipt.json").read_text())
    assert receipt["selection"] == {"experiments": ["cpu/other"]}
    wait_for(lambda: record(batch_dir, "cpu/other/000"), "the task's record")
    assert not (batch_dir / "tasks" / "cpu" / "c0").exists()
    assert naavik(["status", receipt["batch_id"]]) == 0
    status = capsys.readouterr().out
    assert "cpu/other/000" in status and "cpu/c0/000" not in status
