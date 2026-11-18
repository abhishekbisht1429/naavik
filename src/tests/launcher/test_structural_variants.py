"""An experiment may name its own training config (`config:`): one manifest
compares structures, each experiment checked against its own session, and
each task records which config file it was made from."""

from __future__ import annotations

import copy
import json

import pytest
import yaml

from tests.hpc.batch_repo import MANIFEST, TRAINING_CONFIG, compile_repo, make_repo
from training_framework.launcher.compile.render import plan_files
from training_framework.launcher.errors import PlanError

ATTNPOOL = copy.deepcopy(TRAINING_CONFIG)
ATTNPOOL["sessions"][0]["attention_pool"] = {"heads": 4}
del ATTNPOOL["sessions"][0]["fine_tuned_model"]


def two_structures(tmp_path, attnpool_experiment=None, other=ATTNPOOL):
    manifest = copy.deepcopy(MANIFEST)
    manifest["groups"]["probes"]["experiments"].append(attnpool_experiment or {
        "name": "attnpool", "config": "configs/attnpool.yaml",
        "matrix": [{"component": "attention_pool", "path": "heads", "values": [4, 8]}]})
    return make_repo(tmp_path, manifest=manifest,
                     extra_files={"configs/attnpool.yaml": yaml.safe_dump(other)})


def by_experiment(plan):
    found = {}
    for task in plan.tasks:
        found.setdefault(task.experiment, []).append(task)
    return found


def test_each_experiment_runs_its_own_config(tmp_path):
    plan = compile_repo(two_structures(tmp_path)).plan
    tasks = by_experiment(plan)
    assert len(tasks["frozen"]) == 4 and len(tasks["attnpool"]) == 2
    for task in tasks["frozen"]:
        session = task.config["sessions"][0]
        assert "fine_tuned_model" in session and "attention_pool" not in session
        assert task.config_file == "configs/train.yaml"
    assert [task.config["sessions"][0]["attention_pool"]["heads"] for task in tasks["attnpool"]] == [4, 8]
    assert all(task.config_file == "configs/attnpool.yaml" for task in tasks["attnpool"])


def test_a_task_takes_the_rest_of_its_own_config_file(tmp_path):
    other = dict(copy.deepcopy(ATTNPOOL), stage_note="attention pooling")
    tasks = by_experiment(compile_repo(two_structures(tmp_path, other=other)).plan)
    assert all(task.config["stage_note"] == "attention pooling" for task in tasks["attnpool"])
    assert all("stage_note" not in task.config for task in tasks["frozen"])


def test_the_plan_names_every_tasks_config_file(tmp_path):
    compiled = compile_repo(two_structures(tmp_path))
    plan = json.loads(plan_files(compiled.plan, compiled.manifest.raw)["plan.json"])
    assert {task["name"]: task["config_file"] for task in plan["tasks"]}["probes/attnpool/001"] == \
        "configs/attnpool.yaml"
    assert {task.get("config_file") for task in plan["tasks"]} == {"configs/train.yaml",
                                                                   "configs/attnpool.yaml"}


def test_a_batch_of_one_config_names_none(tmp_path):
    compiled = compile_repo(make_repo(tmp_path))
    plan = json.loads(plan_files(compiled.plan, compiled.manifest.raw)["plan.json"])
    assert all("config_file" not in task for task in plan["tasks"])


def test_overrides_are_checked_against_the_experiments_own_session(tmp_path):
    # `fine_tuned_model` is only in the base config.
    root = two_structures(tmp_path, {
        "name": "attnpool", "config": "configs/attnpool.yaml",
        "overrides": [{"component": "fine_tuned_model", "path": "frozen", "value": []}]})
    with pytest.raises(PlanError, match=r"recipes\[attnpool\].*fine_tuned_model"):
        compile_repo(root)


def test_a_missing_config_is_named_with_its_experiment(tmp_path):
    root = two_structures(tmp_path, {"name": "attnpool", "config": "configs/nowhere.yaml"})
    with pytest.raises(PlanError, match=r"jobs.probes.recipes\[attnpool\].config_file "
                                        r"'configs/nowhere.yaml' does not exist"):
        compile_repo(root)


def test_a_config_without_the_selected_session_is_named(tmp_path):
    root = two_structures(tmp_path, {"name": "attnpool", "config": "configs/attnpool.yaml"},
                          other={"sessions": []})
    with pytest.raises(PlanError, match=r"recipes\[attnpool\].config_file 'configs/attnpool.yaml' "
                                        r"has no 'sessions' list"):
        compile_repo(root)


def test_one_config_named_twice_is_loaded_once_and_matches(tmp_path):
    root = two_structures(tmp_path, {"name": "again", "config": "configs/train.yaml",
                                     "overrides": [{"path": "session_config.rng_seed", "value": 7}]})
    tasks = by_experiment(compile_repo(root).plan)
    assert tasks["again"][0].config_file == "configs/train.yaml"
    assert tasks["again"][0].config["sessions"][0]["session_config"]["rng_seed"] == 7


# End to end on this machine -------------------------------------------------------------------

from tests.hpc.test_end_to_end import environment, session  # noqa: E402,F401 (fixture)
from tests.launcher.test_host_end_to_end import here, launch, record, wait_for  # noqa: E402,F401


def test_each_task_records_its_config_file_and_summaries_compare_it(tmp_path, here, capsys):
    from tests.launcher.test_code_check import project
    from training_framework.reporting import discover_runs

    longer = dict(session(max_iterations=4))
    root = project(tmp_path / "p", session(), extra={
        "configs/longer.yaml": yaml.safe_dump({"sessions": [longer]})})
    manifest = yaml.safe_load((root / "batch.yaml").read_text())
    manifest["groups"]["cpu"]["experiments"].append({"name": "longer",
                                                     "config": "configs/longer.yaml"})
    (root / "batch.yaml").write_text(yaml.safe_dump(manifest))
    import subprocess

    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "two"],
                   cwd=root, check=True)
    batch_dir, _ = launch(root, here, capsys)
    for name, file in (("cpu/c0/000", "configs/c0.yaml"), ("cpu/longer/000", "configs/longer.yaml")):
        wait_for(lambda: record(batch_dir, name), f"{name}'s record")
        assert record(batch_dir, name)["exit_status"] == 0
        assert record(batch_dir, name)["config_file"] == file
    # (These runs write no scalars to summarize; what summaries compare is read.)
    runs = discover_runs(batch_dir / "tasks").runs
    assert sorted(run.config_file for run in runs) == ["configs/c0.yaml", "configs/longer.yaml"]
