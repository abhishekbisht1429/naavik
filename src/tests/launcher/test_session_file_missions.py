"""Missions with session files: a config file is one session.

The launcher reads both forms, notes the earlier one and a written
`base.session_index`, refuses any other index, and still writes each
experiment's config as `sessions: [session]` (every framework version a
mission may run reads that), without the engine warning about it."""

from __future__ import annotations

import copy

import pytest
import yaml

from tests.hpc.batch_repo import MANIFEST, TRAINING_CONFIG, compile_repo, make_repo
from tests.hpc.test_end_to_end import COMPONENTS, environment, session  # noqa: F401 (fixture)
from tests.launcher.test_host_end_to_end import here, launch, record, wait_for  # noqa: F401
from naavik.launcher.errors import PlanError

SESSION = TRAINING_CONFIG["sessions"][0]


def _notes(compiled) -> list[str]:
    return list(compiled.plan.notes)


def test_a_session_file_compiles_without_a_note_about_its_form(tmp_path):
    compiled = compile_repo(make_repo(tmp_path, config=copy.deepcopy(SESSION)))
    assert not [note for note in _notes(compiled) if "one session" in note]
    for task in compiled.plan.tasks:
        assert list(task.config) == ["sessions"] and len(task.config["sessions"]) == 1


def test_a_session_file_and_its_earlier_form_give_the_same_experiments(tmp_path):
    new = compile_repo(make_repo(tmp_path / "new", config=copy.deepcopy(SESSION))).plan
    old = compile_repo(make_repo(tmp_path / "old", config=copy.deepcopy(TRAINING_CONFIG))).plan
    strip = lambda config: {  # noqa: E731 (sessions_dir names the mission)
        **config["sessions"][0], "session_config": {
            key: value for key, value in config["sessions"][0]["session_config"].items()
            if key != "sessions_dir"}}
    assert [strip(task.config) for task in new.tasks] == [strip(task.config) for task in old.tasks]


def test_an_earlier_form_file_is_noted(tmp_path):
    compiled = compile_repo(make_repo(tmp_path))
    assert any("configs/train.yaml: a config file is now one session" in note
               for note in _notes(compiled))


def test_a_written_session_index_of_0_is_noted_as_no_longer_needed(tmp_path):
    manifest = copy.deepcopy(MANIFEST)
    manifest["training"]["session"] = 0
    compiled = compile_repo(make_repo(tmp_path, manifest=manifest, config=copy.deepcopy(SESSION)))
    assert "base.session_index is no longer needed: a config file is one session; " \
           "remove it" in _notes(compiled)


def test_another_session_index_is_refused(tmp_path):
    manifest = copy.deepcopy(MANIFEST)
    manifest["training"]["session"] = 1
    with pytest.raises(PlanError, match=r"base.session_index: a config file is now one session"):
        compile_repo(make_repo(tmp_path, manifest=manifest, config=copy.deepcopy(SESSION)))


def test_a_config_file_with_several_sessions_is_refused(tmp_path):
    config = {"sessions": [copy.deepcopy(SESSION), copy.deepcopy(SESSION)]}
    with pytest.raises(PlanError, match="2 sessions in one file"):
        compile_repo(make_repo(tmp_path, config=config))


def test_an_experiment_runs_a_session_file_and_its_log_has_no_form_warning(tmp_path, here,
                                                                          capsys):
    manifest = {
        "version": 1, "name": "local",
        "code": {"framework": "dependency", "paths": ["."]},
        "training": {"config": "configs/train.yaml",
                     "engine_options": ["--process_timeout_on_join", "30"]},
        "groups": {"cpu": {
            "resources": {"walltime": "00:30:00", "cpus": 2, "memory": "4G"},
            "experiments": [{"name": "one"}],
        }},
    }
    root = make_repo(tmp_path, manifest=manifest, config=session(), requirements=None,
                     extra_files={"my_project/components.py": COMPONENTS.read_text()})
    batch_dir, receipt = launch(root, here, capsys)
    wait_for(lambda: record(batch_dir, "cpu/one/000"), "the experiment's record")
    assert record(batch_dir, "cpu/one/000")["exit_status"] == 0
    assert yaml.safe_load((batch_dir / "tasks" / "cpu" / "one" / "000" / "config.yaml")
                          .read_text())["sessions"]
    ref = receipt["nodes"]["group-cpu"]["job_ref"]
    log = (batch_dir / "logs" / "cpu" / f"{ref}_0.out").read_text()
    assert "SessionsListWarning" not in log and "a config file is now one session" not in log
