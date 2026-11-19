"""Summaries compare the code each run ran: from an HPC task's record, or
from the run registry. Runs on different code are never one seed group;
reports of runs whose code is unknown are as before."""

from __future__ import annotations

import json

import pytest

from naavik.reporting import DEFAULT_IGNORED_KEYS, discover_runs, summarize_runs
from naavik.runs.ids import new_run_id
from naavik.runs.registry import RunWriter

from .helpers import config, losses, write_run

A, B = "a" * 40, "b" * 40


@pytest.fixture(autouse=True)
def registry(tmp_path, monkeypatch):
    root = tmp_path / "registry"
    monkeypatch.setenv("TRAINING_FRAMEWORK_RUNS", str(root))
    return root


def task(root, index, *, seed, sha=None, record=None, **extra):
    """An HPC task directory whose record names its batch's commit `sha`."""
    task_dir = root / "tasks" / "g" / "e" / f"{index:03d}"
    write_run(task_dir / "sessions", losses(2.0, 1.0 + seed / 100),
              config(seed=seed, sessions_dir=str(task_dir / "sessions"), **extra))
    (task_dir / "variant.json").write_text("[]")
    if record is not None:
        (task_dir / "record.json").write_text(record)
    elif sha is not None:
        (task_dir / "record.json").write_text(json.dumps({"sha": sha, "exit_status": 0}))
    return task_dir


def summary(*roots, **options):
    return summarize_runs(discover_runs(*roots), ["loss"], **options)


def code_of(result):
    return {run.run_id: run.hyperparameters.get("code") for run in result.runs}


def test_runs_on_different_code_are_different_groups(tmp_path):
    for index, (seed, sha) in enumerate([(1, A), (2, A), (1, B), (2, B)]):
        task(tmp_path, index, seed=seed, sha=sha)
    result = summary(tmp_path)
    assert "code" in [column.name for column in result.columns]
    assert code_of(result) == {"g/e/000": A[:12], "g/e/001": A[:12],
                               "g/e/002": B[:12], "g/e/003": B[:12]}
    assert sorted(sorted(group.members) for group in result.groups) == [
        ["g/e/000", "g/e/001"], ["g/e/002", "g/e/003"]]


def test_runs_on_the_same_code_share_a_group_and_show_it_once(tmp_path):
    task(tmp_path, 0, seed=1, sha=A)
    task(tmp_path, 1, seed=2, sha=A)
    result = summary(tmp_path)
    assert len(result.groups) == 1 and result.shared["code"] == A[:12]
    assert "code" not in [column.name for column in result.columns]


def test_without_code_records_the_report_is_unchanged(tmp_path):
    task(tmp_path, 0, seed=1)
    task(tmp_path, 1, seed=2)
    result = summary(tmp_path)
    assert "code" not in result.shared and "code" not in [c.name for c in result.columns]
    assert all(run.run.code is None for run in result.runs)


def test_unknown_code_beside_known_code_is_its_own_value(tmp_path):
    task(tmp_path, 0, seed=1, sha=A)
    task(tmp_path, 1, seed=2)                    # an older task: no record
    result = summary(tmp_path)
    assert code_of(result) == {"g/e/000": A[:12], "g/e/001": "unknown"}
    assert len(result.groups) == 2


def test_a_snapshot_batch_is_marked(tmp_path):
    task(tmp_path, 0, seed=1, record=json.dumps({"sha": A, "base_sha": B, "snapshot": True}))
    task(tmp_path, 1, seed=2, sha=A)
    assert code_of(summary(tmp_path)) == {"g/e/000": f"{A[:12]} (snapshot)", "g/e/001": A[:12]}


def test_a_direct_run_takes_its_code_from_the_registry(tmp_path, registry):
    sessions = tmp_path / "runs"
    first = write_run(sessions, losses(1.0), config(seed=1), session="session_20261031_120000")
    second = write_run(sessions, losses(1.0), config(seed=2), session="session_20261031_130000")
    for session_dir, code in ((first, {"sha": A, "dirty": True}), (second, {"sha": B})):
        writer = RunWriter.create(registry, {"run_id": new_run_id(), "code": code})
        writer.event("session", index=0, session_dir=str(session_dir))
    result = summary(sessions)
    assert sorted(code_of(result).values()) == [f"{A[:12]} (dirty)", B[:12]]
    assert len(result.groups) == 2


def test_the_registry_off_is_code_unknown(tmp_path, monkeypatch, registry):
    session_dir = write_run(tmp_path / "runs", losses(1.0), config())
    writer = RunWriter.create(registry, {"run_id": new_run_id(), "code": {"sha": A}})
    writer.event("session", index=0, session_dir=str(session_dir))
    monkeypatch.setenv("TRAINING_FRAMEWORK_RUNS", "off")
    assert summary(tmp_path / "runs").runs[0].run.code is None


def test_ignoring_the_code_key_leaves_it_out(tmp_path):
    task(tmp_path, 0, seed=1, sha=A)
    task(tmp_path, 1, seed=2, sha=B)
    result = summary(tmp_path, ignored_keys=[*DEFAULT_IGNORED_KEYS, "code"])
    assert len(result.groups) == 1 and "code" not in result.shared


def test_a_config_with_its_own_code_key_is_a_problem_not_a_merge(tmp_path):
    task(tmp_path, 0, seed=1, sha=A, code={"style": "x"})
    task(tmp_path, 1, seed=2, sha=B, code={"style": "x"})
    result = summary(tmp_path)
    assert any("the runs' code is not compared: their configs have code.style" in p
               for p in result.problems)
    assert len(result.groups) == 1 and result.shared["code.style"] == "x"


def test_a_damaged_task_record_is_a_problem(tmp_path):
    task(tmp_path, 0, seed=1, record="{not json")
    task(tmp_path, 1, seed=2, sha=A)
    result = summary(tmp_path)
    assert any("record.json cannot be read" in p for p in result.problems)
    assert code_of(result)["g/e/000"] == "unknown"


def test_the_report_says_what_code_is_only_when_it_shows_it(tmp_path):
    from naavik.reporting.render import render_markdown

    task(tmp_path / "known", 0, seed=1, sha=A)
    task(tmp_path / "older", 0, seed=1)
    assert "- `code` is the commit each run ran" in render_markdown(summary(tmp_path / "known"))
    assert "`code`" not in render_markdown(summary(tmp_path / "older"))
