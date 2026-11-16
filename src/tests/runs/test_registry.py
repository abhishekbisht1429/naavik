"""Run records: written once by their one writer, read back with their state
derived, merged across machines by copying."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from training_framework.launcher.remote.runtime import process_start_time
from training_framework.runs.ids import new_run_id
from training_framework.runs.registry import (
    CONFIGS,
    ENDED_WITHOUT_RECORD,
    RUNNING,
    UNKNOWN,
    RunWriter,
    config_sha256,
    find_run,
    list_runs,
    parent_of,
    read_run,
    registry_dir,
    write_rank,
)


def record(run_id=None, **changes):
    run_id = run_id or new_run_id()
    data = {"run_id": run_id, "host": socket.gethostname(), "pid": os.getpid(),
            "process_start_time": process_start_time(os.getpid()),
            "launch": {"operation": "config"}, "sessions": []}
    data.update(changes)
    return data


# Where -----------------------------------------------------------------------------------


def test_the_registry_lives_in_the_data_home_unless_named_or_off(tmp_path, monkeypatch):
    monkeypatch.delenv("TRAINING_FRAMEWORK_RUNS", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    assert registry_dir() == tmp_path / "data" / "training-framework" / "runs"
    monkeypatch.setenv("TRAINING_FRAMEWORK_RUNS", str(tmp_path / "elsewhere"))
    assert registry_dir() == tmp_path / "elsewhere"
    monkeypatch.setenv("TRAINING_FRAMEWORK_RUNS", "off")
    assert registry_dir() is None


# Writing ------------------------------------------------------------------------------------


def test_a_run_is_written_once_and_read_back(tmp_path):
    writer = RunWriter.create(tmp_path, record(note="first"))
    writer.event("session", index=0, session_dir="/runs/session_1")
    writer.event("finished", status="ok", last_iteration=[10])
    write_rank(writer.directory, 0, {"device": "cuda:0", "uuid": "GPU-a"})
    run = read_run(writer.directory)
    assert run.record["note"] == "first"
    assert [event["event"] for event in run.events] == ["session", "finished"]
    assert run.session_dirs == ["/runs/session_1"]
    assert run.ranks[0]["uuid"] == "GPU-a" and run.state == "done"


def test_records_are_never_rewritten(tmp_path):
    writer = RunWriter.create(tmp_path, record())
    with pytest.raises(FileExistsError):
        RunWriter.create(tmp_path, record(writer.run_id))
    write_rank(writer.directory, 0, {"device": "cpu"})
    with pytest.raises(FileExistsError):
        write_rank(writer.directory, 0, {"device": "cuda"})
    assert read_run(writer.directory).ranks[0]["device"] == "cpu"


def test_a_config_is_stored_once_by_content(tmp_path):
    writer = RunWriter.create(tmp_path, record())
    config = {"session_config": {"rng_seed": 1, "sessions_dir": Path("/runs")}, "model": {}}
    first = writer.store_config(config)
    second = RunWriter.create(tmp_path, record()).store_config(dict(config))
    assert first == second == config_sha256(config)
    assert [path.name for path in (tmp_path / CONFIGS).iterdir()] == [f"{first}.yaml"]
    run = read_run(RunWriter.create(tmp_path, record(sessions=[{"config_sha256": first}])).directory)
    assert run.config(tmp_path)["session_config"] == {"rng_seed": 1, "sessions_dir": "/runs"}


def test_an_id_that_is_not_a_run_id_is_refused(tmp_path):
    with pytest.raises(ValueError):
        RunWriter.create(tmp_path, record("../escape"))


# State ----------------------------------------------------------------------------------


@pytest.mark.parametrize("status, state", [("ok", "done"), ("failed", "failed"),
                                           ("interrupted", "interrupted"), ("odd", "unknown")])
def test_a_finished_run_has_its_recorded_outcome(tmp_path, status, state):
    writer = RunWriter.create(tmp_path, record())
    writer.event("finished", status=status)
    assert read_run(writer.directory).state == state


def test_an_unfinished_run_is_running_while_its_process_lives(tmp_path):
    assert read_run(RunWriter.create(tmp_path, record()).directory).state == RUNNING


def test_an_unfinished_run_whose_process_is_gone_ended_without_a_record(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    start = process_start_time(child.pid)
    child.wait()
    writer = RunWriter.create(tmp_path, record(pid=child.pid, process_start_time=start))
    assert read_run(writer.directory).state == ENDED_WITHOUT_RECORD


def test_another_machines_unfinished_run_is_unknown(tmp_path):
    writer = RunWriter.create(tmp_path, record(host="elsewhere"))
    assert read_run(writer.directory).state == UNKNOWN


def test_damaged_lines_and_files_are_reported_not_hidden(tmp_path):
    writer = RunWriter.create(tmp_path, record())
    writer.event("session", index=0, session_dir="/s")
    with open(writer.directory / "events.jsonl", "a") as handle:
        handle.write('{"event": "fini')                      # cut off mid-write
    (writer.directory / "rank-1.json").write_text("{")
    run = read_run(writer.directory)
    assert run.session_dirs == ["/s"]
    assert run.problems == ["events.jsonl line 2 is damaged", "rank-1.json is damaged"]


# Listing and merging -------------------------------------------------------------------------


def test_runs_are_listed_newest_first_and_unreadable_ones_named(tmp_path):
    old = RunWriter.create(tmp_path, record(new_run_id(now_ms=1_000)))
    new = RunWriter.create(tmp_path, record(new_run_id(now_ms=2_000)))
    broken = tmp_path / new_run_id(now_ms=1_500)
    broken.mkdir()
    (broken / "run.json").write_text("{")
    (tmp_path / "not-a-run").mkdir()
    runs, problems = list_runs(tmp_path)
    assert [run.run_id for run in runs] == [new.run_id, old.run_id]
    assert len(problems) == 1 and problems[0].startswith(broken.name)
    assert list_runs(tmp_path / "missing") == ([], [])


def test_registries_of_two_machines_merge_by_copying(tmp_path):
    here, there = tmp_path / "here", tmp_path / "there"
    RunWriter.create(here, record()).store_config({"a": 1})
    RunWriter.create(there, record(host="gpubox")).store_config({"a": 1})
    shutil.copytree(there, here, dirs_exist_ok=True)       # what `runs sync` does
    runs, problems = list_runs(here)
    assert len(runs) == 2 and not problems
    assert len(list((here / CONFIGS).iterdir())) == 1


def test_a_run_is_found_by_a_unique_prefix(tmp_path):
    first = RunWriter.create(tmp_path, record(new_run_id(now_ms=10, random_bits=1)))
    RunWriter.create(tmp_path, record(new_run_id(now_ms=10, random_bits=2 ** 79)))
    assert find_run(tmp_path, first.run_id[:20].lower()).run_id == first.run_id
    with pytest.raises(LookupError, match="2 runs"):
        find_run(tmp_path, first.run_id[:10])
    with pytest.raises(LookupError, match="no run"):
        find_run(tmp_path, "ZZZZ")


def test_a_resume_continues_the_latest_earlier_run_of_its_session(tmp_path):
    def run(now_ms, operation, session_dir):
        writer = RunWriter.create(tmp_path, record(new_run_id(now_ms=now_ms),
                                                   launch={"operation": operation}))
        writer.event("session", index=0, session_dir=session_dir)
        return writer.run_id

    first = run(1, "config", "/s/a")
    run(2, "config", "/s/b")
    second = run(3, "resume", "/s/a")
    third = run(4, "extend", "/s/a")
    runs, _ = list_runs(tmp_path)
    by_id = {r.run_id: r for r in runs}
    assert parent_of(by_id[second], runs).run_id == first
    assert parent_of(by_id[third], runs).run_id == second
    assert parent_of(by_id[first], runs) is None
