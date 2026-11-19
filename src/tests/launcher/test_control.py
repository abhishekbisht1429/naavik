"""`cancel`, `logs`, `resume --restart`, and the device check, on batches
launched on this machine with the `host` scheduler and the real engine."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from tests.hpc.services.test_resume import FS, PreflightExecutor, plan_json
from tests.hpc.test_end_to_end import environment, session  # noqa: F401 (fixture)
from tests.launcher.test_host_end_to_end import here, launch, record, repo, wait_for  # noqa: F401
from naavik.cli import main as naavik_cli
from naavik.launcher.config.schema import Resources
from naavik.launcher.config.topology import check_topology
from naavik.launcher.errors import HpcError
from naavik.launcher.remote import host_runner
from naavik.launcher.services import resume as resuming

SRC = Path(__file__).resolve().parents[2]


def _status(name, capsys) -> dict[str, str]:
    """Task or step name -> its `status` line."""
    assert naavik_cli(["status", name]) == 0
    lines = capsys.readouterr().out.splitlines()[2:]          # batch line, header
    return {line.split()[0]: line for line in lines if line.strip()}


def _two_tasks(tmp_path, here, capsys, lengths=(1_000_000, 3)):
    """Task 000 runs until stopped; task 001 (short) waits for its turn."""
    config = session(tmp_path / "progress", max_iterations=lengths[0])
    root = repo(tmp_path, config, [1])
    manifest = yaml.safe_load((root / "batch.yaml").read_text())
    group = manifest["groups"]["cpu"]
    group["max_parallel"] = 1
    group["experiments"][0]["matrix"] = [
        {"path": "session_config.max_iterations", "values": list(lengths)}]
    (root / "batch.yaml").write_text(yaml.safe_dump(manifest))
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "two"],
                   cwd=root, check=True)
    batch_dir, receipt = launch(root, here, capsys)
    wait_for(lambda: (tmp_path / "progress" / "rank_0.progress").exists(), "task 000 to run")
    return batch_dir, receipt


# logs ----------------------------------------------------------------------------------


def test_logs_prints_a_tasks_log_and_names_what_exists(tmp_path, here, capsys):
    batch_dir, receipt = launch(repo(tmp_path, session(), [1]), here, capsys)
    wait_for(lambda: record(batch_dir, "cpu/seeds/000"), "the task's record")
    assert naavik_cli(["logs", batch_dir.name, "cpu/seeds/000"]) == 0
    output = capsys.readouterr().out
    assert "task cpu/seeds/000" in output and "finished: exit 0" in output
    assert naavik_cli(["logs", batch_dir.name, "cpu/seeds/000", "--attempt", "2"]) == 2
    assert "has no attempt 2 (attempts: 1)" in capsys.readouterr().err
    assert naavik_cli(["logs", batch_dir.name, "cpu/seeds/999"]) == 2
    assert "not an experiment or step of this mission (have: cpu/seeds/000)" in capsys.readouterr().err


def test_logs_follow_streams_until_interrupted(tmp_path, here, capsys):
    batch_dir, _ = _two_tasks(tmp_path, here, capsys)
    follower = subprocess.Popen(
        [sys.executable, "-m", "naavik.cli", "logs", batch_dir.name, "cpu/seeds/000",
         "--follow"], env={**os.environ, "PYTHONPATH": str(SRC)},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    first = follower.stdout.readline()
    assert "[hpc" in first
    follower.send_signal(signal.SIGINT)
    assert follower.wait(timeout=30) == 0
    assert naavik_cli(["cancel", batch_dir.name, "--yes"]) == 0


# cancel ------------------------------------------------------------------------------


def test_cancelling_a_batch_stops_its_running_task_cooperatively(tmp_path, here, capsys):
    batch_dir, receipt = _two_tasks(tmp_path, here, capsys)
    assert naavik_cli(["cancel", batch_dir.name, "--yes"]) == 0
    assert "cancelled group-cpu" in capsys.readouterr().out
    wait_for(lambda: record(batch_dir, "cpu/seeds/000"), "the stopped task's record")
    assert record(batch_dir, "cpu/seeds/000")["interrupted"] is True
    saved = json.loads((batch_dir / "receipt.json").read_text())
    assert saved["nodes"]["group-cpu"]["status"] == "cancelled"
    ref = receipt["nodes"]["group-cpu"]["job_ref"]
    wait_for(lambda: not host_runner.ref_alive(ref), "the runner to stop")
    states = _status(batch_dir.name, capsys)
    assert " interrupted " in states["cpu/seeds/000"] + " "
    assert " cancelled" in states["cpu/seeds/001"]


def test_cancelling_one_task_leaves_the_others_and_shows_it_cancelled(tmp_path, here, capsys):
    batch_dir, receipt = _two_tasks(tmp_path, here, capsys)
    assert naavik_cli(["cancel", batch_dir.name, "--task", "cpu/seeds/001", "--yes"]) == 0
    capsys.readouterr()
    ref = receipt["nodes"]["group-cpu"]["job_ref"]
    # The runner applies a task's cancel at its next poll.
    job = lambda: host_runner.read_job(host_runner.state_dir(), ref)  # noqa: E731
    wait_for(lambda: job()["elements"]["1"] == "cancelled", "the runner to drop task 001")
    states = _status(batch_dir.name, capsys)
    assert " running " in states["cpu/seeds/000"] and " cancelled" in states["cpu/seeds/001"]
    assert naavik_cli(["cancel", batch_dir.name, "--yes"]) == 0    # stop task 000 too
    wait_for(lambda: host_runner.read_job(host_runner.state_dir(), ref)["final"], "the job to end")
    assert record(batch_dir, "cpu/seeds/001") is None             # it never ran


def test_cancel_asks_unless_told_not_to_and_skips_what_ended(tmp_path, here, capsys, monkeypatch):
    batch_dir, _ = launch(repo(tmp_path, session(), [1]), here, capsys)
    wait_for(lambda: record(batch_dir, "cpu/seeds/000"), "the task's record")
    ref = json.loads((batch_dir / "receipt.json").read_text())["nodes"]["group-cpu"]["job_ref"]
    wait_for(lambda: not host_runner.ref_alive(ref), "the runner to exit")
    assert naavik_cli(["cancel", batch_dir.name]) == 0
    output = capsys.readouterr().out
    assert "already ended" in output and "nothing to cancel" in output
    assert naavik_cli(["cancel", batch_dir.name, "--task", "cpu/seeds/000"]) == 0
    output = capsys.readouterr().out
    assert f"cpu/seeds/000: its job {ref}_0 already ended" in output and "nothing to cancel" in output


def test_a_non_interactive_cancel_needs_yes(tmp_path, here, capsys, monkeypatch):
    batch_dir, _ = _two_tasks(tmp_path, here, capsys)
    monkeypatch.setattr("sys.stdin", open("/dev/null"))
    assert naavik_cli(["cancel", batch_dir.name]) == 1
    assert "pass --yes" in capsys.readouterr().err
    assert naavik_cli(["cancel", batch_dir.name, "--yes"]) == 0


# resume --restart -------------------------------------------------------------------------


def test_a_task_that_never_ran_restarts_from_its_config(tmp_path, here, capsys):
    batch_dir, receipt = _two_tasks(tmp_path, here, capsys)
    assert naavik_cli(["cancel", batch_dir.name, "--task", "cpu/seeds/001", "--yes"]) == 0
    assert naavik_cli(["cancel", batch_dir.name, "--yes"]) == 0
    ref = receipt["nodes"]["group-cpu"]["job_ref"]
    wait_for(lambda: not host_runner.ref_alive(ref), "the first attempt to end")
    capsys.readouterr()

    assert naavik_cli(["resume", batch_dir.name, "--task", "cpu/seeds/001", "--yes"]) == 2
    captured = capsys.readouterr()
    assert "skip cpu/seeds/001" in captured.out and "--restart runs experiments" in captured.err

    assert naavik_cli(["resume", batch_dir.name, "--task", "cpu/seeds/001", "--restart", "--yes"]) == 0
    assert "restart from its config (no checkpoint)" in capsys.readouterr().out
    wait_for(lambda: record(batch_dir, "cpu/seeds/001"), "the restarted task's record")
    restarted = record(batch_dir, "cpu/seeds/001")
    assert restarted["exit_status"] == 0 and restarted["attempt"] == 2
    assert restarted["restarted"] is True and "resumed_from" not in restarted


def test_an_environment_that_cannot_restart_is_refused(tmp_path):
    batch_dir = tmp_path / "b"
    (batch_dir / "source").mkdir(parents=True)
    plan = dict(plan_json(str(batch_dir)), environment={"managed": False},
                preflight={"protocol_version": 1,
                           "framework": {"location": "environment", "distribution": "0.5.0"}})
    resuming.check_cluster(plan, FS, PreflightExecutor(["resume"]))               # no restarts
    with pytest.raises(HpcError, match="predates restarting"):
        resuming.check_cluster(plan, FS, PreflightExecutor(["resume"]), restart=True)
    resuming.check_cluster(plan, FS, PreflightExecutor(["resume", "restart"]), restart=True)


# The device ------------------------------------------------------------------------------


def _session(device):
    return {"session_config": {"device": device}}


@pytest.mark.parametrize("device, gpus, problem", [
    ("cpu", 0, None),
    ("cuda", 1, None),
    ("cuda:1", 2, None),
    ("cuda", 0, "requests no GPUs"),
    ("cuda:0", 0, "requests no GPUs"),
    ("cuda:4", 2, "sees only its group's 2 GPU(s)"),
    ("cuda:1", 1, "use 'cuda' (or cuda:0..cuda:0)"),
])
def test_the_device_must_be_one_the_task_will_see(device, gpus, problem):
    problems = [problem for problem in check_topology(
        _session(device), Resources("01:00:00", 3600, gpus, 4, "8G", {})) if "device" in problem]
    if problem is None:
        assert problems == []
    else:
        assert len(problems) == 1 and problem in problems[0]


def test_a_task_that_asks_for_no_gpu_sees_none(tmp_path, here, capsys):
    batch_dir, _ = launch(repo(tmp_path, session(), [1]), here, capsys)
    wait_for(lambda: record(batch_dir, "cpu/seeds/000"), "the task's record")
    assert naavik_cli(["logs", batch_dir.name, "cpu/seeds/000"]) == 0
    # The task's log line ends with its CUDA_VISIBLE_DEVICES: empty, not unset.
    (line,) = [line for line in capsys.readouterr().out.splitlines() if "| GPUs" in line]
    assert line.endswith("| GPUs ")


@pytest.mark.parametrize("attempt", [1, 2])
def test_a_cancelled_task_shows_cancelled_while_its_job_runs_on(attempt):
    from naavik.launcher.services.status import BatchFiles, task_rows

    nodes = {"group-g": {"status": "submitted", "job_ref": "5"}}
    if attempt == 2:
        nodes["group-g.resume-2"] = {"status": "submitted", "job_ref": "6", "attempt": 2,
                                     "group": "g", "tasks": [0, 1]}
    receipt = {"nodes": nodes, "cancelled_tasks": {
        "g/e/000": {"attempt": attempt, "ok": True, "detail": "cancelled"}}}
    table = ("index\ttask\tvariant\tconfig\tconfig_hash\n"
             "0\tg/e/000\t[]\ttasks/g/e/000/config.yaml\th\n"
             "1\tg/e/001\t[]\ttasks/g/e/001/config.yaml\th\n")
    job = "5" if attempt == 1 else "6"
    live = {job: "running", f"{job}_1": "running"}           # element 0 left the queue
    rows = {row.name: row.state for row in task_rows(receipt, BatchFiles({}, {"g": table}), live)}
    assert rows["g/e/000"] == "cancelled" and rows["g/e/001"] == "running"
    # A cancel recorded for another attempt says nothing about this one.
    receipt["cancelled_tasks"]["g/e/000"]["attempt"] = 3
    rows = {row.name: row.state for row in task_rows(receipt, BatchFiles({}, {"g": table}), live)}
    assert rows["g/e/000"] == "running"
