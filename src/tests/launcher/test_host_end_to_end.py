"""A batch launched on this machine with the `host` scheduler, end to end:
`naavik launch` stages it and starts runners that run the real engine
(one-rank gloo); `status` follows it; a cancelled task records that it was
interrupted, and `resume` continues it from its checkpoint as attempt 2."""

from __future__ import annotations

import copy
import json
import os
import signal
import time
from pathlib import Path

import pytest
import torch
import yaml

from tests.hpc.batch_repo import PROFILE, make_repo
from tests.hpc.fake_cluster import fake_cluster  # noqa: F401 (fixture)
from tests.hpc.test_end_to_end import COMPONENTS, environment, session  # noqa: F401 (fixture)
from training_framework.cli import main as naavik
from training_framework.launcher.ports.local import LocalExecutor
from training_framework.launcher.remote import host_runner
from training_framework.launcher.scheduler.host_submitter import HostSubmitter

pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="the tasks run a one-rank gloo DDP session",
)
TIMEOUT = 240


@pytest.fixture
def here(tmp_path, monkeypatch, environment):
    """A repository, a local `host` target using the test environment, and a
    clean-up that stops any runner the test left."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    targets = copy.deepcopy(PROFILE)
    entry = targets["profiles"]["cluster"]
    del entry["ssh_host"], entry["slurm"]
    entry.update(local=True, scheduler="host", host={}, hooks={},
                 roots={key: str(tmp_path / "here" / key) for key in ("batches", "envs", "data")},
                 env={"existing": str(environment)})
    path = tmp_path / "targets.yaml"
    path.write_text(yaml.safe_dump(targets))
    yield path
    jobs = Path(host_runner.state_dir(), "jobs")
    for item in jobs.glob("h*.json") if jobs.exists() else ():
        ref = item.stem
        if host_runner.ref_alive(ref):
            os.killpg(host_runner.parse_ref(ref)[0], signal.SIGKILL)


def repo(tmp_path, config_session, seeds):
    manifest = {
        "version": 1, "name": "local",
        "code": {"framework": "dependency", "paths": ["."]},
        "training": {"config": "configs/train.yaml",
                     "engine_options": ["--process_timeout_on_join", "30"]},
        "groups": {"cpu": {
            "resources": {"walltime": "00:30:00", "cpus": 2, "memory": "4G"},
            "max_parallel": 2,
            "experiments": [{"name": "seeds", "matrix": [
                {"path": "session_config.rng_seed", "values": seeds}]}],
        }},
    }
    return make_repo(tmp_path, manifest=manifest, config={"sessions": [config_session]},
                     requirements=None,
                     extra_files={"my_project/components.py": COMPONENTS.read_text()})


def launch(root, targets, capsys) -> tuple[Path, dict]:
    assert naavik(["launch", str(root / "batch.yaml"), "--targets", str(targets), "--yes"]) == 0
    output = capsys.readouterr().out
    batch_dir = Path(output.split("submitted: ")[1].splitlines()[0])
    return batch_dir, json.loads((batch_dir / "receipt.json").read_text())


def wait_for(predicate, what):
    deadline = time.monotonic() + TIMEOUT
    while not predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.2)


def record(batch_dir, task):
    path = batch_dir / "tasks" / task / "record.json"
    return json.loads(path.read_text()) if path.exists() else None


def test_a_local_host_batch_runs_its_tasks_to_the_end(tmp_path, here, capsys):
    batch_dir, receipt = launch(repo(tmp_path, session(), [1, 2]), here, capsys)
    assert receipt["scheduler"] == "host" and receipt["target"]["transport"] == "local"
    tasks = ["cpu/seeds/000", "cpu/seeds/001"]
    wait_for(lambda: all(record(batch_dir, task) for task in tasks), "both task records")
    for task in tasks:
        assert record(batch_dir, task)["exit_status"] == 0, record(batch_dir, task)
    ref = receipt["nodes"]["group-cpu"]["job_ref"]
    job = lambda: host_runner.read_job(host_runner.state_dir(), ref)  # noqa: E731
    wait_for(lambda: job()["final"], "the runner to finish")
    assert job()["elements"] == {
        "0": "completed", "1": "completed"}
    assert (batch_dir / "logs" / "cpu" / f"{ref}_0.out").exists()
    wait_for(lambda: not host_runner.ref_alive(ref), "the runner to exit")
    assert naavik(["status", batch_dir.name]) == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert [row[2] for row in rows if row and row[0] in tasks] == ["done", "done"]


def test_a_cancelled_task_is_interrupted_and_resumes_from_its_checkpoint(tmp_path, here, capsys):
    progress = tmp_path / "progress"
    config = session(progress, max_iterations=1_000_000)
    config["checkpointer"] = {"checkpoint_every": 20}
    batch_dir, receipt = launch(repo(tmp_path, config, [1]), here, capsys)
    task = "cpu/seeds/000"
    sessions = batch_dir / "tasks" / task / "sessions"
    checkpoints = lambda: list(sessions.glob("session_*/**/manifest.json"))  # noqa: E731
    wait_for(lambda: checkpoints(), "a first checkpoint")

    ref = receipt["nodes"]["group-cpu"]["job_ref"]
    assert HostSubmitter(LocalExecutor()).cancel(ref).ok        # USR1, as at a walltime
    wait_for(lambda: record(batch_dir, task), "the interrupted task's record")
    first = record(batch_dir, task)
    assert first["interrupted"] is True and first["exit_status"] == 75

    wait_for(lambda: not host_runner.ref_alive(ref), "the runner to exit")
    assert naavik(["resume", batch_dir.name, "--yes"]) == 0
    capsys.readouterr()
    attempt = json.loads((batch_dir / "receipt.json").read_text())["attempts"][-1]
    assert attempt["attempt"] == 2
    second_ref = next(node["job_ref"] for node_id, node in
                      json.loads((batch_dir / "receipt.json").read_text())["nodes"].items()
                      if node_id.endswith("resume-2"))
    log = batch_dir / "logs" / "cpu" / f"{second_ref}_0.out"
    wait_for(lambda: log.exists() and "resuming from" in log.read_text(), "attempt 2 to start")
    assert HostSubmitter(LocalExecutor()).cancel(second_ref).ok
    wait_for(lambda: (record(batch_dir, task) or {}).get("attempt") == 2, "attempt 2's record")
    second = record(batch_dir, task)
    assert second["resumed_from"] and second["previous_attempts"][0]["interrupted"] is True


def test_a_host_batch_over_ssh_runs_on_the_machine_ssh_reaches(tmp_path, here, fake_cluster,
                                                              monkeypatch, capsys):
    # The fake `ssh` runs the remote command here, as a remote shell would receive it.
    targets = yaml.safe_load(here.read_text())
    entry = targets["profiles"]["cluster"]
    del entry["local"]
    entry["ssh_host"] = "box"
    here.write_text(yaml.safe_dump(targets))
    log = tmp_path / "ssh.log"
    monkeypatch.setenv("FAKE_SSH_LOG", str(log))
    batch_dir, receipt = launch(repo(tmp_path, session(), [1]), here, capsys)
    assert receipt["ssh_host"] == "box" and "target" not in receipt
    wait_for(lambda: record(batch_dir, "cpu/seeds/000"), "the task's record")
    assert record(batch_dir, "cpu/seeds/000")["exit_status"] == 0
    assert set(log.read_text().split()) == {"box"}
    assert naavik(["status", batch_dir.name]) == 0
    assert f"box:{batch_dir}" in capsys.readouterr().out


def test_launch_stops_before_staging_when_the_target_lacks_a_pool_gpu(tmp_path, here, monkeypatch,
                                                                     capsys):
    fake = tmp_path / "gpus-bin"
    fake.mkdir()
    (fake / "nvidia-smi").write_text("#!/bin/sh\necho 0\necho 1\n")
    (fake / "nvidia-smi").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ['PATH']}")
    targets = yaml.safe_load(here.read_text())
    targets["profiles"]["cluster"]["host"] = {"gpus": [0, 42]}
    here.write_text(yaml.safe_dump(targets))
    root = repo(tmp_path, session(), [1])
    assert naavik(["launch", str(root / "batch.yaml"), "--targets", str(here), "--yes"]) == 2
    error = capsys.readouterr().err
    assert "the target cannot run this batch" in error and "[42]" in error
    assert not (tmp_path / "here" / "batches").exists()            # nothing staged
