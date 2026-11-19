"""The `host` scheduler's runner, as real detached processes on this machine:
task sets and `max_parallel`, the GPU pool shared across jobs, busy GPUs,
dependencies, the advance signal and the walltime, cancelling, and a runner
that dies. Job scripts are written by hand (only their `#TFHOST` lines
matter to the runner); `nvidia-smi` is a fake on PATH."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from naavik.launcher.compile.plan import ExecutionContext, JobNode
from naavik.launcher.ports import Owner
from naavik.launcher.ports.local import LocalExecutor
from naavik.launcher.remote import host_runner
from naavik.launcher.remote.bootstrap import render_runner
from naavik.launcher.remote.runtime import current_owner, owner_probe
from naavik.launcher.scheduler.host_submitter import (
    HostSubmitter,
    job_states,
    target_check,
    user_job_states,
)
from naavik.launcher.scheduler.host import HostSettings

TIMEOUT = 60
FAKE_NVIDIA_SMI = r'''
import os, sys
gpus = [item.split(":") for item in os.environ.get("FAKE_GPUS", "").split(",") if item]
query = sys.argv[1]
if query == "--query-gpu=index,uuid":
    for index, uuid in gpus:
        print("%s, %s" % (index, uuid))
elif query == "--query-gpu=index":
    for index, _ in gpus:
        print(index)
elif query == "--query-gpu=uuid,memory.used":
    used = dict(item.split(":") for item in os.environ.get("FAKE_MEMORY", "").split(",") if item)
    for _, uuid in gpus:
        print("%s, %s MiB" % (uuid, used.get(uuid, "0")))
elif query == "--query-compute-apps=gpu_uuid":
    path = os.environ.get("FAKE_BUSY")
    if path and os.path.exists(path):
        print(open(path).read())
else:
    sys.exit(2)
'''
CONTEXT = ExecutionContext(phase="task", interpreter="python3", environment=None,
                           working_directory="/", pythonpath=(), set_variables=(),
                           unset_variables=(), hooks=())


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A batch directory with the runner staged, a per-test state directory,
    and two fake GPUs."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "nvidia-smi"
    fake.write_text(f"#!{sys.executable}\n{FAKE_NVIDIA_SMI}")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_GPUS", "0:GPU-a,1:GPU-b")
    monkeypatch.setenv("FAKE_BUSY", str(tmp_path / "busy"))
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "host_runner.py").write_text(render_runner())
    (batch / "events").touch()
    started = []
    yield batch, started
    for ref in started:                      # never leave a runner behind
        pid = host_runner.parse_ref(ref)[0]
        if host_runner.ref_alive(ref):
            os.kill(pid, signal.SIGKILL)


def job(batch, name, body, elements=(0,), max_parallel=0, gpus=0, pool="0,1", busy="wait",
        walltime="", lead=""):
    """A job script whose elements append `start|end <job> <index> <time> <GPUs>`
    lines to `events` around `body`."""
    directory = batch / "groups" / name
    directory.mkdir(parents=True)
    (batch / "logs" / name).mkdir(parents=True)
    directives = {
        "job-name": f"b.{name}", "batch-dir": str(batch), "kind": "task_set",
        "elements": ",".join(str(i) for i in elements), "max-parallel": str(max_parallel),
        "gpus": str(gpus), "gpu-pool": pool, "busy-gpus": busy, "walltime": walltime,
        "signal-lead": lead, "log": f"{batch}/logs/{name}/{{job}}_{{index}}.out",
        "chdir": str(batch),
    }
    event = (f'echo "KIND {name} $TF_HOST_TASK_INDEX $(date +%s.%N) $CUDA_VISIBLE_DEVICES" '
             f'>> {batch}/events')
    script = "\n".join(["#!/bin/bash", *(f"#TFHOST {k}={v}" for k, v in directives.items()),
                        event.replace("KIND", "start"), body, event.replace("KIND", "end"), ""])
    path = directory / f"{name}.sh"
    path.write_text(script)
    return path


def submit(batch, started, path, after=()):
    node = JobNode(id=path.stem, kind="task_set", location="job", depends_on=(),
                   context=CONTEXT, group=path.parent.name)
    ref = HostSubmitter(LocalExecutor()).submit(node, str(path), list(after))
    started.append(ref)
    return ref


def wait_final(ref):
    deadline = time.monotonic() + TIMEOUT
    while True:
        data = host_runner.read_job(host_runner.state_dir(), ref)
        if data and data["final"]:
            return data
        assert time.monotonic() < deadline, f"{ref} did not finish: {data}"
        time.sleep(0.2)


def wait_for(predicate, what):
    deadline = time.monotonic() + TIMEOUT
    while not predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.1)


def events(batch):
    rows = []
    for line in (batch / "events").read_text().splitlines():
        kind, name, index, stamp, *gpus = line.split()
        rows.append((kind, name, int(index), float(stamp), gpus[0] if gpus else ""))
    return rows


def intervals(batch):
    """(job, index) -> (start, end, GPUs)."""
    found = {}
    for kind, name, index, stamp, gpus in events(batch):
        start, end, _ = found.get((name, index), (None, None, gpus))
        found[(name, index)] = (stamp, end, gpus) if kind == "start" else (start, stamp, gpus)
    return found


def most_at_once(spans):
    points = sorted([(s, 1) for s, _ in spans] + [(e, -1) for _, e in spans])
    level = peak = 0
    for _, change in points:
        level += change
        peak = max(peak, level)
    return peak


# Task sets -------------------------------------------------------------------------------


def test_a_task_set_runs_every_element_at_most_max_parallel_at_once(host):
    batch, started = host
    ref = submit(batch, started, job(batch, "g", "sleep 1", elements=range(4), max_parallel=2))
    data = wait_final(ref)
    assert data["elements"] == {str(i): "completed" for i in range(4)}
    spans = [(s, e) for s, e, _ in intervals(batch).values()]
    assert len(spans) == 4 and most_at_once(spans) == 2
    log = batch / "logs" / "g" / f"{ref}_3.out"
    assert log.exists()                                  # one log per element, Slurm's naming


def test_an_element_that_fails_is_failed_and_the_others_still_run(host):
    batch, started = host
    data = wait_final(submit(batch, started, job(
        batch, "g", '[ "$TF_HOST_TASK_INDEX" != 1 ] || exit 3', elements=range(3))))
    assert data["elements"] == {"0": "completed", "1": "failed", "2": "completed"}
    assert data["exit"]["1"] == 3


# GPUs ------------------------------------------------------------------------------------


def test_elements_get_their_gpus_by_uuid(host):
    batch, started = host
    wait_final(submit(batch, started, job(batch, "g", "sleep 0.5", elements=range(2), gpus=1)))
    gpus = sorted(g for _, _, g in intervals(batch).values())
    assert set(gpus) <= {"GPU-a", "GPU-b"} and all(gpus)


def test_two_jobs_never_share_a_gpu(host):
    batch, started = host
    first = submit(batch, started, job(batch, "one", "sleep 1", elements=range(2), gpus=1, pool="0"))
    second = submit(batch, started, job(batch, "two", "sleep 1", elements=range(2), gpus=1, pool="0"))
    wait_final(first), wait_final(second)
    spans = [(s, e) for s, e, gpus in intervals(batch).values() if gpus == "GPU-a"]
    assert len(spans) == 4 and most_at_once(spans) == 1


def test_a_busy_gpu_is_skipped_with_wait_and_used_with_use(host, tmp_path):
    batch, started = host
    (tmp_path / "busy").write_text("GPU-a\n")             # another user's process
    wait_final(submit(batch, started, job(batch, "waits", "true", gpus=1, busy="wait")))
    wait_final(submit(batch, started, job(batch, "uses", "true", gpus=1, busy="use")))
    found = intervals(batch)
    assert found[("waits", 0)][2] == "GPU-b"
    assert found[("uses", 0)][2] == "GPU-a"


def test_a_gpu_whose_memory_is_in_use_counts_as_busy(host, monkeypatch):
    # Another container's process: not listed by nvidia-smi, but its memory is.
    batch, started = host
    monkeypatch.setenv("FAKE_MEMORY", "GPU-a:18000")
    wait_final(submit(batch, started, job(batch, "waits", "true", gpus=1, busy="wait")))
    assert intervals(batch)[("waits", 0)][2] == "GPU-b"


def test_a_slot_held_by_a_dead_runner_is_taken_over(host):
    batch, started = host
    lock = Path(host_runner.state_dir(), "slots", "gpu-GPU-a.tf-hpc-lock")
    lock.mkdir(parents=True)
    (lock / "owner.json").write_text(json.dumps({"host": "x", "pid": 2 ** 22 + 7, "start_time": "1"}))
    wait_final(submit(batch, started, job(batch, "g", "true", gpus=1, pool="0")))
    assert intervals(batch)[("g", 0)][2] == "GPU-a"


def test_a_pool_gpu_the_machine_lacks_fails_the_job_with_the_reason(host):
    batch, started = host
    data = wait_final(submit(batch, started, job(batch, "g", "true", gpus=1, pool="7")))
    assert data["elements"] == {"0": "cancelled"} and "do not exist here" in data["error"]


# Dependencies -----------------------------------------------------------------------------


def test_a_job_runs_after_the_jobs_it_depends_on_completed(host):
    batch, started = host
    first = submit(batch, started, job(batch, "first", "sleep 1"))
    second = submit(batch, started, job(batch, "second", "true"), after=[first])
    assert wait_final(second)["elements"] == {"0": "completed"}
    found = intervals(batch)
    assert found[("second", 0)][0] >= found[("first", 0)][1]


def test_a_failed_dependency_cancels_the_job_without_running_it(host):
    batch, started = host
    first = submit(batch, started, job(batch, "first", "exit 1"))
    second = submit(batch, started, job(batch, "second", "true", elements=range(2)), after=[first])
    data = wait_final(second)
    assert data["elements"] == {"0": "cancelled", "1": "cancelled"}
    assert ("second", 0) not in intervals(batch)


def test_a_dependency_that_died_unfinished_cancels_the_job(host):
    batch, started = host
    first = submit(batch, started, job(batch, "first", "sleep 60"))
    second = submit(batch, started, job(batch, "second", "true"), after=[first])
    wait_for(lambda: ("first", 0) in intervals(batch), "the first job to start")
    os.kill(host_runner.parse_ref(first)[0], signal.SIGKILL)
    assert wait_final(second)["elements"] == {"0": "cancelled"}


# The walltime -------------------------------------------------------------------------------


def test_the_advance_signal_arrives_signal_lead_before_the_walltime(host):
    batch, started = host
    body = ('trap \'echo "usr1 g $TF_HOST_TASK_INDEX $(date +%s.%N)" >> ' + str(batch)
            + '/events; exit 75\' USR1\nsleep 30 & wait')
    data = wait_final(submit(batch, started, job(batch, "g", body, walltime="4", lead="2")))
    assert data["elements"] == {"0": "failed"} and data["exit"]["0"] == 75
    (start,) = [stamp for kind, _, _, stamp, _ in events(batch) if kind == "start"]
    (usr1,) = [stamp for kind, _, _, stamp, _ in events(batch) if kind == "usr1"]
    assert 1.5 <= usr1 - start <= 3.5


def test_an_element_still_running_at_the_walltime_is_killed(host):
    batch, started = host
    data = wait_final(submit(batch, started, job(batch, "g", "trap '' USR1\nsleep 60",
                                                 walltime="2", lead="1")))
    assert data["elements"] == {"0": "timeout"}
    assert not any(kind == "end" for kind, *_ in events(batch))


# Cancelling and dying ------------------------------------------------------------------------


def test_cancelling_a_job_stops_its_running_elements_and_drops_the_pending_ones(host):
    batch, started = host
    body = ('trap \'echo "usr1 g $TF_HOST_TASK_INDEX $(date +%s.%N)" >> ' + str(batch)
            + '/events; exit 75\' USR1\nsleep 60 & wait')
    ref = submit(batch, started, job(batch, "g", body, elements=range(3), max_parallel=1, lead="5"))
    wait_for(lambda: ("g", 0) in intervals(batch), "the first element to start")
    result = HostSubmitter(LocalExecutor()).cancel(ref)
    assert result.ok, result.detail
    data = wait_final(ref)
    assert data["elements"] == {"0": "cancelled", "1": "cancelled", "2": "cancelled"}
    assert [index for kind, _, index, *_ in events(batch) if kind == "usr1"] == [0]
    assert [index for kind, _, index, *_ in events(batch) if kind == "start"] == [0]


def test_cancelling_one_element_leaves_the_others(host):
    batch, started = host
    ref = submit(batch, started, job(batch, "g", "sleep 2", elements=range(3), max_parallel=1))
    assert HostSubmitter(LocalExecutor()).cancel(f"{ref}_2").ok
    data = wait_final(ref)
    assert data["elements"] == {"0": "completed", "1": "completed", "2": "cancelled"}
    assert ("g", 2) not in intervals(batch)                  # never started


def test_a_finished_job_cannot_be_cancelled(host):
    batch, started = host
    ref = submit(batch, started, job(batch, "g", "true"))
    wait_final(ref)
    wait_for(lambda: not host_runner.ref_alive(ref), "the runner to exit")
    result = HostSubmitter(LocalExecutor()).cancel(ref)
    assert not result.ok and result.detail == "not running"


def test_live_states_list_what_is_still_queued_or_running(host):
    batch, started = host
    ref = submit(batch, started, job(batch, "g", "sleep 60", elements=range(2), max_parallel=1))
    wait_for(lambda: ("g", 0) in intervals(batch), "the first element to start")
    states = job_states(LocalExecutor())([ref])
    assert states == {ref: "running", f"{ref}_0": "running", f"{ref}_1": "pending"}
    assert user_job_states(LocalExecutor()) == states
    assert job_states(LocalExecutor())(["h1.1"]) == {}


def test_a_dead_runners_unfinished_elements_leave_the_live_states(host):
    batch, started = host
    ref = submit(batch, started, job(batch, "g", "sleep 60", elements=range(2), max_parallel=1))
    wait_for(lambda: ("g", 0) in intervals(batch), "the first element to start")
    os.kill(host_runner.parse_ref(ref)[0], signal.SIGKILL)
    wait_for(lambda: not host_runner.ref_alive(ref), "the runner to die")
    # As a job that left the queue: status then says "ended without a record".
    assert job_states(LocalExecutor())([ref]) == {}


# Owners and the target ------------------------------------------------------------------------


def test_steps_of_host_jobs_are_owned_by_their_processes(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "77")             # ignored for host jobs
    owner = current_owner("host")
    assert owner["pid"] == os.getpid() and "job_ref" not in owner
    assert owner_probe("host").is_alive(owner) is True
    assert owner_probe("host").is_alive({"job_ref": "h1.1"}) is None


def test_the_target_check_finds_a_missing_pool_gpu(host):
    executor = LocalExecutor()
    assert target_check(executor, HostSettings(gpus=(0, 1))) == []
    (problem,) = target_check(executor, HostSettings(gpus=(0, 5)))
    assert "[5]" in problem and "[0, 1]" in problem


def test_the_target_check_needs_nvidia_smi_only_for_a_gpu_pool(host, tmp_path, monkeypatch):
    broken = tmp_path / "broken-bin"
    broken.mkdir()
    (broken / "nvidia-smi").write_text("#!/bin/sh\necho 'NVIDIA-SMI has failed' >&2\nexit 9\n")
    (broken / "nvidia-smi").chmod(0o755)
    monkeypatch.setenv("PATH", f"{broken}{os.pathsep}{os.environ['PATH']}")
    assert target_check(LocalExecutor(), HostSettings()) == []
    (problem,) = target_check(LocalExecutor(), HostSettings(gpus=(0,)))
    assert "needs nvidia-smi" in problem and "NVIDIA-SMI has failed" in problem


def test_the_remote_process_probe_checks_owners_on_their_host():
    from naavik.launcher.ports.remote_probe import RemoteProcessProbe

    me = Owner(**current_owner("host"))
    probe = RemoteProcessProbe(LocalExecutor())
    assert probe.is_alive(me) is True
    assert probe.is_alive(Owner(host=me.host, pid=me.pid, start_time="0")) is False
    assert probe.is_alive(Owner(host="elsewhere", pid=me.pid, start_time=me.start_time)) is None
    assert probe.is_alive(Owner(job_ref="h1.1")) is None
