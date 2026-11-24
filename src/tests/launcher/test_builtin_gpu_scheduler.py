"""The builtin scheduler as a simple GPU scheduler, and resources it does not
limit.

The host's `scheduler.gpus` is the pool and the limit: `all` (the default),
a count (any that many GPUs at once, whichever are free), or a list of
indices. The mission file says how many GPUs each experiment gets. A request
the pool can never satisfy is refused before staging and, should one reach
the runner, fails at once; an experiment waiting for GPUs says why. CPUs and
memory are never limited, and `walltime`, `cpus`, `memory` may be left out
(no limit) on a builtin host but not on a Slurm one."""

from __future__ import annotations

import copy
import json
import os

import pytest

from tests.hpc.batch_repo import MANIFEST, PROFILE, compile_repo, make_repo
from tests.launcher.test_host_dialect import _directives, _profile, _task_set
from tests.launcher.test_host_runner import (  # noqa: F401 (fixture)
    host,
    intervals,
    job,
    most_at_once,
    submit,
    wait_final,
    wait_for,
)
from naavik.launcher.compile.plan import JobNode
from naavik.launcher.config.schema import Resources, parse_profiles
from naavik.launcher.errors import PlanError
from naavik.launcher.ports.local import LocalExecutor
from naavik.launcher.remote import host_runner
from naavik.launcher.scheduler.host import HostDialect, HostSettings
from naavik.launcher.scheduler.host_submitter import job_states, target_check
from naavik.launcher.services.follow import Follower
from naavik.launcher.services.status import Row


def _limited(path, limit):
    """`job()`'s script with a `gpu-limit` directive (`scheduler.gpus: N`)."""
    lines = path.read_text().splitlines()
    lines.insert(1, f"#TFHOST gpu-limit={limit}")
    path.write_text("\n".join(lines) + "\n")
    return path


def _resources(gpus=1, walltime="02:00:00", seconds=7200, cpus=8, memory="64G"):
    return Resources(walltime, seconds, gpus, cpus, memory, {})


# Settings ---------------------------------------------------------------------------------


@pytest.mark.parametrize("written, gpus, indices, limit", [
    ("all", "all", None, None),
    (3, 3, None, 3),
    ([4, 5], (4, 5), (4, 5), 2),
])
def test_scheduler_gpus_is_all_a_count_or_a_list_and_gives_the_limit(written, gpus, indices,
                                                                      limit):
    settings = parse_profiles(_profile(gpus=written), None, "t").scheduler_settings
    assert settings.gpus == gpus
    assert settings.gpu_indices == indices and settings.gpu_limit == limit


def test_a_count_below_zero_is_refused():
    with pytest.raises(PlanError, match="gpus must be all, a count >= 0"):
        parse_profiles(_profile(gpus=-1), None, "t")


@pytest.mark.parametrize("gpus", ["all", 2, [0, 1]])
def test_a_plan_records_the_pool_as_written_and_reads_it_back(tmp_path, gpus):
    plan = compile_repo(make_repo(tmp_path), profile_data=_profile(gpus=gpus)).plan
    recorded = json.loads(plan.canonical())["scheduler_settings"]
    assert recorded == {"gpus": gpus, "busy_gpus": "wait"}
    from naavik.launcher.services.resume import plan_dialect
    dialect = plan_dialect({"batch_id": "b", "scheduler": "host", "scheduler_settings": recorded})
    assert dialect.settings == HostSettings(gpus=tuple(gpus) if isinstance(gpus, list) else gpus)


# Checks before staging --------------------------------------------------------------------


def test_a_request_above_a_count_is_refused_when_compiling():
    (problem,) = HostDialect(HostSettings(gpus=2)).check_resources(_resources(gpus=3), "jobs.g")
    assert "requests 3 GPU(s)" in problem and "the pool has 2" in problem


def test_with_all_the_request_is_checked_against_the_machine_when_launching(host):
    # The fake machine has two GPUs; validation, offline, cannot know.
    assert HostDialect(HostSettings()).check_resources(_resources(gpus=3), "jobs.g") == []
    plan = _plan_asking(3)
    (problem,) = target_check(LocalExecutor(), HostSettings(), plan)
    assert "requests 3 GPU(s)" in problem and "at most 2" in problem and "has 2" in problem
    assert target_check(LocalExecutor(), HostSettings(), _plan_asking(2)) == []
    (problem,) = target_check(LocalExecutor(), HostSettings(gpus=1), _plan_asking(2))
    assert "at most 1" in problem


def test_a_machine_without_gpus_launches_cpu_jobs_with_the_default_pool(host, tmp_path,
                                                                       monkeypatch):
    empty = tmp_path / "no-gpus-bin"
    empty.mkdir()
    (empty / "nvidia-smi").write_text("#!/bin/sh\nexit 127\n")
    (empty / "nvidia-smi").chmod(0o755)
    monkeypatch.setenv("PATH", f"{empty}{os.pathsep}{os.environ['PATH']}")
    assert target_check(LocalExecutor(), HostSettings(), _plan_asking(0)) == []
    (problem,) = target_check(LocalExecutor(), HostSettings(), _plan_asking(1))
    assert "needs nvidia-smi" in problem


def _plan_asking(gpus):
    class Plan:
        nodes = (_task_set(resources=_resources(gpus=gpus)),)
    return Plan()


@pytest.mark.parametrize("gpus, pool, limit", [
    ("all", "all", None),
    (3, "all", "3"),
    ((0, 2), "0,2", None),
])
def test_the_script_tells_the_runner_the_pool_and_the_limit(gpus, pool, limit):
    script = HostDialect(HostSettings(gpus=gpus)).render(_task_set(), "b", "/b", ["run"], 40)
    directives = _directives(script)
    assert directives["gpu-pool"] == pool
    assert directives.get("gpu-limit") == limit


# The runner -------------------------------------------------------------------------------


def test_with_all_experiments_use_every_free_gpu_at_once(host):
    batch, started = host
    wait_final(submit(batch, started, job(batch, "g", "sleep 1", elements=range(2), gpus=1,
                                          pool="all")))
    spans = intervals(batch)
    assert {gpus for _, _, gpus in spans.values()} == {"GPU-a", "GPU-b"}
    assert most_at_once([(s, e) for s, e, _ in spans.values()]) == 2


def test_a_count_limits_gpus_held_at_once_across_jobs_whichever_are_free(host):
    batch, started = host
    first = submit(batch, started, _limited(
        job(batch, "one", "sleep 1", elements=range(2), gpus=1, pool="all"), 1))
    second = submit(batch, started, _limited(
        job(batch, "two", "sleep 1", elements=range(2), gpus=1, pool="all"), 1))
    wait_final(first), wait_final(second)
    spans = [(s, e) for s, e, _ in intervals(batch).values()]
    assert len(spans) == 4 and most_at_once(spans) == 1


def test_an_experiment_waiting_for_gpus_says_why(host):
    batch, started = host
    holder = submit(batch, started, _limited(job(batch, "one", "sleep 60", gpus=1, pool="all"), 1))
    wait_for(lambda: ("one", 0) in intervals(batch), "the first job to hold the GPU")
    waiter = submit(batch, started, _limited(job(batch, "two", "true", gpus=1, pool="all"), 1))
    expected = "waiting for GPUs: needs 1; 0 free now (limit 1; 1 held by your experiments)"
    wait_for(lambda: (host_runner.read_job(host_runner.state_dir(), waiter) or {})
             .get("waiting", {}).get("0") == expected, "the reason in the job file")
    states = job_states(LocalExecutor())([waiter])
    assert states[f"{waiter}_0"] == "pending"
    assert states.details == {f"{waiter}_0": expected}
    os.kill(host_runner.parse_ref(holder)[0], 15)
    assert wait_final(waiter)["elements"] == {"0": "completed"}
    assert wait_final(waiter)["waiting"] == {}


def test_an_experiment_held_back_by_max_parallel_says_so(host):
    batch, started = host
    ref = submit(batch, started, job(batch, "g", "sleep 60", elements=range(2), max_parallel=1))
    wait_for(lambda: ("g", 0) in intervals(batch), "the first element to start")
    wait_for(lambda: job_states(LocalExecutor())([ref]).details.get(f"{ref}_1")
             == "waiting: at most 1 experiment(s) of this job run at once", "the reason")


def test_a_request_the_pool_can_never_satisfy_fails_at_once_with_the_reason_in_the_log(host):
    batch, started = host
    ref = submit(batch, started, job(batch, "g", "true", elements=range(2), gpus=3, pool="all"))
    data = wait_final(ref)
    assert data["elements"] == {"0": "failed", "1": "failed"}
    assert "requests 3 GPU(s) per experiment, but at most 2" in data["error"]
    log = (batch / "logs" / "g" / f"{ref}_0.out").read_text()
    assert "[host runner] requests 3 GPU(s)" in log and "scheduler.gpus: all" in log
    assert not intervals(batch)                             # never started


def test_a_job_with_no_walltime_runs_until_it_ends(host):
    batch, started = host
    data = wait_final(submit(batch, started, job(batch, "g", "sleep 1", walltime="", lead="")))
    assert data["elements"] == {"0": "completed"}


# What the user sees -----------------------------------------------------------------------


def test_status_shows_why_a_queued_experiment_waits():
    from naavik.launcher.ports import LiveStates
    from naavik.launcher.services.status import _live_row

    live = LiveStates({"h1.2_0": "pending"}, {"h1.2_0": "waiting for GPUs: needs 2; 1 free now"})
    row = _live_row("run/x/000", "", live, "h1.2_0")
    assert row.detail == "job h1.2_0: waiting for GPUs: needs 2; 1 free now"
    assert row.waiting == "waiting for GPUs: needs 2; 1 free now" and row.live


def test_naavik_run_says_why_the_experiment_waits_once_and_again_when_it_changes():
    said = []
    rows = iter([
        [Row("run/x/000", "", "pending", "", True, waiting="needs 2; 0 free now")],
        [Row("run/x/000", "", "pending", "", True, waiting="needs 2; 0 free now")],
        [Row("run/x/000", "", "pending", "", True, waiting="needs 2; 1 free now")],
        [Row("run/x/000", "", "running", "", True)],
        [Row("run/x/000", "", "done", "")],
    ])
    outcome = Follower(task="run/x/000", rows=lambda: next(rows), log=lambda: None,
                       read=lambda path, offset: b"", cancel=lambda: None,
                       write=lambda chunk: None, say=said.append, sleep=lambda s: None).run()
    assert outcome.status == 0
    assert said == ["run/x/000: needs 2; 0 free now", "run/x/000: needs 2; 1 free now"]


# Resources a builtin host does not limit --------------------------------------------------


def _manifest(**resources):
    data = copy.deepcopy(MANIFEST)
    data["groups"]["probes"]["resources"] = resources
    return data


def test_a_builtin_host_runs_experiments_without_walltime_cpus_or_memory(tmp_path):
    compiled = compile_repo(make_repo(tmp_path, manifest=_manifest(gpus=1)),
                            profile_data=_profile())
    (group,) = [node for node in compiled.plan.nodes if node.kind == "task_set"]
    assert (group.resources.walltime, group.resources.cpus, group.resources.memory) == \
        (None, None, None)
    directives = _directives(group.script)
    assert directives["walltime"] == "" and directives["signal-lead"] != ""
    assert not [note for note in compiled.plan.notes if "does not limit" in note]
    assert group.resources.describe() == \
        "walltime no limit, 1 GPU(s), no limit on CPU(s), no limit on memory"


def test_cpus_and_memory_given_for_a_builtin_host_are_noted_as_not_enforced(tmp_path):
    compiled = compile_repo(make_repo(tmp_path, manifest=_manifest(gpus=1, cpus=4,
                                                                    memory="8G")),
                            profile_data=_profile())
    assert "host 'cluster' (builtin scheduler) does not limit cpus or memory: the values " \
           "are recorded, not enforced" in compiled.plan.notes


def test_a_slurm_host_refuses_experiments_without_walltime_cpus_or_memory(tmp_path):
    with pytest.raises(PlanError, match="walltime, cpus, memory must be given on a Slurm host"):
        compile_repo(make_repo(tmp_path, manifest=_manifest(gpus=1)), profile_data=PROFILE)


def test_without_cpus_gloo_is_not_checked_against_them(tmp_path):
    compile_repo(make_repo(tmp_path, manifest=_manifest(gpus=0)), profile_data=_profile())
