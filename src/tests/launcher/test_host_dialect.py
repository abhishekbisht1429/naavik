"""The `host` scheduler's profile settings, checks, and job scripts."""

from __future__ import annotations

import copy
import json

import pytest

from tests.hpc.batch_repo import PROFILE, compile_repo, make_repo
from training_framework.launcher.compile.plan import ExecutionContext, JobNode
from training_framework.launcher.compile.render import plan_files
from training_framework.launcher.config.schema import Resources, parse_profiles
from training_framework.launcher.errors import PlanError
from training_framework.launcher.remote.bootstrap import render_runner
from training_framework.launcher.scheduler import get_scheduler, script_path
from training_framework.launcher.scheduler.host import HostDialect, HostSettings

CONTEXT = ExecutionContext(
    phase="task", interpreter="/env/bin/python", environment="/env",
    working_directory="/b/source", pythonpath=("/b/source",),
    set_variables=(("PYTHONNOUSERSITE", "1"),), unset_variables=("WORLD_SIZE",),
    hooks=("source /opt/setup.sh",),
)


def _profile(**host):
    data = copy.deepcopy(PROFILE)
    cluster = data["profiles"]["cluster"]
    del cluster["slurm"]
    cluster["scheduler"] = "host"
    cluster["host"] = host
    return data


def _resources(gpus=1, options=None):
    return Resources("02:00:00", 7200, gpus, 8, "64G", options or {})


def _task_set(**changes):
    values = dict(id="group-g", kind="task_set", location="job", depends_on=(), context=CONTEXT,
                  resources=_resources(), group="g", task_count=3, max_parallel=2)
    values.update(changes)
    return JobNode(**values)


# Settings -------------------------------------------------------------------------------


def test_the_host_block_names_a_gpu_pool_and_what_to_do_with_busy_gpus():
    profile = parse_profiles(_profile(gpus=[4, 5], busy_gpus="use"), None, "targets.yaml")
    assert profile.scheduler == "host"
    assert profile.scheduler_settings == HostSettings(gpus=(4, 5), busy_gpus="use")
    assert parse_profiles(_profile(), None, "t").scheduler_settings == HostSettings((), "wait")


@pytest.mark.parametrize("host, message", [
    ({"gpus": [4, 4]}, "lists a GPU more than once"),
    ({"gpus": [-1]}, "list of GPU indices"),
    ({"gpus": "4,5"}, "list of GPU indices"),
    ({"gpus": [True]}, "list of GPU indices"),
    ({"busy_gpus": "maybe"}, "busy_gpus must be one of wait, use"),
    ({"partitions": {}}, "unknown"),
])
def test_a_bad_host_block_is_refused(host, message):
    with pytest.raises(PlanError, match=message):
        parse_profiles(_profile(**host), None, "targets.yaml")


# Checks ---------------------------------------------------------------------------------


def test_a_group_may_not_ask_for_more_gpus_than_the_pool_has():
    dialect = HostDialect(HostSettings(gpus=(4, 5)))
    assert dialect.check_resources(_resources(gpus=2), "groups.g") == []
    (problem,) = dialect.check_resources(_resources(gpus=3), "groups.g")
    assert "requests 3 GPU(s)" in problem and "pool has 2" in problem


def test_scheduler_options_are_refused_on_a_host():
    dialect = HostDialect(HostSettings(gpus=(0,)))
    problems = dialect.check_resources(_resources(options={"host": {"x": 1}, "slurm": {"y": 2}}),
                                       "groups.g")
    assert any("takes no options" in problem for problem in problems)
    assert any("other than the selected 'host'" in problem for problem in problems)


def test_a_batch_on_a_host_compiles_with_no_array_limit(tmp_path):
    plan = compile_repo(make_repo(tmp_path), profile_data=_profile(gpus=[0, 1])).plan
    assert plan.scheduler == "host"
    # As plan.json records them, for a later attempt to render the same.
    assert json.loads(plan.canonical())["scheduler_settings"] == {"gpus": [0, 1], "busy_gpus": "wait"}
    assert HostDialect(HostSettings()).max_task_set_size >= 1_000_000


# Job scripts ----------------------------------------------------------------------------


def _directives(script):
    return dict(line[len("#TFHOST "):].split("=", 1) for line in script.splitlines()
                if line.startswith("#TFHOST "))


def test_a_task_set_script_carries_what_the_runner_needs():
    dialect = HostDialect(HostSettings(gpus=(4, 5), busy_gpus="wait"))
    script = dialect.render(_task_set(), "b1", "/b", ["/env/bin/python", "-m", "x",
                                                      "--config", "{config}"], 150)
    assert _directives(script) == {
        "job-name": "b1.group-g", "batch-dir": "/b", "kind": "task_set", "elements": "0,1,2",
        "max-parallel": "2", "gpus": "1", "gpu-pool": "4,5", "busy-gpus": "wait",
        "walltime": "7200", "signal-lead": "150", "log": "/b/logs/g/{job}_{index}.out",
        "chdir": "/b/source",
    }
    lines = script.splitlines()
    assert lines[0] == "#!/bin/bash"
    for line in ("source /opt/setup.sh", "unset WORLD_SIZE", "export PYTHONNOUSERSITE=1",
                 "trap on_signal USR1"):
        assert line in lines, line
    assert '-v i="$TF_HOST_TASK_INDEX"' in script and "SLURM" not in script
    assert '--config "$config" &' in script


def test_a_later_attempt_runs_only_its_elements_and_reads_its_checkpoints():
    script = HostDialect(HostSettings(gpus=(0,))).render(
        _task_set(id="group-g.resume-2"), "b1", "/b", ["py", "--resume", "{checkpoint}"], 10,
        array_indices=[5, 2], resume_table="/b/groups/g/resume-2.tsv")
    assert _directives(script)["elements"] == "2,5"
    assert "/b/groups/g/resume-2.tsv" in script


def test_a_setup_node_runs_once_without_a_signal():
    node = JobNode(id="setup", kind="setup", location="job", depends_on=(), context=CONTEXT,
                   resources=Resources("01:00:00", 3600, 0, 4, "16G", {}))
    script = HostDialect(HostSettings()).render(node, "b1", "/b", ["/py", "/b/bootstrap.py"], None)
    directives = _directives(script)
    assert directives["elements"] == "0" and directives["signal-lead"] == ""
    assert directives["gpus"] == "0" and directives["log"] == "/b/logs/setup/{job}.out"
    assert script.rstrip().endswith("exec /py /b/bootstrap.py")


def test_control_characters_are_never_rendered_into_directives():
    with pytest.raises(ValueError, match="control characters"):
        HostDialect(HostSettings()).render(_task_set(), "b1\n#TFHOST gpus=8", "/b", ["py"], 1)


def test_a_host_batch_stages_its_runner_and_names_scripts_dot_sh(tmp_path):
    compiled = compile_repo(make_repo(tmp_path), profile_data=_profile(gpus=[0]))
    files = plan_files(compiled.plan, compiled.manifest.raw)
    assert files["host_runner.py"] == render_runner()
    assert "groups/probes/group-probes.sh" in files
    assert script_path(get_scheduler("host"), compiled.plan.node("group-probes")) == \
        "groups/probes/group-probes.sh"
