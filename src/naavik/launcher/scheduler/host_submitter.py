"""The `host` scheduler's effectful side: starting a job's runner detached on
the target, cancelling it, reading what is still queued or running, and
checking the target before anything is staged.

Everything runs through the target's executor (this machine or SSH) with the
target's `python3`: the runner file staged in the batch directory starts a
job; the same bundled program, passed with `-c`, answers state and cancel
queries (which know no batch directory, like `squeue`/`scancel`).
"""

from __future__ import annotations

import json
import posixpath
import re
import subprocess

from naavik.launcher.compile.plan import JobNode
from naavik.launcher.ports import CancelResult, LiveStates, SubmissionFailed, SubmissionUncertain
from naavik.launcher.ports.remote_probe import RemoteProcessProbe
from naavik.launcher.ports.ssh import TransportError

#: The interpreter that runs the runner on the target (standard library only).
RUNNER_PYTHON = "python3"
RUNNER_FILE = "host_runner.py"
_REF = re.compile(r"\Ah\d+\.\d+\Z")

#: Starts the runner in its own session (it outlives this command and an SSH
#: connection), output to its log, and prints the job reference the runner
#: derives for itself: `h<pid>.<start time>`.
_LAUNCH = r"""
import subprocess, sys
log = open(sys.argv[1], "ab")
process = subprocess.Popen(sys.argv[2:], stdin=subprocess.DEVNULL, stdout=log,
                           stderr=subprocess.STDOUT, start_new_session=True)
with open("/proc/%d/stat" % process.pid) as handle:
    start = handle.read().rsplit(")", 1)[1].split()[19]
print("h%d.%s" % (process.pid, start))
"""


def _runner_program() -> str:
    from naavik.launcher.remote.bootstrap import render_runner

    return render_runner()


def batch_dir_of(script_path: str) -> str:
    """The batch directory of a job script (`<batch>/groups/<g>/<script>` or
    `<batch>/nodes/<id>/<script>`, see `scheduler.script_path`)."""
    return posixpath.dirname(posixpath.dirname(posixpath.dirname(script_path)))


class HostSubmitter:
    def __init__(self, executor):
        self.executor = executor

    def submit(self, node: JobNode, script_path: str, dependencies: list[str]) -> str:
        batch_dir = batch_dir_of(script_path)
        log = f"{batch_dir}/logs/{node.group or node.id}/{node.id}.runner.log"
        argv = [RUNNER_PYTHON, "-c", _LAUNCH, log,
                RUNNER_PYTHON, f"{batch_dir}/{RUNNER_FILE}", "run", script_path]
        if dependencies:
            argv += ["--after", *dependencies]
        try:
            result = self.executor.run(argv)
        except (TransportError, subprocess.TimeoutExpired) as error:
            raise SubmissionUncertain(str(error)) from error
        if result.status != 0:
            raise SubmissionFailed(result.stderr.strip() or f"starting the runner exited {result.status}")
        ref = result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else ""
        if not _REF.match(ref):
            raise SubmissionUncertain(f"unparseable runner start output: {result.stdout!r}")
        return ref

    def cancel(self, job_ref: str) -> CancelResult:
        """Cancel a job (`h<pid>.<start>`) or one of its elements
        (`h<pid>.<start>_<index>`)."""
        ref, _, index = job_ref.partition("_")
        argv = [RUNNER_PYTHON, "-c", _runner_program(), "cancel", ref]
        if index:
            argv.append(index)
        result = self.executor.run(argv)
        detail = (result.stdout.strip() or result.stderr.strip() or "cancelled")
        return CancelResult(result.status == 0 and detail != "not running", detail)


def _states(executor, refs: list[str] | None) -> LiveStates:
    result = executor.run([RUNNER_PYTHON, "-c", _runner_program(), "states", *(refs or [])])
    if result.status != 0:
        raise RuntimeError(f"reading the host jobs failed: {result.stderr.strip()}")
    data = json.loads(result.stdout or "{}")
    return LiveStates(data.get("states") or {}, data.get("details") or {})


def job_states(executor):
    """Live states of jobs and their elements: `<ref>` and `<ref>_<index>`
    -> pending | running; finished ones are absent, as in a queue. Their
    `details` say why pending elements wait (GPUs, `max_parallel`, a job
    they run after)."""
    return lambda job_refs: _states(executor, list(job_refs)) if job_refs else LiveStates()


def user_job_states(executor) -> dict:
    return _states(executor, None)


def owner_probe(executor):
    """Owners of leases taken by host jobs are their processes."""
    return RemoteProcessProbe(executor)


def requested_gpus(plan) -> int:
    """The most GPUs one experiment (or step) of `plan` asks for."""
    return max((node.resources.gpus for node in plan.nodes if node.resources is not None),
               default=0)


def target_check(executor, settings, plan=None) -> list[str]:
    """What the target needs before anything is staged: a Python 3.8+
    `python3` for the runner; and, when a job asks for GPUs or the pool
    names some, `nvidia-smi`, every GPU the pool names, and enough GPUs for
    the largest request (with `all` or a count, the machine's are counted
    here: validation, offline, cannot)."""
    problems = []
    # Through `sh`, so a missing program is an exit status on every executor.
    result = executor.run(["sh", "-c", f'{RUNNER_PYTHON} -c "import sys; '
                           'print(sys.version_info >= (3, 8))"'])
    if result.status != 0 or result.stdout.strip() != "True":
        problems.append(f"the host needs {RUNNER_PYTHON} 3.8 or newer for the builtin "
                        f"scheduler's runner ({(result.stderr or result.stdout).strip() or 'too old'})")
    wanted = requested_gpus(plan) if plan is not None else 0
    if not wanted and not settings.gpu_indices:
        return problems
    result = executor.run(["sh", "-c", "nvidia-smi --query-gpu=index --format=csv,noheader"])
    if result.status != 0:
        problems.append(f"scheduler.gpus ({settings.describe_gpus()}) needs nvidia-smi on the "
                        f"host ({result.stderr.strip() or 'not found'})")
        return problems
    present = sorted({int(line) for line in result.stdout.split() if line.strip().isdigit()})
    indices = settings.gpu_indices
    if indices is not None:
        missing = [index for index in indices if index not in present]
        if missing:
            problems.append(f"scheduler.gpus lists GPU(s) {missing} that the host does not "
                            f"have (it has {present})")
    usable = len(indices) if indices is not None else min(
        len(present), settings.gpu_limit if settings.gpu_limit is not None else len(present))
    if wanted > usable:
        problems.append(
            f"an experiment requests {wanted} GPU(s), but at most {usable} may be in use at "
            f"once on this host (scheduler.gpus: {settings.describe_gpus()}; the machine has "
            f"{len(present)})")
    return problems


def extra_files(plan) -> dict[str, str]:
    """Staged next to the plan: the runner every job of the batch starts."""
    return {RUNNER_FILE: _runner_program()}


def reconcile_hint(job_name: str) -> str:
    return (f"look for a host job named {job_name}: grep -l '\"{job_name}\"' "
            "~/.local/state/naavik/host/$(hostname)/jobs/*.json "
            "~/.local/state/training-framework/host/$(hostname)/jobs/*.json")
