"""Slurm's effectful adapters: the submitter (`sbatch`, `scancel`) and the
cluster-side owner probe (`squeue` for jobs; processes only on their own
host). Together with the dialect, the only code that knows Slurm."""

from __future__ import annotations

import re
import subprocess

from training_framework.launcher.compile.plan import JobNode
from training_framework.launcher.ports import (
    CancelResult,
    ExecResult,
    Owner,
    SubmissionFailed,
    SubmissionUncertain,
)
from training_framework.launcher.ports.remote_probe import RemoteProcessProbe
from training_framework.launcher.ports.ssh import TransportError

_JOB_REF = re.compile(r"\A(\d+)(?:;\S+)?\Z")


def reconcile_hint(job_name: str) -> str:
    """How to find a job whose `sbatch` had an uncertain outcome: job names
    are deterministic (`<batch-id>.<node-id>`) for exactly this."""
    return f"squeue --name={job_name}  (or sacct --name={job_name})"


class SlurmSubmitter:
    def __init__(self, executor):
        self.executor = executor

    def submit(self, node: JobNode, script_path: str, dependencies: list[str]) -> str:
        argv = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes"]
        if dependencies:
            argv.append(f"--dependency=afterok:{':'.join(dependencies)}")
        argv.append(script_path)
        try:
            result = self.executor.run(argv)
        except (TransportError, subprocess.TimeoutExpired) as error:
            raise SubmissionUncertain(str(error)) from error
        if result.status != 0:
            raise SubmissionFailed(result.stderr.strip() or f"sbatch exited {result.status}")
        lines = result.stdout.strip().splitlines()
        match = _JOB_REF.match(lines[-1].strip()) if lines else None
        if match is None:
            raise SubmissionUncertain(f"unparseable sbatch output: {result.stdout!r}")
        return match.group(1)

    def cancel(self, job_ref: str) -> CancelResult:
        result = self.executor.run(["scancel", job_ref])
        return CancelResult(result.status == 0, result.stderr.strip() or "cancelled")


class _Completed:
    def __init__(self, result: ExecResult):
        self.returncode, self.stdout, self.stderr = result.status, result.stdout, result.stderr


class RemoteOwnerProbe:
    """`LeaseOwnerProbe` on the cluster: Slurm jobs via `squeue` (any host);
    processes through `RemoteProcessProbe`."""

    def __init__(self, executor):
        self.executor = executor

    def is_alive(self, owner: Owner) -> bool | None:
        from training_framework.launcher.remote.runtime import SlurmJobProbe

        if owner.job_ref is not None:
            return SlurmJobProbe(lambda argv: _Completed(self.executor.run(argv))).is_alive(
                owner.as_json())
        return RemoteProcessProbe(self.executor).is_alive(owner)


def job_states(executor):
    """Live states of jobs and array elements (`squeue --array`); finished
    jobs are simply absent."""
    def query(job_refs: list[str]) -> dict:
        states = {}
        for job in job_refs:
            result = executor.run(["squeue", "--noheader", "--array", "--jobs", job,
                                   "--format", "%i|%T"])
            if result.status != 0:
                if "Invalid job id" in result.stderr:
                    continue
                raise RuntimeError(f"squeue failed: {result.stderr.strip()}")
            for line in result.stdout.splitlines():
                element, _, state = line.strip().partition("|")
                if element:
                    states[element] = state.lower()
        return states
    return query


def user_job_states(executor) -> dict:
    """Live states of all the user's jobs and array elements, in one call."""
    result = executor.run(["sh", "-c", 'squeue --noheader --array --user "$(id -un)" '
                                       '--format "%i|%T"'])
    if result.status != 0:
        raise RuntimeError(f"squeue failed: {result.stderr.strip()}")
    states = {}
    for line in result.stdout.splitlines():
        element, _, state = line.strip().partition("|")
        if element:
            states[element] = state.lower()
    return states

