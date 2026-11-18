"""`cancel` and `logs`: acting on a launched batch's jobs, task by task.

A task's job in an attempt is the group's job (attempt 1) or a resumed
attempt's job that ran it; its log is `logs/<group>/<job>_<index>.out` and a
step's (setup, prepare) is `logs/<step>/<job>.out`, on every scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass

from training_framework.launcher.errors import HpcError
from training_framework.launcher.services.resume import task_index
from training_framework.launcher.services.status import expand_ranges


@dataclass(frozen=True)
class TaskJob:
    task: str
    group: str
    index: int
    attempt: int
    node_id: str
    job_ref: str

    @property
    def element(self) -> str:
        """The scheduler's name for this task's element of the job."""
        return f"{self.job_ref}_{self.index}"

    @property
    def log(self) -> str:
        return f"logs/{self.group}/{self.job_ref}_{self.index}.out"


def task_jobs(receipt: dict, tables: dict[str, str]) -> dict[str, list[TaskJob]]:
    """Task name -> the jobs that ran it, by attempt (first to latest)."""
    found: dict[str, list[TaskJob]] = {}
    for name, (group, index) in task_index(tables).items():
        jobs = []
        for node_id, node in receipt["nodes"].items():
            ref = node.get("job_ref")
            if not ref:
                continue
            if node_id == f"group-{group}" and not node.get("attempt"):
                jobs.append(TaskJob(name, group, index, 1, node_id, ref))
            elif (node.get("attempt") and node.get("group") == group
                  and index in node.get("tasks", [])):
                jobs.append(TaskJob(name, group, index, node["attempt"], node_id, ref))
        found[name] = sorted(jobs, key=lambda job: job.attempt)
    return found


def _steps(receipt: dict) -> dict[str, dict]:
    return {node_id: node for node_id, node in receipt["nodes"].items()
            if not node_id.startswith("group-")}


# logs ----------------------------------------------------------------------------------


def log_path(receipt: dict, tables: dict[str, str], name: str, attempt: int | None = None) -> str:
    """The log of task (or step) `name`, relative to the batch directory: its
    latest attempt's, or attempt `attempt`'s."""
    steps = _steps(receipt)
    if name in steps:
        if attempt not in (None, 1):
            raise HpcError(f"{name} is a step; it has no attempt {attempt}")
        ref = steps[name].get("job_ref")
        if not ref:
            raise HpcError(f"{name} did not run as a job ({steps[name].get('status')}); it has no log")
        return f"logs/{name}/{ref}.out"
    jobs = task_jobs(receipt, tables)
    if name not in jobs:
        known = sorted(jobs) + sorted(steps)
        raise HpcError(f"'{name}' is not an experiment or step of this mission (have: {', '.join(known)})")
    if not jobs[name]:
        raise HpcError(f"{name} never reached the scheduler; it has no log")
    if attempt is None:
        return jobs[name][-1].log
    chosen = [job for job in jobs[name] if job.attempt == attempt]
    if not chosen:
        raise HpcError(f"{name} has no attempt {attempt} (attempts: "
                       f"{', '.join(str(job.attempt) for job in jobs[name])})")
    return chosen[0].log


# cancel --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cancel:
    """One thing to cancel: a job (`task` None) or one task's element."""
    label: str
    job_ref: str
    node_id: str
    task: str | None = None
    attempt: int | None = None


def to_cancel(receipt: dict, tables: dict[str, str], names: list[str],
              live: dict) -> tuple[list[Cancel], list[str]]:
    """What `cancel` acts on, and notes on what it leaves (already ended).
    Without names: every job of the batch still queued or running. With
    names: each task's latest job element, if it is still queued or running.
    `live` is the scheduler's states of jobs and elements."""
    live = expand_ranges(live)
    notes = []
    if not names:
        chosen = []
        for node_id, node in receipt["nodes"].items():
            ref = node.get("job_ref")
            if not ref or node.get("status") != "submitted":
                continue
            if ref in live or any(key.startswith(f"{ref}_") for key in live):
                chosen.append(Cancel(node_id, ref, node_id))
            else:
                notes.append(f"{node_id}: job {ref} already ended")
        return chosen, notes
    jobs = task_jobs(receipt, tables)
    unknown = [name for name in names if name not in jobs]
    if unknown:
        raise HpcError(f"not experiments of this mission: {', '.join(unknown)}")
    chosen = []
    for name in dict.fromkeys(names):
        if not jobs[name]:
            notes.append(f"{name}: never reached the scheduler")
            continue
        job = jobs[name][-1]
        if job.element in live:
            chosen.append(Cancel(name, job.element, job.node_id, name, job.attempt))
        else:
            notes.append(f"{name}: its job {job.element} already ended")
    return chosen, notes


def record_cancel(receipt, item: Cancel, ok: bool, detail: str) -> None:
    """Into the receipt: a cancelled job's node, or a cancelled task."""
    if item.task is None:
        receipt.node_cancelled(item.node_id, ok, detail)
    else:
        receipt.task_cancelled(item.task, item.attempt, ok, detail)
