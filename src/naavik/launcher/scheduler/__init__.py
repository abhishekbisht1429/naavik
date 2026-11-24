"""The scheduler composition point.

Everything scheduler-specific comes from one bundle here: how its profile
block is parsed, its pure dialect (compiler), its submitter and owner probe
(submit). Adding a scheduler means adding a bundle; the schema, compiler,
and CLI only look bundles up. The remote, standard-library side has the
matching table in `remote/runtime.py` (`JOB_SCHEDULERS`). These two tables
are the only places in the launcher that name a scheduler; the Slurm code
itself lives in `naavik.hpc`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from naavik.launcher.errors import HpcError


@dataclass(frozen=True)
class SchedulerBundle:
    name: str
    # (raw, where, problems, reader_class, names="hosts_file"|"plan") -> settings;
    # "plan": the names a plan records (launcher-internal, kept for resume).
    parse_settings: Callable[..., Any]
    dialect: Callable[[Any], Any]          # settings -> SchedulerDialect
    submitter: Callable[[Any], Any]        # RemoteExecutor -> SchedulerSubmitter
    owner_probe: Callable[[Any], Any]      # RemoteExecutor -> LeaseOwnerProbe
    #: RemoteExecutor -> (job refs -> {job or array element id: state}).
    job_states: Callable[[Any], Callable[[list[str]], dict]] | None = None
    #: RemoteExecutor -> {job or array element id: state} of all the user's
    #: queued and running jobs, in one call.
    user_job_states: Callable[[Any], dict] | None = None
    #: File name suffix of the rendered job scripts (Slurm: `.sbatch`).
    script_suffix: str = ".sh"
    #: Job name -> how to look for a job whose submission had an uncertain
    #: outcome (it may exist without the receipt knowing its reference).
    reconcile_hint: Callable[[str], str] = lambda job_name: f"look for a job named {job_name}"
    #: (RemoteExecutor, settings, ExecutionPlan) -> problems found on the
    #: target itself before anything is staged (None: nothing to check).
    target_check: Callable[[Any, Any, Any], list[str]] | None = None
    #: ExecutionPlan -> {path relative to the batch directory: content} the
    #: scheduler needs staged with the plan (None: nothing).
    extra_files: Callable[[Any], dict[str, str]] | None = None


#: The scheduler of receipts and plans written before they recorded theirs:
#: Slurm was the only one.
LEGACY_SCHEDULER = "slurm"


def scheduler_of(data) -> SchedulerBundle:
    """The scheduler a receipt or plan (as a mapping) names."""
    return get_scheduler(data.get("scheduler") or LEGACY_SCHEDULER)


def script_path(bundle: SchedulerBundle, node) -> str:
    """Where a job node's script lives, relative to the batch directory."""
    if node.kind == "task_set":
        return f"groups/{node.group}/{node.id}{bundle.script_suffix}"
    return f"nodes/{node.id}/{node.id}{bundle.script_suffix}"


def _slurm() -> SchedulerBundle:
    from naavik.hpc import slurm, slurm_submitter

    return SchedulerBundle(
        name="slurm",
        parse_settings=slurm.parse_settings,
        dialect=slurm.SlurmDialect,
        submitter=slurm_submitter.SlurmSubmitter,
        owner_probe=slurm_submitter.RemoteOwnerProbe,
        job_states=slurm_submitter.job_states,
        user_job_states=slurm_submitter.user_job_states,
        script_suffix=".sbatch",
        reconcile_hint=slurm_submitter.reconcile_hint,
    )


class _Registry(dict):
    """Bundles by name, created on first use (no import cycles)."""

    def __init__(self, factories):
        super().__init__()
        self._factories = dict(factories)

    def __iter__(self):
        return iter(self._factories)

    def __contains__(self, name):
        return name in self._factories

    def __getitem__(self, name):
        if not super().__contains__(name):
            if name not in self._factories:
                raise KeyError(name)
            super().__setitem__(name, self._factories[name]())
        return super().__getitem__(name)

    def __setitem__(self, name, bundle):
        self._factories[name] = lambda: bundle
        super().__setitem__(name, bundle)

    def __delitem__(self, name):
        self._factories.pop(name)
        super().pop(name, None)

    def keys(self):
        return self._factories.keys()


def _host() -> SchedulerBundle:
    from naavik.launcher.scheduler import host, host_submitter

    return SchedulerBundle(
        name="host",
        parse_settings=host.parse_settings,
        dialect=host.HostDialect,
        submitter=host_submitter.HostSubmitter,
        owner_probe=host_submitter.owner_probe,
        job_states=host_submitter.job_states,
        user_job_states=host_submitter.user_job_states,
        reconcile_hint=host_submitter.reconcile_hint,
        target_check=host_submitter.target_check,
        extra_files=host_submitter.extra_files,
    )


SCHEDULERS = _Registry({"slurm": _slurm, "host": _host})


def get_scheduler(name: str) -> SchedulerBundle:
    try:
        return SCHEDULERS[name]
    except KeyError:
        raise HpcError(f"unknown scheduler '{name}' (known: {', '.join(SCHEDULERS)})") from None
