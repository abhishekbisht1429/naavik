"""The `host` scheduler's dialect: pure validation and job-script rendering
for a machine without a scheduler (this one, or one reached over SSH).

A job script has the same shape as Slurm's: `#TFHOST key=value` directive
lines, the node's execution context, and the same task body (indexed by
`$TF_HOST_TASK_INDEX`), so the cooperative stop, records, and logs are the
ones Slurm batches use. The runner (`remote/host_runner.py`) reads the
directives and does what Slurm would: task sets, `max_parallel`, GPUs from
the host's pool, the advance signal before the walltime, dependencies.
CPUs and memory are never enforced: given, they are recorded in the plan;
absent, an experiment uses what the machine has. Without a walltime an
experiment runs until it finishes.

The GPU pool (`scheduler.gpus`) is `all` (the default: every GPU of the
machine), a count N (any N of them at once, whichever are free), or a list of
indices (only those). It is also the limit: the most GPUs held at once by all
of the user's experiments on the machine.
"""

from __future__ import annotations

import math
import shlex
from dataclasses import dataclass

from naavik.launcher.compile.plan import JobNode
from naavik.launcher.config.schema import Resources
from naavik.launcher.ports.shell import launched, render_context
from naavik.launcher.scheduler.common import task_body

#: The variable holding a task-set element's index in its job script.
INDEX_VARIABLE = "TF_HOST_TASK_INDEX"
DIRECTIVE = "#TFHOST"
BUSY_GPU_MODES = ("wait", "use")


#: `scheduler.gpus: all`: every GPU of the machine.
ALL_GPUS = "all"


@dataclass(frozen=True)
class HostSettings:
    #: The GPU pool, which is also the limit on GPUs held at once:
    #: `"all"`, a count (any that many), or a tuple of nvidia-smi indices
    #: (only those). A plan records it as written (a list for indices).
    gpus: tuple[int, ...] | int | str = ALL_GPUS
    #: `wait`: skip a GPU on which any compute process runs (anyone's);
    #: `use`: only the pool's own slots decide.
    busy_gpus: str = "wait"

    @property
    def gpu_indices(self) -> tuple[int, ...] | None:
        """The GPUs experiments may use, or None for any of the machine's."""
        return self.gpus if isinstance(self.gpus, tuple) else None

    @property
    def gpu_limit(self) -> int | None:
        """The most GPUs held at once, or None for "as many as the machine
        has" (known only on the host)."""
        if isinstance(self.gpus, tuple):
            return len(self.gpus)
        return None if self.gpus == ALL_GPUS else self.gpus

    def describe_gpus(self) -> str:
        """`scheduler.gpus` as written, for messages."""
        if isinstance(self.gpus, tuple):
            return f"[{', '.join(str(index) for index in self.gpus)}]"
        return str(self.gpus)


#: `scheduler.when_gpus_busy` as written -> its value inside.
WHEN_GPUS_BUSY = {"wait": "wait", "use_anyway": "use"}


def parse_settings(raw, where, problems, reader_class, names: str = "hosts_file") -> HostSettings:
    """The settings of a host with `scheduler: {kind: builtin, ...}`
    (`names="plan"`: as a plan records them: `busy_gpus: wait|use`). A plan
    written before `all` and counts existed records a list, read as before."""
    busy_key, busy_values = (("when_gpus_busy", WHEN_GPUS_BUSY) if names == "hosts_file"
                             else ("busy_gpus", {mode: mode for mode in BUSY_GPU_MODES}))
    reader = reader_class(raw if raw is not None else {}, where, problems, {"gpus", busy_key})
    gpus = _gpus(reader.get("gpus", ALL_GPUS), where, problems)
    busy = reader.choice(busy_key, tuple(busy_values), "wait")
    return HostSettings(gpus=gpus, busy_gpus=busy_values[busy or "wait"])


def _gpus(value, where, problems) -> tuple[int, ...] | int | str:
    if value == ALL_GPUS:
        return ALL_GPUS
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            problems.add(f"{where}.gpus must be all, a count >= 0, or a list of GPU indices; "
                         f"got {value}")
            return ()
        return value
    if (isinstance(value, (list, tuple))
            and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0
                    for item in value)):
        if len(set(value)) != len(value):
            problems.add(f"{where}.gpus lists a GPU more than once")
        return tuple(value)
    problems.add(f"{where}.gpus must be all, a count (any that many GPUs), or a list of GPU "
                 f"indices (integers >= 0); got {value!r}")
    return ()


def _unsafe(value) -> bool:
    """Control characters (a newline would start another directive line)."""
    return any(ord(char) < 32 or ord(char) == 127 for char in str(value))


class HostDialect:
    name = "host"
    #: No array limit: the runner holds a task set's elements itself.
    max_task_set_size = 1_000_000

    def __init__(self, settings: HostSettings):
        self.settings = settings

    def check_profile(self) -> list[str]:
        return []

    def check_resources(self, resources: Resources, where: str) -> list[str]:
        problems = []
        for scheduler in resources.scheduler_options:
            if scheduler == self.name:
                problems.append(f"{where}.scheduler_options.host: the host scheduler takes no options")
            else:
                problems.append(
                    f"{where}.scheduler_options.{scheduler}: options for a "
                    f"scheduler other than the selected '{self.name}'")
        limit = self.settings.gpu_limit
        if limit is not None and resources.gpus > limit:
            # With `all`, the machine's GPUs are counted when `launch` checks
            # the host (`host_submitter.target_check`).
            problems.append(
                f"{where}: requests {resources.gpus} GPU(s) per experiment, but the pool has "
                f"{limit} (scheduler.gpus: {self.settings.describe_gpus()}): at most that many "
                "are in use at once on this host")
        return problems

    def signal_lead(self, process_timeout_on_join: float, stop_grace: int) -> int:
        return int(math.ceil(process_timeout_on_join)) + stop_grace

    def render(self, node: JobNode, batch_id: str, batch_dir: str,
               command: list[str], signal_lead: int | None, *,
               array_indices: list[int] | None = None, resume_table: str | None = None) -> str:
        log_dir = f"{batch_dir}/logs/{node.group or node.id}"
        if node.kind == "task_set":
            elements = sorted(array_indices) if array_indices else list(range(node.task_count))
            log = f"{log_dir}/{{job}}_{{index}}.out"
        else:
            elements = [0]
            log = f"{log_dir}/{{job}}.out"
        resources = node.resources
        directives = {
            "job-name": f"{batch_id}.{node.id}",
            "batch-dir": batch_dir,
            "kind": node.kind,
            "elements": ",".join(str(index) for index in elements),
            "max-parallel": str(node.max_parallel or 0),
            "gpus": str(resources.gpus if resources is not None else 0),
            # `all`, or the indices; a runner from before `all` reads the
            # indices of the plans it was staged with.
            "gpu-pool": (ALL_GPUS if self.settings.gpu_indices is None
                         else ",".join(str(index) for index in self.settings.gpu_indices)),
            "busy-gpus": self.settings.busy_gpus,
            "walltime": (str(resources.walltime_seconds)
                         if resources is not None and resources.walltime_seconds is not None
                         else ""),
            "signal-lead": "" if signal_lead is None else str(signal_lead),
            "log": log,
            "chdir": node.context.working_directory,
        }
        if self.settings.gpu_indices is None and self.settings.gpu_limit is not None:
            # `gpus: N`: at most N held at once, any of the machine's.
            directives["gpu-limit"] = str(self.settings.gpu_limit)
        unsafe = [key for key, value in directives.items() if _unsafe(value)]
        if unsafe:   # validation refuses these first; never render them
            raise ValueError(f"refusing to render directives with control characters: {unsafe!r}")
        lines = ["#!/bin/bash"]
        lines += [f"{DIRECTIVE} {key}={value}" for key, value in directives.items()]
        lines.append("set -eo pipefail")
        lines += render_context(node.context)
        command = launched(node.context, command)
        if node.kind == "task_set":
            lines += task_body(batch_dir, node.group, command, INDEX_VARIABLE, resume_table)
        else:
            lines.append("exec " + shlex.join(command))
        return "\n".join(lines) + "\n"
