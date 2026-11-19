"""The `host` scheduler's dialect: pure validation and job-script rendering
for a machine without a scheduler (this one, or one reached over SSH).

A job script has the same shape as Slurm's: `#TFHOST key=value` directive
lines, the node's execution context, and the same task body (indexed by
`$TF_HOST_TASK_INDEX`), so the cooperative stop, records, and logs are the
ones Slurm batches use. The runner (`remote/host_runner.py`) reads the
directives and does what Slurm would: task sets, `max_parallel`, GPUs from
the profile's pool, the advance signal before the walltime, dependencies.
CPUs and memory are recorded in the plan but not enforced.
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


@dataclass(frozen=True)
class HostSettings:
    #: GPU indices (as nvidia-smi numbers them) jobs may use, in order.
    gpus: tuple[int, ...] = ()
    #: `wait`: skip a GPU on which any compute process runs (anyone's);
    #: `use`: only the pool's own slots decide.
    busy_gpus: str = "wait"


#: `scheduler.when_gpus_busy` as written -> its value inside.
WHEN_GPUS_BUSY = {"wait": "wait", "use_anyway": "use"}


def parse_settings(raw, where, problems, reader_class, names: str = "hosts_file") -> HostSettings:
    """The settings of a host with `scheduler: {kind: builtin, ...}`
    (`names="plan"`: as a plan records them: `busy_gpus: wait|use`)."""
    busy_key, busy_values = (("when_gpus_busy", WHEN_GPUS_BUSY) if names == "hosts_file"
                             else ("busy_gpus", {mode: mode for mode in BUSY_GPU_MODES}))
    reader = reader_class(raw if raw is not None else {}, where, problems, {"gpus", busy_key})
    gpus = reader.get("gpus", [])
    if (not isinstance(gpus, (list, tuple))
            or not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0
                       for item in gpus)):
        problems.add(f"{where}.gpus must be a list of GPU indices (integers >= 0)")
        gpus = []
    elif len(set(gpus)) != len(gpus):
        problems.add(f"{where}.gpus lists a GPU more than once")
    busy = reader.choice(busy_key, tuple(busy_values), "wait")
    return HostSettings(gpus=tuple(gpus), busy_gpus=busy_values[busy or "wait"])


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
        if resources.gpus > len(self.settings.gpus):
            problems.append(
                f"{where}: requests {resources.gpus} GPU(s) but the profile's host.gpus "
                f"pool has {len(self.settings.gpus)}")
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
            "gpu-pool": ",".join(str(index) for index in self.settings.gpus),
            "busy-gpus": self.settings.busy_gpus,
            "walltime": str(resources.walltime_seconds) if resources is not None else "",
            "signal-lead": "" if signal_lead is None else str(signal_lead),
            "log": log,
            "chdir": node.context.working_directory,
        }
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
