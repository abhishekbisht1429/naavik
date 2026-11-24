"""Per-task launch checks: DDP topology against the group's resources, and
the engine options a task may be started with."""

from __future__ import annotations

import math
from collections.abc import Mapping

from naavik.launcher import framework
from naavik.launcher.config.schema import Resources

#: Engine options a manifest may pass, each taking one value. Everything
#: else is refused: `--config`, `--resume-session`, `--extend-session`, and
#: `--override` would make the task run something other than its config,
#: `--debug` bypasses the supervision loop that handles interrupts, and
#: argparse prefix abbreviations would hide either.
ENGINE_OPTIONS = (
    "--heartbeat-timeout",
    "--stop-sync-grace-period",
    "--stop-sync-poll-interval",
    "--process_timeout_on_join",
)
DEFAULT_PROCESS_TIMEOUT_ON_JOIN = 30.0


def check_topology(session: Mapping, resources: Resources) -> list[str]:
    return _device_problems(session, resources) + _ddp_problems(session, resources)


def _device_problems(session: Mapping, resources: Resources) -> list[str]:
    """`session_config.device`: a device a session can name at all, and
    against the GPUs the task will see: its group's, renumbered from 0 (so
    `cuda:N` needs N < gpus), and none in a group that asks for none. Caught
    here, not in the running task."""
    config = session.get("session_config")
    device = config.get("device") if isinstance(config, Mapping) else None
    if not isinstance(device, str):
        return []
    spelling_problem = framework.device_spelling_problem(device)
    if spelling_problem is not None:
        return [spelling_problem]
    if not device.startswith("cuda"):
        return []
    gpus = resources.gpus
    if gpus == 0:
        return [f"session_config.device is {device!r} but the group requests no GPUs"]
    _, _, index = device.partition(":")
    if index.isdigit() and int(index) >= gpus:
        return [
            f"session_config.device is {device!r} but the task sees only its group's "
            f"{gpus} GPU(s), numbered from 0; use 'cuda' (or cuda:0..cuda:{gpus - 1})"
        ]
    return []


def _ddp_problems(session: Mapping, resources: Resources) -> list[str]:
    gpus = resources.gpus
    if not framework.is_listed_component(session, "ddp"):
        if gpus > 1:
            return [
                f"requests {gpus} GPUs but configures no 'ddp'; a single "
                "process uses at most one"
            ]
        return []
    ddp = framework.component_entry_of(session, "ddp")
    if not isinstance(ddp, Mapping):
        return ["'ddp' must be configured with a mapping"]
    world_size = ddp.get("world_size")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        return [
            "ddp.world_size must be set explicitly to a positive integer "
            f"(got {world_size!r}); it is never derived from the resources"
        ]
    backend = ddp.get("backend")
    if backend == "nccl":
        if gpus < 1 or world_size != gpus:
            return [
                f"ddp backend 'nccl' needs world_size == gpus >= 1; got "
                f"world_size {world_size}, gpus {gpus}"
            ]
        return []
    if backend not in (None, "gloo"):
        return [f"ddp backend {backend!r} is not supported for HPC batches (nccl, gloo)"]
    problems = []
    if backend is None and gpus >= 1:
        problems.append(
            "the group requests GPUs but ddp.backend is not set (the engine "
            "default is 'gloo'); set it explicitly"
        )
    if resources.cpus is not None and world_size > resources.cpus:
        problems.append(
            f"ddp backend 'gloo' runs world_size {world_size} processes but "
            f"the group requests only {resources.cpus} CPUs"
        )
    return problems


def parse_engine_options(options: tuple[str, ...]) -> tuple[float, list[str]]:
    """The `--process_timeout_on_join` value the task will use, and the
    problems with `options`."""
    problems: list[str] = []
    timeout = DEFAULT_PROCESS_TIMEOUT_ON_JOIN
    index = 0
    while index < len(options):
        token = options[index]
        name, _, inline = token.partition("=")
        if name not in ENGINE_OPTIONS:
            problems.append(
                f"training.engine_options: '{token}' is not allowed (allowed: "
                f"{', '.join(ENGINE_OPTIONS)}; config changes belong in "
                "overrides)"
            )
            index += 1
            continue
        if inline:
            value = inline
            index += 1
        elif index + 1 < len(options):
            value = options[index + 1]
            index += 2
        else:
            problems.append(f"training.engine_options: '{name}' needs a value")
            break
        try:
            number = float(value)
        except ValueError:
            number = math.nan
        if not math.isfinite(number) or number < 0:
            problems.append(
                f"training.engine_options: '{name}' needs a finite "
                f"non-negative number; got {value!r}"
            )
        elif name == "--process_timeout_on_join":
            timeout = number
    return timeout, problems
