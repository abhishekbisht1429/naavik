"""Launching batches of training experiments on a target.

The workstation compiles a batch manifest and a target profile into an
immutable `ExecutionPlan` (`compiler.compile_plan`); `validate` and `render`
stop there. Nothing here depends on a particular scheduler: schedulers plug
in through `scheduler.SCHEDULERS` (Slurm lives in `naavik.hpc`).
See `.context/plan-hpc-batch-training.md` for the protocol.
"""

from naavik.launcher.errors import HpcError, PlanError

EXECUTION_PROTOCOL_VERSION = 2
"""The contract between the workstation that compiles a plan and the remote
side that consumes it. Any change to the plan files, record formats, or
wrapper/task arguments bumps it. Version 2: the host runs the framework
under its name naavik (version 1 ran it as training_framework)."""

#: What a plan of each protocol version runs on the host: its task entry
#: point and its data module. A mission keeps running the modules its own
#: plan's version names (its environment may hold only those), so resume,
#: preflight and login-node steps ask by the plan's version.
TASK_MODULES = {1: "training_framework.hpc.task", 2: "naavik.launcher.task"}
DATA_MODULES = {1: "training_framework.data", 2: "naavik.data"}


def task_module(protocol_version: int) -> str:
    """The task entry point a plan of `protocol_version` runs."""
    try:
        return TASK_MODULES[protocol_version]
    except KeyError:
        raise HpcError(f"the plan uses execution protocol {protocol_version}, which this "
                       f"naavik does not know (it knows {sorted(TASK_MODULES)})") from None


__all__ = ["DATA_MODULES", "EXECUTION_PROTOCOL_VERSION", "HpcError", "PlanError",
           "TASK_MODULES", "task_module"]
