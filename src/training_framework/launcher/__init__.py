"""Launching batches of training experiments on a target.

The workstation compiles a batch manifest and a target profile into an
immutable `ExecutionPlan` (`compiler.compile_plan`); `validate` and `render`
stop there. Nothing here depends on a particular scheduler: schedulers plug
in through `scheduler.SCHEDULERS` (Slurm lives in `training_framework.hpc`).
See `.context/plan-hpc-batch-training.md` for the protocol.
"""

from training_framework.launcher.errors import HpcError, PlanError

EXECUTION_PROTOCOL_VERSION = 1
"""The contract between the workstation that compiles a plan and the remote
side that consumes it. Any change to the plan files, record formats, or
wrapper/task arguments bumps it."""

__all__ = ["EXECUTION_PROTOCOL_VERSION", "HpcError", "PlanError"]
