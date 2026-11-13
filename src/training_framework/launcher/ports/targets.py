"""Where a batch runs, and the ports that reach it.

A `Target` is how the workstation reaches the machine a batch runs on. It
comes from a profile (when launching) or from a receipt (afterwards: the
batch is followed where it went, whatever the current profile says). Every
command builds its executor, filesystem, and file transfer here, so a new
way of reaching a machine is added in one place.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable

from training_framework.launcher.ports.ssh import RsyncTransfer, SshExecutor, SshFilesystem

#: How a target is reached.
TRANSPORTS = ("ssh",)


@dataclass(frozen=True)
class Target:
    transport: str
    #: The machine: for `ssh`, the alias from ~/.ssh/config.
    host: str

    def __post_init__(self):
        if self.transport not in TRANSPORTS:
            raise ValueError(f"unknown transport {self.transport!r} (known: {', '.join(TRANSPORTS)})")

    def __str__(self) -> str:
        return self.host

    def tools(self, *, transfer: bool) -> tuple[str, ...]:
        """Programs this machine needs to reach the target (`transfer`: to
        copy files too, not only to run commands)."""
        return ("ssh", "rsync") if transfer else ("ssh",)


def target_of_profile(profile) -> Target:
    return Target("ssh", profile.ssh_host)


def target_of_receipt(receipt: dict) -> Target | None:
    """The target a receipt's batch went to; None for receipts written
    before receipts recorded it."""
    host = receipt.get("ssh_host")
    return Target("ssh", host) if host else None


@dataclass(frozen=True)
class Ports:
    executor: Any
    filesystem: Any
    transfer: Any


def ports_for(target: Target, progress: Callable[[str], None] | None = None) -> Ports:
    """The executor, filesystem, and file transfer reaching `target`.
    `progress` receives the transfer's periodic progress lines when standard
    error is not a terminal (an IDE console, a log); on a terminal the
    transfer shows its own live line."""
    executor = SshExecutor(target.host)
    if sys.stderr.isatty() or progress is None:
        transfer = RsyncTransfer(target.host, show_progress=sys.stderr.isatty())
    else:
        transfer = RsyncTransfer(target.host, report=progress)
    return Ports(executor, SshFilesystem(executor), transfer)
