"""Where a batch runs, and the ports that reach it.

A `Target` is how the workstation reaches the machine a batch runs on. It
comes from a profile (when launching) or from a receipt (afterwards: the
batch is followed where it went, whatever the current profile says). Every
command builds its executor, filesystem, and file transfer here, so a new
way of reaching a machine is added in one place.
"""

from __future__ import annotations

import socket
import sys
from dataclasses import dataclass
from typing import Any, Callable

from training_framework.launcher.errors import HpcError
from training_framework.launcher.ports.local import LocalExecutor, LocalFilesystem, LocalFileTransfer
from training_framework.launcher.ports.ssh import RsyncTransfer, SshExecutor, SshFilesystem

#: How a target is reached: over SSH, or this machine itself.
TRANSPORTS = ("ssh", "local")


@dataclass(frozen=True)
class Target:
    transport: str
    #: The machine: for `ssh`, the alias from ~/.ssh/config; for `local`,
    #: this machine's host name (recorded, so a receipt says where it ran).
    host: str

    def __post_init__(self):
        if self.transport not in TRANSPORTS:
            raise ValueError(f"unknown transport {self.transport!r} (known: {', '.join(TRANSPORTS)})")

    def __str__(self) -> str:
        return self.host

    @property
    def is_local(self) -> bool:
        return self.transport == "local"

    def tools(self, *, transfer: bool) -> tuple[str, ...]:
        """Programs this machine needs to reach the target (`transfer`: to
        copy files too, not only to run commands)."""
        if self.is_local:
            return ()
        return ("ssh", "rsync") if transfer else ("ssh",)

    def as_json(self) -> dict:
        return {"transport": self.transport, "host": self.host}

    def receipt_fields(self) -> dict:
        """What a receipt records about its target: one key, so it has one
        source of truth. SSH receipts keep the `ssh_host` that earlier
        versions write and read; other transports record `target`."""
        if self.transport == "ssh":
            return {"ssh_host": self.host}
        return {"target": self.as_json()}


def this_machine() -> Target:
    return Target("local", socket.gethostname())


def target_of_profile(profile) -> Target:
    return this_machine() if profile.local else Target("ssh", profile.ssh_host)


def target_of_receipt(receipt: dict) -> Target | None:
    """The target a receipt's batch went to; None for receipts written
    before receipts recorded it. `ssh_host` means SSH; `target` names any
    other transport; a receipt with both is refused."""
    recorded = receipt.get("target")
    if recorded is not None and receipt.get("ssh_host"):
        raise HpcError("the receipt records both ssh_host and target; it can only have one")
    if recorded is not None:
        try:
            return Target(recorded["transport"], recorded["host"])
        except (KeyError, TypeError, ValueError) as error:
            raise HpcError(f"the receipt's target {recorded!r} is not valid ({error})") from None
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
    if target.is_local:
        return Ports(LocalExecutor(), LocalFilesystem(), LocalFileTransfer())
    executor = SshExecutor(target.host)
    if sys.stderr.isatty() or progress is None:
        transfer = RsyncTransfer(target.host, show_progress=sys.stderr.isatty())
    else:
        transfer = RsyncTransfer(target.host, report=progress)
    return Ports(executor, SshFilesystem(executor), transfer)
