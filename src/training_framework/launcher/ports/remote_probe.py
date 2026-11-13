"""Liveness of a process owner on the target, checked through its executor."""

from __future__ import annotations

from training_framework.launcher.ports import Owner


class RemoteProcessProbe:
    """`LeaseOwnerProbe` for process owners on the target: True or False only
    when the command runs on the owner's own host, None otherwise (and for
    job owners, which belong to a scheduler's probe)."""

    def __init__(self, executor):
        self.executor = executor

    def is_alive(self, owner: Owner) -> bool | None:
        if owner.job_ref is not None:
            return None
        if owner.pid is None:
            return None
        result = self.executor.run(
            ["sh", "-c", 'hostname; cat "/proc/$1/stat" 2>/dev/null || true', "sh", str(owner.pid)])
        lines = result.stdout.splitlines()
        if not lines or lines[0].strip() != owner.host:
            return None
        if len(lines) < 2 or ")" not in lines[1]:
            return False
        return lines[1].rsplit(")", 1)[1].split()[19] == owner.start_time
