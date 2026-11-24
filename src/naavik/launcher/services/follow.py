"""Follow one task of a launched batch: print its log as it grows, until the
task has finished, then give its exit status.

Works on every target: the log is read through the target's executor (new
bytes since the last read) and the task's state comes from the same rows as
`naavik status` (the scheduler's queue, then the task's record, which the
task wrapper writes only when it ends).

While the task waits to start, why it waits is said once, and again when
it changes (the builtin scheduler says: GPUs, `max_parallel`, a job it runs
after).

Ctrl-C: the first cancels the task (as `naavik cancel --task`) and keeps
following until it has stopped; the second stops following (the cancel
stands, the task is not waited for).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable

from naavik.launcher.errors import HpcError

#: Exit status of a follow that was detached, or of a task that ended without
#: an exit status of its own (cancelled, no record).
DETACHED = 130
NO_EXIT_STATUS = 1


@dataclass(frozen=True)
class Outcome:
    #: The task's exit status, `NO_EXIT_STATUS` when it has none, or
    #: `DETACHED` when following stopped first.
    status: int
    #: The task's final state as `naavik status` shows it (None: detached).
    state: str | None
    cancelled: bool = False


class Follower:
    """`rows()` -> the batch's rows (`services.status.Row`), `log()` -> the
    task's log path or None while it has none, `read(path, offset)` -> the
    bytes from `offset` (b"" when there are none yet), `cancel()` asks the
    scheduler to stop the task."""

    def __init__(self, *, task: str, rows: Callable, log: Callable, read: Callable,
                 cancel: Callable, write: Callable[[bytes], None], say: Callable[[str], None],
                 interval: float = 1.0, sleep: Callable[[float], None] = time.sleep):
        self.task, self.rows, self.log, self.read = task, rows, log, read
        self.cancel, self.write, self.say = cancel, write, say
        self.interval, self.sleep = interval, sleep
        self.offset = 0
        self.cancelled = False
        #: The last reason for waiting that was said (said again on change).
        self.waiting: str | None = None

    def _row(self):
        for row in self.rows():
            if row.name == self.task:
                return row
        raise HpcError(f"'{self.task}' is not an experiment of this mission")

    def _drain(self) -> None:
        path = self.log()
        if path is None:
            return
        chunk = self.read(path, self.offset)
        if chunk:
            self.offset += len(chunk)
            self.write(chunk)

    def _step(self) -> Outcome | None:
        row = self._row()
        if row.waiting and row.waiting != self.waiting:
            self.say(f"{self.task}: {row.waiting}")
        self.waiting = row.waiting
        # Read after the row: once the task has ended, this has its last line.
        self._drain()
        if row.live:
            return None
        return Outcome(_exit_status(row.state), row.state, self.cancelled)

    def run(self) -> Outcome:
        while True:
            try:
                outcome = self._step()
                if outcome is not None:
                    return outcome
                self.sleep(self.interval)
            except KeyboardInterrupt:
                if self.cancelled:
                    self.say("detached; the experiment was asked to stop")
                    return Outcome(DETACHED, None, True)
                self.cancelled = True
                self.say("cancelling the experiment (Ctrl-C again to stop following)")
                try:
                    self.cancel()
                except KeyboardInterrupt:
                    self.say("detached; the cancel may not have reached the scheduler")
                    return Outcome(DETACHED, None, True)


_FAILED = re.compile(r"\Afailed \(exit (-?\d+)\)\Z")


def _exit_status(state: str) -> int:
    """The exit status a finished task's state stands for."""
    if state == "done":
        return 0
    if match := _FAILED.match(state):
        status = int(match.group(1))
        return status if status > 0 else NO_EXIT_STATUS
    return NO_EXIT_STATUS          # interrupted, cancelled, failed, ended without a record
