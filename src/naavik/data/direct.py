"""Declared data in a direct engine run: before the engine builds a new run's
sessions, its parent process ensures every requirement whose component
config asks for a download (`download: true`), under the lease and ready
marker `ensure` uses everywhere. Two runs on one data path then fetch once;
the second waits for the first.

Not here: batch tasks (their target's `data_modes` saw to their data),
components that do not declare data, and configs with `download: false`
(their constructor reads what is there, as before).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from typing import Any

#: Set for an engine run that is a batch task (`runs/recorder.py`).
BATCH_TASK_CONTEXT = "NAAVIK_RUN_CONTEXT"


def _log(message: str) -> None:
    print(f"[data] {message}", file=sys.stderr, flush=True)


def asks_for_download(config: Any) -> bool:
    value = config.get("download") if isinstance(config, Mapping) else getattr(config, "download", None)
    return value is True


def ensure_downloads(sessions: list[tuple[Mapping, str | None]],
                     notify: Callable[[str], None] = _log) -> list[str]:
    """Ensure the downloads of every session (config, session type) of a new
    run; returns the locations made ready. A session whose config does not
    plan is skipped: building it reports why."""
    if os.environ.get(BATCH_TASK_CONTEXT):
        return []
    from naavik.data import ensure, requirements_of
    from naavik.session import check_session_config

    done: list[str] = []
    for config, session_type in sessions:
        check = check_session_config(config, session_type)
        if check.problems:
            continue
        for planned in check.components:
            for located in requirements_of(planned.component_class, planned.config):
                if located.location in done or not asks_for_download(located.config):
                    continue
                outcome = ensure(planned.component_class, located, notify=notify)
                if outcome == "fetched":
                    notify(f"{located.requirement.name} is ready at {located.location}")
                done.append(located.location)
    return done
