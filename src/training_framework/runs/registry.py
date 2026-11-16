"""Writing and reading run records (see the package docstring for the
layout). The launching process writes `run.json` and `events.jsonl`; each
worker writes its own `rank-<r>.json`; the CLI only reads."""

from __future__ import annotations

import json
import os
import socket
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from training_framework.runs.ids import is_run_id

ENV_DIRECTORY = "TRAINING_FRAMEWORK_RUNS"
#: `TRAINING_FRAMEWORK_RUNS=off` turns the registry off.
OFF = "off"
RUN_FILE = "run.json"
EVENTS_FILE = "events.jsonl"
CONFIGS = "configs"

#: Run states, derived (never stored): from the `finished` event, else from
#: whether the launching process still runs.
DONE, FAILED, INTERRUPTED = "done", "failed", "interrupted"
RUNNING, ENDED_WITHOUT_RECORD, UNKNOWN = "running", "ended without a record", "unknown"
_FINISHED_STATES = {"ok": DONE, "failed": FAILED, "interrupted": INTERRUPTED}


def registry_dir() -> Path | None:
    """The registry of this machine, or None when it is turned off."""
    configured = os.environ.get(ENV_DIRECTORY)
    if configured == OFF:
        return None
    if configured:
        return Path(configured)
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return Path(base) / "training-framework" / "runs"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    # Values that are not JSON (a path, a dtype) are recorded as their text.
    return json.dumps(value, sort_keys=True, default=str)


def _write_once(path: Path, data: bytes) -> None:
    """Atomic, and never over an existing file (records are immutable)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)          # fails if the file exists
    finally:
        os.unlink(temporary)


def config_sha256(config: Mapping) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


# Writing --------------------------------------------------------------------------


class RunWriter:
    """The launching process's handle on its run's record."""

    def __init__(self, root: Path, run_id: str):
        if not is_run_id(run_id):
            raise ValueError(f"{run_id!r} is not a run id")
        self.root = Path(root)
        self.run_id = run_id
        self.directory = self.root / run_id

    @classmethod
    def create(cls, root: Path, record: Mapping) -> "RunWriter":
        """Start a run's record: `record` becomes its `run.json`."""
        writer = cls(root, record["run_id"])
        writer.directory.mkdir(parents=True, exist_ok=False)
        _write_once(writer.directory / RUN_FILE, (_json(dict(record)) + "\n").encode())
        return writer

    def store_config(self, config: Mapping) -> str:
        """Store a resolved config once (by content); returns its sha256."""
        sha = config_sha256(config)
        path = self.root / CONFIGS / f"{sha}.yaml"
        if not path.exists():
            try:
                _write_once(path, yaml.safe_dump(json.loads(_json(config)), sort_keys=False).encode())
            except FileExistsError:        # another run stored the same config just now
                pass
        return sha

    def event(self, event: str, **fields) -> None:
        """Append one event (only the launching process writes events)."""
        line = _json({"event": event, "at": now(), **fields}) + "\n"
        with open(self.directory / EVENTS_FILE, "a") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def write_rank(run_directory: Path, rank: int, facts: Mapping) -> None:
    """A worker's own record (each rank writes its file once)."""
    _write_once(Path(run_directory) / f"rank-{rank}.json",
                (_json({"rank": rank, "at": now(), **facts}) + "\n").encode())


# Reading --------------------------------------------------------------------------


@dataclass
class Run:
    run_id: str
    directory: Path
    record: dict
    events: list[dict] = field(default_factory=list)
    ranks: dict[int, dict] = field(default_factory=dict)
    #: What could not be read (a damaged line or file), kept, not hidden.
    problems: list[str] = field(default_factory=list)

    @property
    def finished(self) -> dict | None:
        return next((e for e in reversed(self.events) if e.get("event") == "finished"), None)

    @property
    def session_dirs(self) -> list[str]:
        return [e["session_dir"] for e in self.events
                if e.get("event") == "session" and e.get("session_dir")]

    @property
    def state(self) -> str:
        finished = self.finished
        if finished is not None:
            return _FINISHED_STATES.get(finished.get("status"), UNKNOWN)
        if self.record.get("host") != socket.gethostname():
            return UNKNOWN                 # another machine's run: sync it again
        from training_framework.launcher.remote.runtime import process_start_time

        start = process_start_time(self.record.get("pid"))
        if start is not None and start == self.record.get("process_start_time"):
            return RUNNING
        return ENDED_WITHOUT_RECORD

    def config(self, root: Path, index: int = 0) -> dict | None:
        sessions = self.record.get("sessions") or []
        if index >= len(sessions):
            return None
        path = Path(root) / CONFIGS / f"{sessions[index]['config_sha256']}.yaml"
        return yaml.safe_load(path.read_text()) if path.exists() else None


def read_run(directory: Path) -> Run:
    directory = Path(directory)
    record = json.loads((directory / RUN_FILE).read_text())
    run = Run(record["run_id"], directory, record)
    events = directory / EVENTS_FILE
    if events.exists():
        for number, line in enumerate(events.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                run.events.append(json.loads(line))
            except ValueError:
                run.problems.append(f"{EVENTS_FILE} line {number} is damaged")
    for path in sorted(directory.glob("rank-*.json")):
        try:
            data = json.loads(path.read_text())
            run.ranks[int(data["rank"])] = data
        except (ValueError, KeyError, TypeError):
            run.problems.append(f"{path.name} is damaged")
    return run


def list_runs(root: Path) -> tuple[list[Run], list[str]]:
    """Every run of a registry, newest first, and what could not be read."""
    runs, problems = [], []
    root = Path(root)
    if not root.exists():
        return runs, problems
    for directory in sorted(root.iterdir(), reverse=True):
        if not is_run_id(directory.name):
            continue
        try:
            runs.append(read_run(directory))
        except (OSError, ValueError, KeyError) as error:
            problems.append(f"{directory.name}: {type(error).__name__}: {error}")
    return runs, problems


def find_run(root: Path, prefix: str) -> Run:
    """The one run whose id starts with `prefix` (case-insensitive)."""
    prefix = prefix.upper()
    matches = [path for path in Path(root).iterdir()
               if is_run_id(path.name) and path.name.startswith(prefix)] if Path(root).exists() else []
    if len(matches) != 1:
        found = "no run" if not matches else f"{len(matches)} runs ({', '.join(sorted(p.name for p in matches)[:5])})"
        raise LookupError(f"{found} with an id starting with {prefix!r} in {root}")
    return read_run(matches[0])


def parent_of(run: Run, runs: list[Run]) -> Run | None:
    """The run a resume or extension continues: the latest earlier run whose
    sessions include a session directory this run restored."""
    if (run.record.get("launch") or {}).get("operation") not in ("resume", "extend"):
        return None
    mine = set(run.session_dirs)
    earlier = [other for other in runs if other.run_id < run.run_id and mine & set(other.session_dirs)]
    return max(earlier, key=lambda other: other.run_id) if earlier else None
