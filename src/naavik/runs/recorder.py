"""The engine's side of the registry: start a run's record, add its sessions,
finish it. Every step is fenced, so a registry problem (an unwritable
directory, a broken repository) is one warning on stderr and never fails
the run; after a failure the recorder stops writing.

A batch task tells the engine what it is through `NAAVIK_RUN_CONTEXT` (JSON:
batch, task, attempt, target, and the batch's code), which the remote task
entry point sets; the engine then does not inspect the (staged) code itself.
"""

from __future__ import annotations

import contextlib
import getpass
import json
import os
import socket
import sys
from pathlib import Path

from naavik.runs.ids import new_run_id
from naavik.runs.registry import RunWriter, now, registry_dir

#: Set by the remote task entry point for a batch task (JSON object).
ENV_CONTEXT = "NAAVIK_RUN_CONTEXT"
#: A worker's run directory, in its environment from the moment it is spawned.
ENV_RUN_DIRECTORY = "TRAINING_FRAMEWORK_RUN_DIRECTORY"
_OPERATIONS = {"new": "config", "resume": "resume", "extend": "extend"}


def _warn(message: str) -> None:
    print(f"warning: run registry: {message}", file=sys.stderr, flush=True)


def _components_package(configurator) -> str | None:
    try:
        definition = configurator.session_configs[0]
        return definition["session_config"]["components_package"]
    except Exception:                      # resume/extend, or no such key
        return None


class RunRecorder:
    def __init__(self, writer: RunWriter | None = None):
        self.writer = writer

    @property
    def run_id(self) -> str | None:
        return self.writer.run_id if self.writer else None

    @property
    def directory(self) -> str | None:
        """Where workers write their rank files (None: not recording)."""
        return str(self.writer.directory) if self.writer else None

    def _fenced(self, what: str, action) -> None:
        if self.writer is None:
            return
        try:
            action()
        except Exception as error:
            _warn(f"{what} failed ({type(error).__name__}: {error}); this run's record "
                  f"{self.writer.directory} stays incomplete")
            self.writer = None

    @classmethod
    def start(cls, configurator, argv: list[str] | None = None) -> "RunRecorder":
        """Start the record of a launch (before anything is built, so a
        launch that fails while building is recorded too)."""
        root = registry_dir()
        if root is None:
            return cls()
        try:
            record = cls._record(configurator, argv)
            writer = RunWriter.create(root, record)
        except Exception as error:
            _warn(f"cannot start a record in {root} ({type(error).__name__}: {error}); "
                  "the run goes on unrecorded")
            return cls()
        print(f"run {writer.run_id} (naavik runs show {writer.run_id})", file=sys.stderr, flush=True)
        return cls(writer)

    @staticmethod
    def _record(configurator, argv) -> dict:
        from naavik.launcher.remote.runtime import process_start_time
        from naavik.runs import facts

        run_id = new_run_id()
        mode = getattr(configurator, "mode", None)
        launch = {
            "operation": _OPERATIONS.get(mode, mode),
            "command": list(sys.argv if argv is None else argv),
            "working_directory": os.getcwd(),
        }
        if mode in ("resume", "extend"):
            launch["checkpoint"] = str(getattr(configurator, "checkpoint_path", None))
        record = {"run_id": run_id, "started": now(), "host": socket.gethostname(),
                  "pid": os.getpid(), "process_start_time": process_start_time(os.getpid()),
                  "user": getpass.getuser(), "launch": launch}
        context = os.environ.get(ENV_CONTEXT)
        if context:
            try:
                given = json.loads(context)
                record["code"] = given.pop("code", None)
                launch.update(given)
            except (ValueError, AttributeError) as error:
                record["context_error"] = f"{ENV_CONTEXT} is not a JSON object: {error}"
        if "code" not in record:
            start = facts.package_directory(_components_package(configurator)) or Path.cwd()
            record["code"] = facts.code_identity(start, run_id)
        record["environment"] = facts.environment_facts()
        return record

    def sessions(self, sessions: list[tuple]) -> None:
        """One event per built session: (session, config_sources or None)."""
        def write():
            for index, (session, sources) in enumerate(sessions):
                self.writer.event(
                    "session", index=index, type=session.session_type,
                    session_dir=str(session.session_config.session_dir),
                    config_sha256=self.writer.store_config(session.full_config),
                    config_sources=[source.as_json() if hasattr(source, "as_json") else str(source)
                                    for source in (sources or [])],
                )
        self._fenced("recording the sessions", write)

    def finished(self, status: str, error: str | None = None, workers=()) -> None:
        """The launch's end: its outcome and each worker's last iteration
        (read from its progress once more: the last poll may predate it)."""
        def write():
            for worker in workers:
                worker.check_progress()
            self.writer.event("finished", status=status, error=error,
                              last_iteration=[worker.last_iteration for worker in workers])
        self._fenced("recording the end", write)


@contextlib.contextmanager
def worker_run_directory(directory: str | None):
    """`ENV_RUN_DIRECTORY` set (or, when not recording, removed) for the
    duration of a worker's spawn, then put back as it was."""
    saved = os.environ.get(ENV_RUN_DIRECTORY)
    if directory:
        os.environ[ENV_RUN_DIRECTORY] = directory
    else:
        os.environ.pop(ENV_RUN_DIRECTORY, None)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(ENV_RUN_DIRECTORY, None)
        else:
            os.environ[ENV_RUN_DIRECTORY] = saved


def record_worker_device(directory: str | None, rank: int, session) -> None:
    """In a worker, once its session is set up: its device (fenced)."""
    if not directory:
        return
    try:
        import torch.distributed as dist

        from naavik.runs.facts import device_facts
        from naavik.runs.registry import write_rank

        initialized = dist.is_available() and dist.is_initialized()
        write_rank(Path(directory), rank, device_facts(
            session.device, rank,
            world_size=dist.get_world_size() if initialized else 1,
            backend=dist.get_backend() if initialized else None))
    except Exception as error:
        _warn(f"rank {rank} could not record its device ({type(error).__name__}: {error})")
