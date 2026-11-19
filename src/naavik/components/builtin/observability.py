from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, override

import torch

from torch.utils.tensorboard import SummaryWriter

from naavik.components import (
    ExtendableComponent,
    LifecycleHook,
    Resource,
    hook,
    rank_zero_only,
    requires_resource,
    resource,
)
from naavik.util import format_execution_time
from naavik.components.config_schema import (
    drop_unread_keys,
    parse_component_config,
    positive_every,
)

if TYPE_CHECKING:
    from naavik.session import Session, TrainingSession


def reported(session: Session, attribute: str) -> Any:
    """The first non-None `attribute` any hook or resource exposes.

    Duck-typed so the value is found whatever the component is called or
    bound as -- `current_lrs`, `grad_norm`, `last_loss` and
    `completed_steps` come from the optimizer resource, and a hook may
    expose them too.
    """
    for listing in ("get_all_hooks", "get_all_resources"):
        get_all = getattr(session, listing, None)
        if get_all is None:
            continue
        for component in get_all():
            value = getattr(component, attribute, None)
            if value is not None:
                return value
    return None


@dataclass(frozen=True)
class LoggerConfig:
    """The `logger` config (`log_every` comes from the session's defaults
    when the entry leaves it out)."""

    log_every: Any
    log_file: Any = None
    #: Restoring a checkpoint from before this schema: keys the logger never
    #: read are dropped with a warning, so the checkpoint still resumes.
    unknown_keys_dropped_on_restore: ClassVar[bool] = True

    def __post_init__(self) -> None:
        positive_every(self.log_every, "logger.log_every")
        if self.log_file is not None and (not isinstance(self.log_file, str) or not self.log_file):
            raise ValueError(f"logger.log_file must be a file path; got {self.log_file!r}")


@dataclass(frozen=True)
class TimerConfig:
    """The `timer` config."""

    call_every: Any
    unknown_keys_dropped_on_restore: ClassVar[bool] = True

    def __post_init__(self) -> None:
        positive_every(self.call_every, "timer.call_every")


def _port_problem(port) -> bool:
    return (isinstance(port, bool)
            or not (isinstance(port, int) or (isinstance(port, str) and port.isdigit()))
            or not 0 < int(port) < 65536)


@dataclass(frozen=True)
class TensorboardConfig:
    """The `tensorboard` config: the writer always; with `serve` (default),
    also a TensorBoard server on `host`:`port`, which it then needs."""

    serve: Any = True
    host: Any = None
    port: Any = None
    logdir: Any = None
    unknown_keys_dropped_on_restore: ClassVar[bool] = True

    #: Keys that only configure the server.
    SERVER_KEYS: ClassVar[tuple[str, ...]] = ("host", "port", "logdir")

    def __post_init__(self) -> None:
        if not isinstance(self.serve, bool):
            raise ValueError(f"serve must be a boolean; got {self.serve!r}")
        given = [key for key in self.SERVER_KEYS if getattr(self, key) is not None]
        if not self.serve:
            if given:
                raise ValueError(
                    "tensorboard starts no server with serve: false, so "
                    f"{', '.join(given)} would have no effect; remove "
                    f"{'it' if len(given) == 1 else 'them'}"
                )
            return
        missing = [key for key in ("host", "port") if getattr(self, key) is None]
        if missing:
            raise ValueError(
                f"tensorboard serves on host:port, so it needs {' and '.join(missing)} "
                "(or serve: false to write events only)"
            )
        if not isinstance(self.host, str) or not self.host:
            raise ValueError(f"tensorboard.host must be a host name or address; got {self.host!r}")
        if _port_problem(self.port):
            raise ValueError(f"tensorboard.port must be a port number (1-65535); got {self.port!r}")
        if self.logdir is not None and (not isinstance(self.logdir, str) or not self.logdir):
            raise ValueError(f"tensorboard.logdir must be a directory path; got {self.logdir!r}")


@rank_zero_only
@hook("logger")
class Logger(LifecycleHook, ExtendableComponent):

    config_schema = LoggerConfig

    def __init__(self, config: dict):
        settings = parse_component_config(type(self), config)
        self._config = config
        self.call_every = settings.log_every
        # The configured path (None: stdout), and the stream written to,
        # which is stdout whenever no file is open.
        self._log_path = settings.log_file
        self._log_file = sys.stdout

    def pre_session(self, session: Session) -> Any:
        if self._log_path is None:
            return
        # A session resumed from a checkpoint continues the log of the run
        # it came from; a fresh one starts it clean.
        mode = "a" if session.iteration > 0 else "w"
        try:
            os.makedirs(os.path.dirname(self._log_path) or ".", exist_ok=True)
            self._log_file = open(self._log_path, mode)
        except OSError as error:
            raise OSError(
                f"Logger cannot open log_file {self._log_path!r} for "
                f"writing: {error}"
            ) from error

    def post_session(self, session) -> None:
        log_file, self._log_file = self._log_file, sys.stdout
        if log_file is not sys.stdout:
            log_file.close()

    def print(self, *args, **kwargs):
        print(*args, **kwargs, file=self._log_file)

    def pre_iteration_callback(self, session: Session) -> None:
        line = (
            f"Iteration {session.iteration}/"
            f"{session.session_config.max_iterations}"
        )
        lrs = self._reported(session, "current_lrs")
        if lrs is not None:
            line += " | lr: " + ", ".join(f"{lr:.3e}" for lr in lrs)
        grad_norm = self._reported(session, "grad_norm")
        if grad_norm is not None:
            line += f" | grad_norm: {grad_norm:.3e}"
        loss = self._reported(session, "last_loss")
        if loss is not None:
            line += f" | loss: {loss:.4e}"
        self.print(line)

    @staticmethod
    def _reported(session: Session, attribute: str) -> Any:
        return reported(session, attribute)

    def post_iteration_callback(self, session: Session) -> None:
        pass

    def __getstate__(self) -> Any:
        return {"config": self._config}

    def __setstate__(self, state: Any) -> None:
        # An unpickled session (a single-file checkpoint) may hold keys the
        # logger once ignored.
        self.__init__(drop_unread_keys(type(self), "logger", state["config"]))

    def apply_extension_config(
            self,
            config: Mapping,
            changed_paths: frozenset[tuple[str, ...]],
    ) -> None:
        unsupported = changed_paths - {("log_every",)}
        if unsupported:
            names = ", ".join(".".join(path) for path in sorted(unsupported))
            raise ValueError(
                "Logger session extension does not allow changes to: " + names
            )
        settings = parse_component_config(type(self), config)
        self._config = deepcopy(dict(config))
        self.call_every = settings.log_every


@rank_zero_only
@resource("tensorboard")
class Tensorboard(Resource):

    config_schema = TensorboardConfig

    def __init__(self, config: dict):
        parse_component_config(type(self), config)
        self._config = config
        self._tb_process: subprocess.Popen | None = None
        self._tb_summary_writer = None

    @property
    def summary_writer(self):
        return self._tb_summary_writer

    def setup(self, session: Session):
        serve = self._config.get("serve", True)
        if serve:
            logdir = self._config.get(
                "logdir",
                session.session_config.session_dir,
            )
            tensorboard_args = [
                "tensorboard",
                "--logdir", logdir,
                "--host", self._config["host"],
                "--port", str(self._config["port"]),
            ]
            print("Tensorboard Arguments: ", " ".join(tensorboard_args))
            self._tb_process = subprocess.Popen(tensorboard_args)
        # Named per instance, so two writers do not merge their scalars into
        # one event directory. The port cannot be shared and is not defaulted:
        # a second serving instance must be given one of its own.
        writer_dir = f"{type(self).__name__}_tensorboard"
        if self.instance_suffix is not None:
            writer_dir = f"{writer_dir}_{self.instance_suffix}"
        self._tb_summary_writer = SummaryWriter(
            log_dir=os.path.join(
                session.session_config.session_dir,
                writer_dir,
            )
        )
        if serve:
            time.sleep(3)
            if self._tb_process.poll() is not None:
                print("Failed to start tensorboard process...")
                raise RuntimeError("Failed to start tensorboard process...")

    def teardown(self, session):
        print("releasing resources...")
        self._release_partial_setup()
        print("resources released...")

    def rollback_setup(self, session: Session) -> None:
        self._release_partial_setup()

    def _release_partial_setup(self) -> None:
        try:
            if self._tb_summary_writer is not None:
                self._tb_summary_writer.close()
        finally:
            self._tb_summary_writer = None

            process = self._tb_process
            self._tb_process = None
            if process is not None and process.poll() is None:
                process.terminate()


# -- tensorboard_summaries ------------------------------------------------------


@dataclass
class TensorboardSummariesConfig:
    """What is written, by kind of summary.

    `scalars` maps a TensorBoard tag to the iteration_context key whose
    value is written under it. `optimizer` adds the optimizer's learning
    rates (one tag per group), gradient norm, loss and completed steps.
    """

    call_every: int = 1
    scalars: Any = field(default_factory=dict)
    optimizer: bool = False

    def __post_init__(self):
        if (
                isinstance(self.call_every, bool)
                or not isinstance(self.call_every, int)
                or self.call_every <= 0
        ):
            raise ValueError(
                f"call_every must be a positive integer; got {self.call_every!r}"
            )
        if not isinstance(self.scalars, Mapping):
            raise ValueError(
                "scalars must be a mapping of TensorBoard tag to "
                f"iteration_context key; got {self.scalars!r}"
            )
        for tag, key in self.scalars.items():
            for name, value in (("tag", tag), ("key", key)):
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"scalars maps non-empty tags to non-empty keys; got "
                        f"{name} {value!r}"
                    )
        self.scalars = dict(self.scalars)
        if not isinstance(self.optimizer, bool):
            raise ValueError(
                f"optimizer must be true or false; got {self.optimizer!r}"
            )
        if not self.scalars and not self.optimizer:
            raise ValueError(
                "tensorboard_summaries writes nothing: give scalars, or set "
                "optimizer: true"
            )


def _scalars(tag: str, value: Any) -> list[tuple[str, float]]:
    """`value` as (tag, number) pairs: a number or one-element tensor under
    `tag`, a mapping as one tag per entry (`tag/name`)."""
    if isinstance(value, Mapping):
        pairs = []
        for name, item in value.items():
            pairs.extend(_scalars(f"{tag}/{name}", item))
        return pairs
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError(
                f"tensorboard_summaries scalar {tag!r} is a tensor of shape "
                f"{tuple(value.shape)}; a scalar needs one element"
            )
        return [(tag, float(value.detach()))]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [(tag, float(value))]
    raise TypeError(
        f"tensorboard_summaries scalar {tag!r} is a {type(value).__name__}, "
        "not a number, a one-element tensor or a mapping of them"
    )


@rank_zero_only
@requires_resource("tensorboard")
@hook("tensorboard_summaries")
class TensorboardSummaries(LifecycleHook):
    """Write summaries of the run to the `tensorboard` resource's writer.

    Every `call_every` iterations, after the iteration's steps, each value
    `scalars` names is written at `session.iteration`. The values are read
    from the iteration context like any step's inputs, so a key no step
    writes is an error when the session is built. With `optimizer: true`,
    the optimizer's learning rates, gradient norm, loss and completed steps
    are written too (`optimizer/...`). Written on rank 0 only.
    """

    config_schema = TensorboardSummariesConfig

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__(config)
        self.call_every = self._cfg.call_every

    @override
    def context_reads(self) -> dict[str, str]:
        # Keys need not be identifiers, so values arrive by key.
        return {key: key for key in dict.fromkeys(self._cfg.scalars.values())}

    def pre_session(self, session: Session) -> None:
        pass

    def post_session(self, session: Session) -> None:
        writer = self.get_dependency("tensorboard").summary_writer
        if writer is not None:
            writer.flush()

    def pre_iteration_callback(self, session: Session) -> None:
        return None

    def post_iteration_callback(self, session: Session, /, *args: Any, **reads: Any) -> None:
        writer = self.get_dependency("tensorboard").summary_writer
        if writer is None:
            raise RuntimeError(
                "tensorboard_summaries has no writer: the tensorboard resource "
                "is not set up"
            )
        step = session.iteration
        for tag, key in self._cfg.scalars.items():
            for name, value in _scalars(tag, reads[key]):
                writer.add_scalar(name, value, step)
        if self._cfg.optimizer:
            for name, value in self._optimizer_scalars(session):
                writer.add_scalar(name, value, step)

    @staticmethod
    def _optimizer_scalars(session: Session) -> list[tuple[str, float]]:
        values = []
        lrs = reported(session, "current_lrs")
        for index, lr in enumerate(lrs or ()):
            values.append((f"optimizer/lr/group_{index}", float(lr)))
        for attribute in ("grad_norm", "last_loss", "completed_steps"):
            value = reported(session, attribute)
            if value is not None:
                values.append((f"optimizer/{attribute}", float(value)))
        return values


@rank_zero_only
@hook("timer", session_type="training")
class Timer(LifecycleHook):

    config_schema = TimerConfig

    def __init__(self, config) -> None:
        self.call_every = parse_component_config(type(self), config).call_every
        self._session_start = None
        self._iter_start = None

    def pre_session(self, session: TrainingSession):
        self._session_start = time.time_ns()

    def post_session(self, session: Session):
        pass

    def pre_iteration_callback(self, session: Session) -> None:
        self._iter_start = time.time_ns()

    def post_iteration_callback(self, session: Session) -> None:
        iter_end = time.time_ns()
        iter_duration = iter_end - self._iter_start
        elapsed_time = iter_end - self._session_start
        print(
            f"Time taken for the iteration {session.iteration}: "
            f"{format_execution_time(iter_duration)}"
        )
        print(f"Elapsed time: {format_execution_time(elapsed_time)}")
        print()
