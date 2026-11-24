import argparse
import math
import re
import warnings
from collections.abc import Mapping
from copy import deepcopy

from omegaconf import OmegaConf

from naavik.config_files import (
    ConfigFileError,
    SessionsListWarning,
    check_no_interpolation,
    load_session,
    unescape,
)
from naavik.components.config import (
    parse_session_config,
    reject_legacy_components_entry,
)
from naavik.engine.topology import TOPOLOGY_KEYS
from naavik.session.checkpoint_format import (
    CheckpointFormatError,
    is_checkpoint,
    latest_checkpoint,
)


#: Overrides that describe the launch rather than the session, and so may be
#: given when resuming a checkpoint that changes nothing else.
_TOPOLOGY_OVERRIDES = frozenset(f"ddp.{name}" for name in TOPOLOGY_KEYS)

#: An override key written for the earlier `sessions: [entry]` form:
#: `sessions[0].a.b` or `sessions.0.a.b` (group 1 or 2: the index; group 3:
#: the rest, `a.b`).
_SESSIONS_PREFIX = re.compile(r"\Asessions(?:\[(\d+)\]|\.(\d+))?(?:\.(.*))?\Z")


def _session_overrides(overrides, path: str, wrapped: bool) -> list[str]:
    """`--override` items as dotlist items relative to the session.

    `a.b=v` is relative to the session. `sessions[0].a.b=v` (the earlier
    form) is accepted only for a file in the earlier form, and means `a.b=v`;
    any other `sessions...` key is refused, since a file is one session.
    """
    relative = []
    for item in overrides:
        key, separator, value = item.partition("=")
        match = _SESSIONS_PREFIX.match(key.strip())
        if match is None:
            relative.append(item)
            continue
        index = match.group(1) or match.group(2)
        rest = match.group(3)
        if not wrapped:
            hint = f" (write {rest}={value})" if rest else ""
            raise ConfigFileError(
                f"--override {item!r}: {path} is one session, so overrides are "
                f"relative to it{hint}"
            )
        if index != "0" or not rest:
            raise ConfigFileError(
                f"--override {item!r}: a config file is one session; address a "
                "value in it as sessions[0].<path> or, better, <path>"
            )
        relative.append(f"{rest}{separator}{value}")
    return relative


class Configurator:
    @staticmethod
    def _non_negative_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0:
            raise argparse.ArgumentTypeError(
                "must be a finite, non-negative number"
            )
        return parsed

    @staticmethod
    def _positive_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0:
            raise argparse.ArgumentTypeError(
                "must be a finite number greater than zero"
            )
        return parsed

    def __init__(self, argv=None, *, session=None):
        """Read the engine's options from `argv` (default: the command
        line, `sys.argv[1:]`).

        `session`: `(definition, sources)` of a new session its caller has
        already loaded (includes merged, overrides applied, `\\${`
        unescaped), with `sources` as `load_session` returns them; `argv`
        then holds only the other options (`naavik run` does this, so the
        run's record names the user's file)."""
        self._parser = argparse.ArgumentParser()

        group = self._parser.add_mutually_exclusive_group(required=session is None)
        group.add_argument("--config", help="Path to session config file")
        group.add_argument(
            "--resume-session",
            metavar="SESSION_DIR",
            help="Continue the session in SESSION_DIR from its latest checkpoint",
        )
        group.add_argument(
            "--resume-session-from-checkpoint",
            metavar="CHECKPOINT",
            help="Continue a session from this checkpoint",
        )
        group.add_argument(
            "--extend-session",
            nargs="+",
            metavar="SESSION_DIR",
            help="Extend the session in SESSION_DIR from its latest checkpoint",
        )
        group.add_argument(
            "--extend-session-from-checkpoint",
            nargs="+",
            metavar="VALUE",
            help=(
                "Extend a session from this checkpoint, optionally followed "
                "by the deprecated positional maximum iteration count"
            ),
        )

        self._parser.add_argument("--override", type=str, nargs="*", default=None)
        self._parser.add_argument(
            "--debug",
            action="store_true",
            help="Wait for worker processes using joins without monitoring",
        )
        self._parser.add_argument("--heartbeat-timeout", type=float, default=30.0)
        self._parser.add_argument(
            "--stop-sync-grace-period",
            type=self._non_negative_finite_float,
            default=0.01,
            help=(
                "Seconds to poll a DDP stop collective before sleeping"
            ),
        )
        self._parser.add_argument(
            "--stop-sync-poll-interval",
            type=self._positive_finite_float,
            default=0.005,
            help=(
                "Seconds to sleep between DDP stop-collective polls after "
                "the grace period"
            ),
        )
        self._parser.add_argument(
            "--process_timeout_on_join",
            type=float,
            default=30.0,
        )

        self._args = self._parser.parse_args(argv)

        self._session_configs = None
        self._config_sources = None
        self._checkpoint_path = None
        self._new_max_iters = None
        self._extension_overrides = None
        self._topology_overrides = {}
        self._mode = None

        if session is not None:
            operations = ("config", "resume_session", "resume_session_from_checkpoint",
                          "extend_session", "extend_session_from_checkpoint")
            if any(getattr(self._args, name) for name in operations) or self._args.override:
                raise ValueError("a Configurator given a session takes no operation "
                                 "and no --override in argv")
            definition, sources = session
            self._mode = "new"
            self._session_configs = [deepcopy(dict(definition))]
            self._config_sources = [list(sources)]
        elif self._args.config:
            self._mode = "new"
            try:
                loaded = load_session(self._args.config)
                overrides = _session_overrides(
                    self._args.override or [], self._args.config, loaded.wrapped,
                )
            except ConfigFileError as error:
                self._parser.error(str(error))
            if loaded.wrapped:
                warnings.warn(
                    SessionsListWarning(loaded.unwrap_advice(self._args.config)),
                    stacklevel=2,
                )
            config = OmegaConf.create(loaded.session)
            if overrides:
                config.merge_with_dotlist(overrides)
            merged = OmegaConf.to_container(config)
            try:
                # Again after the overrides, which can bring one in too.
                check_no_interpolation(merged, f"{self._args.config} with --override")
            except ConfigFileError as error:
                self._parser.error(str(error))
            self._session_configs = [unescape(merged)]
            self._config_sources = [loaded.sources]
        elif self._args.extend_session or self._args.extend_session_from_checkpoint:
            self._mode = "extend"
            if self._args.extend_session:
                values = self._args.extend_session
                if len(values) > 1 or is_checkpoint(values[0]):
                    self._parser.error(
                        "--extend-session takes a session directory and extends it from "
                        "its latest checkpoint; to extend from a given checkpoint (and "
                        "for the deprecated NEW_MAX_ITERATIONS) use "
                        "--extend-session-from-checkpoint"
                    )
                self._checkpoint_path = self._latest(values[0], "extending")
                values = [self._checkpoint_path]
            else:
                values = self._args.extend_session_from_checkpoint
                if len(values) > 2:
                    self._parser.error(
                        "--extend-session-from-checkpoint accepts CHECKPOINT and an "
                        "optional deprecated NEW_MAX_ITERATIONS value"
                    )
                self._checkpoint_path = values[0]
                self._say_checkpoint("extending", values[0])
            overrides = list(self._args.override or [])
            self._topology_overrides, overrides = (
                self._split_topology_overrides(overrides)
            )
            if len(values) == 2:
                warnings.warn(
                    "The positional NEW_MAX_ITERATIONS argument is deprecated; "
                    "use --override session_config.max_iterations=VALUE",
                    DeprecationWarning,
                    stacklevel=2,
                )
                try:
                    self._new_max_iters = int(values[1])
                except ValueError:
                    self._parser.error(
                        "The deprecated positional NEW_MAX_ITERATIONS value "
                        "must be an integer"
                    )
                if any(
                    item.split("=", 1)[0].strip()
                    == "session_config.max_iterations"
                    for item in overrides
                ):
                    self._parser.error(
                        "max_iterations cannot be supplied both positionally "
                        "and through --override"
                    )
                overrides.append(
                    f"session_config.max_iterations={self._new_max_iters}"
                )
            if not overrides and not self._topology_overrides:
                self._parser.error(
                    "--extend-session requires at least one --override"
                )
            try:
                check_no_interpolation(overrides, "--override")
            except ConfigFileError as error:
                self._parser.error(str(error))
            self._extension_overrides = tuple(overrides)
        elif self._args.resume_session or self._args.resume_session_from_checkpoint:
            self._mode = "resume"
            if self._args.resume_session:
                if is_checkpoint(self._args.resume_session):
                    self._parser.error(
                        f"--resume-session takes a session directory and resumes it from "
                        f"its latest checkpoint; {self._args.resume_session} is a "
                        "checkpoint: use --resume-session-from-checkpoint"
                    )
                self._checkpoint_path = self._latest(self._args.resume_session, "resuming")
            else:
                self._checkpoint_path = self._args.resume_session_from_checkpoint
                self._say_checkpoint("resuming", self._checkpoint_path)
            self._topology_overrides, unsupported = (
                self._split_topology_overrides(self._args.override or [])
            )
            if unsupported:
                names = ", ".join(sorted(unsupported))
                self._parser.error(
                    "--resume-session only accepts launch-topology "
                    f"overrides ({', '.join(sorted(_TOPOLOGY_OVERRIDES))}), "
                    f"but got: {names}. Extend the session (--extend-session) to "
                    "change its configuration."
                )

    def _latest(self, session_dir, verb):
        """The latest checkpoint of the session in `session_dir`, said in
        the log before anything is built."""
        try:
            path, iteration = latest_checkpoint(session_dir)
        except CheckpointFormatError as error:
            self._parser.error(str(error))
        at = "" if iteration is None else f" (iteration {iteration})"
        print(f"{verb} session {session_dir} from its latest checkpoint: {path}{at}",
              flush=True)
        return path

    @staticmethod
    def _say_checkpoint(verb, path):
        print(f"{verb} from checkpoint {path}", flush=True)

    @staticmethod
    def _split_topology_overrides(
            overrides,
    ) -> tuple[dict[str, str], list[str]]:
        """Separate launch-topology overrides from session-config ones.

        Topology overrides never reach the session extension machinery: they
        describe the machine this launch runs on, not the run itself.
        """
        topology: dict[str, str] = {}
        remaining: list[str] = []
        for item in overrides:
            key, separator, value = item.partition("=")
            key = key.strip()
            if separator and key in _TOPOLOGY_OVERRIDES:
                topology[key.split(".", 1)[1]] = value
            else:
                remaining.append(item)
        return topology, remaining

    def get_session_definition(self, index):
        if not self._session_configs:
            raise KeyError("Cannot use this function in the current operation!")
        session_definition = self._session_configs[index]
        reject_legacy_components_entry(session_definition)
        return deepcopy(session_definition)

    def _component_entries(self, session_index: int):
        if not self._session_configs:
            raise KeyError("Cannot use this function in the current operation!")
        session_config = self._session_configs[session_index]
        return parse_session_config(
            session_config,
            session_type=session_config.get("session_type"),
        ).components

    def get_component_config(self, session_index: int, key: str):
        """The configuration component `key` is constructed with, whether it
        is listed at the top level or in a group, without `role_bindings`."""
        entries = self._component_entries(session_index)
        if key not in entries:
            raise KeyError(key)
        return deepcopy(entries[key].config)

    def get_all_component_configs(self, session_index):
        return {
            key: deepcopy(entry.config)
            for key, entry in self._component_entries(session_index).items()
        }

    @property
    def session_configs(self):
        if not self._session_configs:
            raise KeyError("Cannot use this property in the current operation!")
        for session_config in self._session_configs:
            reject_legacy_components_entry(session_config)
        return deepcopy(self._session_configs)

    @property
    def config_sources(self):
        """Per session, the files its config was built from
        (`ConfigSource`s: the `--config` file, then the files it includes)."""
        if self._config_sources is None:
            raise KeyError("Cannot use this property in the current operation!")
        return list(self._config_sources)

    @property
    def checkpoint_path(self):
        if not self._checkpoint_path:
            raise KeyError("Cannot use this property in the current operation!")
        return self._checkpoint_path

    @property
    def topology_overrides(self):
        return dict(self._topology_overrides)

    @property
    def new_max_iters(self):
        if self._new_max_iters is None:
            raise KeyError("Cannot use this property in the current operation!")
        return self._new_max_iters

    @property
    def extension_overrides(self):
        if self._extension_overrides is None:
            raise KeyError("Cannot use this property in the current operation!")
        return tuple(self._extension_overrides)

    @property
    def process_timeout_on_join(self):
        return self._args.process_timeout_on_join

    @property
    def mode(self):
        return self._mode

    @property
    def heartbeat_timeout(self):
        return self._args.heartbeat_timeout

    @property
    def stop_sync_grace_period(self):
        return self._args.stop_sync_grace_period

    @property
    def stop_sync_poll_interval(self):
        return self._args.stop_sync_poll_interval

    @property
    def debug(self):
        return self._args.debug
