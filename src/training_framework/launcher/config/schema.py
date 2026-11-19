"""Strict schemas for the site profile and the committed batch manifest.

Unknown fields are errors in both. Site concerns (SSH, roots, scheduler
placement, hooks, environment sources) live only in the profile; the
manifest holds only experiment concerns, so it runs unchanged on another
cluster.
"""

from __future__ import annotations

import posixpath
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from training_framework.launcher.errors import Problems
from training_framework.launcher.names import check_name

_SIZE = re.compile(r"\A(\d+)([KMGT]?)\Z")
_SIZE_FACTORS = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
_WALLTIME = re.compile(r"\A(?:(\d+)-)?(\d{1,2}):(\d{2}):(\d{2})\Z")
_DURATION = re.compile(r"\A(\d+)([smh])\Z")
_DURATION_FACTORS = {"s": 1, "m": 60, "h": 3600}

#: The keys of one host in the hosts file (`hosts.yaml`).
HOST_KEYS = frozenset({
    "connection", "scheduler", "directories", "shell_commands_before",
    "python_environment", "environment_build", "data_preparation",
    "local_to_host_paths", "confirm_uploads_over", "run_code_from",
    "working_tree_untracked_limit", "dataset_delivery",
})
#: Every key a host has had, current and earlier: a mission file refuses
#: them all (they are site settings).
PROFILE_KEYS = HOST_KEYS | frozenset({
    "ssh_host", "local", "roots", "slurm", "hooks", "env", "setup_location",
    "prepare_location", "path_map", "upload_limit", "setup_resources",
    "worktree", "snapshot_limit", "data_modes",
})
#: How a data requirement reaches a host (`dataset_delivery`), and the
#: name each has inside the launcher.
DATASET_DELIVERY = {"download_on_host": "fetch", "copy_from_here": "upload",
                    "already_on_host": "present"}
DATA_MODES = tuple(DATASET_DELIVERY.values())
#: `environment_build.on` / `data_preparation.on`, and the launcher's names.
STEP_LOCATIONS = {"compute_node": "job", "login_node": "login"}
#: `run_code_from`, and the launcher's names.
CODE_FROM = {"commit": "commit", "working_tree": "snapshot"}
#: `code_transfer.method` and `code.framework_from`, and the launcher's names.
TRANSFER_METHODS = {"git_bundle": "bundle", "git_clone": "fetch"}
FRAMEWORK_FROM = {"installed": "dependency", "this_repository": "checkout"}
#: How a batch takes your code: the commit you made (`commit`), or a commit
#: made of your working tree as it is (`snapshot`).
WORKTREE_MODES = ("commit", "snapshot")
DEFAULT_SNAPSHOT_LIMIT = "50M"
DEFAULT_SETUP_RESOURCES = {"walltime": "01:00:00", "cpus": 4, "memory": "16G"}
#: The keys of a mission file (`missions/*.yaml`).
MISSION_FILE_KEYS = frozenset({
    "format_version", "name", "note", "project_file", "code_transfer", "code",
    "python_packages", "base", "data_sources", "jobs",
})
#: The project file (`naavik-project.yaml`): what a project's batches share.
PROJECT_FILE = "naavik-project.yaml"
#: Keys a project file shares with mission files (a mission file naming the
#: project file may not repeat them: one setting has one place).
PROJECT_MISSION_FILE_KEYS = ("code_transfer", "code", "python_packages", "data_sources")
PROJECT_KEYS = frozenset(PROJECT_MISSION_FILE_KEYS) | {"quick_run_defaults"}
HOOK_PHASES = ("common", "setup", "prepare", "task")
LOCATIONS = ("job", "login")


def parse_size(value: Any) -> int | None:
    match = _SIZE.match(value) if isinstance(value, str) else None
    if match is None:
        return None
    return int(match.group(1)) * _SIZE_FACTORS[match.group(2)]


def parse_walltime(value: Any) -> int | None:
    """Seconds in `[D-]HH:MM:SS`, or None."""
    match = _WALLTIME.match(value) if isinstance(value, str) else None
    if match is None:
        return None
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    if minutes > 59 or seconds > 59:
        return None
    total = ((days * 24 + hours) * 60 + minutes) * 60 + seconds
    return total or None


def parse_duration(value: Any) -> int | None:
    """Seconds in `<n>s|m|h`, or None."""
    match = _DURATION.match(value) if isinstance(value, str) else None
    if match is None:
        return None
    return int(match.group(1)) * _DURATION_FACTORS[match.group(2)]


class _Reader:
    """Reads one mapping strictly, collecting problems."""

    def __init__(self, raw: Any, where: str, problems: Problems,
                 allowed: frozenset[str] | set[str],
                 refused: Mapping[str, str] | None = None):
        self.where = where
        self.problems = problems
        if not isinstance(raw, Mapping):
            problems.add(f"{where} must be a mapping")
            raw = {}
        self.raw = raw
        for key in raw:
            if refused and key in refused:
                problems.add(f"{where}.{key}: {refused[key]}")
            elif key not in allowed:
                problems.add(
                    f"{where}: unknown field '{key}' (allowed: "
                    f"{', '.join(sorted(allowed))})"
                )

    def get(self, key, default=None, *, required=False):
        if key not in self.raw:
            if required:
                self.problems.add(f"{self.where}.{key} is required")
            return default
        return self.raw[key]

    def string(self, key, default=None, *, required=False):
        value = self.get(key, default, required=required)
        if value is not None and (not isinstance(value, str) or not value):
            self.problems.add(f"{self.where}.{key} must be a non-empty string")
            return default
        return value

    def choice(self, key, choices, default=None, *, required=False):
        value = self.get(key, default, required=required)
        if value is not None and value not in choices:
            self.problems.add(
                f"{self.where}.{key} must be one of {', '.join(choices)}; "
                f"got {value!r}"
            )
            return default
        return value

    def positive_int(self, key, default=None, *, required=False, minimum=1):
        value = self.get(key, default, required=required)
        if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
                or value < minimum
        ):
            self.problems.add(
                f"{self.where}.{key} must be an integer >= {minimum}; "
                f"got {value!r}"
            )
            return default
        return value

    def boolean(self, key, default=None):
        value = self.get(key, default)
        if not isinstance(value, bool):
            self.problems.add(f"{self.where}.{key} must be a boolean")
            return default
        return value

    def strings(self, key, default=()):
        value = self.get(key, list(default))
        if not isinstance(value, list) or not all(
                isinstance(item, str) and item for item in value
        ):
            self.problems.add(
                f"{self.where}.{key} must be a list of non-empty strings"
            )
            return tuple(default)
        return tuple(value)

    def absolute_path(self, key, default=None, *, required=False):
        value = self.string(key, default, required=required)
        if value is not None and not posixpath.isabs(value):
            self.problems.add(f"{self.where}.{key} must be an absolute path")
            return default
        return value


# Profile ------------------------------------------------------------------

#: `env.existing: current`: the environment of the interpreter running the
#: command (e.g. an IDE's project interpreter), for a target on this machine.
CURRENT_ENVIRONMENT = "current"


def _existing_environment(env_reader, local: bool, where: str, problems) -> str | None:
    if env_reader.raw.get("use_existing") != CURRENT_ENVIRONMENT:
        return env_reader.absolute_path("use_existing")
    if not local:
        problems.add(f"{where}.python_environment.use_existing: 'current' (this interpreter's "
                     "environment) needs a host on this machine (connection: local)")
        return None
    # Resolved now, so the plan records the path the tasks will use.
    return sys.prefix


@dataclass(frozen=True)
class PythonIdentity:
    implementation: str
    version: str
    executable: str
    platform: str

    def as_json(self) -> dict:
        return {
            "implementation": self.implementation,
            "version": self.version,
            "executable": self.executable,
            "platform": self.platform,
        }


@dataclass(frozen=True)
class EnvSettings:
    backend: str = "venv"
    python_source: str = "site"
    site_python_identity: PythonIdentity | None = None
    existing: str | None = None
    conda_packages: tuple[str, ...] = ()
    conda_run: bool = False
    bootstrap_python: str | None = None


@dataclass(frozen=True)
class Profile:
    name: str
    #: The SSH alias of the target, or None for `local: true` (this machine).
    ssh_host: str | None
    roots: dict[str, str]
    scheduler: str
    #: The selected scheduler's own settings, parsed by its bundle.
    scheduler_settings: Any
    hooks: dict[str, tuple[str, ...]]
    env: EnvSettings
    setup_location: str
    prepare_location: str
    path_map: tuple[tuple[str, str], ...]
    upload_limit: int
    setup_resources: "Resources | None" = None
    #: True for a target that is this machine (`local: true`).
    local: bool = False
    #: `commit` (a clean tree; batches run HEAD) or `snapshot`.
    worktree: str = "commit"
    #: Most bytes of untracked files a snapshot may take into Git.
    snapshot_limit: int = 50 * 1024 ** 2
    #: How data requirements reach this host; None: not managed.
    data: "DataModes | None" = None
    #: What a data-preparation job is given (`data_preparation.allocation`).
    preparation_resources: "Resources | None" = None
    #: Earlier names this host's settings were written with (`old -> new`),
    #: for one warning.
    renamed: tuple[str, ...] = ()

    @property
    def slurm(self):
        """Compatibility alias for the Slurm settings of a Slurm profile."""
        if self.scheduler != "slurm":
            raise AttributeError("this profile does not use Slurm")
        return self.scheduler_settings


@dataclass(frozen=True)
class DataModes:
    """`dataset_delivery: {default: MODE, <requirement name>: MODE}` (the
    launcher's names for the modes: fetch, upload, present)."""

    default: str
    named: tuple[tuple[str, str], ...]

    def mode(self, name: str) -> str:
        return dict(self.named).get(name, self.default)


def _parse_data_modes(raw: Any, where: str, problems: Problems) -> DataModes | None:
    if raw is None:
        return None
    modes = ", ".join(DATASET_DELIVERY)
    if not isinstance(raw, Mapping):
        problems.add(f"{where} must be a mapping of a requirement name (or 'default') to "
                     f"one of {modes}")
        return None
    default, named = "fetch", []
    for key, written in raw.items():
        mode = DATASET_DELIVERY.get(written)
        if mode is None:
            problems.add(f"{where}.{key} must be one of {modes}; got {written!r}")
            continue
        if key == "default":
            default = mode
        elif problem := check_name(key, f"{where} (a requirement name)"):
            problems.add(problem)
        else:
            named.append((key, mode))
    return DataModes(default, tuple(sorted(named)))


def parse_profiles(raw: Any, selected: str | None, source: str) -> Profile:
    """Select a host from a hosts file: `{default_host: <name>, hosts: {<name>:
    {...}}}`; `selected` (from `--host`) wins over `default_host`. A file
    written with the earlier names (`targets:` / `profiles:`, `default`, the
    earlier keys of a host) is read too; the host says what was renamed."""
    from training_framework.launcher.config.names import hosts_file_from_earlier
    from training_framework.launcher.scheduler import SCHEDULERS

    problems = Problems()
    raw, renamed, renaming_problems = hosts_file_from_earlier(raw, source, tuple(SCHEDULERS))
    for problem in renaming_problems:
        problems.add(problem)
    problems.raise_if_any()
    top = _Reader(raw, source, problems, {"default_host", "hosts"})
    hosts = top.get("hosts", required=True)
    if not isinstance(hosts, Mapping) or not hosts:
        if hosts is not None:
            problems.add(f"{source}.hosts must be a non-empty mapping")
        problems.raise_if_any()
    name = selected or top.string("default_host")
    if name is None:
        problems.add(f"{source}: no host selected; pass --host or set default_host")
    elif name not in hosts:
        problems.add(f"{source}: host '{name}' not found (have: {', '.join(sorted(hosts))})")
    problems.raise_if_any()
    profile = _parse_profile(hosts[name], f"{source}:hosts.{name}", name, problems)
    problems.raise_if_any()
    mine = tuple(item for item in renamed if not item.startswith("hosts.")
                 or item.startswith(f"hosts.{name}:"))
    return replace(profile, renamed=mine)


def target_names(raw: Any) -> list[str]:
    """The names of every host in a hosts file (`hosts:`, or its earlier
    names `targets:` / `profiles:`), sorted; [] when it has none."""
    if not isinstance(raw, Mapping):
        return []
    entries = raw.get("hosts", raw.get("targets", raw.get("profiles")))
    return sorted(entries) if isinstance(entries, Mapping) else []


def _parse_profile(raw, where, name, problems) -> Profile:
    """One host, in the current names (`config/names.py` turned earlier ones
    into these)."""
    from training_framework.launcher.scheduler import SCHEDULERS

    reader = _Reader(raw, where, problems, HOST_KEYS)
    # How the host is reached.
    connection = reader.get("connection", required=True)
    ssh_host, local = None, False
    if connection == "local":
        local = True
    elif isinstance(connection, Mapping) and set(connection) == {"ssh"} \
            and isinstance(connection["ssh"], str) and connection["ssh"]:
        ssh_host = connection["ssh"]
    elif connection is not None:
        problems.add(f"{where}.connection must be `local` (this machine) or "
                     f"{{ssh: <host or ~/.ssh/config alias>}}; got {connection!r}")

    directories = _Reader(reader.get("directories", {}, required=True), f"{where}.directories",
                          problems, {"missions", "python_environments", "datasets"})
    roots = {inside: directories.absolute_path(written, required=True)
             for written, inside in (("missions", "batches"), ("python_environments", "envs"),
                                     ("datasets", "data"))}

    # `scheduler.kind` as written -> the scheduler's id (builtin -> host).
    kinds = {_kind_name(scheduler): scheduler for scheduler in SCHEDULERS}
    scheduler_raw = reader.get("scheduler", required=True)
    scheduler = settings = None
    if scheduler_raw is not None and not isinstance(scheduler_raw, Mapping):
        problems.add(f"{where}.scheduler must be a mapping: {{kind: {' | '.join(sorted(kinds))}, ...}}")
    elif scheduler_raw is not None:
        kind = scheduler_raw.get("kind")
        if kind not in kinds:
            problems.add(f"{where}.scheduler.kind must be one of {', '.join(sorted(kinds))}; got {kind!r}")
        else:
            scheduler = kinds[kind]
            settings = SCHEDULERS[scheduler].parse_settings(
                {key: value for key, value in scheduler_raw.items() if key != "kind"},
                f"{where}.scheduler", problems, _Reader)

    shell = _Reader(reader.get("shell_commands_before", {}), f"{where}.shell_commands_before",
                    problems, {"everything", "environment_build", "data_preparation", "experiments"})
    hooks = {inside: shell.strings(written) for written, inside in (
        ("everything", "common"), ("environment_build", "setup"),
        ("data_preparation", "prepare"), ("experiments", "task"))}

    env_reader = _Reader(
        reader.get("python_environment", {}), f"{where}.python_environment", problems,
        {"kind", "interpreter_from", "site_interpreter", "use_existing",
         "conda_packages", "activate_with_conda_run", "bootstrap_interpreter"},
    )
    identity = None
    if "site_interpreter" in env_reader.raw:
        id_reader = _Reader(
            env_reader.raw["site_interpreter"],
            f"{where}.python_environment.site_interpreter", problems,
            {"implementation", "version", "executable", "platform"},
        )
        fields = {
            key: id_reader.string(key, required=True)
            for key in ("implementation", "version", "executable", "platform")
        }
        if all(fields.values()):
            identity = PythonIdentity(**fields)
    env = EnvSettings(
        backend=env_reader.choice("kind", ("venv", "conda"), "venv"),
        python_source=env_reader.choice("interpreter_from", ("site", "uv", "conda"), "site"),
        site_python_identity=identity,
        existing=_existing_environment(env_reader, local, where, problems),
        conda_packages=env_reader.strings("conda_packages"),
        conda_run=env_reader.boolean("activate_with_conda_run", False),
        bootstrap_python=env_reader.string("bootstrap_interpreter"),
    )

    def step(key):
        """(location, resources) of `environment_build` / `data_preparation`."""
        step_reader = _Reader(reader.get(key, {}), f"{where}.{key}", problems, {"on", "allocation"})
        location = STEP_LOCATIONS[step_reader.choice("on", tuple(STEP_LOCATIONS), "compute_node")
                                  or "compute_node"]
        resources = parse_resources(step_reader.get("allocation", DEFAULT_SETUP_RESOURCES),
                                    f"{where}.{key}.allocation", problems)
        return location, resources

    setup_location, setup_resources = step("environment_build")
    prepare_location, preparation_resources = step("data_preparation")

    path_map_raw = reader.get("local_to_host_paths", {})
    path_map = []
    if not isinstance(path_map_raw, Mapping):
        problems.add(f"{where}.local_to_host_paths must map local prefixes to prefixes on the host")
    else:
        for local_prefix, remote in path_map_raw.items():
            if not (isinstance(local_prefix, str) and posixpath.isabs(local_prefix)
                    and isinstance(remote, str) and posixpath.isabs(remote)):
                problems.add(
                    f"{where}.local_to_host_paths: '{local_prefix}' -> '{remote}' must map an "
                    "absolute local prefix to an absolute prefix on the host"
                )
                continue
            path_map.append((local_prefix.rstrip("/") or "/", remote.rstrip("/") or "/"))

    upload_limit = parse_size(reader.get("confirm_uploads_over", "20G"))
    if upload_limit is None:
        problems.add(f"{where}.confirm_uploads_over must be a size such as 20G")
    snapshot_limit = parse_size(reader.get("working_tree_untracked_limit", DEFAULT_SNAPSHOT_LIMIT))
    if snapshot_limit is None:
        problems.add(f"{where}.working_tree_untracked_limit must be a size such as 50M")
    code_from = reader.choice("run_code_from", tuple(CODE_FROM), "commit") or "commit"

    return Profile(
        name=name,
        ssh_host=ssh_host,
        roots=roots,
        scheduler=scheduler,
        scheduler_settings=settings,
        hooks=hooks,
        env=env,
        setup_location=setup_location,
        prepare_location=prepare_location,
        path_map=tuple(sorted(path_map)),
        upload_limit=upload_limit or 0,
        setup_resources=setup_resources,
        local=local,
        worktree=CODE_FROM[code_from],
        snapshot_limit=snapshot_limit or 0,
        data=_parse_data_modes(reader.get("dataset_delivery"), f"{where}.dataset_delivery",
                               problems),
        preparation_resources=preparation_resources,
    )


def _kind_name(scheduler: str) -> str:
    from training_framework.launcher.config.names import kind_of

    return kind_of(scheduler)


# Experiment file (a batch's experiments; was called the manifest) ----------


@dataclass(frozen=True)
class Resources:
    walltime: str
    walltime_seconds: int
    gpus: int
    cpus: int
    memory: str
    scheduler_options: dict


#: `from_mission.checkpoint`: each source experiment's newest complete
#: checkpoint, or every complete one.
CHECKPOINT_CHOICES = ("latest", "all")
#: `from_mission.into` when not given: where an analysis session loads a
#: trained model from (`components/builtin/model.py`).
DEFAULT_CHECKPOINT_INTO = {"component": "trained_model", "path": "model_checkpoint_path"}
FROM_MISSION_KEYS = frozenset({"id", "experiments", "checkpoint", "into"})


@dataclass(frozen=True)
class FromMission:
    """A recipe's `from_mission`: one experiment per chosen checkpoint of
    another mission's experiments."""

    id: str
    #: Globs over the source's experiment names; () = all of them.
    experiments: tuple[str, ...]
    checkpoint: str               # latest | all
    #: Raw address {component?, path} the checkpoint's path is written at.
    into: dict

    def as_json(self) -> dict:
        return {"id": self.id, "experiments": list(self.experiments),
                "checkpoint": self.checkpoint, "into": dict(self.into)}


@dataclass(frozen=True)
class Experiment:
    name: str
    overrides: tuple[dict, ...]   # raw {component?, path, value}
    matrix: tuple[dict, ...]      # raw {component?, path, values}
    #: Its own training config (a repository path); None: `training.config`.
    config: str | None = None
    #: `from_mission`; None: the recipe's experiments come from the file only.
    from_mission: FromMission | None = None
    #: `config_files`: several session configs, one axis (each file with
    #: every `vary` combination); () otherwise (then `config`, or the base).
    configs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Group:
    name: str
    resources: Resources
    max_parallel: int | None
    stop_grace_seconds: int
    experiments: tuple[Experiment, ...]


@dataclass(frozen=True)
class MissionFile:
    version: int
    name: str
    source_strategy: str
    source_ref: str
    source_url: str | None
    code_framework: str
    code_paths: tuple[str, ...]
    requirements: str | None
    python_pin: str | None
    training_config: str
    training_session: int
    engine_options: tuple[str, ...]
    data_sources: tuple[Any, ...]
    groups: tuple[Group, ...]
    raw: dict = field(repr=False)
    #: Free text: why this mission runs (`note`; `naavik run --note`).
    description: str | None = None
    #: Earlier names it was written with (`old -> new`), for one warning.
    renamed: tuple[str, ...] = ()


_PROFILE_KEY_MESSAGE = (
    "is a site setting: it belongs in hosts.yaml, not in the committed mission file"
)


def parse_resources(raw, where, problems, *, require_cpus=True) -> Resources | None:
    reader = _Reader(
        raw, where, problems,
        {"walltime", "gpus", "cpus", "memory"},
        refused={
            key: f"{_PROFILE_KEY_MESSAGE} (use scheduler_options only for "
                 "options the dialect allows)"
            for key in ("partition", "qos", "account")
        },
    )
    walltime = reader.string("walltime", required=True)
    seconds = parse_walltime(walltime) if walltime else None
    if walltime and seconds is None:
        problems.add(f"{where}.walltime must be [D-]HH:MM:SS; got {walltime!r}")
    gpus = reader.positive_int("gpus", 0, minimum=0)
    cpus = reader.positive_int("cpus", required=require_cpus, default=1)
    memory = reader.string("memory", required=True)
    if memory and parse_size(memory) is None:
        problems.add(f"{where}.memory must be a size such as 64G; got {memory!r}")
    if seconds is None or memory is None or cpus is None:
        return None
    return Resources(walltime, seconds, gpus or 0, cpus, memory, {})


def _scheduler_options(raw, where, problems) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping) or not all(
            isinstance(key, str) and isinstance(value, Mapping)
            for key, value in raw.items()
    ):
        problems.add(f"{where} must map a scheduler name to its options")
        return {}
    return {key: dict(value) for key, value in raw.items()}


@dataclass(frozen=True)
class RunDefaults:
    """`run_defaults` of a project file: what `naavik run` uses unless told."""

    resources: dict
    max_parallel: int | None
    target: str | None


@dataclass(frozen=True)
class Project:
    #: The mission-file keys it provides (current names).
    mission_file_keys: dict
    run_defaults: RunDefaults
    #: Earlier names it was written with (`old -> new`), for one warning.
    renamed: tuple[str, ...] = ()


def parse_project(raw: Any, source: str) -> Project:
    """A project file: `code_transfer`, `code`, `python_packages`,
    `data_sources` (as in a mission file) and `quick_run_defaults:
    {per_experiment, max_parallel_experiments, host}`. Earlier names are
    read too (`config/names.py`)."""
    from training_framework.launcher.config.names import project_file_from_earlier

    problems = Problems()
    raw, renamed, renaming_problems = project_file_from_earlier(raw, source)
    for problem in renaming_problems:
        problems.add(problem)
    problems.raise_if_any()
    reader = _Reader(raw, source, problems, PROJECT_KEYS)
    defaults = _Reader(reader.get("quick_run_defaults", {}), f"{source}.quick_run_defaults",
                       problems, {"per_experiment", "max_parallel_experiments", "host"})
    resources = defaults.get("per_experiment", {})
    if not isinstance(resources, Mapping):
        problems.add(f"{source}.quick_run_defaults.per_experiment must be a mapping")
        resources = {}
    max_parallel = defaults.get("max_parallel_experiments")
    if max_parallel is not None and (isinstance(max_parallel, bool)
                                     or not isinstance(max_parallel, int) or max_parallel < 1):
        problems.add(f"{source}.quick_run_defaults.max_parallel_experiments must be a positive "
                     "integer")
    host = defaults.string("host")
    problems.raise_if_any()
    return Project(
        mission_file_keys={key: reader.raw[key] for key in PROJECT_MISSION_FILE_KEYS
                           if key in reader.raw},
        run_defaults=RunDefaults(dict(resources), max_parallel, host),
        renamed=tuple(renamed),
    )


def with_project(raw_mission: Any, project: Project, mission_source: str,
                 project_source: str) -> dict:
    """The mission file (current names) with the project file's keys added;
    a key in both is an error (one setting has one place)."""
    if not isinstance(raw_mission, Mapping):
        return raw_mission
    both = sorted(key for key in project.mission_file_keys if key in raw_mission)
    if both:
        problems = Problems()
        problems.add(f"{mission_source} and its project file {project_source} both set "
                     f"{', '.join(both)}; keep each in one of them")
        problems.raise_if_any()
    return {**raw_mission, **project.mission_file_keys}


def parse_mission_file(raw: Any, source: str) -> MissionFile:
    """A mission file, in the current names (earlier ones are read too, and
    listed in `renamed`)."""
    from training_framework.launcher.config.names import mission_file_from_earlier

    problems = Problems()
    raw, renamed, renaming_problems = mission_file_from_earlier(raw, source)
    for problem in renaming_problems:
        problems.add(problem)
    problems.raise_if_any()
    refused = {
        key: _PROFILE_KEY_MESSAGE
        for key in PROFILE_KEYS | {"profile", "account", "partition", "qos"}
    }
    reader = _Reader(raw, source, problems, MISSION_FILE_KEYS, refused=refused)

    version = reader.get("format_version", required=True)
    if version is not None and version != 1:
        problems.add(f"{source}.format_version must be 1; got {version!r}")
    name = reader.get("name", required=True)
    if name is not None and (problem := check_name(name, f"{source}.name")):
        problems.add(problem)

    transfer = _Reader(reader.get("code_transfer", {}), f"{source}.code_transfer", problems,
                       {"method", "commit", "clone_url"})
    method = transfer.choice("method", tuple(TRANSFER_METHODS), "git_bundle")
    strategy = TRANSFER_METHODS[method or "git_bundle"]
    ref = transfer.string("commit", "HEAD")
    url = transfer.string("clone_url")
    if strategy == "fetch" and url is None:
        problems.add(f"{source}.code_transfer.clone_url is required with method 'git_clone'")
    if strategy == "bundle" and url is not None:
        problems.add(f"{source}.code_transfer.clone_url is only used with method 'git_clone'")

    code = _Reader(reader.get("code", {}, required=True), f"{source}.code",
                   problems, {"framework_from", "python_paths"})
    framework_from = code.choice("framework_from", tuple(FRAMEWORK_FROM), required=True)
    framework = FRAMEWORK_FROM.get(framework_from)
    paths = code.get("python_paths", required=True)
    code_paths: tuple[str, ...] = ()
    if paths is not None:
        if not isinstance(paths, list) or not paths or not all(
                isinstance(path, str) and path and not posixpath.isabs(path)
                and ".." not in posixpath.normpath(path).split("/")
                for path in paths
        ):
            problems.add(
                f"{source}.code.python_paths must be a non-empty list of relative "
                "paths inside the repository"
            )
        else:
            code_paths = tuple(posixpath.normpath(path) for path in paths)

    packages = _Reader(reader.get("python_packages", {}), f"{source}.python_packages", problems,
                       {"requirements_file", "python_version"})
    requirements = packages.string("requirements_file")
    python_pin = packages.get("python_version")
    if python_pin is not None:
        if isinstance(python_pin, (int, float)) and not isinstance(python_pin, bool):
            problems.add(
                f"{source}.python_packages.python_version must be a quoted string such as "
                f"\"3.13\" (YAML reads {python_pin!r} as a number)"
            )
            python_pin = None
        elif not isinstance(python_pin, str) or not re.match(
                r"\A\d+\.\d+(\.\d+)?\Z", python_pin):
            problems.add(
                f"{source}.python_packages.python_version must be X.Y or X.Y.Z; got "
                f"{python_pin!r}"
            )
            python_pin = None

    base = _Reader(reader.get("base", {}, required=True), f"{source}.base", problems,
                   {"config_file", "session_index", "engine_options"})
    config = base.string("config_file", required=True)
    session = base.positive_int("session_index", 0, minimum=0)
    engine_options = base.strings("engine_options")

    sources = reader.get("data_sources", [])
    if not isinstance(sources, list):
        problems.add(f"{source}.data_sources must be a list")
        sources = []

    jobs_raw = reader.get("jobs", required=True)
    groups = []
    if jobs_raw is not None:
        if not isinstance(jobs_raw, Mapping) or not jobs_raw:
            problems.add(f"{source}.jobs must be a non-empty mapping")
        else:
            for job_name, job_raw in jobs_raw.items():
                group = _parse_group(job_name, job_raw, f"{source}.jobs.{job_name}", problems)
                if group is not None:
                    groups.append(group)

    problems.raise_if_any()
    return MissionFile(
        version=version,
        name=name,
        source_strategy=strategy,
        source_ref=ref,
        source_url=url,
        code_framework=framework,
        code_paths=code_paths,
        requirements=requirements,
        python_pin=python_pin,
        training_config=config,
        training_session=session,
        engine_options=engine_options,
        data_sources=tuple(sources),
        groups=tuple(groups),
        raw=dict(raw),
        description=reader.string("note"),
        renamed=tuple(renamed),
    )


def _parse_group(name, raw, where, problems) -> Group | None:
    """One job (inside: a group) of a mission file."""
    if problem := check_name(name, f"{where} (job name)"):
        problems.add(problem)
    reader = _Reader(raw, where, problems,
                     {"per_experiment", "scheduler_options", "max_parallel_experiments",
                      "stop_grace_period", "recipes"})
    resources = parse_resources(reader.get("per_experiment", {}, required=True),
                                f"{where}.per_experiment", problems)
    options = _scheduler_options(reader.get("scheduler_options"),
                                 f"{where}.scheduler_options", problems)
    max_parallel = reader.positive_int("max_parallel_experiments")
    grace = reader.get("stop_grace_period", "120s")
    grace_seconds = parse_duration(grace)
    if grace_seconds is None:
        problems.add(f"{where}.stop_grace_period must be a duration such as 120s; got {grace!r}")
    recipes_raw = reader.get("recipes", required=True)
    experiments = []
    if recipes_raw is not None:
        if not isinstance(recipes_raw, list) or not recipes_raw:
            problems.add(f"{where}.recipes must be a non-empty list")
        else:
            seen = set()
            for index, recipe_raw in enumerate(recipes_raw):
                experiment = _parse_experiment(recipe_raw, f"{where}.recipes[{index}]", problems)
                if experiment is None:
                    continue
                if experiment.name in seen:
                    problems.add(f"{where}: recipe name '{experiment.name}' is used twice")
                seen.add(experiment.name)
                experiments.append(experiment)
    if resources is None or grace_seconds is None:
        return None
    resources = Resources(**{**resources.__dict__, "scheduler_options": options})
    return Group(name, resources, max_parallel, grace_seconds, tuple(experiments))


def _parse_experiment(raw, where, problems) -> Experiment | None:
    """One recipe (inside: an experiment) of a job."""
    reader = _Reader(raw, where, problems,
                     {"name", "overrides", "vary", "config_file", "config_files", "from_mission"})
    name = reader.get("name", required=True)
    config = reader.string("config_file")
    configs = _config_files(reader.get("config_files"), f"{where}.config_files", problems)
    if config is not None and reader.get("config_files") is not None:
        problems.add(f"{where}: config_file and config_files cannot both be given (config_file "
                     "names one session config, config_files several)")
    from_mission = None
    if reader.get("from_mission") is not None:
        from_mission = _parse_from_mission(reader.get("from_mission"), f"{where}.from_mission", problems)
    if name is not None and (problem := check_name(name, f"{where}.name")):
        problems.add(problem)
        name = None
    overrides = reader.get("overrides", [])
    matrix = reader.get("vary", [])
    for key, value, field_name in (
            ("overrides", overrides, "value"), ("vary", matrix, "values"),
    ):
        if not isinstance(value, list):
            problems.add(f"{where}.{key} must be a list")
            continue
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                problems.add(f"{where}.{key}[{index}] must be a mapping")
                continue
            unknown = set(item) - {"component", "path", field_name}
            if unknown:
                problems.add(
                    f"{where}.{key}[{index}]: unknown field(s) "
                    f"{', '.join(sorted(unknown))} (allowed: component, "
                    f"path, {field_name})"
                )
            if field_name not in item:
                problems.add(f"{where}.{key}[{index}].{field_name} is required")
            elif field_name == "values" and (
                    not isinstance(item["values"], list) or not item["values"]
            ):
                problems.add(
                    f"{where}.{key}[{index}].values must be a non-empty list"
                )
    if name is None or not isinstance(overrides, list) or not isinstance(matrix, list):
        return None
    return Experiment(name, tuple(overrides), tuple(matrix), config, from_mission, configs)


def _config_files(raw, where, problems) -> tuple[str, ...]:
    """A recipe's `config_files`: two or more distinct repository paths."""
    if raw is None:
        return ()
    if (not isinstance(raw, list) or len(raw) < 2
            or not all(isinstance(path, str) and path for path in raw)):
        problems.add(f"{where} must be a list of two or more session config paths "
                     "(for one, use config_file)")
        return ()
    if len(set(raw)) != len(raw):
        twice = sorted({path for path in raw if raw.count(path) > 1})
        problems.add(f"{where} names {', '.join(twice)} more than once")
        return ()
    return tuple(raw)


def _parse_from_mission(raw, where, problems) -> FromMission | None:
    """A recipe's `from_mission` (its checkpoints are listed when compiling)."""
    from training_framework.launcher.names import check_mission_id

    if raw is not None and not isinstance(raw, Mapping):
        problems.add(f"{where} must be a mapping with id (and experiments, checkpoint, into)")
        return None
    reader = _Reader(raw, where, problems, FROM_MISSION_KEYS)
    mission_id = reader.get("id", required=True)
    if mission_id is not None and (problem := check_mission_id(mission_id, f"{where}.id")):
        problems.add(problem)
        mission_id = None
    patterns = reader.get("experiments", [])
    if not isinstance(patterns, list) or not all(isinstance(p, str) and p for p in patterns):
        problems.add(f"{where}.experiments must be a list of experiment names or globs "
                     "(such as short/*); leave it out for all of them")
        patterns = None
    elif len(set(patterns)) != len(patterns):
        problems.add(f"{where}.experiments names an experiment or glob twice")
    checkpoint = reader.choice("checkpoint", CHECKPOINT_CHOICES, "latest")
    into = reader.get("into", DEFAULT_CHECKPOINT_INTO)
    if not isinstance(into, Mapping):
        problems.add(f"{where}.into must be an address {{component?, path}}")
        into = None
    elif set(into) - {"component", "path"}:
        problems.add(f"{where}.into: unknown field(s) "
                     f"{', '.join(sorted(set(into) - {'component', 'path'}))} (allowed: component, path)")
        into = None
    if mission_id is None or patterns is None or checkpoint is None or into is None:
        return None
    return FromMission(mission_id, tuple(patterns), checkpoint, dict(into))


# The earlier names (the mission file was called the manifest, a word a
# checkpoint's `manifest.json` already used), kept for code that imports them.
Manifest = MissionFile
parse_manifest = parse_mission_file
MANIFEST_KEYS = MISSION_FILE_KEYS


def __getattr__(name):
    # `SlurmSettings` lives with the Slurm dialect; kept importable from here.
    if name == "SlurmSettings":
        from training_framework.hpc.slurm import SlurmSettings

        return SlurmSettings
    raise AttributeError(name)
