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
from dataclasses import dataclass, field
from typing import Any

from training_framework.launcher.errors import Problems
from training_framework.launcher.names import check_name

_SIZE = re.compile(r"\A(\d+)([KMGT]?)\Z")
_SIZE_FACTORS = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
_WALLTIME = re.compile(r"\A(?:(\d+)-)?(\d{1,2}):(\d{2}):(\d{2})\Z")
_DURATION = re.compile(r"\A(\d+)([smh])\Z")
_DURATION_FACTORS = {"s": 1, "m": 60, "h": 3600}

PROFILE_KEYS = frozenset({
    "ssh_host", "local", "roots", "scheduler", "slurm", "hooks", "env",
    "setup_location", "prepare_location", "path_map", "upload_limit",
    "setup_resources", "worktree", "snapshot_limit", "data_modes",
})
#: How a data requirement reaches a target (profile `data_modes:`; not
#: `data:`, which is the manifest's data sources).
DATA_MODES = ("fetch", "upload", "present")
#: How a batch takes your code: the commit you made (`commit`), or a commit
#: made of your working tree as it is (`snapshot`).
WORKTREE_MODES = ("commit", "snapshot")
DEFAULT_SNAPSHOT_LIMIT = "50M"
DEFAULT_SETUP_RESOURCES = {"walltime": "01:00:00", "cpus": 4, "memory": "16G"}
MANIFEST_KEYS = frozenset({
    "version", "name", "source", "code", "environment", "training", "data",
    "groups", "project", "description",
})
#: The project file (`naavik-project.yaml`): what a project's batches share.
PROJECT_FILE = "naavik-project.yaml"
#: Manifest keys a project file may hold (a manifest naming the project may
#: not repeat them: one source per key).
PROJECT_MANIFEST_KEYS = ("source", "code", "environment", "data")
PROJECT_KEYS = frozenset(PROJECT_MANIFEST_KEYS) | {"run_defaults"}
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
    if env_reader.raw.get("existing") != CURRENT_ENVIRONMENT:
        return env_reader.absolute_path("existing")
    if not local:
        problems.add(f"{where}.env.existing: 'current' (this interpreter's environment) "
                     "needs a target on this machine (local: true)")
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
    #: How data requirements reach this target; None: not managed.
    data: "DataModes | None" = None

    @property
    def slurm(self):
        """Compatibility alias for the Slurm settings of a Slurm profile."""
        if self.scheduler != "slurm":
            raise AttributeError("this profile does not use Slurm")
        return self.scheduler_settings


@dataclass(frozen=True)
class DataModes:
    """Profile `data_modes: {default: MODE, <requirement name>: MODE}`."""

    default: str
    named: tuple[tuple[str, str], ...]

    def mode(self, name: str) -> str:
        return dict(self.named).get(name, self.default)


def _parse_data_modes(raw: Any, where: str, problems: Problems) -> DataModes | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        problems.add(f"{where} must be a mapping of a requirement name (or 'default') to "
                     f"one of {', '.join(DATA_MODES)}")
        return None
    default, named = "fetch", []
    for key, mode in raw.items():
        if mode not in DATA_MODES:
            problems.add(f"{where}.{key} must be one of {', '.join(DATA_MODES)}; got {mode!r}")
            continue
        if key == "default":
            default = mode
        elif problem := check_name(key, f"{where} (a requirement name)"):
            problems.add(problem)
        else:
            named.append((key, mode))
    return DataModes(default, tuple(sorted(named)))


def parse_profiles(raw: Any, selected: str | None, source: str) -> Profile:
    """Select a profile from a target file: `{default: <name>, targets:
    {<name>: {...}}}` (`profiles:` is the earlier name of `targets:`; a file
    has exactly one of them); `selected` (from `--target`) wins over
    `default`."""
    problems = Problems()
    top = _Reader(raw, source, problems, {"default", "profiles", "targets"})
    keys = [key for key in ("targets", "profiles") if key in top.raw]
    if len(keys) == 2:
        problems.add(f"{source}: has both 'targets' and 'profiles' ('profiles' is the "
                     "earlier name of 'targets'); keep one")
        problems.raise_if_any()
    entries_key = keys[0] if keys else "targets"
    profiles = top.get(entries_key, required=True)
    if not isinstance(profiles, Mapping) or not profiles:
        if profiles is not None:
            problems.add(f"{source}.{entries_key} must be a non-empty mapping")
        problems.raise_if_any()
    name = selected or top.string("default")
    if name is None:
        problems.add(
            f"{source}: no profile selected; pass --profile or set 'default'"
        )
    elif name not in profiles:
        problems.add(
            f"{source}: profile '{name}' not found (have: "
            f"{', '.join(sorted(profiles))})"
        )
    problems.raise_if_any()
    profile = _parse_profile(profiles[name], f"{source}:profiles.{name}", name, problems)
    problems.raise_if_any()
    return profile


def _parse_profile(raw, where, name, problems) -> Profile:
    from training_framework.launcher.scheduler import SCHEDULERS

    reader = _Reader(raw, where, problems, PROFILE_KEYS | set(SCHEDULERS))
    # How the target is reached: exactly one of `ssh_host` and `local: true`.
    ssh_host = reader.string("ssh_host")
    local = False
    if "local" in reader.raw:
        if reader.raw["local"] is not True:
            problems.add(f"{where}.local must be true (to use this machine) or left out")
        else:
            local = True
    if local and ssh_host is not None:
        problems.add(f"{where}: set either ssh_host (a machine reached over SSH) or "
                     "local: true (this machine), not both")
    elif not local and ssh_host is None and "ssh_host" not in reader.raw:
        problems.add(f"{where}: set ssh_host (a machine reached over SSH) or "
                     "local: true (this machine)")

    roots_reader = _Reader(
        reader.get("roots", {}, required=True), f"{where}.roots", problems,
        {"batches", "envs", "data"},
    )
    roots = {
        key: roots_reader.absolute_path(key, required=True)
        for key in ("batches", "envs", "data")
    }

    from training_framework.launcher.scheduler import SCHEDULERS

    scheduler = reader.choice("scheduler", tuple(SCHEDULERS), required=True)
    settings = None
    if scheduler is not None:
        settings = SCHEDULERS[scheduler].parse_settings(
            reader.get(scheduler, {}, required=True), f"{where}.{scheduler}", problems, _Reader,
        )
    for other in SCHEDULERS:
        if other != scheduler and other in reader.raw:
            problems.add(f"{where}.{other}: settings for scheduler '{other}', but "
                         f"the profile uses '{scheduler}'")

    hooks_reader = _Reader(
        reader.get("hooks", {}), f"{where}.hooks", problems, set(HOOK_PHASES),
    )
    hooks = {phase: hooks_reader.strings(phase) for phase in HOOK_PHASES}

    env_reader = _Reader(
        reader.get("env", {}), f"{where}.env", problems,
        {"backend", "python_source", "site_python_identity", "existing",
         "conda_packages", "conda_run", "bootstrap_python"},
    )
    identity = None
    if "site_python_identity" in env_reader.raw:
        id_reader = _Reader(
            env_reader.raw["site_python_identity"],
            f"{where}.env.site_python_identity", problems,
            {"implementation", "version", "executable", "platform"},
        )
        fields = {
            key: id_reader.string(key, required=True)
            for key in ("implementation", "version", "executable", "platform")
        }
        if all(fields.values()):
            identity = PythonIdentity(**fields)
    env = EnvSettings(
        backend=env_reader.choice("backend", ("venv", "conda"), "venv"),
        python_source=env_reader.choice(
            "python_source", ("site", "uv", "conda"), "site",
        ),
        site_python_identity=identity,
        existing=_existing_environment(env_reader, local, where, problems),
        conda_packages=env_reader.strings("conda_packages"),
        conda_run=env_reader.boolean("conda_run", False),
        bootstrap_python=env_reader.string("bootstrap_python"),
    )

    path_map_raw = reader.get("path_map", {})
    path_map = []
    if not isinstance(path_map_raw, Mapping):
        problems.add(f"{where}.path_map must map local prefixes to cluster prefixes")
    else:
        for local, remote in path_map_raw.items():
            if not (isinstance(local, str) and posixpath.isabs(local)
                    and isinstance(remote, str) and posixpath.isabs(remote)):
                problems.add(
                    f"{where}.path_map: '{local}' -> '{remote}' must map an "
                    "absolute local prefix to an absolute cluster prefix"
                )
                continue
            path_map.append((local.rstrip("/") or "/", remote.rstrip("/") or "/"))

    upload_limit = parse_size(reader.get("upload_limit", "20G"))
    if upload_limit is None:
        problems.add(f"{where}.upload_limit must be a size such as 20G")

    snapshot_limit = parse_size(reader.get("snapshot_limit", DEFAULT_SNAPSHOT_LIMIT))
    if snapshot_limit is None:
        problems.add(f"{where}.snapshot_limit must be a size such as 50M")

    setup_resources = parse_resources(
        reader.get("setup_resources", DEFAULT_SETUP_RESOURCES),
        f"{where}.setup_resources", problems,
    )

    return Profile(
        name=name,
        ssh_host=ssh_host,
        roots=roots,
        scheduler=scheduler,
        scheduler_settings=settings,
        hooks=hooks,
        env=env,
        setup_location=reader.choice("setup_location", LOCATIONS, "job"),
        prepare_location=reader.choice("prepare_location", LOCATIONS, "job"),
        path_map=tuple(sorted(path_map)),
        upload_limit=upload_limit or 0,
        setup_resources=setup_resources,
        local=local,
        worktree=reader.choice("worktree", WORKTREE_MODES, "commit"),
        snapshot_limit=snapshot_limit or 0,
        data=_parse_data_modes(reader.get("data_modes"), f"{where}.data_modes", problems),
    )


# Manifest -----------------------------------------------------------------


@dataclass(frozen=True)
class Resources:
    walltime: str
    walltime_seconds: int
    gpus: int
    cpus: int
    memory: str
    scheduler_options: dict


@dataclass(frozen=True)
class Experiment:
    name: str
    overrides: tuple[dict, ...]   # raw {component?, path, value}
    matrix: tuple[dict, ...]      # raw {component?, path, values}


@dataclass(frozen=True)
class Group:
    name: str
    resources: Resources
    max_parallel: int | None
    stop_grace_seconds: int
    experiments: tuple[Experiment, ...]


@dataclass(frozen=True)
class Manifest:
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
    #: Free text: why this batch runs (`naavik run --note`).
    description: str | None = None


_PROFILE_KEY_MESSAGE = (
    "is a site concern and belongs in the profile, not the committed manifest"
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
    #: The manifest keys it provides, as written.
    manifest_keys: dict
    run_defaults: RunDefaults


def parse_project(raw: Any, source: str) -> Project:
    """A project file: `source`, `code`, `environment`, `data` (as in a
    manifest) and `run_defaults: {resources, max_parallel, target}`."""
    problems = Problems()
    reader = _Reader(raw, source, problems, PROJECT_KEYS)
    defaults = _Reader(reader.get("run_defaults", {}), f"{source}.run_defaults", problems,
                       {"resources", "max_parallel", "target"})
    resources = defaults.get("resources", {})
    if not isinstance(resources, Mapping):
        problems.add(f"{source}.run_defaults.resources must be a mapping")
        resources = {}
    max_parallel = defaults.get("max_parallel")
    if max_parallel is not None and (isinstance(max_parallel, bool)
                                     or not isinstance(max_parallel, int) or max_parallel < 1):
        problems.add(f"{source}.run_defaults.max_parallel must be a positive integer")
    target = defaults.string("target")
    problems.raise_if_any()
    return Project(
        manifest_keys={key: reader.raw[key] for key in PROJECT_MANIFEST_KEYS if key in reader.raw},
        run_defaults=RunDefaults(dict(resources), max_parallel, target),
    )


def with_project(raw_manifest: Any, project: Project, manifest_source: str,
                 project_source: str) -> dict:
    """The manifest with the project file's keys added; a key in both is an
    error (one source per key)."""
    if not isinstance(raw_manifest, Mapping):
        return raw_manifest
    both = sorted(key for key in project.manifest_keys if key in raw_manifest)
    if both:
        problems = Problems()
        problems.add(f"{manifest_source} and its project file {project_source} both set "
                     f"{', '.join(both)}; keep each in one of them")
        problems.raise_if_any()
    return {**raw_manifest, **project.manifest_keys}


def parse_manifest(raw: Any, source: str) -> Manifest:
    problems = Problems()
    refused = {
        key: _PROFILE_KEY_MESSAGE
        for key in PROFILE_KEYS | {"profile", "account", "partition", "qos"}
    }
    reader = _Reader(raw, source, problems, MANIFEST_KEYS, refused=refused)

    version = reader.get("version", required=True)
    if version is not None and version != 1:
        problems.add(f"{source}.version must be 1; got {version!r}")
    name = reader.get("name", required=True)
    if name is not None and (problem := check_name(name, f"{source}.name")):
        problems.add(problem)

    src = _Reader(reader.get("source", {}), f"{source}.source", problems,
                  {"strategy", "ref", "url"})
    strategy = src.choice("strategy", ("bundle", "fetch"), "bundle")
    ref = src.string("ref", "HEAD")
    url = src.string("url")
    if strategy == "fetch" and url is None:
        problems.add(f"{source}.source.url is required with strategy 'fetch'")
    if strategy == "bundle" and url is not None:
        problems.add(f"{source}.source.url is only used with strategy 'fetch'")

    code = _Reader(reader.get("code", {}, required=True), f"{source}.code",
                   problems, {"framework", "paths"})
    framework = code.choice("framework", ("checkout", "dependency"), required=True)
    paths = code.get("paths", required=True)
    code_paths: tuple[str, ...] = ()
    if paths is not None:
        if not isinstance(paths, list) or not paths or not all(
                isinstance(path, str) and path and not posixpath.isabs(path)
                and ".." not in posixpath.normpath(path).split("/")
                for path in paths
        ):
            problems.add(
                f"{source}.code.paths must be a non-empty list of relative "
                "paths inside the repository"
            )
        else:
            code_paths = tuple(posixpath.normpath(path) for path in paths)

    environment = _Reader(reader.get("environment", {}),
                          f"{source}.environment", problems,
                          {"requirements", "python"})
    requirements = environment.string("requirements")
    python_pin = environment.get("python")
    if python_pin is not None:
        if isinstance(python_pin, (int, float)) and not isinstance(python_pin, bool):
            problems.add(
                f"{source}.environment.python must be a quoted string such as "
                f"\"3.13\" (YAML reads {python_pin!r} as a number)"
            )
            python_pin = None
        elif not isinstance(python_pin, str) or not re.match(
                r"\A\d+\.\d+(\.\d+)?\Z", python_pin):
            problems.add(
                f"{source}.environment.python must be X.Y or X.Y.Z; got "
                f"{python_pin!r}"
            )
            python_pin = None

    training = _Reader(reader.get("training", {}, required=True),
                       f"{source}.training", problems,
                       {"config", "session", "engine_options"})
    config = training.string("config", required=True)
    session = training.positive_int("session", 0, minimum=0)
    engine_options = training.strings("engine_options")

    data = _Reader(reader.get("data", {}), f"{source}.data", problems,
                   {"sources"})
    sources = data.get("sources", [])
    if not isinstance(sources, list):
        problems.add(f"{source}.data.sources must be a list")
        sources = []

    groups_raw = reader.get("groups", required=True)
    groups = []
    if groups_raw is not None:
        if not isinstance(groups_raw, Mapping) or not groups_raw:
            problems.add(f"{source}.groups must be a non-empty mapping")
        else:
            for group_name, group_raw in groups_raw.items():
                group = _parse_group(group_name, group_raw,
                                     f"{source}.groups.{group_name}", problems)
                if group is not None:
                    groups.append(group)

    problems.raise_if_any()
    return Manifest(
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
        description=reader.string("description"),
    )


def _parse_group(name, raw, where, problems) -> Group | None:
    if problem := check_name(name, f"{where} (group name)"):
        problems.add(problem)
    reader = _Reader(raw, where, problems,
                     {"resources", "scheduler_options", "max_parallel",
                      "stop_grace", "experiments"})
    resources = parse_resources(reader.get("resources", {}, required=True),
                                f"{where}.resources", problems)
    options = _scheduler_options(reader.get("scheduler_options"),
                                 f"{where}.scheduler_options", problems)
    max_parallel = reader.positive_int("max_parallel")
    grace = reader.get("stop_grace", "120s")
    grace_seconds = parse_duration(grace)
    if grace_seconds is None:
        problems.add(f"{where}.stop_grace must be a duration such as 120s; got {grace!r}")
    experiments_raw = reader.get("experiments", required=True)
    experiments = []
    if experiments_raw is not None:
        if not isinstance(experiments_raw, list) or not experiments_raw:
            problems.add(f"{where}.experiments must be a non-empty list")
        else:
            seen = set()
            for index, experiment_raw in enumerate(experiments_raw):
                experiment = _parse_experiment(
                    experiment_raw, f"{where}.experiments[{index}]", problems,
                )
                if experiment is None:
                    continue
                if experiment.name in seen:
                    problems.add(
                        f"{where}: experiment name '{experiment.name}' is used twice"
                    )
                seen.add(experiment.name)
                experiments.append(experiment)
    if resources is None or grace_seconds is None:
        return None
    resources = Resources(**{**resources.__dict__, "scheduler_options": options})
    return Group(name, resources, max_parallel, grace_seconds, tuple(experiments))


def _parse_experiment(raw, where, problems) -> Experiment | None:
    reader = _Reader(raw, where, problems, {"name", "overrides", "matrix"})
    name = reader.get("name", required=True)
    if name is not None and (problem := check_name(name, f"{where}.name")):
        problems.add(problem)
        name = None
    overrides = reader.get("overrides", [])
    matrix = reader.get("matrix", [])
    for key, value, field_name in (
            ("overrides", overrides, "value"), ("matrix", matrix, "values"),
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
    return Experiment(name, tuple(overrides), tuple(matrix))


def __getattr__(name):
    # `SlurmSettings` lives with the Slurm dialect; kept importable from here.
    if name == "SlurmSettings":
        from training_framework.hpc.slurm import SlurmSettings

        return SlurmSettings
    raise AttributeError(name)
