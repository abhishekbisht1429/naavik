"""Earlier names of naavik's settings, and the current ones.

The hosts file (`hosts.yaml`) was the target file and, before that, the
profile file (`hpc.yaml`); its keys were renamed so each says what it holds
(`.context/plan-naming.md`). A file written with the earlier names is still
read: `host_entry_from_earlier` turns one entry into the current names and
says what it renamed, so the caller can warn once. The same table drives
`naavik migrate`, which rewrites a file for good.

Both a name and its earlier name in one place is an error: one setting has
one key.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Scheduler kinds as the hosts file names them -> the scheduler's id in
#: plans and receipts (kept: missions already written name it).
SCHEDULER_KINDS = {"builtin": "host", "slurm": "slurm"}
#: The earlier scheduler names -> the current kinds.
_EARLIER_SCHEDULERS = {"host": "builtin", "slurm": "slurm"}

_DIRECTORIES = {"batches": "missions", "envs": "python_environments", "data": "datasets"}
_SHELL_PHASES = {"common": "everything", "setup": "environment_build",
                 "prepare": "data_preparation", "task": "experiments"}
_PYTHON_ENVIRONMENT = {"existing": "use_existing", "python_source": "interpreter_from",
                       "site_python_identity": "site_interpreter", "backend": "kind",
                       "conda_packages": "conda_packages", "conda_run": "activate_with_conda_run",
                       "bootstrap_python": "bootstrap_interpreter"}
_LOCATIONS = {"job": "compute_node", "login": "login_node"}
_CODE_FROM = {"commit": "commit", "snapshot": "working_tree"}
_DELIVERY = {"fetch": "download_on_host", "upload": "copy_from_here", "present": "already_on_host"}
_SLURM = {"defaults": "sbatch_options", "max_array_size": "max_array_size"}
_SLURM_PARTITIONS = {"gpu": "gpu_jobs", "cpu": "cpu_jobs"}
_BUILTIN_BUSY = {"wait": "wait", "use": "use_anyway"}
#: Top-level keys that keep their meaning under a new name.
_PLAIN = {"path_map": "local_to_host_paths", "upload_limit": "confirm_uploads_over",
          "snapshot_limit": "working_tree_untracked_limit"}


class _Renaming:
    def __init__(self, where: str):
        self.where = where
        self.renamed: list[str] = []
        self.problems: list[str] = []

    def note(self, old: str, new: str) -> None:
        self.renamed.append(f"{old} -> {new}")

    def clash(self, entry: Mapping, old: str, new: str) -> bool:
        if old in entry and new in entry:
            self.problems.append(f"{self.where}: {old} and {new} are the same setting "
                                 f"({new} is its current name); keep {new}")
            return True
        return False


def _renamed_keys(value: Any, names: Mapping, renaming: _Renaming, prefix: str) -> Any:
    """`value` (a mapping) with its keys renamed by `names`; other keys kept
    (the parser reports what it does not know)."""
    if not isinstance(value, Mapping):
        return value
    result = {}
    for key, item in value.items():
        new = names.get(key, key)
        if new != key:
            if new in value:
                renaming.problems.append(f"{renaming.where}.{prefix}: {key} and {new} are the "
                                         f"same setting; keep {new}")
                continue
            renaming.note(f"{prefix}.{key}", f"{prefix}.{new}")
        result[new] = item
    return result


def _renamed_value(value: Any, values: Mapping, renaming: _Renaming, what: str) -> Any:
    if isinstance(value, str) and value in values and values[value] != value:
        renaming.note(f"{what}: {value}", f"{what}: {values[value]}")
        return values[value]
    return value


def kind_of(scheduler_id: str) -> str:
    """The hosts file's name for a scheduler (its id elsewhere)."""
    return {value: key for key, value in SCHEDULER_KINDS.items()}.get(scheduler_id, scheduler_id)


def scheduler_id(kind: str) -> str:
    """The scheduler id of a hosts file's `scheduler.kind`."""
    return SCHEDULER_KINDS.get(kind, kind)


def host_entry_from_earlier(entry: Any, where: str, schedulers=()) -> tuple[Any, list[str], list[str]]:
    """(the entry with current names, what was renamed, problems). An entry
    already in the current names comes back unchanged. `schedulers`: the
    registered scheduler ids (their settings sections had their names)."""
    if not isinstance(entry, Mapping):
        return entry, [], []
    renaming = _Renaming(where)
    entry = dict(entry)

    # How the host is reached: now one key.
    if "ssh_host" in entry and "local" in entry:
        renaming.problems.append(f"{where}: set either ssh_host or local, not both "
                                 "(both are now `connection`)")
    elif not renaming.clash(entry, "ssh_host", "connection") and "ssh_host" in entry:
        renaming.note("ssh_host", "connection: {ssh: ...}")
        entry["connection"] = {"ssh": entry.pop("ssh_host")}
    elif not renaming.clash(entry, "local", "connection") and "local" in entry:
        renaming.note("local: true", "connection: local")
        value = entry.pop("local")
        entry["connection"] = "local" if value is True else {"local": value}

    # The scheduler and its settings: now one mapping with a `kind`.
    sections = {name for name in (*schedulers, *_EARLIER_SCHEDULERS) if name in entry}
    scheduler = entry.get("scheduler")
    if isinstance(scheduler, str):
        kind = _EARLIER_SCHEDULERS.get(scheduler, scheduler)
        renaming.note(f"scheduler: {scheduler} + {scheduler}: {{...}}",
                      f"scheduler: {{kind: {kind}, ...}}")
        settings = entry.pop(scheduler, None) if scheduler in sections else None
        merged: dict = {"kind": kind}
        if settings is not None and not isinstance(settings, Mapping):
            renaming.problems.append(f"{where}.{scheduler} must be a mapping")
        elif settings:
            merged.update(_scheduler_settings(kind, settings, renaming))
        entry["scheduler"] = merged
        sections.discard(scheduler)
    for other in sorted(sections):
        current = entry["scheduler"].get("kind") if isinstance(entry.get("scheduler"), Mapping) \
            else entry.get("scheduler")
        renaming.problems.append(f"{where}.{other}: settings for scheduler '{other}', "
                                 f"but the host uses '{current}'")
        entry.pop(other)

    for old, new, names in (("roots", "directories", _DIRECTORIES),
                            ("hooks", "shell_commands_before", _SHELL_PHASES),
                            ("env", "python_environment", _PYTHON_ENVIRONMENT)):
        if not renaming.clash(entry, old, new) and old in entry:
            renaming.note(old, new)
            entry[new] = _renamed_keys(entry.pop(old), names, renaming, new)

    # Where the environment is built and the data prepared, and with what.
    for location, step in (("setup_location", "environment_build"),
                           ("prepare_location", "data_preparation")):
        if location in entry:
            step_entry = dict(entry.get(step) or {})
            if "on" in step_entry:
                renaming.problems.append(f"{where}: {location} and {step}.on are the same "
                                         f"setting; keep {step}.on")
                continue
            renaming.note(location, f"{step}.on")
            value = entry.pop(location)
            step_entry["on"] = _LOCATIONS.get(value, value)
            entry[step] = step_entry
    if "setup_resources" in entry:
        build = dict(entry.get("environment_build") or {})
        if "allocation" in build:
            renaming.problems.append(f"{where}: setup_resources and environment_build."
                                     "allocation are the same setting; keep the second")
        else:
            allocation = entry.pop("setup_resources")
            renaming.note("setup_resources", "environment_build.allocation")
            build["allocation"] = allocation
            entry["environment_build"] = build
            # Data steps used the environment build's resources before.
            preparation = dict(entry.get("data_preparation") or {})
            preparation.setdefault("allocation", allocation)
            entry["data_preparation"] = preparation

    for old, new in _PLAIN.items():
        if not renaming.clash(entry, old, new) and old in entry:
            renaming.note(old, new)
            entry[new] = entry.pop(old)
    if not renaming.clash(entry, "worktree", "run_code_from") and "worktree" in entry:
        renaming.note("worktree", "run_code_from")
        value = entry.pop("worktree")
        entry["run_code_from"] = _CODE_FROM.get(value, value)
    if not renaming.clash(entry, "data_modes", "dataset_delivery") and "data_modes" in entry:
        renaming.note("data_modes", "dataset_delivery")
        modes = entry.pop("data_modes")
        entry["dataset_delivery"] = ({key: _DELIVERY.get(value, value) for key, value in modes.items()}
                                     if isinstance(modes, Mapping) else modes)
    return entry, renaming.renamed, renaming.problems


def _scheduler_settings(kind: str, settings: Mapping, renaming: _Renaming) -> dict:
    """A scheduler's settings with current names (other kinds: as given)."""
    if kind == "slurm":
        settings = _renamed_keys(settings, _SLURM, renaming, "scheduler")
        if isinstance(settings.get("partitions"), Mapping):
            settings["partitions"] = _renamed_keys(settings["partitions"], _SLURM_PARTITIONS,
                                                   renaming, "scheduler.partitions")
    elif kind == "builtin":
        settings = _renamed_keys(settings, {"busy_gpus": "when_gpus_busy"}, renaming, "scheduler")
        if "when_gpus_busy" in settings:
            settings["when_gpus_busy"] = _renamed_value(settings["when_gpus_busy"], _BUILTIN_BUSY,
                                                        renaming, "scheduler.when_gpus_busy")
    return dict(settings)


def hosts_file_from_earlier(raw: Any, source: str, schedulers=()) -> tuple[Any, list[str], list[str]]:
    """The whole hosts file with current names: top-level keys, then every
    entry. (file, what was renamed, problems)."""
    if not isinstance(raw, Mapping):
        return raw, [], []
    renaming = _Renaming(source)
    raw = dict(raw)
    if not renaming.clash(raw, "default", "default_host") and "default" in raw:
        renaming.note("default", "default_host")
        raw["default_host"] = raw.pop("default")
    earlier = [key for key in ("targets", "profiles") if key in raw]
    if len(earlier) == 2:
        renaming.problems.append(f"{source}: has both 'targets' and 'profiles' (earlier names "
                                 "of 'hosts'); keep one, named hosts")
    elif earlier and "hosts" in raw:
        renaming.problems.append(f"{source}: has both '{earlier[0]}' and 'hosts' ('{earlier[0]}' "
                                 "is its earlier name); keep hosts")
    elif earlier:
        renaming.note(earlier[0], "hosts")
        raw["hosts"] = raw.pop(earlier[0])
    hosts = raw.get("hosts")
    if isinstance(hosts, Mapping):
        converted = {}
        for name, entry in hosts.items():
            new, renamed, problems = host_entry_from_earlier(entry, f"{source}:hosts.{name}",
                                                             schedulers)
            converted[name] = new
            renaming.renamed += [f"hosts.{name}: {item}" for item in renamed]
            renaming.problems += problems
        raw["hosts"] = converted
    return raw, renaming.renamed, renaming.problems


# Mission files and the project file -----------------------------------------
#
# The mission file was the experiment file and, before that, the manifest;
# a job was a group, a recipe an experiment. The project file shares the
# mission file's names for what both may hold.

_CODE_TRANSFER = {"strategy": "method", "ref": "commit", "url": "clone_url"}
_TRANSFER_METHODS = {"bundle": "git_bundle", "fetch": "git_clone"}
_CODE = {"framework": "framework_from", "paths": "python_paths"}
_FRAMEWORK_FROM = {"dependency": "installed", "checkout": "this_repository"}
_PYTHON_PACKAGES = {"requirements": "requirements_file", "python": "python_version"}
_BASE = {"config": "config_file", "session": "session_index"}
_JOB = {"resources": "per_experiment", "max_parallel": "max_parallel_experiments",
        "stop_grace": "stop_grace_period", "experiments": "recipes"}
_RECIPE = {"config": "config_file", "matrix": "vary"}
_SOURCE_KINDS = {"upload": "copy_from_here", "cluster": "already_on_host",
                 "prepare": "prepare_on_host"}
_SOURCE_SPEC = {"version": "content_version", "inputs": "rerun_when_changed",
                "resources": "allocation"}
_QUICK_RUN = {"resources": "per_experiment", "max_parallel": "max_parallel_experiments",
              "target": "host"}


def _section(entry: dict, old: str, new: str, names: Mapping, renaming: _Renaming) -> None:
    """Rename section `old` to `new` and its keys by `names` (in place)."""
    if renaming.clash(entry, old, new) or old not in entry:
        return
    renaming.note(old, new)
    entry[new] = _renamed_keys(entry.pop(old), names, renaming, new)


def _shared_sections(entry: dict, renaming: _Renaming) -> None:
    """What a mission file and the project file both hold (in place)."""
    _section(entry, "source", "code_transfer", _CODE_TRANSFER, renaming)
    transfer = entry.get("code_transfer")
    if isinstance(transfer, Mapping) and "method" in transfer:
        transfer["method"] = _renamed_value(transfer["method"], _TRANSFER_METHODS, renaming,
                                            "code_transfer.method")
    code = entry.get("code")
    if isinstance(code, Mapping):
        code = entry["code"] = _renamed_keys(code, _CODE, renaming, "code")
        if "framework_from" in code:
            code["framework_from"] = _renamed_value(code["framework_from"], _FRAMEWORK_FROM,
                                                    renaming, "code.framework_from")
    _section(entry, "environment", "python_packages", _PYTHON_PACKAGES, renaming)
    if "data" in entry:
        if renaming.clash(entry, "data", "data_sources"):
            return
        data = entry.pop("data")
        renaming.note("data.sources", "data_sources")
        sources = data.get("sources", []) if isinstance(data, Mapping) else data
        if isinstance(data, Mapping) and set(data) - {"sources"}:
            renaming.problems.append(f"{renaming.where}.data: unknown key(s) "
                                     f"{', '.join(sorted(set(data) - {'sources'}))}")
        entry["data_sources"] = sources
    if isinstance(entry.get("data_sources"), list):
        entry["data_sources"] = [_data_source(item, index, renaming)
                                 for index, item in enumerate(entry["data_sources"])]


def _data_source(item: Any, index: int, renaming: _Renaming) -> Any:
    if not isinstance(item, Mapping):
        return item
    item = dict(item)
    where = f"data_sources[{index}]"
    if not renaming.clash(item, "target", "address") and "target" in item:
        renaming.note(f"{where}.target", f"{where}.address")
        item["address"] = item.pop("target")
    for old, new in _SOURCE_KINDS.items():
        if not renaming.clash(item, old, new) and old in item:
            renaming.note(f"{where}.{old}", f"{where}.{new}")
            item[new] = _renamed_keys(item.pop(old), _SOURCE_SPEC, renaming, f"{where}.{new}")
    return item


def mission_file_from_earlier(raw: Any, source: str) -> tuple[Any, list[str], list[str]]:
    """A mission file with current names: (file, what was renamed, problems)."""
    if not isinstance(raw, Mapping):
        return raw, [], []
    renaming = _Renaming(source)
    entry = dict(raw)
    for old, new in (("version", "format_version"), ("description", "note"),
                     ("project", "project_file")):
        if not renaming.clash(entry, old, new) and old in entry:
            renaming.note(old, new)
            entry[new] = entry.pop(old)
    _shared_sections(entry, renaming)
    _section(entry, "training", "base", _BASE, renaming)
    if not renaming.clash(entry, "groups", "jobs") and "groups" in entry:
        renaming.note("groups", "jobs")
        entry["jobs"] = entry.pop("groups")
    jobs = entry.get("jobs")
    if isinstance(jobs, Mapping):
        renamed_jobs = {}
        for name, job in jobs.items():
            job = _renamed_keys(job, _JOB, renaming, f"jobs.{name}")
            if isinstance(job, Mapping) and isinstance(job.get("recipes"), list):
                job["recipes"] = [_renamed_keys(recipe, _RECIPE, renaming, f"jobs.{name}.recipes")
                                  for recipe in job["recipes"]]
            renamed_jobs[name] = job
        entry["jobs"] = renamed_jobs
    return entry, renaming.renamed, renaming.problems


def project_file_from_earlier(raw: Any, source: str) -> tuple[Any, list[str], list[str]]:
    """The project file with current names: (file, what was renamed, problems)."""
    if not isinstance(raw, Mapping):
        return raw, [], []
    renaming = _Renaming(source)
    entry = dict(raw)
    _shared_sections(entry, renaming)
    _section(entry, "run_defaults", "quick_run_defaults", _QUICK_RUN, renaming)
    return entry, renaming.renamed, renaming.problems
