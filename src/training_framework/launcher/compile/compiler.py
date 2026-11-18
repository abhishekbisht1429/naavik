"""The pure compiler: manifest + profile + dialect + repository snapshot +
local facts + clock -> `ExecutionPlan`.

Every offline check (building block B8) happens here; problems are collected
and raised together as one `PlanError`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import dataclasses
import itertools
import shlex
import tomllib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any

import yaml

from training_framework.config_files import (
    ConfigFileError,
    RepositoryFiles,
    check_no_interpolation,
    load_config,
)
from training_framework.launcher import EXECUTION_PROTOCOL_VERSION
from training_framework.launcher.config import address as addresses
from training_framework.launcher.compile import data as data_module
from training_framework.launcher.compile.data import LocalFacts
from training_framework.launcher.errors import HpcError, Problems
from training_framework.launcher.names import hash_bytes, hash_value
from training_framework.launcher.compile.plan import (
    Artifact,
    ExecutionContext,
    ExecutionPlan,
    JobNode,
    Task,
)
from training_framework.launcher.config.python_rules import (
    interpreter_expectation,
    parse_requires_python,
)
from training_framework.launcher.ports.repository import RepositoryReader
from training_framework.launcher.config.requirements import (
    Requirements,
    RequirementsError,
    parse_requirements,
)
from training_framework.launcher.config.schema import (
    ExperimentFile,
    Profile,
    parse_experiment_file,
    parse_project,
    with_project,
)
from training_framework.launcher.ports.shell import exec_script
from training_framework.launcher.scheduler import get_scheduler
from training_framework.launcher.config.topology import check_topology, parse_engine_options

UNSET_VARIABLES = ("WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
SET_VARIABLES = (("PYTHONNOUSERSITE", "1"),)
FRAMEWORK_DISTRIBUTION = "training-framework"


@dataclasses.dataclass(frozen=True)
class Compiled:
    plan: ExecutionPlan
    manifest: ExperimentFile
    suggestions: list[dict]
    #: What the batch's own code said about its tasks; None: not asked.
    code_facts: "CodeFacts | None" = None


@dataclass(frozen=True)
class TaskRequirement:
    """Data one component of one task declares (`training_framework.data`)."""

    session: int
    component: str
    #: The component class, dotted.
    class_path: str
    #: `DataRequirement.as_json()`.
    requirement: dict
    #: The value of the requirement's config field, as the config has it.
    path: str
    location: str


@dataclass
class CodeFacts:
    """The answer of the `code_facts` port of `compile_plan`."""

    problems: list[str] = field(default_factory=list)
    #: Printed once by the caller; not part of the plan.
    notes: list[str] = field(default_factory=list)
    #: Task name -> its data requirements.
    requirements: dict[str, tuple[TaskRequirement, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Selection:
    """Part of an experiment file: whole groups, and `GROUP/EXPERIMENT`s."""

    groups: tuple[str, ...] = ()
    experiments: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.groups or self.experiments)

    def as_json(self) -> dict:
        return {key: list(value) for key, value in
                (("groups", self.groups), ("experiments", self.experiments)) if value}


def _select(manifest: ExperimentFile, selection: Selection | None, problems: Problems) -> ExperimentFile:
    """The manifest with only the chosen groups and experiments (task names
    and numbers stay as the whole manifest makes them)."""
    import dataclasses as _dataclasses

    if not selection:
        return manifest
    groups = {group.name: group for group in manifest.groups}
    for name in selection.groups:
        if name not in groups:
            problems.add(f"--group {name}: the experiment file has no such group "
                         f"(it has: {', '.join(groups)})")
    chosen: dict[str, set[str] | None] = {name: None for name in selection.groups if name in groups}
    for item in selection.experiments:
        group, slash, experiment = item.partition("/")
        if not slash or not experiment:
            problems.add(f"--experiment {item}: name it GROUP/EXPERIMENT")
            continue
        if group not in groups:
            problems.add(f"--experiment {item}: the experiment file has no group '{group}' "
                         f"(it has: {', '.join(groups)})")
            continue
        names = [e.name for e in groups[group].experiments]
        if experiment not in names:
            problems.add(f"--experiment {item}: group '{group}' has no experiment "
                         f"'{experiment}' (it has: {', '.join(names)})")
            continue
        if chosen.get(group, set()) is not None:            # not already the whole group
            chosen.setdefault(group, set()).add(experiment)
    kept = []
    for group in manifest.groups:
        if group.name not in chosen:
            continue
        wanted = chosen[group.name]
        experiments = tuple(e for e in group.experiments if wanted is None or e.name in wanted)
        kept.append(_dataclasses.replace(group, experiments=experiments))
    return _dataclasses.replace(manifest, groups=tuple(kept))


#: `code_facts(sha, manifest, managed, tasks)`, tasks being (name, config) in
#: order: the batch's own code asked about its expanded task configs (as the
#: manifest makes them, before data paths are rewritten for the target).
CodeFactsPort = Callable[[str, ExperimentFile, bool, list[tuple[str, dict]]], CodeFacts]


def _load_yaml(data: bytes, source: str, problems: Problems) -> Any:
    try:
        return yaml.safe_load(data)
    except yaml.YAMLError as error:
        problems.add(f"{source} is not valid YAML ({error})")
        return None


def compile_plan(
        *,
        manifest_path: str,
        repository: RepositoryReader,
        profile: Profile,
        facts: LocalFacts,
        clock: datetime,
        fresh_env: bool = False,
        manifest_data: bytes | None = None,
        code_facts: CodeFactsPort | None = None,
        selection: Selection | None = None,
) -> Compiled:
    """`manifest_data`: a manifest given as data rather than read from the
    repository (`naavik run` builds one); `manifest_path` then only names
    it in messages, and everything else is still read at the batch's
    commit. `code_facts`: asked once the tasks are expanded (problems it
    reports stop the compile there); None: not asked. `selection`: only
    these groups and experiments become tasks (None: all)."""
    problems = Problems()

    # The batch runs one exact tree: HEAD (worktree: commit, which must then
    # be clean), or the working tree made into a commit on HEAD (snapshot).
    snapshot = None
    if profile.worktree == "snapshot":
        try:
            snapshot = repository.snapshot(profile.snapshot_limit)
        except HpcError as error:
            problems.add(str(error))
            problems.raise_if_any()
        head = snapshot.sha
    else:
        if not repository.is_clean():
            problems.add(
                "the repository has uncommitted changes or untracked files; "
                "commit or ignore them (missions run an exact commit), or set "
                "run_code_from: working_tree on the host (hosts.yaml)"
            )
        head = repository.resolve("HEAD")
    head_bytes = manifest_data if manifest_data is not None else repository.read(head, manifest_path)
    if head_bytes is None:
        problems.add(f"{manifest_path} is not committed")
        problems.raise_if_any()
    raw_manifest = _load_yaml(head_bytes, manifest_path, problems)
    problems.raise_if_any()
    # A project file the manifest names adds the keys a project's batches
    # share; read at the same commit as the manifest.
    project_path = project_bytes = None
    if isinstance(raw_manifest, Mapping) and raw_manifest.get("project") is not None:
        project_path = raw_manifest["project"]
        if not isinstance(project_path, str) or not project_path:
            problems.add(f"{manifest_path}.project must be a path in the repository")
        else:
            project_bytes = repository.read(head, project_path)
            if project_bytes is None:
                problems.add(f"project file '{project_path}' does not exist at {head[:12]} "
                             f"(named by {manifest_path}.project)")
        problems.raise_if_any()
        project_raw = _load_yaml(project_bytes, project_path, problems)
        problems.raise_if_any()
        raw_manifest = with_project(raw_manifest, parse_project(project_raw, project_path),
                                    manifest_path, project_path)
    manifest = parse_experiment_file(raw_manifest, manifest_path)

    if snapshot is not None:
        if manifest.source_ref != "HEAD":
            problems.add(f"source.ref '{manifest.source_ref}' cannot be used with run_code_from: "
                         "working_tree (the working tree is taken on HEAD)")
        if manifest.source_strategy == "fetch":
            problems.add("source.strategy 'fetch' cannot stage a snapshot (it exists only in "
                         "this repository); use 'bundle'")
        sha = head
    else:
        sha = repository.resolve(manifest.source_ref)
    if manifest_data is None and sha != head and repository.read(sha, manifest_path) != head_bytes:
        problems.add(
            f"{manifest_path} differs between HEAD and source.ref "
            f"'{manifest.source_ref}'; check out that ref first"
        )
    if project_path and sha != head and repository.read(sha, project_path) != project_bytes:
        problems.add(
            f"{project_path} differs between HEAD and source.ref "
            f"'{manifest.source_ref}'; check out that ref first"
        )

    dialect = get_scheduler(profile.scheduler).dialect(profile.scheduler_settings)
    for problem in dialect.check_profile():
        problems.add(problem)

    # Each training config once: `training.config`, and every experiment's own.
    configs = {manifest.training_config: _training_config(
        manifest, manifest.training_config, repository, sha, problems)}
    for group in manifest.groups:
        for experiment in group.experiments:
            if experiment.config is not None and experiment.config not in configs:
                configs[experiment.config] = _training_config(
                    manifest, experiment.config, repository, sha, problems,
                    where=f"groups.{group.name}.experiments[{experiment.name}].config")
    several_configs = any(experiment.config is not None
                          for group in manifest.groups for experiment in group.experiments)

    def config_of(experiment) -> str:
        return experiment.config or manifest.training_config

    # Only now: what is recorded (config files) is the whole manifest's, so
    # the parts of one manifest launched apart compare as one batch.
    manifest = _select(manifest, selection, problems)
    problems.raise_if_any()

    for path in manifest.code_paths:
        if not repository.exists(sha, path):
            problems.add(f"code.paths: '{path}' does not exist at {sha[:12]}")
        elif not repository.is_directory(sha, path):
            problems.add(f"code.paths: '{path}' is a file at {sha[:12]}; code.paths are "
                         "directories put on PYTHONPATH")
    problems.raise_if_any()

    # Environment, Python, code layout ------------------------------------
    managed = profile.env.existing is None
    requires_python, problem = _requires_python(repository, sha)
    notes: list[str] = [snapshot.note()] if snapshot is not None and snapshot.snapshot else []
    if problem:
        problems.add(problem)
    if requires_python is None and not problem:
        notes.append("pyproject.toml declares no requires-python; the interpreter version is not checked against it")
    requirements = _requirements(manifest, repository, sha, managed, problems)
    expectation, python_problems = interpreter_expectation(
        profile.env, manifest.python_pin, requires_python, managed,
    )
    for problem in python_problems:
        problems.add(problem)
    framework_expectation = _framework_expectation(manifest, requirements, managed, problems)

    clock = clock.astimezone(timezone.utc)
    batch_id = f"{manifest.name}-{clock:%Y%m%dT%H%M%SZ}-{sha[:12]}"
    batch_dir = f"{profile.roots['batches']}/{batch_id}"
    source_dir = f"{batch_dir}/source"

    if profile.env.conda_run and profile.env.existing is None and profile.env.backend != "conda":
        problems.add(
            "python_environment.activate_with_conda_run needs a conda environment "
            "(python_environment.kind: conda, or use_existing pointing to one); with a "
            "venv it would have no effect"
        )
    environment, setup_context = _environment(
        manifest, profile, requirements, expectation, batch_id, batch_dir, fresh_env,
    )
    timeout, option_problems = parse_engine_options(manifest.engine_options)
    for problem in option_problems:
        problems.add(problem)

    # Tasks ------------------------------------------------------------------
    tasks_raw = _expand(manifest, lambda experiment: configs[config_of(experiment)][1],
                        dialect, problems)
    problems.raise_if_any()

    facts_found = None
    if code_facts is not None:
        facts_found = code_facts(sha, manifest, bool(environment["managed"]), [
            (name, {**configs[config_of(experiment)][0], "sessions": [
                _with_sessions_dir(session, f"{batch_dir}/tasks/{name}/sessions")]})
            for name, experiment, session in (
                (f"{group.name}/{experiment.name}/{number:03d}", experiment, session)
                for group, experiment, number, _, session, _ in tasks_raw)])
        for problem in facts_found.problems:
            problems.add(problem)
        problems.raise_if_any()

    sessions = [session for _, _, _, _, session, _ in tasks_raw]
    sources = data_module.parse_sources(manifest.data_sources, problems, profile)
    problems.raise_if_any()

    def code_identity(inputs):
        if not inputs:
            return {"sha": sha}
        identity = {}
        for path in inputs:
            object_id = repository.object_id(sha, path)
            if object_id is None:
                problems.add(f"prepare inputs: '{path}' does not exist at {sha[:12]}")
            identity[path] = object_id
        return identity

    data = data_module.compile_sources(
        sources, sessions, profile, facts, environment.get("key"), code_identity, problems,
    )
    _data_requirements(tasks_raw, sessions, sources, profile, facts, facts_found, problems, data)
    suggestions = data_module.suggest_sources(sessions, sources, facts)
    if suggestions:
        notes.append(
            f"{len(suggestions)} absolute local path(s) in the task configs are "
            "not data sources; `validate --suggest-data` prints entries for them"
        )
    problems.raise_if_any()

    tasks: list[Task] = []
    seen_hashes: dict[str, str] = {}
    for index, (group, experiment, number, array_index, session, variant) in enumerate(tasks_raw):
        for target, path in data.rewrites.get(index, {}).items():
            session = addresses.write(session, target, path)
        name = f"{group.name}/{experiment.name}/{number:03d}"
        identity_hash = hash_value(session)
        if identity_hash in seen_hashes:
            problems.add(
                f"tasks {seen_hashes[identity_hash]} and {name} resolve to the same config"
            )
        seen_hashes[identity_hash] = name
        session = _with_sessions_dir(session, f"{batch_dir}/tasks/{name}/sessions")
        config = {**configs[config_of(experiment)][0], "sessions": [session]}
        tasks.append(Task(
            group.name, experiment.name, number, array_index, tuple(variant),
            config, hash_value(config),
            config_of(experiment) if several_configs else None,
        ))
    problems.raise_if_any()

    # Job graph and scripts -----------------------------------------------
    task_context = _context("task", environment, source_dir, manifest, profile)
    prepare_context = _context("prepare", environment, source_dir, manifest, profile)
    plan_file = f"{batch_dir}/plan.json"
    nodes: list[JobNode] = []
    if environment["managed"]:
        nodes.append(JobNode(
            id="setup", kind="setup", location=profile.setup_location,
            depends_on=(), context=setup_context, artifact="environment",
            resources=profile.setup_resources if profile.setup_location == "job" else None,
        ))
    setup_dep = ("setup",) if environment["managed"] else ()
    for node_id, artifact, source, _location in data.prepare_nodes:
        nodes.append(JobNode(
            id=node_id, kind="prepare", location=profile.prepare_location,
            depends_on=setup_dep, context=prepare_context,
            resources=source.resources, artifact=artifact.key,
        ))
    for group in manifest.groups:
        indices = {i for i, raw in enumerate(tasks_raw) if raw[0].name == group.name}
        prepares = tuple(
            node_id for node_id, users in sorted(data.users.items()) if users & indices
        )
        nodes.append(JobNode(
            id=f"group-{group.name}", kind="task_set", location="job",
            depends_on=setup_dep + prepares, context=task_context,
            resources=group.resources, group=group.name,
            task_count=len(indices), max_parallel=group.max_parallel,
            stop_grace_seconds=group.stop_grace_seconds,
        ))

    for node in nodes:
        if node.kind != "task_set" and node.resources is not None:
            where = ("environment_build.allocation" if node.kind == "setup"
                     else f"data preparation step {node.id}")
            for problem in dialect.check_resources(node.resources, where):
                problems.add(problem)

    rendered = []
    for node in nodes:
        if node.kind == "task_set":
            command = [
                node.context.interpreter, "-m", "training_framework.hpc.task",
                "--config", "{config}", "--record", "{record}",
                "--plan", plan_file, *manifest.engine_options,
            ]
            lead = dialect.signal_lead(timeout, node.stop_grace_seconds)
            if lead >= node.resources.walltime_seconds:
                problems.add(
                    f"group '{node.group}': the shutdown signal lead ({lead}s "
                    "= process_timeout_on_join + stop_grace) is not shorter "
                    f"than the walltime ({node.resources.walltime})"
                )
        elif node.kind == "prepare":
            command = [
                node.context.interpreter, "-m", "training_framework.hpc.task",
                "--prepare", node.id, "--plan", plan_file,
            ]
            lead = None
        else:
            command = [node.context.interpreter, f"{batch_dir}/bootstrap.py", "--plan", plan_file]
            lead = None
        script = dialect.render(node, batch_id, batch_dir, command, lead)
        rendered.append(dataclasses.replace(node, script=script))
    problems.raise_if_any()

    if environment["managed"]:
        environment["preflight_script"] = exec_script(task_context)
    components_packages = sorted({
        task.config["sessions"][0]["session_config"]["components_package"]
        for task in tasks
    })
    artifacts = list(data.artifacts)
    if environment["managed"]:
        artifacts.insert(0, Artifact(
            "environment", "environment", environment["key"],
            environment["location"], "key-derived",
            {"backend": profile.env.backend, "requirements": requirements.key()},
        ))
    plan = ExecutionPlan(
        protocol_version=EXECUTION_PROTOCOL_VERSION,
        batch_id=batch_id,
        batch_dir=batch_dir,
        sha=sha,
        # With a project file, both files make the manifest.
        manifest_hash=hash_bytes(head_bytes + (b"\0" + project_bytes if project_bytes else b"")),
        profile=profile.name,
        scheduler=dialect.name,
        code_mode=manifest.code_framework,
        tasks=tuple(tasks),
        artifacts=tuple(artifacts),
        nodes=tuple(rendered),
        environment=environment,
        preflight={
            "protocol_version": EXECUTION_PROTOCOL_VERSION,
            "interpreter": expectation.as_json(),
            "framework": framework_expectation,
            "components_packages": components_packages,
            "cluster_paths": sorted(data.cluster_paths),
            "requires_python": str(requires_python) if requires_python else None,
        },
        engine_options=manifest.engine_options,
        scheduler_settings=_settings_json(profile.scheduler_settings),
        notes=tuple(notes),
        base_sha=snapshot.base if snapshot is not None and snapshot.snapshot else None,
        description=manifest.description,
        selection=selection.as_json() if selection else None,
    )
    return Compiled(plan, manifest, suggestions, facts_found)


# Helpers -----------------------------------------------------------------------


def _training_config(manifest, path, repository, sha, problems, where="training.config"):
    """(config, selected session) of the training config at `path`."""
    if repository.read(sha, path) is None:
        problems.add(f"{where} '{path}' does not exist at {sha[:12]}")
        return {}, {}
    # Includes are read at the same commit: the commit is the whole config.
    try:
        config = load_config(
            path,
            reader=RepositoryFiles(lambda included: repository.read(sha, included)),
        ).config
    except ConfigFileError as error:
        problems.add(f"{where} at {sha[:12]}: {error}")
        return {}, {}
    sessions = config.get("sessions") if isinstance(config, dict) else None
    if not isinstance(sessions, list) or not sessions:
        problems.add(f"{where} '{path}' has no 'sessions' list")
        return {}, {}
    if manifest.training_session >= len(sessions):
        problems.add(
            f"training.session {manifest.training_session} is out of range "
            f"({len(sessions)} session(s)"
            + (f" in {path})" if where != "training.config" else ")")
        )
        return {}, {}
    session = sessions[manifest.training_session]
    if not isinstance(session, dict):
        problems.add("the selected session must be a mapping"
                     + (f" ({path})" if where != "training.config" else ""))
        return {}, {}
    session_config = session.get("session_config")
    if not isinstance(session_config, dict) or not isinstance(
            session_config.get("components_package"), str):
        problems.add(
            "the selected session needs session_config.components_package "
            "(the package that registers its components)"
            + (f" ({path})" if where != "training.config" else "")
        )
    return config, session


def _requires_python(repository, sha):
    raw = repository.read(sha, "pyproject.toml")
    if raw is None:
        return None, None
    try:
        project = tomllib.loads(raw.decode()).get("project", {})
    except tomllib.TOMLDecodeError as error:
        return None, f"pyproject.toml cannot be read ({error})"
    return parse_requires_python(project.get("requires-python"))


def _requirements(manifest, repository, sha, managed, problems) -> Requirements | None:
    if manifest.requirements is None:
        if managed:
            problems.add(
                "environment.requirements is required for a managed "
                "environment (or set python_environment.use_existing on the host)"
            )
        return None
    raw = repository.read(sha, manifest.requirements)
    if raw is None:
        problems.add(f"environment.requirements '{manifest.requirements}' does not exist at {sha[:12]}")
        return None
    try:
        return parse_requirements(raw.decode(), manifest.requirements)
    except RequirementsError as error:
        for problem in error.problems:
            problems.add(problem)
        return None


def _framework_expectation(manifest, requirements, managed, problems) -> dict:
    pins = requirements.pins_named(FRAMEWORK_DISTRIBUTION) if requirements else []
    if manifest.code_framework == "checkout":
        if pins:
            problems.add(
                "code.framework is 'checkout', so the requirements must not "
                f"contain {FRAMEWORK_DISTRIBUTION} (it comes from the checkout)"
            )
        return {"location": "checkout", "distribution": None}
    if requirements is not None and len(pins) != 1:
        problems.add(
            "code.framework is 'dependency', so the requirements must pin "
            f"{FRAMEWORK_DISTRIBUTION} exactly once"
        )
        return {"location": "environment", "distribution": None}
    version = pins[0].version if pins and managed else None
    return {"location": "environment", "distribution": version}


def _environment(manifest, profile, requirements, expectation, batch_id, batch_dir, fresh_env):
    env = profile.env
    hooks_setup = tuple(profile.hooks["common"]) + tuple(profile.hooks["setup"])
    if env.existing is not None:
        environment = {
            "managed": False, "key": None, "location": env.existing,
            "interpreter": f"{env.existing}/bin/python",
        }
        return environment, None
    key_parts = {
        "requirements": requirements.normalized() if requirements else None,
        "backend": env.backend,
        "interpreter": expectation.key_part(),
        "hooks": list(hooks_setup),
    }
    if env.backend == "conda":
        key_parts["conda"] = {"packages": list(env.conda_packages), "python": manifest.python_pin}
    key = hash_value(key_parts)
    name = f"{key}-{batch_id}" if fresh_env else key
    location = f"{profile.roots['envs']}/{name}"
    environment = {
        "managed": True, "key": key, "location": location,
        "interpreter": f"{location}/bin/python", "fresh": fresh_env,
        "backend": env.backend, "python_source": env.python_source,
        "python": manifest.python_pin, "conda_packages": list(env.conda_packages),
        "requirements": manifest.requirements,
    }
    bootstrap, resolve_hooks = bootstrap_interpreter(env, manifest.python_pin, batch_dir)
    setup_context = ExecutionContext(
        phase="setup", interpreter=bootstrap, environment=location,
        working_directory=batch_dir, pythonpath=(), set_variables=SET_VARIABLES,
        unset_variables=UNSET_VARIABLES, hooks=hooks_setup + resolve_hooks,
    )
    return environment, setup_context


def _settings_json(settings) -> dict:
    """The scheduler settings as plain data for plan.json (what `resume`
    renders a later attempt with). Settings that are neither a dataclass nor
    a mapping are not recorded; such batches cannot be resumed."""
    if dataclasses.is_dataclass(settings) and not isinstance(settings, type):
        return dataclasses.asdict(settings)
    if isinstance(settings, Mapping):
        return dict(settings)
    return {}


def bootstrap_interpreter(env, pin: str | None, batch_dir: str) -> tuple[str, tuple[str, ...]]:
    """The interpreter the setup node runs the bootstrap with, and the hook
    lines that provide it. Commands are always quoted argument vectors, so an
    interpreter found at run time (`uv`, `conda`) is linked to a fixed path in
    the batch directory by a hook instead of being a shell expression."""
    if env.bootstrap_python:
        return env.bootstrap_python, ()
    if env.python_source == "site":
        return "python3", ()
    link = f"{batch_dir}/nodes/setup/python"
    if env.python_source == "uv":
        find = (f"uv python install {shlex.quote(pin)} && "
                f"ln -sfn \"$(uv python find {shlex.quote(pin)})\" {shlex.quote(link)}")
    else:
        find = ("ln -sfn \"$(conda run -n base python -c 'import sys; print(sys.executable)')\" "
                f"{shlex.quote(link)}")
    return link, (f"mkdir -p {shlex.quote(link.rsplit('/', 1)[0])}", find)


def _context(phase, environment, source_dir, manifest, profile) -> ExecutionContext:
    location = environment["location"]
    launcher = (("conda", "run", "--no-capture-output", "-p", location)
                if profile.env.conda_run else ())
    return ExecutionContext(
        path_prepend=(f"{location}/bin",),
        launcher=launcher,
        phase=phase,
        interpreter=environment["interpreter"],
        environment=environment["location"],
        working_directory=source_dir,
        pythonpath=tuple(
            source_dir if path == "." else f"{source_dir}/{path}"
            for path in manifest.code_paths
        ),
        set_variables=SET_VARIABLES,
        unset_variables=UNSET_VARIABLES,
        hooks=tuple(profile.hooks["common"]) + tuple(profile.hooks[phase]),
    )


def _data_requirements(tasks_raw, sessions, sources, profile, facts, code_facts, problems,
                       data) -> None:
    """Declared data (step 4): managed when the target's profile has `data_modes`;
    said once (a note) when it does not."""
    names = [f"{group.name}/{experiment.name}/{number:03d}"
             for group, experiment, number, _, _, _ in tasks_raw]
    if code_facts is None:
        if profile.data is not None:
            problems.add(f"host '{profile.name}' manages data (dataset_delivery), which "
                         "needs the data requirements the code check finds; the code check "
                         "is off (TRAINING_FRAMEWORK_CODE_CHECK=0)")
        return
    per_task = [code_facts.requirements.get(name, ()) for name in names]
    if profile.data is None:
        found = sorted({item.requirement["name"] for items in per_task for item in items})
        if found:
            code_facts.notes.append(
                f"data requirement(s) {', '.join(found)} are not managed on host "
                f"'{profile.name}' (it has no dataset_delivery); each experiment reads what is at "
                "its config's path")
        return
    data_module.compile_requirements(per_task, sessions, sources, profile, facts, problems, data)


def _with_sessions_dir(session: dict, path: str) -> dict:
    session_config = dict(session.get("session_config") or {})
    session_config["sessions_dir"] = path
    return {**session, "session_config": session_config}


def _parse_entries(entries, field, where, base, problems):
    parsed = []
    seen = set()
    for index, entry in enumerate(entries):
        target, target_problems = addresses.parse_address(entry, f"{where}[{index}]")
        for problem in target_problems:
            problems.add(problem)
        if target is None:
            continue
        if problem := addresses.check_target(target, base):
            problems.add(f"{where}[{index}]: {problem}")
            continue
        if target in seen:
            problems.add(f"{where}[{index}]: {target.describe()} is addressed twice")
            continue
        seen.add(target)
        parsed.append((target, entry[field]))
    return parsed


def _expand(manifest, session_of, dialect, problems):
    """(group, experiment, number, array index, session, variant) per task;
    `session_of(experiment)`: the selected session of its training config."""
    tasks = []
    for group in manifest.groups:
        where = f"groups.{group.name}"
        for problem in dialect.check_resources(group.resources, f"{where}.resources"):
            problems.add(problem)
        array_index = 0
        for experiment in group.experiments:
            e_where = f"{where}.experiments[{experiment.name}]"
            base_session = session_of(experiment)
            overrides = _parse_entries(experiment.overrides, "value", f"{e_where}.overrides", base_session, problems)
            matrix = _parse_entries(experiment.matrix, "values", f"{e_where}.matrix", base_session, problems)
            session = base_session
            for target, value in overrides:
                try:
                    session = addresses.write(session, target, value)
                except ValueError as error:
                    problems.add(f"{e_where}: {error}")
            combinations = itertools.product(*(values for _, values in matrix)) if matrix else [()]
            for number, combination in enumerate(combinations):
                task_session = session
                variant = []
                for (target, _), value in zip(matrix, combination):
                    try:
                        task_session = addresses.write(task_session, target, value)
                    except ValueError as error:
                        problems.add(f"{e_where}: {error}")
                    variant.append({"address": target.as_json(), "value": value})
                try:
                    # Overrides and matrix values can bring one in too.
                    check_no_interpolation(task_session, f"{group.name}/{experiment.name}/{number:03d}")
                except ConfigFileError as error:
                    problems.add(str(error))
                for problem in check_topology(task_session, group.resources):
                    problems.add(f"{group.name}/{experiment.name}/{number:03d}: {problem}")
                tasks.append((group, experiment, number, array_index, task_session, variant))
                array_index += 1
        if array_index > dialect.max_task_set_size:
            problems.add(
                f"group '{group.name}' has {array_index} tasks, more than the "
                f"maximum task-set size {dialect.max_task_set_size}; split it"
            )
    return tasks

