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

from naavik.config_files import (
    ConfigFileError,
    RepositoryFiles,
    check_no_interpolation,
    load_config,
)
from naavik.launcher import EXECUTION_PROTOCOL_VERSION
from naavik.launcher.config import address as addresses
from naavik.launcher.compile import data as data_module
from naavik.launcher.compile.data import LocalFacts
from naavik.launcher.errors import HpcError, Problems
from naavik.launcher.names import hash_bytes, hash_value
from naavik.launcher.compile.plan import (
    Artifact,
    ExecutionContext,
    ExecutionPlan,
    JobNode,
    Task,
)
from naavik.launcher.config.python_rules import (
    interpreter_expectation,
    parse_requires_python,
)
from naavik.launcher.ports.repository import RepositoryReader
from naavik.launcher.config.requirements import (
    Requirements,
    RequirementsError,
    parse_requirements,
)
from naavik.launcher.config.names import mission_file_from_earlier
from naavik.launcher.config.schema import (
    MissionFile,
    Profile,
    parse_mission_file,
    parse_project,
    with_project,
)
from naavik.launcher.ports.shell import exec_script
from naavik.launcher.scheduler import get_scheduler
from naavik.launcher.config.topology import check_topology, parse_engine_options

UNSET_VARIABLES = ("WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
SET_VARIABLES = (("PYTHONNOUSERSITE", "1"),)
#: The framework's distribution name (its earlier one is read too:
#: `naavik.distribution`).
FRAMEWORK_DISTRIBUTION = "naavik"


@dataclasses.dataclass(frozen=True)
class Compiled:
    plan: ExecutionPlan
    manifest: MissionFile
    suggestions: list[dict]
    #: What the batch's own code said about its tasks; None: not asked.
    code_facts: "CodeFacts | None" = None
    #: Earlier names the mission and project files were written with
    #: (`file: old -> new`), for one warning.
    renamed: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskRequirement:
    """Data one component of one task declares (`naavik.data`)."""

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
    """Part of a mission file: whole jobs, and `JOB/RECIPE`s."""

    jobs: tuple[str, ...] = ()
    recipes: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.jobs or self.recipes)

    def as_json(self) -> dict:
        return {key: list(value) for key, value in
                (("jobs", self.jobs), ("recipes", self.recipes)) if value}


def _select(manifest: MissionFile, selection: Selection | None, problems: Problems) -> MissionFile:
    """The mission file with only the chosen jobs and recipes (experiment
    names and numbers stay as the whole mission file makes them)."""
    import dataclasses as _dataclasses

    if not selection:
        return manifest
    groups = {group.name: group for group in manifest.groups}
    for name in selection.jobs:
        if name not in groups:
            problems.add(f"--job {name}: the mission file has no such job "
                         f"(it has: {', '.join(groups)})")
    chosen: dict[str, set[str] | None] = {name: None for name in selection.jobs if name in groups}
    for item in selection.recipes:
        group, slash, recipe = item.partition("/")
        if not slash or not recipe:
            problems.add(f"--recipe {item}: name it JOB/RECIPE")
            continue
        if group not in groups:
            problems.add(f"--recipe {item}: the mission file has no job '{group}' "
                         f"(it has: {', '.join(groups)})")
            continue
        names = [e.name for e in groups[group].experiments]
        if recipe not in names:
            problems.add(f"--recipe {item}: job '{group}' has no recipe "
                         f"'{recipe}' (it has: {', '.join(names)})")
            continue
        if chosen.get(group, set()) is not None:            # not already the whole job
            chosen.setdefault(group, set()).add(recipe)
    kept = []
    for group in manifest.groups:
        if group.name not in chosen:
            continue
        wanted = chosen[group.name]
        experiments = tuple(e for e in group.experiments if wanted is None or e.name in wanted)
        kept.append(_dataclasses.replace(group, experiments=experiments))
    return _dataclasses.replace(manifest, groups=tuple(kept))


@dataclass(frozen=True)
class SourceCheckpoint:
    """One complete checkpoint of another mission's experiment."""

    path: str
    iteration: int


@dataclass
class SourceMission:
    """The answer of the `checkpoint_facts` port of `compile_plan`: a
    mission's experiments, each with its complete checkpoints."""

    #: Experiment name -> its complete checkpoints (any order), in the
    #: mission's own order of experiments.
    experiments: dict[str, tuple[SourceCheckpoint, ...]] = field(default_factory=dict)
    #: Problems reading it (no receipt, unreachable host...): they stop the compile.
    problems: list[str] = field(default_factory=list)
    #: Where it was read, for the note: "on workstation", "from a fetched copy".
    where: str = ""
    #: Experiment name -> its varied values, as its plan has them
    #: ([{address, value}]), so summaries compare what trained each checkpoint.
    variants: dict[str, tuple[dict, ...]] = field(default_factory=dict)


#: `checkpoint_facts(mission_id)`: another mission's complete checkpoints,
#: asked once per mission a recipe's `from_mission` names.
CheckpointFactsPort = Callable[[str], SourceMission]


#: `code_facts(sha, manifest, managed, tasks)`, tasks being (name, config) in
#: order: the batch's own code asked about its expanded task configs (as the
#: manifest makes them, before data paths are rewritten for the target).
CodeFactsPort = Callable[[str, MissionFile, bool, list[tuple[str, dict]]], CodeFacts]


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
        checkpoint_facts: CheckpointFactsPort | None = None,
) -> Compiled:
    """`manifest_data`: a manifest given as data rather than read from the
    repository (`naavik run` builds one); `manifest_path` then only names
    it in messages, and everything else is still read at the batch's
    commit. `code_facts`: asked once the tasks are expanded (problems it
    reports stop the compile there); None: not asked. `selection`: only
    these groups and experiments become tasks (None: all).
    `checkpoint_facts`: lists the checkpoints of the missions recipes take
    theirs from (`from_mission`); needed only when one does."""
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
    # Earlier names become current ones first: the project file is named by
    # `project_file` (earlier: `project`).
    raw_manifest, renamed, renaming_problems = mission_file_from_earlier(raw_manifest, manifest_path)
    for problem in renaming_problems:
        problems.add(problem)
    problems.raise_if_any()
    renamed = [f"{manifest_path}: {item}" for item in renamed]
    # A project file the mission file names adds the keys a project's
    # missions share; read at the same commit as the mission file.
    project_path = project_bytes = None
    if isinstance(raw_manifest, Mapping) and raw_manifest.get("project_file") is not None:
        project_path = raw_manifest["project_file"]
        if not isinstance(project_path, str) or not project_path:
            problems.add(f"{manifest_path}.project_file must be a path in the repository")
        else:
            project_bytes = repository.read(head, project_path)
            if project_bytes is None:
                problems.add(f"project file '{project_path}' does not exist at {head[:12]} "
                             f"(named by {manifest_path}.project_file)")
        problems.raise_if_any()
        project_raw = _load_yaml(project_bytes, project_path, problems)
        problems.raise_if_any()
        project = parse_project(project_raw, project_path)
        renamed += [f"{project_path}: {item}" for item in project.renamed]
        raw_manifest = with_project(raw_manifest, project, manifest_path, project_path)
    manifest = parse_mission_file(raw_manifest, manifest_path)

    if snapshot is not None:
        if manifest.source_ref != "HEAD":
            problems.add(f"code_transfer.commit '{manifest.source_ref}' cannot be used with "
                         "run_code_from: working_tree (the working tree is taken on HEAD)")
        if manifest.source_strategy == "fetch":
            problems.add("code_transfer.method 'git_clone' cannot stage a working tree (it "
                         "exists only in this repository); use 'git_bundle'")
        sha = head
    else:
        sha = repository.resolve(manifest.source_ref)
    if manifest_data is None and sha != head and repository.read(sha, manifest_path) != head_bytes:
        problems.add(
            f"{manifest_path} differs between HEAD and code_transfer.commit "
            f"'{manifest.source_ref}'; check out that ref first"
        )
    if project_path and sha != head and repository.read(sha, project_path) != project_bytes:
        problems.add(
            f"{project_path} differs between HEAD and code_transfer.commit "
            f"'{manifest.source_ref}'; check out that ref first"
        )

    dialect = get_scheduler(profile.scheduler).dialect(profile.scheduler_settings)
    for problem in dialect.check_profile():
        problems.add(problem)

    # Each session config once: `base.config_file`, and every recipe's own
    # (`config_file`, or each of its `config_files`).
    configs = {manifest.training_config: _training_config(
        manifest, manifest.training_config, repository, sha, problems)}
    for group in manifest.groups:
        for experiment in group.experiments:
            recipe = f"jobs.{group.name}.recipes[{experiment.name}]"
            named = ([(experiment.config, f"{recipe}.config_file")] if experiment.config is not None
                     else [(path, f"{recipe}.config_files[{index}]")
                           for index, path in enumerate(experiment.configs)])
            for path, where in named:
                if path not in configs:
                    configs[path] = _training_config(manifest, path, repository, sha, problems,
                                                     where=where)
    several_configs = any(experiment.config is not None or experiment.configs
                          for group in manifest.groups for experiment in group.experiments)

    def files_of(experiment) -> tuple[str, ...]:
        """A recipe's session config files: its `config_files`, its
        `config_file`, or `base.config_file`."""
        return experiment.configs or (experiment.config or manifest.training_config,)

    # Only now: what is recorded (config files) is the whole manifest's, so
    # the parts of one manifest launched apart compare as one batch.
    manifest = _select(manifest, selection, problems)
    problems.raise_if_any()

    for path in manifest.code_paths:
        if not repository.exists(sha, path):
            problems.add(f"code.python_paths: '{path}' does not exist at {sha[:12]}")
        elif not repository.is_directory(sha, path):
            problems.add(f"code.python_paths: '{path}' is a file at {sha[:12]}; code.python_paths are "
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
    chosen = _choose_checkpoints(manifest, checkpoint_facts, problems, notes)
    problems.raise_if_any()
    tasks_raw = _expand(manifest, lambda path: configs[path][1], dialect, problems, chosen,
                        files_of)
    problems.raise_if_any()

    facts_found = None
    if code_facts is not None:
        facts_found = code_facts(sha, manifest, bool(environment["managed"]), [
            (name, {**configs[path][0], "sessions": [
                _with_sessions_dir(session, f"{batch_dir}/tasks/{name}/sessions")]})
            for name, path, session in (
                (f"{group.name}/{experiment.name}/{number:03d}", path, session)
                for group, experiment, number, _, session, _, _, path in tasks_raw)])
        for problem in facts_found.problems:
            problems.add(problem)
        problems.raise_if_any()

    sessions = [session for _, _, _, _, session, *_ in tasks_raw]
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
    suggestions = data_module.suggest_sources(sessions, sources, facts, [
        addresses.parse_address(experiment.from_mission.into, "")[0]
        if experiment.from_mission is not None else None
        for _, experiment, *_ in tasks_raw])
    if suggestions:
        notes.append(
            f"{len(suggestions)} absolute local path(s) in the experiment configs are "
            "not data sources; `validate --suggest-data` prints entries for them"
        )
    problems.raise_if_any()

    tasks: list[Task] = []
    seen_hashes: dict[str, str] = {}
    for index, (group, experiment, number, array_index, session, variant, source, config_file) in enumerate(tasks_raw):
        for target, path in data.rewrites.get(index, {}).items():
            session = addresses.write(session, target, path)
        name = f"{group.name}/{experiment.name}/{number:03d}"
        identity_hash = hash_value(session)
        if identity_hash in seen_hashes:
            problems.add(
                f"experiments {seen_hashes[identity_hash]} and {name} resolve to the same config"
            )
        seen_hashes[identity_hash] = name
        session = _with_sessions_dir(session, f"{batch_dir}/tasks/{name}/sessions")
        config = {**configs[config_file][0], "sessions": [session]}
        tasks.append(Task(
            group.name, experiment.name, number, array_index, tuple(variant),
            config, hash_value(config),
            config_file if several_configs else None,
            source,
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
                    f"job '{node.group}': the shutdown signal lead ({lead}s "
                    "= process_timeout_on_join + stop_grace_period) is not shorter "
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
    return Compiled(plan, manifest, suggestions, facts_found, tuple(renamed))


# Helpers -----------------------------------------------------------------------


def _training_config(manifest, path, repository, sha, problems, where="base.config_file"):
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
            f"base.session_index {manifest.training_session} is out of range "
            f"({len(sessions)} session(s)"
            + (f" in {path})" if where != "base.config_file" else ")")
        )
        return {}, {}
    session = sessions[manifest.training_session]
    if not isinstance(session, dict):
        problems.add("the selected session must be a mapping"
                     + (f" ({path})" if where != "base.config_file" else ""))
        return {}, {}
    session_config = session.get("session_config")
    if not isinstance(session_config, dict) or not isinstance(
            session_config.get("components_package"), str):
        problems.add(
            "the selected session needs session_config.components_package "
            "(the package that registers its components)"
            + (f" ({path})" if where != "base.config_file" else "")
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
                "python_packages.requirements_file is required for a managed "
                "environment (or set python_environment.use_existing on the host)"
            )
        return None
    raw = repository.read(sha, manifest.requirements)
    if raw is None:
        problems.add(f"python_packages.requirements_file '{manifest.requirements}' does not exist at {sha[:12]}")
        return None
    try:
        return parse_requirements(raw.decode(), manifest.requirements)
    except RequirementsError as error:
        for problem in error.problems:
            problems.add(problem)
        return None


def _framework_expectation(manifest, requirements, managed, problems) -> dict:
    from naavik.distribution import DESCRIBED, framework_pins

    pins = framework_pins(requirements) if requirements else []
    if manifest.code_framework == "checkout":
        if pins:
            pinned = " and ".join(sorted({pin.name for pin in pins}))
            problems.add(
                "code.framework is 'checkout', so the requirements must not "
                f"contain {pinned} (it comes from the checkout)"
            )
        return {"location": "checkout", "distribution": None}
    if requirements is not None and len({pin.name for pin in pins}) > 1:
        problems.add(
            "the requirements pin both naavik and training-framework: they are the same "
            "framework under its current and earlier name; keep one"
        )
        return {"location": "environment", "distribution": None}
    if requirements is not None and len(pins) != 1:
        problems.add(
            "code.framework is 'dependency', so the requirements must pin "
            f"{DESCRIBED} exactly once"
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


def _code_check_switch() -> str:
    from naavik.launcher.services.evaluation import switch_name

    return switch_name()


def _data_requirements(tasks_raw, sessions, sources, profile, facts, code_facts, problems,
                       data) -> None:
    """Declared data (step 4): managed when the target's profile has `data_modes`;
    said once (a note) when it does not."""
    names = [f"{group.name}/{experiment.name}/{number:03d}"
             for group, experiment, number, *_ in tasks_raw]
    if code_facts is None:
        if profile.data is not None:
            problems.add(f"host '{profile.name}' manages data (dataset_delivery), which "
                         "needs the data requirements the code check finds; the code check "
                         f"is off ({_code_check_switch()}=0)")
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


def _choose_checkpoints(manifest, checkpoint_facts, problems, notes):
    """(job, recipe) -> [(source experiment, SourceCheckpoint, its variant)] for every
    recipe with `from_mission`, in the source's order of experiments, then
    by iteration. Each mission is asked for once."""
    import fnmatch

    wanted = [(group, experiment) for group in manifest.groups
              for experiment in group.experiments if experiment.from_mission is not None]
    if not wanted:
        return {}
    if checkpoint_facts is None:
        for group, experiment in wanted:
            problems.add(f"jobs.{group.name}.recipes[{experiment.name}].from_mission: this "
                         "command cannot list another mission's checkpoints")
        return {}
    missions: dict[str, SourceMission] = {}
    chosen = {}
    for group, experiment in wanted:
        origin = experiment.from_mission
        where = f"jobs.{group.name}.recipes[{experiment.name}].from_mission"
        if origin.id not in missions:
            missions[origin.id] = found = checkpoint_facts(origin.id)
            for problem in found.problems:
                problems.add(f"{where}: {problem}")
            if not found.problems:
                notes.append(f"listed the checkpoints of {origin.id}"
                             + (f" {found.where}" if found.where else ""))
        source = missions[origin.id]
        if source.problems:
            continue
        names = list(source.experiments)
        for pattern in origin.experiments:
            if not fnmatch.filter(names, pattern):
                problems.add(f"{where}.experiments: '{pattern}' matches no experiment of "
                             f"{origin.id} (it has: {', '.join(names) or 'none'})")
        selected = [name for name in names if not origin.experiments
                    or any(fnmatch.fnmatchcase(name, p) for p in origin.experiments)]
        picked, left_out = [], []
        for name in selected:
            checkpoints = sorted(source.experiments[name], key=lambda c: (c.iteration, c.path))
            if not checkpoints:
                left_out.append(name)
                continue
            for checkpoint in (checkpoints[-1:] if origin.checkpoint == "latest" else checkpoints):
                picked.append((name, checkpoint, tuple(source.variants.get(name, ()))))
        if left_out:
            notes.append(f"{group.name}/{experiment.name}: {', '.join(left_out)} of {origin.id} "
                         "has no complete checkpoint; left out")
        if selected and not picked:
            problems.add(f"{where}: no experiment of {origin.id} it chooses has a complete "
                         "checkpoint, so it makes no experiment")
        chosen[(group.name, experiment.name)] = picked
    return chosen


def _check_into(experiment, origin, base_session, overrides, matrix, e_where, problems):
    """The address `from_mission` writes each checkpoint's path at, or None."""
    session_type = base_session.get("session_type", "training")
    if not isinstance(session_type, str) or session_type.strip() != "analysis":
        problems.add(f"{e_where}.from_mission: the recipe's session is a {session_type!r} "
                     "session; from_mission needs an analysis session (session_type: "
                     "analysis), which loads the checkpoints it is given")
        return None
    parsed = _parse_entries([{**origin.into, "value": None}], "value",
                            f"{e_where}.from_mission.into", base_session, problems)
    if not parsed:
        return None
    into = parsed[0][0]
    for key, entries in (("overrides", overrides), ("vary", matrix)):
        for target, _ in entries:
            if _overlaps(target, into):
                problems.add(f"{e_where}.{key}: {target.describe()} is where from_mission "
                             f"writes each checkpoint ({into.describe()}); set one or the other")
    return into


def _overlaps(a, b) -> bool:
    """Whether writing one address changes the other."""
    if a.component != b.component:
        return False
    shorter = min(len(a.path), len(b.path))
    return a.path[:shorter] == b.path[:shorter]


def _expand(manifest, session_of, dialect, problems, chosen=None, files_of=None):
    """(group, experiment, number, array index, session, variant, source,
    config file) per task. `files_of(experiment)`: its session config
    files (several with `config_files`); `session_of(path)`: the selected
    session of one. `chosen`: the checkpoints of recipes with
    `from_mission` (`_choose_checkpoints`). Axes, slowest first: the
    checkpoint, the config file, then `vary` (its last entry fastest). A
    task's source is {mission, experiment, checkpoint, iteration, into,
    variant} for a checkpoint, None otherwise."""
    chosen = chosen or {}
    files_of = files_of or (lambda experiment: (manifest.training_config,))
    tasks = []
    for group in manifest.groups:
        where = f"jobs.{group.name}"
        for problem in dialect.check_resources(group.resources, f"{where}.resources"):
            problems.add(problem)
        array_index = 0
        for experiment in group.experiments:
            e_where = f"{where}.recipes[{experiment.name}]"
            origin = experiment.from_mission
            paths = files_of(experiment)
            # Each file's session, with the recipe's overrides, and its own
            # reading of the addresses: they are checked against every file.
            per_file = []
            for path in paths:
                found = Problems() if len(paths) > 1 else problems
                base_session = session_of(path)
                overrides = _parse_entries(experiment.overrides, "value", f"{e_where}.overrides",
                                           base_session, found)
                matrix = _parse_entries(experiment.matrix, "values", f"{e_where}.vary", base_session, found)
                session = base_session
                for target, value in overrides:
                    try:
                        session = addresses.write(session, target, value)
                    except ValueError as error:
                        found.add(f"{e_where}: {error}")
                into = None
                if origin is not None:
                    into = _check_into(experiment, origin, base_session, overrides, matrix,
                                       e_where, found)
                if found is not problems:
                    for problem in found._items:
                        problems.add(f"{problem} (in {path})")
                per_file.append((path, session, matrix, into))
            sources: list = [None]
            if origin is not None:
                into = per_file[0][3]
                # `into` and the source's own varied values: what summaries
                # compare in place of the checkpoint's path.
                sources = [{"mission": origin.id, "experiment": name, "checkpoint": checkpoint.path,
                            "iteration": checkpoint.iteration,
                            "into": into.as_json() if into is not None else None,
                            "variant": [dict(item) for item in variant]}
                           for name, checkpoint, variant in chosen.get((group.name, experiment.name), [])]
                if any(item[3] is None for item in per_file):
                    sources = []
            number = -1
            for source in sources:
                for path, session, matrix, into in per_file:
                    combinations = list(itertools.product(*(values for _, values in matrix))) if matrix else [()]
                    for combination in combinations:
                        number += 1
                        task_session = session
                        if source is not None:
                            try:
                                task_session = addresses.write(task_session, into, source["checkpoint"])
                            except ValueError as error:
                                problems.add(f"{e_where}.from_mission.into: {error}")
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
                        tasks.append((group, experiment, number, array_index, task_session, variant,
                                      source, path))
                        array_index += 1
        if array_index > dialect.max_task_set_size:
            problems.add(
                f"job '{group.name}' has {array_index} experiments, more than the "
                f"most one job may hold ({dialect.max_task_set_size}); split it"
            )
    return tasks

