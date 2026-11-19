"""The compiler asks the batch's own code about its expanded tasks through a
port (`code_facts`), before its data stage: problems stop the compile there,
notes stay out of the plan, and each task's data requirements come back."""

from __future__ import annotations

import copy
import importlib

import pytest

from tests.data import data_components
from tests.hpc.test_end_to_end import environment  # noqa: F401 (fixture)
from tests.launcher.test_host_end_to_end import here  # noqa: F401 (fixture)
from tests.hpc.batch_repo import CLOCK, MANIFEST, PROFILE, FakeFacts, make_repo
from naavik.launcher.compile.compiler import CodeFacts, compile_plan
from naavik.launcher.compile.render import plan_files
from naavik.launcher.config.schema import parse_profiles
from naavik.launcher.errors import PlanError
from naavik.launcher.evaluate import evaluate
from naavik.launcher.ports.repository import GitRepository

DATA_PACKAGE = "tests.data.data_components"


def compile_with(root, port):
    return compile_plan(manifest_path="batch.yaml", repository=GitRepository(root),
                        profile=parse_profiles(PROFILE, None, "profiles.yaml"),
                        facts=FakeFacts(root), clock=CLOCK, code_facts=port)


def with_cluster_data(tmp_path):
    manifest = copy.deepcopy(MANIFEST)
    manifest["data"] = {"sources": [{"name": "flowers",
                                     "target": {"component": "flowers102", "path": "root"},
                                     "cluster": {"path": "/scratch/flowers"}}]}
    return make_repo(tmp_path, manifest=manifest)


class Recorder:
    def __init__(self, answer=None):
        self.calls, self.answer = [], answer or CodeFacts()

    def __call__(self, sha, manifest, managed, tasks):
        self.calls.append((sha, manifest.name, managed, tasks))
        return self.answer


def test_the_port_sees_every_task_as_the_manifest_makes_it(tmp_path):
    recorder = Recorder()
    compiled = compile_with(with_cluster_data(tmp_path), recorder)
    (sha, name, managed, tasks), = recorder.calls
    plan = compiled.plan
    assert (sha, name, managed) == (plan.sha, "sweep", True)
    assert [task_name for task_name, _ in tasks] == [task.name for task in plan.tasks]
    for (_, config), task in zip(tasks, plan.tasks):
        asked, planned = config["sessions"][0], task.config["sessions"][0]
        assert asked["flowers102"]["root"] == "/placeholder/flowers"     # before the data stage
        assert planned["flowers102"]["root"] == "/scratch/flowers"       # after it
        assert asked["session_config"]["sessions_dir"] == planned["session_config"]["sessions_dir"]
    assert compiled.code_facts is recorder.answer


def test_problems_stop_the_compile(tmp_path):
    recorder = Recorder(CodeFacts(problems=["probes/frozen/000: flowers102: bad split"]))
    with pytest.raises(PlanError, match="probes/frozen/000: flowers102: bad split"):
        compile_with(make_repo(tmp_path), recorder)


def test_notes_and_requirements_stay_out_of_the_plan(tmp_path):
    root = make_repo(tmp_path)
    without = compile_with(root, None)
    answer = CodeFacts(notes=["probes/frozen/000: not checked: imports"],
                       requirements={"probes/frozen/000": ()})
    with_facts = compile_with(root, Recorder(answer))
    assert without.code_facts is None
    assert plan_files(with_facts.plan, with_facts.manifest.raw) == \
        plan_files(without.plan, without.manifest.raw)


# The evaluator's answer -----------------------------------------------------------------------


@pytest.fixture
def registered():
    """Registered again for each test (the suite resets registrations)."""
    importlib.reload(data_components)


def session(tmp_path, **components):
    return {"sessions": [{"session_config": {
        "rng_seed": 1, "sessions_dir": str(tmp_path), "max_iterations": 1,
        "components_package": DATA_PACKAGE, "show_execution_graph": False}, **components}]}


def test_the_evaluator_lists_each_tasks_data(tmp_path, registered):
    found = evaluate([{"name": "g/e/000", "config": session(tmp_path, fake_data={"root": "/d/x"})},
                      {"name": "g/e/001", "config": session(tmp_path)}])
    assert found["tasks"]["g/e/000"]["problems"] == []
    assert found["tasks"]["g/e/000"]["requirements"] == [{
        "session": 0, "component": "fake_data",
        "class": "tests.data.data_components.FakeData",
        "requirement": {"name": "fake", "path": "root", "version": "1", "key": {},
                        "subpath": "", "fetchable": True, "size_hint": None},
        "path": "/d/x", "location": "/d/x"}]
    assert found["tasks"]["g/e/001"]["requirements"] == []


def test_a_wrong_declaration_is_a_problem_of_the_task(tmp_path, registered):
    found = evaluate([{"name": "g/e/000",
                       "config": session(tmp_path, wrongly_declared_data={"root": "/d/x"})}])
    (problem,) = found["tasks"]["g/e/000"]["problems"]
    assert problem.startswith("wrongly_declared_data: its data requirements: ")
    assert "must return a tuple of DataRequirement" in problem


def test_a_config_with_problems_reports_no_requirements(tmp_path, registered):
    found = evaluate([{"name": "g/e/000",
                       "config": session(tmp_path, fake_data={"root": "/d/x", "colour": 1})}])
    (problem,) = found["tasks"]["g/e/000"]["problems"]          # said once, by the check
    assert "colour" in problem and found["tasks"]["g/e/000"]["requirements"] == []


def test_the_batchs_own_code_reports_its_requirements(tmp_path, here, registered):
    import yaml

    from tests.launcher.test_code_check import project
    from naavik.launcher.compile.compiler import TaskRequirement
    from naavik.launcher.services.evaluation import evaluate_batch

    config = session(tmp_path, fake_data={"root": "/d/x", "version": "3"})["sessions"][0]
    root = project(tmp_path / "repo", config)
    profile = parse_profiles(yaml.safe_load(here.read_text()), None, "targets.yaml")
    repository = GitRepository(root)

    def port(sha, manifest, managed, tasks):
        return evaluate_batch(sha, manifest, managed, tasks, repository, profile)

    compiled = compile_plan(manifest_path="batch.yaml", repository=repository, profile=profile,
                            facts=FakeFacts(root), clock=CLOCK, code_facts=port)
    assert compiled.code_facts.problems == []
    assert compiled.code_facts.requirements == {"cpu/c0/000": (TaskRequirement(
        0, "fake_data", "tests.data.data_components.FakeData",
        {"name": "fake", "path": "root", "version": "3", "key": {}, "subpath": "",
         "fetchable": True, "size_hint": None}, "/d/x", "/d/x"),)}
