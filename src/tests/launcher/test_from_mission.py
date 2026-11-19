"""A recipe may take its experiments from another mission's checkpoints
(`from_mission`): one experiment per chosen checkpoint, its path written at
`into`, each recording its source. The checkpoints come from a port (a fake
one here); missions without `from_mission` are unchanged."""

from __future__ import annotations

import copy
import json

import pytest
import yaml

from tests.hpc.batch_repo import CLOCK, PROFILE, FakeFacts, make_repo
from training_framework.launcher.compile.compiler import (
    SourceCheckpoint,
    SourceMission,
    compile_plan,
)
from training_framework.launcher.compile.render import plan_files
from training_framework.launcher.config.schema import parse_mission_file, parse_profiles
from training_framework.launcher.errors import PlanError
from training_framework.launcher.ports.repository import GitRepository

SOURCE = "arch-20261118T090000Z-9e04101571a2"

ANALYSIS = {"sessions": [{
    "session_type": "analysis",
    "session_config": {"rng_seed": 0, "sessions_dir": "./runs",
                       "components_package": "my_project.components"},
    "trained_model": {"model_checkpoint_path": "unset"},
    "test_set": {"split": "test", "batch_size": 64},
}]}


def mission(**recipe):
    return {
        "format_version": 1, "name": "eval-arch",
        "code": {"framework_from": "installed", "python_paths": ["."]},
        "python_packages": {"requirements_file": "requirements/hpc.txt"},
        "base": {"config_file": "configs/analysis.yaml"},
        "jobs": {"eval": {
            "per_experiment": {"walltime": "00:30:00", "gpus": 1, "cpus": 8, "memory": "32G"},
            "recipes": [{"name": "test-set", "from_mission": {"id": SOURCE}, **recipe}],
        }},
    }


def checkpoint(name, iteration):
    return SourceCheckpoint(f"/m/{SOURCE}/tasks/{name}/sessions/s/checkpoints/{iteration}", iteration)


#: The source mission: three experiments; short/base/002 has no checkpoint yet.
FOUND = {
    "short/base/000": (checkpoint("short/base/000", 2000), checkpoint("short/base/000", 1000)),
    "short/base/001": (checkpoint("short/base/001", 1000),),
    "short/base/002": (),
    "long/full/000": (checkpoint("long/full/000", 5000),),
}


class Port:
    def __init__(self, found=None, problems=()):
        self.found = FOUND if found is None else found
        self.problems = list(problems)
        self.calls = []

    def __call__(self, mission_id):
        self.calls.append(mission_id)
        return SourceMission(dict(self.found), list(self.problems), "on cluster")


def compile_with(tmp_path, manifest, port=None, config=ANALYSIS):
    root = make_repo(tmp_path, manifest=manifest,
                     extra_files={"configs/analysis.yaml": yaml.safe_dump(config)})
    return compile_plan(
        manifest_path="batch.yaml", repository=GitRepository(root),
        profile=parse_profiles(PROFILE, None, "profiles.yaml"), facts=FakeFacts(root),
        clock=CLOCK, checkpoint_facts=port if port is not None else Port())


def checkpoint_of(task):
    return task.config["sessions"][0]["trained_model"]["model_checkpoint_path"]


def test_one_experiment_per_latest_checkpoint_with_its_source(tmp_path):
    compiled = compile_with(tmp_path, mission())
    tasks = compiled.plan.tasks
    assert [task.name for task in tasks] == ["eval/test-set/000", "eval/test-set/001", "eval/test-set/002"]
    assert [task.source["experiment"] for task in tasks] == ["short/base/000", "short/base/001", "long/full/000"]
    assert tasks[0].source == {"mission": SOURCE, "experiment": "short/base/000",
                               "checkpoint": checkpoint("short/base/000", 2000).path, "iteration": 2000}
    assert [checkpoint_of(task) for task in tasks] == [task.source["checkpoint"] for task in tasks]
    notes = "\n".join(compiled.plan.notes)
    assert f"listed the checkpoints of {SOURCE} on cluster" in notes
    assert "short/base/002 of " in notes and "left out" in notes


def test_the_plan_and_so_each_record_carry_the_source(tmp_path):
    compiled = compile_with(tmp_path, mission())
    plan = json.loads(plan_files(compiled.plan, compiled.manifest.raw)["plan.json"])
    assert [task["source"]["iteration"] for task in plan["tasks"]] == [2000, 1000, 5000]


def test_all_checkpoints_and_vary_make_each_combination_checkpoint_slowest(tmp_path):
    manifest = mission(vary=[{"component": "test_set", "path": "batch_size", "values": [32, 64]}])
    manifest["jobs"]["eval"]["recipes"][0]["from_mission"].update(
        checkpoint="all", experiments=["short/*"])
    tasks = compile_with(tmp_path, manifest).plan.tasks
    assert [(task.source["experiment"], task.source["iteration"],
             task.config["sessions"][0]["test_set"]["batch_size"]) for task in tasks] == [
        ("short/base/000", 1000, 32), ("short/base/000", 1000, 64),
        ("short/base/000", 2000, 32), ("short/base/000", 2000, 64),
        ("short/base/001", 1000, 32), ("short/base/001", 1000, 64)]
    # The checkpoint is not a varied value: it is the source.
    assert all([item["address"]["path"] for item in task.variant] == [["batch_size"]] for task in tasks)


def test_a_mission_is_asked_for_once(tmp_path):
    manifest = mission()
    manifest["jobs"]["eval"]["recipes"][0]["from_mission"]["experiments"] = ["short/*"]
    manifest["jobs"]["eval"]["recipes"].append(
        {"name": "long", "from_mission": {"id": SOURCE, "experiments": ["long/*"]}})
    port = Port()
    compile_with(tmp_path, manifest, port)
    assert port.calls == [SOURCE]


def test_one_checkpoint_analysed_twice_alike_is_refused(tmp_path):
    manifest = mission()
    manifest["jobs"]["eval"]["recipes"].append(
        {"name": "again", "from_mission": {"id": SOURCE, "experiments": ["long/*"]}})
    with pytest.raises(PlanError, match="eval/test-set/002 and eval/again/000 resolve to the same config"):
        compile_with(tmp_path, manifest)


def test_another_into_address(tmp_path):
    config = copy.deepcopy(ANALYSIS)
    config["sessions"][0]["probe"] = {"weights": "unset"}
    manifest = mission()
    manifest["jobs"]["eval"]["recipes"][0]["from_mission"]["into"] = {"component": "probe", "path": "weights"}
    tasks = compile_with(tmp_path, manifest, config=config).plan.tasks
    assert [task.config["sessions"][0]["probe"]["weights"] for task in tasks] == \
        [task.source["checkpoint"] for task in tasks]


@pytest.mark.parametrize("change, message", [
    ({"experiments": ["nope/*"]}, r"experiments: 'nope/\*' matches no experiment of " + SOURCE),
    ({"experiments": ["short/base/002"]}, "no experiment of .* it chooses has a complete checkpoint"),
    ({"into": {"component": "nowhere", "path": "x"}}, "component 'nowhere' is not listed"),
])
def test_what_cannot_be_chosen_is_an_error(tmp_path, change, message):
    manifest = mission()
    manifest["jobs"]["eval"]["recipes"][0]["from_mission"].update(change)
    with pytest.raises(PlanError, match=message):
        compile_with(tmp_path, manifest)


@pytest.mark.parametrize("key, entry", [
    ("vary", {"component": "trained_model", "path": "model_checkpoint_path", "values": ["a", "b"]}),
    ("overrides", {"component": "trained_model", "path": [], "value": {}}),
])
def test_into_set_by_vary_or_overrides_too_is_an_error(tmp_path, key, entry):
    if entry["path"] == []:
        entry = {"component": "trained_model", "path": "model_checkpoint_path", "value": "x"}
    with pytest.raises(PlanError, match=rf"recipes\[test-set\]\.{key}: .* is where from_mission writes"):
        compile_with(tmp_path, mission(**{key: [entry]}))


def test_a_training_session_cannot_take_checkpoints(tmp_path):
    config = copy.deepcopy(ANALYSIS)
    del config["sessions"][0]["session_type"]
    with pytest.raises(PlanError, match="session is a 'training' session; from_mission needs an analysis"):
        compile_with(tmp_path, mission(), config=config)


def test_problems_reading_the_source_are_named_with_the_recipe(tmp_path):
    port = Port(problems=[f"no receipt for {SOURCE} on this machine"])
    with pytest.raises(PlanError, match=rf"recipes\[test-set\]\.from_mission: no receipt for {SOURCE}"):
        compile_with(tmp_path, mission(), port)


def test_without_a_port_from_mission_is_refused_not_ignored(tmp_path):
    root = make_repo(tmp_path, manifest=mission(),
                     extra_files={"configs/analysis.yaml": yaml.safe_dump(ANALYSIS)})
    with pytest.raises(PlanError, match="cannot list another mission's checkpoints"):
        compile_plan(manifest_path="batch.yaml", repository=GitRepository(root),
                     profile=parse_profiles(PROFILE, None, "profiles.yaml"),
                     facts=FakeFacts(root), clock=CLOCK)


@pytest.mark.parametrize("change, message", [
    ({"id": "arch"}, r"from_mission.id must be a mission id"),
    ({"id": None}, r"from_mission.id is required"),
    ({"checkpoint": "best"}, r"from_mission.checkpoint must be one of latest, all"),
    ({"experiments": "short/*"}, r"from_mission.experiments must be a list"),
    ({"experiments": ["a", "a"]}, r"from_mission.experiments names an experiment or glob twice"),
    ({"into": {"component": "x", "key": "y"}}, r"from_mission.into: unknown field\(s\) key"),
    ({"colour": "red"}, r"from_mission: unknown field 'colour'"),
])
def test_from_mission_is_read_strictly(change, message):
    manifest = mission()
    origin = manifest["jobs"]["eval"]["recipes"][0]["from_mission"]
    origin.update(change)
    if origin.get("id") is None:
        del origin["id"]
    with pytest.raises(PlanError, match=message):
        parse_mission_file(manifest, "m.yaml")


def test_from_mission_defaults():
    (job,) = parse_mission_file(mission(), "m.yaml").groups
    origin = job.experiments[0].from_mission
    assert origin.as_json() == {"id": SOURCE, "experiments": [], "checkpoint": "latest",
                                "into": {"component": "trained_model", "path": "model_checkpoint_path"}}


def test_a_mission_without_from_mission_never_asks_and_names_no_source(tmp_path):
    from tests.hpc.batch_repo import compile_repo

    compiled = compile_repo(make_repo(tmp_path))
    plan = json.loads(plan_files(compiled.plan, compiled.manifest.raw)["plan.json"])
    assert all("source" not in task for task in plan["tasks"])
    port = Port()
    root = make_repo(tmp_path / "again")
    compile_plan(manifest_path="batch.yaml", repository=GitRepository(root),
                 profile=parse_profiles(PROFILE, None, "profiles.yaml"), facts=FakeFacts(root),
                 clock=CLOCK, checkpoint_facts=port)
    assert port.calls == []
