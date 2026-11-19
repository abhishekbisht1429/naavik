"""One recipe over several session configs (`config_files`): each file with
every `vary` combination, its addresses checked against each file, and each
experiment recording the file it was made from."""

from __future__ import annotations

import copy
import json

import pytest
import yaml

from tests.hpc.batch_repo import TRAINING_CONFIG, compile_repo, make_repo
from naavik.launcher.compile.compiler import Selection
from naavik.launcher.compile.render import plan_files
from naavik.launcher.config.schema import parse_mission_file
from naavik.launcher.errors import PlanError

#: A second structure: an attention pool where the base config has a frozen backbone.
ATTNPOOL = copy.deepcopy(TRAINING_CONFIG)
ATTNPOOL["sessions"][0]["attention_pool"] = {"heads": 4}

FILES = ["configs/train.yaml", "configs/attnpool.yaml"]


def mission(**recipe):
    return {
        "format_version": 1, "name": "arch",
        "code": {"framework_from": "installed", "python_paths": ["."]},
        "python_packages": {"requirements_file": "requirements/hpc.txt"},
        "base": {"config_file": "configs/train.yaml"},
        "jobs": {"arch": {
            "per_experiment": {"walltime": "01:00:00", "gpus": 1, "cpus": 8, "memory": "64G"},
            "recipes": [{"name": "heads", "config_files": FILES, **recipe}],
        }},
    }


def compile_mission(tmp_path, manifest, other=ATTNPOOL, selection=None):
    from tests.hpc.batch_repo import CLOCK, PROFILE, FakeFacts
    from naavik.launcher.compile.compiler import compile_plan
    from naavik.launcher.config.schema import parse_profiles
    from naavik.launcher.ports.repository import GitRepository

    root = make_repo(tmp_path, manifest=manifest,
                     extra_files={"configs/attnpool.yaml": yaml.safe_dump(other)})
    if selection is None:
        return compile_repo(root)
    return compile_plan(manifest_path="batch.yaml", repository=GitRepository(root),
                        profile=parse_profiles(PROFILE, None, "profiles.yaml"),
                        facts=FakeFacts(root), clock=CLOCK, selection=selection)


VARY = [{"component": "optimizer", "path": "optimizer.kwargs.lr", "values": [0.001, 0.0001]},
        {"path": "session_config.rng_seed", "values": [41, 42]}]


def test_each_file_takes_every_combination_the_file_slowest(tmp_path):
    tasks = compile_mission(tmp_path, mission(vary=VARY)).plan.tasks
    assert [task.name for task in tasks] == [f"arch/heads/{n:03d}" for n in range(8)]
    assert [task.config_file for task in tasks] == [FILES[0]] * 4 + [FILES[1]] * 4
    assert [("attention_pool" in task.config["sessions"][0],
             task.config["sessions"][0]["resources"]["optimizer"]["optimizer"]["kwargs"]["lr"],
             task.config["sessions"][0]["session_config"]["rng_seed"]) for task in tasks] == [
        (pool, lr, seed) for pool in (False, True) for lr in (0.001, 0.0001) for seed in (41, 42)]
    assert [task.array_index for task in tasks] == list(range(8))


def test_overrides_apply_to_every_file(tmp_path):
    tasks = compile_mission(tmp_path, mission(
        overrides=[{"path": "session_config.max_iterations", "value": 99}])).plan.tasks
    assert len(tasks) == 2
    assert {task.config["sessions"][0]["session_config"]["max_iterations"] for task in tasks} == {99}


def test_the_plan_and_so_each_record_name_its_file(tmp_path):
    compiled = compile_mission(tmp_path, mission(vary=VARY[:1]))
    plan = json.loads(plan_files(compiled.plan, compiled.manifest.raw)["plan.json"])
    assert [task["config_file"] for task in plan["tasks"]] == [FILES[0]] * 2 + [FILES[1]] * 2


def test_an_address_missing_in_one_file_is_named_with_that_file(tmp_path):
    with pytest.raises(PlanError) as raised:
        compile_mission(tmp_path, mission(
            vary=[{"component": "attention_pool", "path": "heads", "values": [4, 8]}]))
    (problem,) = [item for item in raised.value.problems if "attention_pool" in item]
    assert problem.startswith("jobs.arch.recipes[heads].vary[0]: component 'attention_pool' is not listed")
    assert problem.endswith(f"(in {FILES[0]})")


def test_two_files_giving_the_same_config_are_refused(tmp_path):
    with pytest.raises(PlanError, match="arch/heads/000 and arch/heads/001 resolve to the same config"):
        compile_mission(tmp_path, mission(), other=TRAINING_CONFIG)


def test_a_missing_file_is_named_with_its_place_in_the_list(tmp_path):
    manifest = mission()
    manifest["jobs"]["arch"]["recipes"][0]["config_files"] = [FILES[0], "configs/nowhere.yaml"]
    with pytest.raises(PlanError, match=r"recipes\[heads\]\.config_files\[1\] 'configs/nowhere.yaml' "
                                        "does not exist"):
        compile_mission(tmp_path, manifest)


def test_recipe_selection_takes_every_file(tmp_path):
    manifest = mission()
    manifest["jobs"]["arch"]["recipes"].append({"name": "other",
                                                "overrides": [{"path": "session_config.rng_seed", "value": 7}]})
    tasks = compile_mission(tmp_path, manifest, selection=Selection(recipes=("arch/heads",))).plan.tasks
    assert [task.config_file for task in tasks] == FILES


def test_with_from_mission_the_checkpoint_is_slowest_then_the_file(tmp_path):
    from tests.launcher.test_from_mission import ANALYSIS, Port
    from tests.launcher.test_from_mission import mission as analysis_mission

    other = copy.deepcopy(ANALYSIS)
    other["sessions"][0]["test_set"]["split"] = "val"
    manifest = analysis_mission()
    manifest["jobs"]["eval"]["recipes"][0]["from_mission"]["experiments"] = ["short/*"]
    manifest["jobs"]["eval"]["recipes"][0]["config_files"] = ["configs/analysis.yaml", "configs/val.yaml"]
    from tests.hpc.batch_repo import CLOCK, PROFILE, FakeFacts
    from naavik.launcher.compile.compiler import compile_plan
    from naavik.launcher.config.schema import parse_profiles
    from naavik.launcher.ports.repository import GitRepository

    root = make_repo(tmp_path, manifest=manifest, extra_files={
        "configs/analysis.yaml": yaml.safe_dump(ANALYSIS), "configs/val.yaml": yaml.safe_dump(other)})
    tasks = compile_plan(manifest_path="batch.yaml", repository=GitRepository(root),
                         profile=parse_profiles(PROFILE, None, "profiles.yaml"), facts=FakeFacts(root),
                         clock=CLOCK, checkpoint_facts=Port()).plan.tasks
    assert [(task.source["experiment"], task.config_file) for task in tasks] == [
        ("short/base/000", "configs/analysis.yaml"), ("short/base/000", "configs/val.yaml"),
        ("short/base/001", "configs/analysis.yaml"), ("short/base/001", "configs/val.yaml")]


@pytest.mark.parametrize("recipe, message", [
    ({"config_files": FILES, "config_file": FILES[0]}, "config_file and config_files cannot both be given"),
    ({"config_files": [FILES[0]]}, r"config_files must be a list of two or more session config paths "
                                   r"\(for one, use config_file\)"),
    ({"config_files": FILES[0]}, "config_files must be a list of two or more"),
    ({"config_files": [FILES[0], ""]}, "config_files must be a list of two or more"),
    ({"config_files": [FILES[0], FILES[0]]}, f"config_files names {FILES[0]} more than once"),
])
def test_config_files_is_read_strictly(recipe, message):
    manifest = mission()
    manifest["jobs"]["arch"]["recipes"][0] = {"name": "heads", **recipe}
    with pytest.raises(PlanError, match=message):
        parse_mission_file(manifest, "m.yaml")


def test_a_recipe_of_one_file_keeps_its_messages_as_they_were(tmp_path):
    manifest = mission(vary=[{"component": "attention_pool", "path": "heads", "values": [4, 8]}])
    del manifest["jobs"]["arch"]["recipes"][0]["config_files"]
    with pytest.raises(PlanError) as raised:
        compile_mission(tmp_path, manifest)
    (problem,) = [item for item in raised.value.problems if "attention_pool" in item]
    assert problem == ("jobs.arch.recipes[heads].vary[0]: component 'attention_pool' is not "
                       "listed in the selected session")
