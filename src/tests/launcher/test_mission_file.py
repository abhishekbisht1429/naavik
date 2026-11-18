"""Mission files and the project file in their current names (jobs, recipes,
`vary`, `base`, `python_packages`, `data_sources`...), and the earlier
names (groups, experiments, `matrix`, `training`, `environment`,
`data.sources`...) still read, with one warning, to the same mission."""

from __future__ import annotations

import copy

import pytest
import yaml

from tests.hpc.batch_repo import MANIFEST, compile_repo, make_repo
from training_framework.launcher.compile.render import plan_files
from training_framework.launcher.config.schema import RunDefaults, parse_mission_file, parse_project
from training_framework.launcher.errors import PlanError

#: MANIFEST (earlier names) in the current names.
CURRENT = {
    "format_version": 1,
    "name": "sweep",
    "code": {"framework_from": "installed", "python_paths": ["."]},
    "python_packages": {"requirements_file": "requirements/hpc.txt"},
    "base": {"config_file": "configs/train.yaml"},
    "jobs": {"probes": {
        "per_experiment": {"walltime": "01:00:00", "gpus": 1, "cpus": 8, "memory": "64G"},
        "max_parallel_experiments": 2,
        "recipes": [{
            "name": "frozen",
            "overrides": [{"component": "fine_tuned_model", "path": "frozen", "value": ["module.*"]}],
            "vary": [{"component": "optimizer", "path": "optimizer.kwargs.lr", "values": [0.001, 0.0001]},
                     {"path": "session_config.rng_seed", "values": [41, 42]}],
        }],
    }},
}


def test_a_mission_compiles_the_same_from_either_names(tmp_path):
    current = compile_repo(make_repo(tmp_path / "current", manifest=CURRENT))
    earlier = compile_repo(make_repo(tmp_path / "earlier", manifest=MANIFEST))
    # Staged in the current names either way: every file the same, the
    # mission file included (the ids differ only by the repositories' commits).
    a = plan_files(current.plan, current.manifest.raw)
    b = plan_files(earlier.plan, earlier.manifest.raw)
    assert yaml.safe_load(a["mission.yaml"]) == yaml.safe_load(b["mission.yaml"])
    def sessions(plan):        # without sessions_dir, which names each mission's directory
        found = []
        for task in plan.tasks:
            session = copy.deepcopy(task.config["sessions"][0])
            del session["session_config"]["sessions_dir"]
            found.append((task.name, session))
        return found

    assert len(current.plan.tasks) == 4 and sessions(current.plan) == sessions(earlier.plan)
    assert current.renamed == () and "groups -> jobs" in "; ".join(earlier.renamed)


def test_the_current_names_say_what_they_hold():
    mission = parse_mission_file(dict(CURRENT, note="why", code_transfer={
        "method": "git_clone", "commit": "v1", "clone_url": "https://example.org/r.git"}), "m.yaml")
    assert (mission.source_strategy, mission.source_ref, mission.source_url) == (
        "fetch", "v1", "https://example.org/r.git")
    assert mission.code_framework == "dependency" and mission.description == "why"
    (job,) = mission.groups
    assert (job.name, job.max_parallel, [recipe.name for recipe in job.experiments]) == (
        "probes", 2, ["frozen"])
    assert len(job.experiments[0].matrix) == 2


@pytest.mark.parametrize("change, message", [
    ({"format_version": 2}, "format_version must be 1"),
    ({"code": {"framework_from": "vendored", "python_paths": ["."]}},
     "framework_from must be one of installed, this_repository"),
    ({"code_transfer": {"method": "git_clone"}}, "clone_url is required with method 'git_clone'"),
    ({"python_packages": {"python_version": 3.12}}, "python_version must be a quoted string"),
    ({"jobs": {}}, "jobs must be a non-empty mapping"),
])
def test_a_wrong_setting_is_named_as_written(change, message):
    with pytest.raises(PlanError, match=message):
        parse_mission_file(dict(CURRENT, **change), "m.yaml")


def test_a_recipe_names_its_parts_as_written():
    data = copy.deepcopy(CURRENT)
    data["jobs"]["probes"]["recipes"][0]["vary"][0]["value"] = 1
    with pytest.raises(PlanError, match=r"recipes\[0\]\.vary\[0\]: unknown field\(s\) value"):
        parse_mission_file(data, "m.yaml")


@pytest.mark.parametrize("both, message", [
    ({"groups": {}}, "groups and jobs are the same setting"),
    ({"training": {}}, "training and base are the same setting"),
    ({"version": 1}, "version and format_version are the same setting"),
])
def test_an_earlier_name_beside_its_current_name_is_refused(both, message):
    with pytest.raises(PlanError, match=message):
        parse_mission_file(dict(CURRENT, **both), "m.yaml")


def test_data_sources_in_their_current_names(tmp_path):
    from tests.hpc.batch_repo import FakeFacts

    mission = dict(CURRENT, data_sources=[
        {"name": "flowers", "address": {"component": "flowers102", "path": "root"},
         "already_on_host": {"path": "/scratch/flowers"}}])
    plan = compile_repo(make_repo(tmp_path, manifest=mission)).plan
    assert {task.config["sessions"][0]["flowers102"]["root"] for task in plan.tasks} == {"/scratch/flowers"}
    wrong = dict(CURRENT, data_sources=[
        {"name": "flowers", "address": {"component": "flowers102", "path": "root"},
         "copy_from_here": {"rerun_when_changed": ["x.py"]}}])
    with pytest.raises(PlanError, match=r"copy_from_here: unknown field\(s\) rerun_when_changed"):
        compile_repo(make_repo(tmp_path / "w", manifest=wrong),
                     facts=FakeFacts(tmp_path, existing={"/placeholder/flowers"}))


def test_the_project_file_in_its_current_names():
    project = parse_project({
        "code": {"framework_from": "installed", "python_paths": ["."]},
        "quick_run_defaults": {"per_experiment": {"gpus": 1}, "max_parallel_experiments": 2,
                               "host": "workstation"}}, "naavik-project.yaml")
    assert project.run_defaults == RunDefaults({"gpus": 1}, 2, "workstation")
    assert project.renamed == ()
    earlier = parse_project({"run_defaults": {"target": "x"}}, "naavik-project.yaml")
    assert earlier.run_defaults.target == "x" and "run_defaults -> quick_run_defaults" in earlier.renamed


def test_earlier_names_are_warned_about_once_with_the_fix(tmp_path, capsys, monkeypatch):
    from tests.launcher.test_hosts_file import CURRENT as HOSTS
    from training_framework.cli import main as naavik

    hosts = tmp_path / "hosts.yaml"
    hosts.write_text(yaml.safe_dump(HOSTS))
    monkeypatch.chdir(tmp_path)                                  # not the repository
    root = make_repo(tmp_path)                                   # MANIFEST: earlier names
    assert naavik(["validate", str(root / "batch.yaml"), "--targets", str(hosts)]) == 0
    warnings = [line for line in capsys.readouterr().err.splitlines()
                if line.startswith("warning:") and "earlier names" in line]
    assert len(warnings) == 1 and "groups -> jobs" in warnings[0]
    # The command works from where it is run.
    assert "`naavik migrate repo/batch.yaml`" in warnings[0]


def test_a_project_file_in_earlier_names_is_part_of_the_warning(tmp_path):
    project = {"code": {"framework": "dependency", "paths": ["."]},
               "environment": {"requirements": "requirements/hpc.txt"},
               "run_defaults": {"resources": {"gpus": 1}}}
    mission = {key: value for key, value in CURRENT.items() if key not in ("code", "python_packages")}
    mission["project_file"] = "naavik-project.yaml"
    root = make_repo(tmp_path, manifest=mission,
                     extra_files={"naavik-project.yaml": yaml.safe_dump(project)})
    renamed = compile_repo(root).renamed
    assert "naavik-project.yaml: run_defaults -> quick_run_defaults" in renamed
    assert "naavik-project.yaml: environment -> python_packages" in renamed
    assert not [item for item in renamed if item.startswith("batch.yaml")]


def test_an_unknown_data_source_setting_is_named_not_a_crash(tmp_path):
    wrong = dict(CURRENT, data_sources=[
        {"name": "flowers", "address": {"component": "flowers102", "path": "root"},
         "already_on_host": {"colour": "red"}}])
    with pytest.raises(PlanError, match=r"already_on_host: unknown field\(s\) colour"):
        compile_repo(make_repo(tmp_path, manifest=wrong))
