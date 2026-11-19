"""The project file: what a project's batches share (`source`, `code`,
`environment`, `data`, and `run_defaults` for `naavik run`), named by a
manifest's `project:` and read at the batch's commit."""

from __future__ import annotations

import copy
import subprocess

import pytest
import yaml

from tests.hpc.batch_repo import MANIFEST, compile_repo, make_repo
from naavik.launcher.config.schema import PROJECT_FILE, RunDefaults, parse_project
from naavik.launcher.errors import PlanError

SHARED = {"code": MANIFEST["code"], "environment": MANIFEST["environment"]}


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True).stdout.strip()


def commit(root, message="change"):
    git(root, "add", "-A")
    git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", message)


def manifest_naming_project(**changes):
    data = {key: value for key, value in copy.deepcopy(MANIFEST).items() if key not in SHARED}
    data["project"] = PROJECT_FILE
    data.update(changes)
    return data


def project_repo(tmp_path, project=None, manifest=None):
    project = SHARED if project is None else project
    return make_repo(tmp_path, manifest=manifest or manifest_naming_project(),
                     extra_files={PROJECT_FILE: yaml.safe_dump(project)})


def test_a_manifest_naming_its_project_compiles_like_one_that_repeats_it(tmp_path):
    inline = compile_repo(make_repo(tmp_path / "inline")).plan
    shared = compile_repo(project_repo(tmp_path / "shared")).plan
    def without_sessions_dir(plan):     # it names the batch, so each repository's commit
        configs = copy.deepcopy([task.config for task in plan.tasks])
        for config in configs:
            del config["sessions"][0]["session_config"]["sessions_dir"]
        return configs

    assert without_sessions_dir(shared) == without_sessions_dir(inline)
    assert shared.environment["key"] == inline.environment["key"]
    assert shared.code_mode == inline.code_mode
    assert shared.manifest_hash != inline.manifest_hash        # both files make the manifest


def test_the_batch_keeps_the_manifest_as_compiled(tmp_path):
    from naavik.launcher.compile.render import plan_files

    compiled = compile_repo(project_repo(tmp_path))
    staged = yaml.safe_load(plan_files(compiled.plan, compiled.manifest.raw)["mission.yaml"])
    # Staged in the current names.
    assert staged["code"] == {"framework_from": "installed", "python_paths": ["."]}
    assert staged["project_file"] == PROJECT_FILE


def test_a_key_in_both_files_is_refused(tmp_path):
    root = project_repo(tmp_path, manifest=manifest_naming_project(code=MANIFEST["code"]))
    with pytest.raises(PlanError, match=r"both set code; keep each in one of them"):
        compile_repo(root)


def test_a_missing_project_file_is_named(tmp_path):
    root = make_repo(tmp_path, manifest=manifest_naming_project(project="nowhere.yaml"))
    with pytest.raises(PlanError, match=r"project file 'nowhere.yaml' does not exist at"):
        compile_repo(root)


@pytest.mark.parametrize("project, message", [
    (dict(SHARED, groups={}), "unknown"),
    (dict(SHARED, run_defaults={"max_parallel": 0}), "max_parallel_experiments must be a positive integer"),
    (dict(SHARED, run_defaults={"resources": "big"}), "per_experiment must be a mapping"),
    (dict(SHARED, run_defaults={"colour": "blue"}), "unknown"),
])
def test_a_bad_project_file_is_refused(tmp_path, project, message):
    with pytest.raises(PlanError, match=message):
        compile_repo(project_repo(tmp_path, project=project))


def test_the_project_file_must_match_at_source_ref(tmp_path):
    root = project_repo(tmp_path, manifest=manifest_naming_project(
        source={"strategy": "bundle", "ref": "older"}))
    git(root, "tag", "older")
    changed = dict(SHARED, environment={"requirements": "requirements/hpc.txt", "python": "3.12"})
    (root / PROJECT_FILE).write_text(yaml.safe_dump(changed))
    commit(root)
    with pytest.raises(PlanError, match=rf"{PROJECT_FILE} differs between HEAD and code_transfer.commit 'older'"):
        compile_repo(root)


def test_run_defaults_are_read():
    project = parse_project(dict(SHARED, run_defaults={
        "resources": {"gpus": 1, "walltime": "04:00:00"}, "max_parallel": 2, "target": "here"}),
        PROJECT_FILE)
    assert set(project.mission_file_keys) == {"code", "python_packages"}    # current names
    assert project.run_defaults == RunDefaults({"gpus": 1, "walltime": "04:00:00"}, 2, "here")
    assert parse_project(SHARED, PROJECT_FILE).run_defaults == RunDefaults({}, None, None)


def test_a_change_to_the_project_file_alone_changes_the_manifest_hash(tmp_path):
    root = project_repo(tmp_path)
    before = compile_repo(root).plan.manifest_hash
    changed = dict(SHARED, data={"sources": []})
    (root / PROJECT_FILE).write_text(yaml.safe_dump(changed))
    commit(root)
    assert compile_repo(root).plan.manifest_hash != before
