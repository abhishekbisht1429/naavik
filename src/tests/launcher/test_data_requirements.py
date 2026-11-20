"""Declared data on a target: the profile's `data_modes` say how each
requirement gets there (`fetch` on the target, `upload` from here, or
`present`); a profile without it manages nothing."""

from __future__ import annotations

import copy
import json
import subprocess

import pytest
import yaml

from tests.hpc.batch_repo import CLOCK, MANIFEST, PROFILE, FakeFacts, make_repo
from naavik.data import DataRequirement, requirement_key
from naavik.launcher.compile.compiler import CodeFacts, TaskRequirement, compile_plan
from naavik.launcher.compile.data import expand_command
from naavik.launcher.compile.render import plan_files
from naavik.launcher.config.schema import parse_profiles
from naavik.launcher.errors import PlanError
from naavik.launcher.ports.repository import GitRepository

FLOWERS = DataRequirement("flowers102", "root", "1", size_hint=1000)
CLASS = "naavik.components.builtin.datasets.flowers102.Flowers102"


def requirement(spec=FLOWERS, path="/placeholder/flowers", component="flowers102"):
    location = f"{path}/{spec.subpath}" if spec.subpath else path
    return TaskRequirement(0, component, CLASS, spec.as_json(), path, location)


def port(*per_task):
    """Every task declares `per_task[0]` (or task i declares per_task[i])."""
    def answer(sha, manifest, managed, tasks):
        chosen = [per_task[min(i, len(per_task) - 1)] for i in range(len(tasks))]
        return CodeFacts(requirements={name: tuple(items)
                                       for (name, _), items in zip(tasks, chosen)})
    return answer


def profile(modes=None, **changes):
    data = copy.deepcopy(PROFILE)
    entry = data["profiles"]["cluster"]
    if modes is not None:
        entry["data_modes"] = modes
    entry.update(changes)
    return parse_profiles(data, None, "targets.yaml")


def compile_with(root, code_facts, target, facts=None):
    return compile_plan(manifest_path="batch.yaml", repository=GitRepository(root),
                        profile=target, facts=facts or FakeFacts(root), clock=CLOCK,
                        code_facts=code_facts)


def roots(plan):
    return {task.name: task.config["sessions"][0]["flowers102"]["root"] for task in plan.tasks}


# The profile ----------------------------------------------------------------------------------


def test_modes_name_requirements_with_a_default():
    target = profile({"default": "upload", "imagenet": "present"})
    assert (target.data.mode("imagenet"), target.data.mode("cifar10")) == ("present", "upload")
    assert profile({"imagenet": "present"}).data.mode("cifar10") == "fetch"
    assert profile().data is None


@pytest.mark.parametrize("modes, message", [
    ({"cifar10": "download"}, "dataset_delivery.cifar10 must be one of download_on_host, "
                              "copy_from_here, already_on_host"),
    ({"has space": "fetch"}, "a requirement name"),
    (["fetch"], "dataset_delivery must be a mapping"),
])
def test_bad_modes_are_refused(modes, message):
    with pytest.raises(PlanError, match=message):
        profile(modes)


# Not managed ----------------------------------------------------------------------------------


def test_without_data_modes_nothing_changes_but_it_is_said(tmp_path):
    root = make_repo(tmp_path)
    plain = compile_with(root, port([]), profile())
    declared = compile_with(root, port([requirement()]), profile())
    assert plan_files(declared.plan, declared.manifest.raw) == plan_files(plain.plan, plain.manifest.raw)
    assert declared.code_facts.notes == [
        "data requirement(s) flowers102 are not managed on host 'cluster' (it has "
        "no dataset_delivery); each experiment reads what is at its config's path"]


def test_managed_data_needs_the_code_check(tmp_path):
    with pytest.raises(PlanError, match="the code check is off"):
        compile_with(make_repo(tmp_path), None, profile({"default": "fetch"}))


# fetch ----------------------------------------------------------------------------------------


def test_fetch_is_one_prepare_step_before_the_tasks(tmp_path):
    target = profile({"default": "fetch"}, path_map={"/placeholder": "/scratch/me/datasets"})
    plan = compile_with(make_repo(tmp_path), port([requirement()]), target).plan
    (node,) = [node for node in plan.nodes if node.kind == "prepare"]
    assert node.id == "fetch-flowers102-0" and node.resources is not None   # setup_resources
    assert all(node.id in group.depends_on for group in plan.nodes if group.kind == "task_set")
    (artifact,) = [a for a in plan.artifacts if a.kind == "prepare"]
    assert artifact.key == requirement_key(FLOWERS)
    assert artifact.location == "/scratch/me/datasets/flowers"
    assert set(roots(plan).values()) == {"/scratch/me/datasets/flowers"}
    command = expand_command(artifact.details["command"], artifact.location)
    assert command[:5] == ["python", "-m", "naavik.data", "fetch", "--component"]
    assert command[-1] == "--under-lease"
    entry = json.loads(command[command.index("--config") + 1])     # its braces survive
    assert entry["root"] == "/scratch/me/datasets/flowers" and entry["split"] == "train"


def test_a_requirement_in_a_subfolder_is_kept_there_but_the_config_keeps_its_root(tmp_path):
    split = DataRequirement("inaturalist", "root", "1", key={"split": "2021_valid"},
                            subpath="2021_valid")
    plan = compile_with(make_repo(tmp_path), port([requirement(split)]),
                        profile({"default": "fetch"})).plan
    (artifact,) = [a for a in plan.artifacts if a.kind == "prepare"]
    assert artifact.location == "/placeholder/flowers/2021_valid"
    assert set(roots(plan).values()) == {"/placeholder/flowers"}


def test_a_new_commit_fetches_nothing_again(tmp_path):
    root = make_repo(tmp_path)
    target = profile({"default": "fetch"})
    before = compile_with(root, port([requirement()]), target).plan
    (root / "notes.txt").write_text("a change")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c"],
                   cwd=root, check=True)
    after = compile_with(root, port([requirement()]), target).plan
    assert after.sha != before.sha
    assert [a.key for a in after.artifacts if a.kind == "prepare"] == \
        [a.key for a in before.artifacts if a.kind == "prepare"]


def test_data_obtained_by_hand_cannot_be_fetched(tmp_path):
    by_hand = DataRequirement("imagenet", "root", "1", fetchable=False)
    with pytest.raises(PlanError, match=r"data 'imagenet' \(flowers102\) cannot be fetched: .*"
                                        r"set dataset_delivery: \{imagenet: copy_from_here\} or \{imagenet: already_on_host\}"):
        compile_with(make_repo(tmp_path), port([requirement(by_hand)]), profile({"default": "fetch"}))


def test_different_content_at_one_location_is_refused(tmp_path):
    other = DataRequirement("flowers102", "root", "2")
    with pytest.raises(PlanError, match="into the same location /placeholder/flowers"):
        compile_with(make_repo(tmp_path), port([requirement()], [requirement(other)]),
                     profile({"default": "fetch"}))


def test_a_relative_path_cannot_be_fetched(tmp_path):
    with pytest.raises(PlanError, match="'data/flowers' is not an absolute path on the host"):
        compile_with(make_repo(tmp_path), port([requirement(path="data/flowers")]),
                     profile({"default": "fetch"}))


def test_a_hand_written_source_wins(tmp_path):
    manifest = copy.deepcopy(MANIFEST)
    manifest["data"] = {"sources": [{"name": "flowers",
                                     "target": {"component": "flowers102", "path": "root"},
                                     "cluster": {"path": "/scratch/flowers"}}]}
    plan = compile_with(make_repo(tmp_path, manifest=manifest), port([requirement()]),
                        profile({"default": "fetch"})).plan
    assert not [node for node in plan.nodes if node.kind == "prepare"]
    assert set(roots(plan).values()) == {"/scratch/flowers"}


# present and upload ---------------------------------------------------------------------------


def test_present_data_is_a_given_path(tmp_path):
    target = profile({"flowers102": "present"}, path_map={"/placeholder": "/shared"})
    plan = compile_with(make_repo(tmp_path), port([requirement()]), target).plan
    assert not [node for node in plan.nodes if node.kind == "prepare"]
    assert [(a.kind, a.location) for a in plan.artifacts if a.kind == "cluster"] == \
        [("cluster", "/shared/flowers")]
    assert set(roots(plan).values()) == {"/shared/flowers"}


def test_upload_needs_the_data_here_and_says_who_fetches_it(tmp_path):
    root = make_repo(tmp_path)
    target = profile({"default": "upload"})
    with pytest.raises(PlanError, match="is not at /placeholder/flowers yet; `naavik launch` "
                                        "fetches it here first"):
        compile_with(root, port([requirement()]), target)
    facts = FakeFacts(root, existing={"/placeholder/flowers"})
    plan = compile_with(root, port([requirement()]), target, facts=facts).plan
    (artifact,) = [a for a in plan.artifacts if a.kind == "upload"]
    assert artifact.details["local"] == "/placeholder/flowers"


def test_data_obtained_by_hand_is_placed_before_an_upload(tmp_path):
    by_hand = DataRequirement("imagenet", "root", "1", fetchable=False)
    with pytest.raises(PlanError, match="it is obtained by hand: place it there"):
        compile_with(make_repo(tmp_path), port([requirement(by_hand)]), profile({"default": "upload"}))


# End to end on this machine -------------------------------------------------------------------

from tests.hpc.test_end_to_end import environment, session  # noqa: E402,F401 (fixture)
from tests.launcher.test_host_end_to_end import here, record, wait_for  # noqa: E402,F401
from tests.launcher.test_code_check import project  # noqa: E402
from naavik.cli import main as naavik_cli  # noqa: E402


@pytest.fixture
def checked(monkeypatch):
    monkeypatch.setenv("TRAINING_FRAMEWORK_CODE_CHECK", "1")


def data_project(tmp_path, data_root):
    from tests.hpc.test_end_to_end import COMPONENTS

    config = dict(session(), fake_data={"root": str(data_root)})
    components = COMPONENTS.read_text() + "\nfrom tests.data import data_components  # noqa\n"
    return project(tmp_path / "repo", config, extra={"my_project/components.py": components})


def with_modes(targets_path, modes):
    data = yaml.safe_load(targets_path.read_text())
    data["profiles"]["cluster"]["data_modes"] = modes
    targets_path.write_text(yaml.safe_dump(data))


def launched(root, targets, capsys):
    assert naavik_cli(["launch", str(root / "batch.yaml"), "--targets", str(targets), "--yes"]) == 0, \
        capsys.readouterr().err
    output = capsys.readouterr().out
    from pathlib import Path

    return Path(output.split("submitted: ")[1].splitlines()[0]), output


def test_the_target_fetches_once_and_the_task_uses_it(tmp_path, here, checked, capsys):
    data_root = tmp_path / "datasets" / "fake"
    with_modes(here, {"default": "fetch"})
    root = data_project(tmp_path, data_root)
    batch_dir, _ = launched(root, here, capsys)
    wait_for(lambda: record(batch_dir, "cpu/c0/000"), "the task's record")
    assert record(batch_dir, "cpu/c0/000")["exit_status"] == 0
    assert (data_root / "data.bin").exists()
    assert len((data_root.parent / "fetches").read_text().split()) == 1
    # A second batch finds it ready (no fetch, whatever the commit).
    (root / "notes.txt").write_text("x")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c"],
                   cwd=root, check=True)
    batch_dir, _ = launched(root, here, capsys)
    wait_for(lambda: record(batch_dir, "cpu/c0/000"), "the second task's record")
    assert len((data_root.parent / "fetches").read_text().split()) == 1


def test_launch_fetches_upload_data_here_first(tmp_path, here, checked, capsys):
    data_root = tmp_path / "datasets" / "fake"
    with_modes(here, {"fake": "upload"})
    root = data_project(tmp_path, data_root)
    assert naavik_cli(["validate", str(root / "batch.yaml"), "--targets", str(here)]) == 2
    assert "`naavik launch` fetches it here first" in capsys.readouterr().err
    batch_dir, output = launched(root, here, capsys)
    assert f"fetching data 'fake' here, into {data_root}" in output
    assert (data_root / "data.bin").exists()
    wait_for(lambda: record(batch_dir, "cpu/c0/000"), "the task's record")
    assert record(batch_dir, "cpu/c0/000")["exit_status"] == 0
