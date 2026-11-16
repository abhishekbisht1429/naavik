"""What a record says about the code, the environment, and a device."""

from __future__ import annotations

import subprocess

import pytest
import torch

from tests.hpc.batch_repo import make_repo
from training_framework.runs.facts import (
    RUN_REFS,
    code_identity,
    device_facts,
    environment_facts,
    package_directory,
)
from training_framework.runs.ids import new_run_id


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True).stdout.strip()


def test_a_clean_tree_is_its_head(tmp_path):
    root = make_repo(tmp_path)
    identity = code_identity(root / "my_project", new_run_id())
    assert identity == {"repository": str(root), "sha": git(root, "rev-parse", "HEAD"),
                        "base_sha": None, "snapshot": False}
    assert RUN_REFS not in git(root, "for-each-ref")


def test_a_dirty_tree_is_kept_as_a_snapshot_under_the_run_id(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINING_FRAMEWORK_RUNS_SNAPSHOT", "1")   # the default (tests turn it off)
    root = make_repo(tmp_path)
    head = git(root, "rev-parse", "HEAD")
    (root / "my_project" / "__init__.py").write_text("CHANGED = 1\n")
    run_id = new_run_id()
    identity = code_identity(root, run_id)
    assert identity["snapshot"] is True and identity["base_sha"] == head
    assert identity["ref"] == f"{RUN_REFS}/{run_id}"
    assert git(root, "rev-parse", identity["ref"]) == identity["sha"] != head
    assert git(root, "show", f"{identity['sha']}:my_project/__init__.py") == "CHANGED = 1"
    assert git(root, "rev-parse", "HEAD") == head                       # nothing committed
    assert git(root, "status", "--porcelain") == "M my_project/__init__.py"   # still the user's change


def test_a_dirty_tree_without_snapshots_says_so_and_keeps_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINING_FRAMEWORK_RUNS_SNAPSHOT", "0")
    root = make_repo(tmp_path)
    (root / "my_project" / "__init__.py").write_text("CHANGED = 1\n")
    identity = code_identity(root, new_run_id())
    assert identity["dirty"] is True and identity["snapshot"] is False
    assert "keeps no snapshot" in identity["error"]
    assert RUN_REFS not in git(root, "for-each-ref")


def test_a_snapshot_that_cannot_be_made_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINING_FRAMEWORK_RUNS_SNAPSHOT", "1")   # the default (tests turn it off)
    root = make_repo(tmp_path)
    (root / "big.bin").write_bytes(b"x" * 4096)
    identity = code_identity(root, new_run_id(), limit=1024)
    assert identity["dirty"] is True and "exceed snapshot_limit" in identity["error"]


def test_outside_a_repository_there_is_no_code_identity(tmp_path):
    assert code_identity(tmp_path, new_run_id()) is None


def test_the_environment_names_python_torch_and_the_packages():
    facts = environment_facts()
    assert facts["torch"] == torch.__version__ and len(facts["packages_sha256"]) == 64
    assert facts["python"].count(".") == 2


def test_a_package_is_located_without_importing_it():
    assert package_directory("training_framework.runs").name == "runs"
    assert package_directory("no_such_package_anywhere") is None
    assert package_directory(None) is None


def test_a_cpu_worker_records_its_device_and_visible_gpus(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    facts = device_facts(torch.device("cpu"), 0, world_size=1, backend="gloo")
    assert facts["device"] == "cpu" and facts["cuda_visible_devices"] == ""
    assert facts["world_size"] == 1 and "uuid" not in facts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_cuda_worker_records_its_gpu_by_name_and_uuid():
    facts = device_facts(torch.device("cuda", 0), 0, world_size=1, backend="nccl")
    properties = torch.cuda.get_device_properties(0)
    assert facts["index"] == 0 and facts["name"] == properties.name
    assert facts["total_memory"] == properties.total_memory
    # Named as nvidia-smi names it, so it matches a host runner's CUDA_VISIBLE_DEVICES.
    smi = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.split()
    assert facts["uuid"] == f"GPU-{properties.uuid}" and facts["uuid"] in smi
