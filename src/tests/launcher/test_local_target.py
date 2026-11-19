"""A target that is this machine (`local: true`): launched, followed, and
fetched without SSH or rsync, through the same commands as a remote one.
Slurm is the fake one of `tests.hpc.fake_cluster` (on PATH here), so the
whole protocol runs: staging, environment build, preflight, submission."""

from __future__ import annotations

import copy
import json
import os
import socket
import sys
from pathlib import Path

import pytest
import yaml

from tests.hpc.batch_repo import MANIFEST, PROFILE, make_repo
from tests.hpc.fake_cluster import fake_cluster, submitted_jobs  # noqa: F401 (fixture)
from tests.hpc.stub_framework import FILES, build_wheel, pinned_line
from naavik.cli import main as naavik_cli
from naavik.launcher.config.schema import parse_profiles
from naavik.launcher.errors import HpcError, PlanError
from naavik.launcher.ports.local import LocalExecutor, LocalFilesystem, LocalFileTransfer
from naavik.launcher.ports.targets import (
    Target,
    ports_for,
    target_of_profile,
    target_of_receipt,
    this_machine,
)


def _local_profile(**changes):
    data = copy.deepcopy(PROFILE)
    cluster = data["profiles"]["cluster"]
    del cluster["ssh_host"]
    cluster["local"] = True
    cluster.update(changes)
    return data


# Profiles -------------------------------------------------------------------------------


def test_a_local_profile_targets_this_machine():
    profile = parse_profiles(_local_profile(), None, "targets.yaml")
    assert profile.local and profile.ssh_host is None
    assert target_of_profile(profile) == Target("local", socket.gethostname()) == this_machine()


@pytest.mark.parametrize("changes, message", [
    ({"ssh_host": "cluster"}, "set either ssh_host"),
    ({"local": False}, "connection must be `local`"),
    ({"local": "yes"}, "connection must be `local`"),
])
def test_a_profile_names_exactly_one_way_to_reach_its_target(changes, message):
    with pytest.raises(PlanError, match=message):
        parse_profiles(_local_profile(**changes), None, "targets.yaml")


def test_a_profile_without_a_target_says_how_to_name_one():
    data = copy.deepcopy(PROFILE)
    del data["profiles"]["cluster"]["ssh_host"]
    with pytest.raises(PlanError, match=r"connection is required"):
        parse_profiles(data, None, "targets.yaml")


# Receipts --------------------------------------------------------------------------------


def test_a_receipt_records_its_target_in_one_key():
    assert Target("ssh", "cluster").receipt_fields() == {"ssh_host": "cluster"}
    assert Target("local", "box").receipt_fields() == {"target": {"transport": "local", "host": "box"}}
    for target in (Target("ssh", "cluster"), Target("local", "box")):
        assert target_of_receipt(target.receipt_fields()) == target


def test_an_earlier_receipt_is_read_as_ssh_or_as_unknown():
    assert target_of_receipt({"ssh_host": "cluster"}) == Target("ssh", "cluster")
    assert target_of_receipt({}) is None


@pytest.mark.parametrize("receipt, message", [
    ({"ssh_host": "a", "target": {"transport": "local", "host": "b"}}, "both ssh_host and target"),
    ({"target": {"transport": "teleport", "host": "b"}}, "is not valid"),
    ({"target": {"host": "b"}}, "is not valid"),
    ({"target": "local"}, "is not valid"),
])
def test_a_damaged_receipt_target_is_an_error(receipt, message):
    with pytest.raises(HpcError, match=message):
        target_of_receipt(receipt)


# Ports -----------------------------------------------------------------------------------


def test_this_machine_is_reached_without_ssh_or_rsync():
    target = Target("local", "box")
    assert target.tools(transfer=True) == ()
    ports = ports_for(target)
    assert isinstance(ports.executor, LocalExecutor)
    assert isinstance(ports.filesystem, LocalFilesystem)
    assert isinstance(ports.transfer, LocalFileTransfer)


# The whole protocol on this machine --------------------------------------------------------


@pytest.fixture
def launched(fake_cluster, tmp_path, monkeypatch, capsys):
    """A batch launched on a local target, with SSH made to fail and logged,
    so any use of it shows."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("FAKE_SSH_DOWN", "1")
    ssh_log = tmp_path / "ssh.log"
    monkeypatch.setenv("FAKE_SSH_LOG", str(ssh_log))
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    requirements = pinned_line("training-framework",
                               build_wheel(wheels, "training-framework", "0.5.0", FILES))
    root = make_repo(tmp_path, manifest=copy.deepcopy(MANIFEST), requirements=requirements,
                     extra_files={"my_project/components.py": "COMPONENTS = True\n"})
    data = _local_profile(
        roots={key: str(tmp_path / "here" / key) for key in ("batches", "envs", "data")},
        hooks={}, setup_location="login", env={"bootstrap_python": sys.executable},
    )
    targets = tmp_path / "targets.yaml"
    targets.write_text(yaml.safe_dump(data))
    assert naavik_cli(["probe-python", "--targets", str(targets)]) == 0
    identity = yaml.safe_load(capsys.readouterr().out)["env"]["site_python_identity"]
    data["profiles"]["cluster"]["env"]["site_python_identity"] = identity
    targets.write_text(yaml.safe_dump(data))

    assert naavik_cli(["launch", str(root / "batch.yaml"), "--targets", str(targets), "--yes"]) == 0
    output = capsys.readouterr().out
    batch_dir = Path(output.split("submitted: ")[1].splitlines()[0])
    return batch_dir, targets, ssh_log, fake_cluster


def test_a_local_batch_runs_the_whole_protocol_without_ssh(launched):
    batch_dir, _, ssh_log, slurm = launched
    assert "/here/batches/" in str(batch_dir)                          # the profile's root, here
    assert (batch_dir / "source" / "batch.yaml").exists()                 # staged here
    assert json.loads((batch_dir / "nodes" / "setup" / "record.json").read_text())["status"] == "built"
    (job,) = submitted_jobs(slurm)
    assert job["argv"][-1] == f"{batch_dir}/groups/probes/group-probes.sbatch"
    assert not ssh_log.exists()                                           # ssh never ran


def test_a_local_receipt_records_this_machine_and_no_ssh_host(launched, tmp_path):
    batch_dir, *_ = launched
    receipt = json.loads((tmp_path / "state" / "training-framework" / "submissions"
                          / f"{batch_dir.name}.json").read_text())
    assert receipt["target"] == {"transport": "local", "host": socket.gethostname()}
    assert "ssh_host" not in receipt
    assert json.loads((batch_dir / "receipt.json").read_text())["target"] == receipt["target"]


def test_status_follows_a_local_batch_here(launched, capsys):
    batch_dir, _, ssh_log, _ = launched
    assert naavik_cli(["status", batch_dir.name]) == 0
    output = capsys.readouterr().out
    assert f"{socket.gethostname()}:{batch_dir}" in output
    # Every task listed (the fake squeue reports no array elements, so their
    # live state is not what this test is about).
    assert all(f"probes/frozen/00{n}" in output for n in range(4))
    assert not ssh_log.exists()


def test_the_listing_of_a_local_target_reads_its_receipts_here(launched, capsys):
    batch_dir, targets, ssh_log, _ = launched
    receipts = Path(str(batch_dir).split("/here/")[0]) / "state" / "training-framework" / "submissions"
    for item in receipts.iterdir():
        item.unlink()                                   # as if launched from elsewhere
    assert naavik_cli(["batches", "--remote", "--targets", str(targets)]) == 0
    assert batch_dir.name in capsys.readouterr().out
    assert not ssh_log.exists()


def test_fetching_a_local_batch_says_where_it_already_is(launched, capsys):
    batch_dir, *_ = launched
    assert naavik_cli(["fetch", batch_dir.name]) == 0
    output = capsys.readouterr().out
    assert "ran on this machine" in output and str(batch_dir) in output


def test_fetching_a_local_batch_elsewhere_copies_without_rsync(launched, tmp_path, monkeypatch, capsys):
    batch_dir, _, ssh_log, _ = launched
    # `fetch` asks for all the user's jobs (`squeue --user`), which the shared
    # fake does not answer: here nothing is queued any more.
    bin_dir = tmp_path / "quiet-bin"
    bin_dir.mkdir()
    (bin_dir / "squeue").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "squeue").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    dest = tmp_path / "copy"
    assert naavik_cli(["fetch", batch_dir.name, "--dest", str(dest), "--state", "all", "--yes"]) == 0
    assert (dest / batch_dir.name / "plan.json").exists()
    assert (dest / batch_dir.name / "tasks" / "probes" / "frozen" / "000" / "config.yaml").exists()
    assert not ssh_log.exists()


# The current interpreter's environment ----------------------------------------------------


def test_a_local_target_can_use_the_environment_running_naavik():
    profile = parse_profiles(_local_profile(env={"existing": "current"}), None, "targets.yaml")
    assert profile.env.existing == sys.prefix
    assert Path(sys.prefix, "bin", "python").exists()


def test_the_current_environment_is_refused_for_a_remote_target():
    data = copy.deepcopy(PROFILE)
    data["profiles"]["cluster"]["env"] = {"existing": "current"}
    with pytest.raises(PlanError, match=r"'current' .* needs a host on this machine"):
        parse_profiles(data, None, "targets.yaml")
