"""What the launcher asks of a scheduler goes through its bundle: script
names, how to reconcile an uncertain submission, the index variable of a
task set's job script. The launcher's services name no scheduler command."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

import training_framework.launcher as launcher
from tests.hpc.batch_repo import PROFILE, compile_repo, make_repo, manifest
from tests.hpc.fakes import FakeSubmitter
from training_framework.launcher.compile.plan import ExecutionContext, JobNode
from training_framework.launcher.compile.render import plan_files
from training_framework.launcher.errors import HpcError
from training_framework.launcher.ports.local import LocalExecutor, LocalFilesystem
from training_framework.launcher.ports.ssh import RsyncTransfer, SshExecutor, SshFilesystem
from training_framework.launcher.ports.targets import (
    Target,
    ports_for,
    target_of_profile,
    target_of_receipt,
)
from training_framework.launcher.scheduler import (
    LEGACY_SCHEDULER,
    SCHEDULERS,
    SchedulerBundle,
    get_scheduler,
    script_path,
)
from training_framework.launcher.scheduler.common import task_body
from training_framework.launcher.config.schema import parse_profiles
from training_framework.launcher.services import submit as submitting

LAUNCHER = Path(launcher.__file__).parent


def _node(kind="task_set"):
    context = ExecutionContext(phase="task", interpreter="/env/bin/python", environment="/env",
                               working_directory="/b/source", pythonpath=(), set_variables=(),
                               unset_variables=(), hooks=())
    return JobNode(id="group-g" if kind == "task_set" else "setup", kind=kind, location="job",
                   depends_on=(), context=context, group="g" if kind == "task_set" else None)


class _Dialect:
    name = "fake"
    max_task_set_size = 100

    def __init__(self, settings):
        self.settings = settings

    def check_profile(self):
        return []

    def check_resources(self, resources, where):
        return []

    def signal_lead(self, timeout, stop_grace):
        return 10

    def render(self, node, batch_id, batch_dir, command, lead, **_):
        return f"# {node.id}\n"


@pytest.fixture
def fake_scheduler(monkeypatch):
    bundle = SchedulerBundle(
        name="fake", parse_settings=lambda raw, where, problems, reader: {},
        dialect=_Dialect, submitter=lambda executor: None, owner_probe=lambda executor: None,
        script_suffix=".job", reconcile_hint=lambda name: f"fakeq --find {name}",
    )
    monkeypatch.setitem(SCHEDULERS, "fake", bundle)
    data = copy.deepcopy(PROFILE)
    cluster = data["profiles"]["cluster"]
    cluster["scheduler"] = "fake"
    cluster["fake"] = {}
    del cluster["slurm"]
    return data


# Bundle fields ------------------------------------------------------------------------


def test_a_bundle_without_the_new_fields_gets_neutral_defaults():
    bundle = SchedulerBundle("x", lambda *a: {}, _Dialect, lambda e: None, lambda e: None)
    assert bundle.script_suffix == ".sh"
    assert bundle.reconcile_hint("b.n") == "look for a job named b.n"
    assert bundle.target_check is None


def test_slurm_keeps_its_script_names_and_reconciliation_hint():
    slurm = get_scheduler("slurm")
    assert script_path(slurm, _node()) == "groups/g/group-g.sbatch"
    assert script_path(slurm, _node("setup")) == "nodes/setup/setup.sbatch"
    assert slurm.reconcile_hint("b.n") == "squeue --name=b.n  (or sacct --name=b.n)"


def test_receipts_without_a_scheduler_are_slurm_receipts():
    assert LEGACY_SCHEDULER == "slurm"


def test_rendered_scripts_are_named_by_the_schedulers_suffix(tmp_path, fake_scheduler):
    compiled = compile_repo(make_repo(tmp_path), profile_data=fake_scheduler)
    files = plan_files(compiled.plan, compiled.manifest.raw)
    assert files["groups/probes/group-probes.job"] == "# group-probes\n"
    assert not [name for name in files if name.endswith(".sbatch")]


def test_an_uncertain_submission_names_the_schedulers_reconciliation(tmp_path, fake_scheduler):
    # The submission harness of the existing submit tests, on the fake scheduler.
    from tests.hpc.services.test_submission import FactsExecutor, FakeStager, profile_for

    data = profile_for(tmp_path)
    data["profiles"]["cluster"].update(scheduler="fake", fake={})
    del data["profiles"]["cluster"]["slurm"]
    compiled = compile_repo(make_repo(tmp_path, manifest=manifest(environment={}), requirements=None),
                            profile_data=data)
    plan = compiled.plan
    with pytest.raises(HpcError) as caught:
        submitting.submit(
            plan, compiled.manifest.raw, stager=FakeStager(), filesystem=LocalFilesystem(),
            executor=FactsExecutor(plan), submitter=FakeSubmitter(fail_at=0, failure="uncertain"),
            receipts_dir=str(tmp_path / "receipts"),
        )
    assert f"fakeq --find {plan.batch_id}.group-probes" in str(caught.value)
    assert "squeue" not in str(caught.value)


# The task-set body ---------------------------------------------------------------------


def test_the_task_body_reads_the_given_index_variable():
    lines = "\n".join(task_body("/b", "g", ["py", "--config", "{config}"], "TF_INDEX",
                                resume_table="/b/r.tsv"))
    assert '-v i="$TF_INDEX"' in lines
    assert "no checkpoint for array index $TF_INDEX" in lines
    assert "SLURM" not in lines


# Targets -------------------------------------------------------------------------------


def test_a_profile_and_its_receipt_name_the_same_target():
    profile = parse_profiles(PROFILE, None, "targets.yaml")
    target = target_of_profile(profile)
    assert target == Target("ssh", "cluster")
    assert target_of_receipt({"ssh_host": "cluster"}) == target
    assert target_of_receipt({}) is None          # written before receipts recorded it


def test_an_unknown_transport_is_refused():
    with pytest.raises(ValueError, match="unknown transport 'carrier-pigeon'"):
        Target("carrier-pigeon", "x")


def test_an_ssh_target_needs_ssh_and_rsync_only_to_copy():
    target = Target("ssh", "cluster")
    assert target.tools(transfer=False) == ("ssh",)
    assert target.tools(transfer=True) == ("ssh", "rsync")


def test_an_ssh_target_is_reached_over_ssh_and_rsync():
    ports = ports_for(Target("ssh", "cluster"))
    assert isinstance(ports.executor, SshExecutor) and ports.executor.host == "cluster"
    assert isinstance(ports.filesystem, SshFilesystem)
    assert isinstance(ports.transfer, RsyncTransfer)


# The services name no scheduler ---------------------------------------------------------


@pytest.mark.parametrize("package", ["services", "compile", "ports"])
def test_the_launchers_services_name_no_scheduler_command(package):
    pattern = re.compile(r"\b(sbatch|squeue|sacct|scancel|SLURM_\w+)\b|\"slurm\"")
    found = [f"{path.relative_to(LAUNCHER)}:{number}: {line.strip()}"
             for path in sorted((LAUNCHER / package).rglob("*.py"))
             for number, line in enumerate(path.read_text().splitlines(), 1)
             if pattern.search(line)]
    assert found == []
