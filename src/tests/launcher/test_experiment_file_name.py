"""The batch file of experiments is the *experiment file* (it was called the
manifest, a word checkpoints' `manifest.json` already use): new batches
stage it as `experiments.yaml`; the earlier code names still import."""

from __future__ import annotations

import pytest

from tests.hpc.batch_repo import MANIFEST, compile_repo, make_repo
from training_framework.launcher.compile.render import plan_files
from training_framework.launcher.config import schema
from training_framework.launcher.errors import PlanError


def test_a_batch_stages_its_experiment_file_as_experiments_yaml(tmp_path):
    compiled = compile_repo(make_repo(tmp_path))
    files = plan_files(compiled.plan, compiled.manifest.raw)
    assert "experiments.yaml" in files and "manifest.yaml" not in files


def test_the_earlier_names_are_the_same_objects():
    assert schema.Manifest is schema.ExperimentFile
    assert schema.parse_manifest is schema.parse_experiment_file
    assert schema.MANIFEST_KEYS is schema.EXPERIMENT_FILE_KEYS


def test_messages_name_the_experiment_file(tmp_path):
    root = make_repo(tmp_path, manifest=dict(MANIFEST, account="lab"))
    with pytest.raises(PlanError, match="belongs in the profile, not the committed experiment file"):
        compile_repo(root)


def test_the_cli_calls_it_the_experiment_file(capsys):
    from training_framework.cli import main as naavik

    with pytest.raises(SystemExit):
        naavik(["validate", "--help"])
    assert "path to the committed experiment file" in capsys.readouterr().out
