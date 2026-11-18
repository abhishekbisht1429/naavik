"""The file of a mission's experiments is the *mission file* (it was the
experiment file, and before that the manifest, a word checkpoints'
`manifest.json` already use): new missions stage it as `mission.yaml`, in
the current names; the earlier code names still import."""

from __future__ import annotations

import pytest
import yaml

from tests.hpc.batch_repo import MANIFEST, compile_repo, make_repo
from training_framework.launcher.compile.render import plan_files
from training_framework.launcher.config import schema
from training_framework.launcher.errors import PlanError


def test_a_mission_stages_its_file_as_mission_yaml_in_the_current_names(tmp_path):
    compiled = compile_repo(make_repo(tmp_path))                 # written with earlier names
    files = plan_files(compiled.plan, compiled.manifest.raw)
    assert "mission.yaml" in files and "manifest.yaml" not in files
    staged = yaml.safe_load(files["mission.yaml"])
    assert {"format_version", "base", "jobs", "python_packages"} <= set(staged)
    assert staged["jobs"]["probes"]["recipes"][0]["vary"]


def test_the_earlier_names_are_the_same_objects():
    assert schema.Manifest is schema.MissionFile
    assert schema.parse_manifest is schema.parse_mission_file
    assert schema.MANIFEST_KEYS is schema.MISSION_FILE_KEYS


def test_messages_name_the_mission_file(tmp_path):
    root = make_repo(tmp_path, manifest=dict(MANIFEST, account="lab"))
    with pytest.raises(PlanError, match="belongs in hosts.yaml, not in the committed mission file"):
        compile_repo(root)


def test_the_cli_calls_it_the_mission_file(capsys):
    from training_framework.cli import main as naavik

    with pytest.raises(SystemExit):
        naavik(["validate", "--help"])
    assert "path to the committed mission file" in capsys.readouterr().out
