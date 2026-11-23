"""Tests for reading session definitions and launch options from the command line."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from naavik.engine import Configurator
from naavik.session import TrainingSession
from tests.test_utils import has_resource_named, resource_named


def _write_yaml(tmp_path: Path, data: dict, name: str = "config.yaml") -> str:
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    return str(path)


def test_configurator_reads_overrides_and_returns_deep_copies(tmp_path, monkeypatch):
    sample_config = {
        "session_config": {
            "max_iterations": 5,
            "sessions_dir": str(tmp_path / "sessions_1"),
            "device": "cpu",
            "rng_seed": 123,
            "components_package": "naavik.components.builtin",
        },
        "logger": {"log_every": 1, "log_file": str(tmp_path / "log_1.txt"), "nested": {"enabled": True}},
        "checkpointer": {"checkpoint_every": 2, "checkpoints_dir": str(tmp_path / "ckpts_1")},
        "tensorboard": {"host": "0.0.0.0", "port": 16040},
    }

    config_path = _write_yaml(tmp_path, sample_config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pytest",
            "--config",
            config_path,
            "--debug",
            "--stop-sync-grace-period",
            "0.02",
            "--stop-sync-poll-interval",
            "0.007",
            "--override",
            "checkpointer.checkpoint_every=11",
            "tensorboard.port=16041",
        ],
    )

    configurator = Configurator()
    session_config = configurator.get_session_definition(0)
    resource_config = configurator.get_component_config(0, "logger")

    assert session_config["checkpointer"]["checkpoint_every"] == 11
    assert configurator.debug is True
    assert configurator.stop_sync_grace_period == 0.02
    assert configurator.stop_sync_poll_interval == 0.007
    assert configurator.get_component_config(0, "tensorboard")["port"] == 16041
    assert resource_config == sample_config["logger"]

    resource_config["nested"]["enabled"] = False
    assert sample_config["logger"]["nested"]["enabled"] is True

    with pytest.raises(KeyError):
        configurator.get_component_config(0, "missing")

    sample_config["logger"]["log_every"] = 99
    assert configurator.get_session_definition(0)["logger"]["log_every"] == 1


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--stop-sync-grace-period", "-0.1"),
        ("--stop-sync-grace-period", "nan"),
        ("--stop-sync-poll-interval", "0"),
        ("--stop-sync-poll-interval", "inf"),
    ],
)
def test_configurator_rejects_invalid_stop_sync_timing(
        monkeypatch,
        option,
        value,
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pytest", "--resume-session", "checkpoint.pt", option, value],
    )

    with pytest.raises(SystemExit):
        Configurator()


def test_configurator_create_sessions_attaches_expected_components(tmp_path, monkeypatch):
    sample_config = {
        "session_config": {
            "max_iterations": 2,
            "sessions_dir": str(tmp_path / "s1"),
            "device": "cpu",
            "rng_seed": 1,
            "components_package": "naavik.components.builtin",
        },
        "tensorboard": {"host": "0.0.0.0", "port": 16050},
    }

    config_path = _write_yaml(tmp_path, sample_config, "config_create_sessions.yaml")
    monkeypatch.setattr(sys, "argv", ["pytest", "--config", config_path])

    configurator = Configurator()
    sessions = [TrainingSession(config) for config in configurator.session_configs]

    assert len(sessions) == 1
    assert configurator.debug is False
    assert configurator.stop_sync_grace_period == 0.01
    assert configurator.stop_sync_poll_interval == 0.005

    hook_names = {component.name for component in sessions[0].get_all_hooks()}

    assert {"logger", "checkpointer"} <= hook_names
    assert has_resource_named(sessions[0], "tensorboard")
    assert resource_named(sessions[0], "tensorboard").name == "tensorboard"
