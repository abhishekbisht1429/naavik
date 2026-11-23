"""A config file is one session: its top level is the session's entry.

The earlier form, `sessions: [entry]` with one entry, is still read with a
`SessionsListWarning`; several entries are refused."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest
import yaml

from naavik.config_files import (
    ConfigFileError,
    SessionsListWarning,
    load_config,
    load_session,
)
from naavik.engine import Configurator

SESSION = {
    "session_config": {
        "max_iterations": 2, "device": "cpu", "rng_seed": 1,
        "components_package": "naavik.components.builtin",
    },
    "logger": {"log_every": 1},
}


def _write(root: Path, files: dict[str, object]) -> Path:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content if isinstance(content, str) else yaml.safe_dump(content),
                        encoding="utf-8")
    return root / next(iter(files))


def _configurator(monkeypatch, path, *overrides) -> Configurator:
    argv = ["pytest", "--config", str(path)]
    if overrides:
        argv += ["--override", *overrides]
    monkeypatch.setattr(sys, "argv", argv)
    return Configurator()


# The loader ------------------------------------------------------------------------------------


def test_the_top_of_the_file_is_the_session(tmp_path):
    loaded = load_session(_write(tmp_path, {"train.yaml": SESSION}))
    assert loaded.session == SESSION
    assert loaded.wrapped is False
    assert [Path(source.path).name for source in loaded.sources] == ["train.yaml"]


def test_includes_at_the_top_of_a_session_file_are_merged_under_it(tmp_path):
    loaded = load_session(_write(tmp_path, {
        "train.yaml": {"includes": ["common/model.yaml"], **SESSION,
                       "model": {"depth": 4}},
        "common/model.yaml": {"model": {"depth": 2, "width": 8}},
    }))
    assert loaded.session["model"] == {"depth": 4, "width": 8}
    assert "includes" not in loaded.session
    assert [Path(source.path).name for source in loaded.sources] == ["train.yaml", "model.yaml"]


def test_an_earlier_form_file_with_one_entry_is_read_as_that_entry(tmp_path):
    loaded = load_session(_write(tmp_path, {
        "train.yaml": {"sessions": [SESSION], "shared": {"anchor": 1}}}))
    assert loaded.session == SESSION
    assert loaded.wrapped is True
    assert loaded.rest == {"shared": {"anchor": 1}}
    advice = loaded.unwrap_advice("train.yaml")
    assert "Move the entry's keys to the top of the file" in advice and "shared" in advice


def test_load_config_keeps_the_earlier_shape_for_both_forms(tmp_path):
    new = load_config(_write(tmp_path, {"new.yaml": SESSION}))
    old = load_config(_write(tmp_path, {"old.yaml": {"sessions": [SESSION]}}))
    assert new.config == old.config == {"sessions": [SESSION]}


@pytest.mark.parametrize("sessions", [[], "not a list"])
def test_a_sessions_key_that_is_not_one_entry_is_refused(tmp_path, sessions):
    with pytest.raises(ConfigFileError, match="'sessions' must be a list with one session"):
        load_session(_write(tmp_path, {"train.yaml": {"sessions": sessions}}))


# The engine ------------------------------------------------------------------------------------


def test_the_engine_runs_a_session_file_without_a_warning(tmp_path, monkeypatch):
    path = _write(tmp_path, {"train.yaml": SESSION})
    with warnings.catch_warnings():
        warnings.simplefilter("error", SessionsListWarning)
        configurator = _configurator(monkeypatch, path)
    assert configurator.session_configs == [SESSION]
    assert [Path(source.path).name for source in configurator.config_sources[0]] == ["train.yaml"]


def test_engine_overrides_are_relative_to_the_session(tmp_path, monkeypatch):
    path = _write(tmp_path, {"train.yaml": SESSION})
    configurator = _configurator(monkeypatch, path, "logger.log_every=7",
                                 "session_config.rng_seed=5")
    (session,) = configurator.session_configs
    assert session["logger"]["log_every"] == 7
    assert session["session_config"]["rng_seed"] == 5


def test_a_sessions_override_on_a_session_file_is_refused_with_the_path_to_use(
        tmp_path, monkeypatch, capsys):
    path = _write(tmp_path, {"train.yaml": SESSION})
    with pytest.raises(SystemExit):
        _configurator(monkeypatch, path, "sessions[0].logger.log_every=7")
    assert "write logger.log_every=7" in capsys.readouterr().err


def test_an_earlier_form_file_warns_and_still_takes_sessions_0_overrides(tmp_path, monkeypatch):
    path = _write(tmp_path, {"train.yaml": {"sessions": [SESSION]}})
    with pytest.warns(SessionsListWarning, match="a config file is now one session"):
        configurator = _configurator(monkeypatch, path, "sessions[0].logger.log_every=7",
                                     "sessions.0.session_config.rng_seed=5",
                                     "logger.nested=3")
    (session,) = configurator.session_configs
    assert session["logger"] == {"log_every": 7, "nested": 3}
    assert session["session_config"]["rng_seed"] == 5


def test_an_override_for_a_second_session_is_refused(tmp_path, monkeypatch, capsys):
    path = _write(tmp_path, {"train.yaml": {"sessions": [SESSION]}})
    with pytest.raises(SystemExit):
        _configurator(monkeypatch, path, "sessions[1].logger.log_every=7")
    assert "a config file is one session" in capsys.readouterr().err


def test_the_engine_refuses_a_file_with_several_sessions(tmp_path, monkeypatch, capsys):
    path = _write(tmp_path, {"train.yaml": {"sessions": [SESSION, SESSION]}})
    with pytest.raises(SystemExit):
        _configurator(monkeypatch, path)
    assert "2 sessions in one file" in capsys.readouterr().err
