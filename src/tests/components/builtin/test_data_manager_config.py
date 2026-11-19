"""The data managers' `config_schema`s: unknown keys refused, the documented
defaults real, values checked; the session check also splits the global
batch over `ddp.world_size`; a checkpoint written while a data manager
ignored keys still restores, dropping them with a warning."""

from __future__ import annotations

import copy
import importlib
import sys

import pytest

from tests.hpc.test_end_to_end import session
from naavik.components.builtin.data import AnalysisDataManager, DataManager
from naavik.session import TrainingSession, check_session_config

PACKAGE = "tests.integration.integration_training_components"


@pytest.fixture(autouse=True)
def components():
    """Registered again for each test (the suite resets registrations)."""
    if PACKAGE in sys.modules:
        importlib.reload(sys.modules[PACKAGE])
    else:
        importlib.import_module(PACKAGE)


def training(tmp_path, **data_manager):
    data = copy.deepcopy(session())
    data["session_config"]["components_package"] = PACKAGE
    data["session_config"]["sessions_dir"] = str(tmp_path / "runs")
    data["ddp"]["master_port"] = "29999"
    data["data_manager"] = data_manager
    return data


def test_the_documented_defaults_are_real():
    manager = DataManager({"batch_size": 4})
    assert (manager._num_workers, manager._pin_memory) == (0, False)
    analysis = AnalysisDataManager({"batch_size": 4})
    assert (analysis._num_workers, analysis._pin_memory, analysis._drop_last) == (0, False, False)


@pytest.mark.parametrize("component", [DataManager, AnalysisDataManager])
@pytest.mark.parametrize("config, error, message", [
    ({"batch_size": 4, "shuffle": True}, ValueError, "Invalid data_manager config: unknown keys ['shuffle']"),
    ({"num_workers": 2}, ValueError, "Invalid data_manager config: missing required keys ['batch_size']"),
    ({"batch_size": 0}, ValueError, "DataManager batch_size must be a positive integer"),
    ({"batch_size": 4, "num_workers": -1}, ValueError, "DataManager num_workers must be a non-negative integer"),
    ({"batch_size": 4, "num_workers": True}, ValueError, "DataManager num_workers must be a non-negative integer"),
    ({"batch_size": 4, "pin_memory": "yes"}, TypeError, "DataManager pin_memory must be a boolean"),
])
def test_a_wrong_config_is_refused_when_built(component, config, error, message):
    with pytest.raises(error) as raised:
        component(config)
    assert message in str(raised.value)


def test_drop_last_belongs_to_the_analysis_data_manager():
    with pytest.raises(TypeError, match="DataManager drop_last must be a boolean"):
        AnalysisDataManager({"batch_size": 4, "drop_last": 1})
    with pytest.raises(ValueError, match=r"unknown keys \['drop_last'\]"):
        DataManager({"batch_size": 4, "drop_last": True})


def test_the_check_finds_an_unknown_key_without_building(tmp_path):
    problems = check_session_config(training(tmp_path, batch_size=4, shuffle=True)).problems
    assert problems == ["data_manager: Invalid data_manager config: unknown keys ['shuffle']. "
                        "Accepted keys: ['batch_size', 'num_workers', 'pin_memory']"]


@pytest.mark.parametrize("batch_size, world_size, message", [
    (3, 2, "DataManager batch_size must be divisible by the DDP world_size (batch_size 3, ddp.world_size 2)"),
    (1, 2, "DataManager batch_size must be at least the DDP world_size so every rank receives a sample"),
])
def test_the_check_splits_the_global_batch_over_the_ranks(tmp_path, batch_size, world_size, message):
    data = training(tmp_path, batch_size=batch_size)
    data["ddp"]["world_size"] = world_size
    (problem,) = check_session_config(data).problems
    assert problem.startswith("data_manager: ") and message in problem


def test_an_invalid_world_size_is_reported_once_by_ddp(tmp_path):
    data = training(tmp_path, batch_size=3)
    data["ddp"]["world_size"] = "2"                           # not a number in the config
    (problem,) = check_session_config(data).problems
    assert problem.startswith("ddp: Invalid ddp config: ddp.world_size must be a positive integer")
    data["ddp"]["world_size"] = 3
    assert check_session_config(data).problems == []


def test_a_checkpoint_with_a_key_the_data_manager_ignored_still_restores(tmp_path):
    state = TrainingSession(training(tmp_path, batch_size=4, num_workers=0, pin_memory=False)).get_state()
    # As a checkpoint written before the schema holds it: a key it never read.
    state["components_state"]["data_manager"]["init_args"]["args"][0]["shuffle"] = True
    with pytest.warns(UserWarning, match="data_manager: the checkpoint's config has shuffle, which "
                                         "data_manager never read; dropped"):
        restored = TrainingSession.from_state(state)
    manager = restored._components.get_resource("data_manager")
    assert manager.batch_size == 4
    # A new session with the same key is refused.
    with pytest.raises(ValueError, match=r"unknown keys \['shuffle'\]"):
        TrainingSession(training(tmp_path, batch_size=4, shuffle=True))


def test_a_component_without_the_flag_still_refuses_on_restore(tmp_path):
    from naavik.session.components import _without_unread_keys

    class Strict:
        config_schema = type("Schema", (), {})

    args = ({"a": 1},)
    assert _without_unread_keys(Strict, "strict", args) == args
