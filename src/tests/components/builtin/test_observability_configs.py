"""Config schemas of `checkpointer`, `logger`, `timer` and `tensorboard`:
found by the session check without building, refused alike when built or
extended; checkpoints and pickles with keys they once ignored still load."""

from __future__ import annotations

import copy
import importlib
import pickle
import sys

import pytest

from tests.hpc.test_end_to_end import session
from naavik.components.builtin.checkpointing import Checkpointer
from naavik.components.builtin.observability import Logger, Tensorboard, Timer
from naavik.session import TrainingSession, check_session_config

PACKAGE = "tests.integration.integration_training_components"


@pytest.fixture(autouse=True)
def components():
    """Registered again for each test (the suite resets registrations)."""
    if PACKAGE in sys.modules:
        importlib.reload(sys.modules[PACKAGE])
    else:
        importlib.import_module(PACKAGE)


def config_with(tmp_path, **entries):
    data = copy.deepcopy(session())
    data["session_config"]["components_package"] = PACKAGE
    data["session_config"]["sessions_dir"] = str(tmp_path / "runs")
    data["ddp"]["master_port"] = "29999"
    data.update(entries)
    return data


CASES = [
    (Checkpointer, "checkpointer", {"checkpoint_every": 100, "keep": 3}, "unknown keys ['keep']"),
    (Checkpointer, "checkpointer", {"checkpoint_first": True}, "missing required keys ['checkpoint_every']"),
    (Checkpointer, "checkpointer", {"checkpoint_every": 0}, "checkpointer.checkpoint_every must be a positive integer; got 0"),
    (Checkpointer, "checkpointer", {"checkpoint_every": 5, "checkpoint_first": "yes"}, "checkpointer.checkpoint_first must be a boolean"),
    (Checkpointer, "checkpointer", {"checkpoint_every": 5, "checkpoints_dir": ""}, "checkpointer.checkpoints_dir must be a directory path"),
    (Logger, "logger", {"log_every": 10, "level": "info"}, "unknown keys ['level']"),
    (Logger, "logger", {"log_every": -1}, "logger.log_every must be a positive integer; got -1"),
    (Logger, "logger", {"log_every": 10, "log_file": ""}, "logger.log_file must be a file path"),
    (Timer, "timer", {}, "missing required keys ['call_every']"),
    (Timer, "timer", {"call_every": 2.5}, "timer.call_every must be a positive integer; got 2.5"),
    (Tensorboard, "tensorboard", {"host": "127.0.0.1"}, "tensorboard serves on host:port, so it needs port"),
    (Tensorboard, "tensorboard", {"host": "127.0.0.1", "port": "web"}, "tensorboard.port must be a port number"),
    (Tensorboard, "tensorboard", {"serve": False, "port": 6006}, "tensorboard starts no server with serve: false, so port"),
    (Tensorboard, "tensorboard", {"serve": "no"}, "serve must be a boolean; got 'no'"),
    (Tensorboard, "tensorboard", {"serve": False, "flush_secs": 5}, "unknown keys ['flush_secs']"),
]


@pytest.mark.parametrize("component, name, config, message", CASES)
def test_a_wrong_config_is_refused_by_the_check_and_when_built(tmp_path, component, name, config, message):
    problems = check_session_config(config_with(tmp_path, **{name: config})).problems
    (problem,) = [item for item in problems if item.startswith(f"{name}: ")]
    assert problem.startswith(f"{name}: Invalid {name} config: ") and message in problem
    with pytest.raises((TypeError, ValueError)) as built:
        component(config)
    assert problem == f"{name}: {built.value}"


@pytest.mark.parametrize("component, config", [
    (Checkpointer, {"checkpoint_every": 100, "checkpoints_dir": "./ckpt", "checkpoint_first": True}),
    (Logger, {"log_every": 10, "log_file": "./runs/train.log"}),
    (Timer, {"call_every": 10}),
    (Tensorboard, {"host": "127.0.0.1", "port": 6006, "logdir": "./runs/tb"}),
    (Tensorboard, {"serve": False}),
])
def test_the_documented_configs_pass(tmp_path, component, config):
    component(config)
    name = {Checkpointer: "checkpointer", Logger: "logger", Timer: "timer", Tensorboard: "tensorboard"}[component]
    assert [p for p in check_session_config(config_with(tmp_path, **{name: config})).problems
            if p.startswith(f"{name}: ")] == []


def test_an_extension_is_parsed_the_same_way():
    logger = Logger({"log_every": 10})
    logger.apply_extension_config({"log_every": 5}, frozenset({("log_every",)}))
    assert logger.call_every == 5
    checkpointer = Checkpointer({"checkpoint_every": 100})
    with pytest.raises(ValueError, match="checkpointer.checkpoint_every must be a positive integer; got 0"):
        checkpointer.apply_extension_config({"checkpoint_every": 0}, frozenset({("checkpoint_every",)}))
    assert checkpointer.call_every == 100


def test_a_checkpoint_with_a_key_the_checkpointer_ignored_still_restores(tmp_path):
    data = config_with(tmp_path, checkpointer={"checkpoint_every": 50})
    state = TrainingSession(data).get_state()
    state["components_state"]["checkpointer"]["init_args"]["args"][0]["keep"] = 3
    with pytest.warns(UserWarning, match="checkpointer: the checkpoint's config has keep, which "
                                         "checkpointer never read"):
        restored = TrainingSession.from_state(state)
    (checkpointer,) = [hook for hook in restored.get_all_hooks() if hook.name == "checkpointer"]
    assert checkpointer.call_every == 50


def test_an_unpickled_logger_drops_keys_it_ignored():
    logger = Logger({"log_every": 10})
    logger._config = {"log_every": 10, "level": "info"}       # as an old pickle holds it
    with pytest.warns(UserWarning, match="logger: the checkpoint's config has level"):
        restored = pickle.loads(pickle.dumps(logger))
    assert restored.call_every == 10
