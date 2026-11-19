"""The `ddp` config schema: `world_size` required; `backend`, `master_addr`
and `master_port` optional, as the engine treats them (it decides the
address and port for each launch); unknown keys and bad values refused,
by the session check without building and when building alike."""

from __future__ import annotations

import copy
import importlib
import sys

import pytest

from tests.hpc.test_end_to_end import session
from naavik.components.builtin.distributed import DDPResource
from naavik.session import TrainingSession, check_session_config

PACKAGE = "tests.integration.integration_training_components"


@pytest.fixture(autouse=True)
def components():
    """Registered again for each test (the suite resets registrations)."""
    if PACKAGE in sys.modules:
        importlib.reload(sys.modules[PACKAGE])
    else:
        importlib.import_module(PACKAGE)


def config_with(tmp_path, **ddp):
    data = copy.deepcopy(session())
    data["session_config"]["components_package"] = PACKAGE
    data["session_config"]["sessions_dir"] = str(tmp_path / "runs")
    data["ddp"] = ddp
    return data


def test_the_engine_decided_keys_may_be_left_out(tmp_path):
    assert check_session_config(config_with(tmp_path, world_size=1)).problems == []
    resource = DDPResource({"world_size": 1})
    assert (resource.backend, resource._master_addr, resource._master_port) == ("gloo", "127.0.0.1", None)
    # The config as given is what the engine reads back.
    assert resource.config == {"world_size": 1}


@pytest.mark.parametrize("ddp, message", [
    ({"world_size": 1, "world_sise": 2}, "unknown keys ['world_sise']"),
    ({"backend": "gloo"}, "missing required keys ['world_size']"),
    ({"world_size": 0}, "ddp.world_size must be a positive integer; got 0"),
    ({"world_size": True}, "ddp.world_size must be a positive integer; got True"),
    ({"world_size": 1, "backend": "nccll"}, "ddp.backend must be one of gloo, nccl, mpi, ucc; got 'nccll'"),
    ({"world_size": 1, "master_port": "http"}, "ddp.master_port must be a port number (1-65535); got 'http'"),
    ({"world_size": 1, "master_port": 70000}, "ddp.master_port must be a port number"),
    ({"world_size": 1, "master_addr": ""}, "ddp.master_addr must be a host name or address"),
    ({"world_size": 1, "rank_zero_components": "logger"}, "ddp.rank_zero_components must be a list of component names"),
])
def test_a_wrong_ddp_config_is_refused_by_the_check_and_when_built(tmp_path, ddp, message):
    (problem,) = check_session_config(config_with(tmp_path, **ddp)).problems
    assert problem.startswith("ddp: Invalid ddp config: ") and message in problem
    with pytest.raises(ValueError) as built:
        DDPResource(ddp)
    assert problem == f"ddp: {built.value}"


@pytest.mark.parametrize("port", [29500, "29500"])
def test_a_port_is_a_number_or_its_digits(port):
    assert DDPResource({"world_size": 1, "master_port": port})._master_port == port


def test_the_deprecated_parallel_components_still_warns():
    with pytest.warns(FutureWarning, match="ddp.parallel_components is deprecated"):
        DDPResource({"world_size": 1, "parallel_components": ["model"]})


def test_setting_up_without_a_port_says_who_chooses_one():
    with pytest.raises(ValueError, match="ddp.master_port is not set: a launch by the engine"):
        DDPResource({"world_size": 1}).setup(None)


def test_a_checkpoint_with_a_key_ddp_ignored_still_restores(tmp_path):
    data = config_with(tmp_path, world_size=1, backend="gloo", master_addr="127.0.0.1",
                       master_port="29999")
    state = TrainingSession(data).get_state()
    state["components_state"]["ddp"]["init_args"]["args"][0]["timeout"] = 30
    with pytest.warns(UserWarning, match="ddp: the checkpoint's config has timeout, which ddp never read"):
        restored = TrainingSession.from_state(state)
    assert restored._components.get_resource("ddp").world_size == 1
