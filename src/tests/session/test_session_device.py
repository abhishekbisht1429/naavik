"""`session_config.device`: where a session runs, and what refuses to run."""

import pytest
import torch
from torch import nn

from tests.test_utils import make_config, resource_named
from naavik.components import (
    ModuleResource,
    requires_resource,
    resource,
)
from naavik.components.builtin import Checkpointer
from naavik.launcher.config.schema import Resources
from naavik.launcher.config.topology import check_topology
from naavik.session import (
    AnalysisSession,
    TrainingSession,
    check_session_config,
)


def _config(tmp_path, device):
    config = make_config(tmp_path)
    config["session_config"]["device"] = device
    config["session_config"]["show_execution_graph"] = False
    return config


@pytest.mark.parametrize("device", ["gpu", "CUDA:0", " cuda", "mps"])
def test_a_device_the_session_cannot_run_on_is_refused(tmp_path, device):
    with pytest.raises(ValueError, match="session_config.device"):
        TrainingSession(_config(tmp_path, device))


def test_cuda_without_cuda_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA is not available"):
        TrainingSession(_config(tmp_path, "cuda"))


def test_a_cuda_ordinal_past_the_visible_devices_is_refused(
        tmp_path, monkeypatch,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    with pytest.raises(ValueError, match="only 2 CUDA device"):
        TrainingSession(_config(tmp_path, "cuda:2"))


def test_cpu_is_the_default_device(tmp_path):
    config = _config(tmp_path, "cpu")
    del config["session_config"]["device"]

    assert TrainingSession(config).device == torch.device("cpu")


def test_a_checkpoint_asking_for_missing_cuda_loads_on_cpu_with_a_warning(
        tmp_path, monkeypatch,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    session = TrainingSession(_config(tmp_path, "cuda"))
    path = Checkpointer.save_checkpoint(session, tmp_path / "checkpoint")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(UserWarning, match="running on cpu instead"):
        restored = Checkpointer.load_checkpoint(path)

    assert restored.device == torch.device("cpu")


# -- checks made before the session reaches the machine that runs it ----------


MISSPELLED_DEVICES = ["gpu", "CUDA:0", "cuda:abc", " cuda", "mps"]
_RESOURCES = Resources("01:00:00", 3600, 1, 8, "16G", {})


@pytest.mark.parametrize("device", MISSPELLED_DEVICES)
def test_the_config_check_reports_a_misspelled_device(tmp_path, device):
    check = check_session_config(_config(tmp_path, device))

    assert not check.ok
    assert any("session_config.device" in problem for problem in check.problems)


def test_the_config_check_leaves_availability_to_the_running_machine(
        tmp_path, monkeypatch,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert check_session_config(_config(tmp_path, "cuda:7")).ok


@pytest.mark.parametrize("device", MISSPELLED_DEVICES)
def test_a_launch_refuses_a_misspelled_device(tmp_path, device):
    problems = check_topology(_config(tmp_path, device), _RESOURCES)

    assert any("session_config.device" in problem for problem in problems)


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0"])
def test_a_launch_accepts_a_device_its_group_provides(tmp_path, device):
    assert not check_topology(_config(tmp_path, device), _RESOURCES)


# -- trained_model: every part reaches the analysis device --------------------


def _declare_model_reaching_its_encoder():
    """A model that holds its encoder nowhere: it takes it from
    get_dependency on every call."""

    @resource("device_encoder")
    class _Encoder(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(x)

    @requires_resource("device_encoder")
    @resource("device_model")
    class _ModelReachingItsEncoder(ModuleResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.head = nn.Linear(4, 2)

        def forward(self, x):
            return self.head(self.get_dependency("device_encoder")(x))


def _trained_checkpoint(tmp_path):
    _declare_model_reaching_its_encoder()
    config = _config(tmp_path / "source", "cpu")
    config["device_encoder"] = {}
    config["device_model"] = {}
    config["role_bindings"] = {"model": "device_model"}
    return Checkpointer.save_checkpoint(
        TrainingSession(config), tmp_path / "trained",
    )


def test_load_component_moves_every_rebuilt_part_to_device(tmp_path):
    path = _trained_checkpoint(tmp_path)

    model = Checkpointer.load_component(path, "model", device="meta")

    assert model.head.weight.device.type == "meta"
    encoder = model.get_dependency("device_encoder")
    assert encoder.linear.weight.device.type == "meta"


def test_load_component_without_device_leaves_parts_where_they_are(tmp_path):
    path = _trained_checkpoint(tmp_path)

    model = Checkpointer.load_component(path, "model")

    encoder = model.get_dependency("device_encoder")
    assert encoder.linear.weight.device == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_analysis_runs_a_part_held_outside_the_model_on_its_device(tmp_path):
    path = _trained_checkpoint(tmp_path)
    config = _config(tmp_path / "analysis", "cuda:0")
    config["trained_model"] = {"model_checkpoint_path": str(path)}
    analysis = AnalysisSession(config)

    with analysis:
        model = resource_named(analysis, "trained_model").model
        encoder = model.get_dependency("device_encoder")
        assert encoder.linear.weight.device == torch.device("cuda", 0)
        output = model(torch.ones(1, 4, device=analysis.device))

    assert output.device == torch.device("cuda", 0)
