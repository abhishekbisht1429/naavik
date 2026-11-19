"""The optimizer's `config_schema`: the session check reads its config
without building anything, and the optimizer's, its parameter groups', and
its schedule's kwargs are checked against the torch class each builds."""

from __future__ import annotations

import copy

import pytest

from tests.hpc.test_end_to_end import session
from training_framework.components.builtin.optimization import OptimizerResource
from training_framework.components.config_schema import parse_component_config
from training_framework.session import check_session_config

PACKAGE = "tests.integration.integration_training_components"


@pytest.fixture(autouse=True)
def components():
    """Registered again for each test (the suite resets registrations)."""
    import importlib
    import sys

    if PACKAGE in sys.modules:
        importlib.reload(sys.modules[PACKAGE])
    else:
        importlib.import_module(PACKAGE)


def session_with(**optimizer):
    data = copy.deepcopy(session())
    data["session_config"]["components_package"] = PACKAGE
    data["optimizer"] = {"optimizer": {"name": "AdamW", "kwargs": {"lr": 0.1}}, **optimizer}
    return data


def problems(**optimizer):
    return check_session_config(session_with(**optimizer)).problems


def test_a_sound_optimizer_config_passes_the_check():
    assert problems(param_groups=[{"match": ["*bias"], "kwargs": {"weight_decay": 0.0}}],
                    lr_scheduler={"stages": [{"name": "CosineAnnealingLR",
                                              "kwargs": {"T_max": "$max_iterations"}}]}) == []


@pytest.mark.parametrize("optimizer, message", [
    ({"optimizer": {"name": "AdamW", "kwargs": {"lrr": 0.1}}},
     "optimizer.optimizer.kwargs: AdamW takes no 'lrr' (it takes: lr, betas, eps"),
    ({"param_groups": [{"match": ["*bias"], "kwargs": {"weight_decy": 0.0}}]},
     "optimizer.param_groups[0].kwargs: AdamW takes no 'weight_decy'"),
    ({"lr_scheduler": {"stages": [{"name": "LinearLR", "kwargs": {"start_factor": 0.1}},
                                  {"name": "CosineAnnealingLR", "kwargs": {"tmax": 10}}],
                       "milestones": [5]}},
     "optimizer.lr_scheduler.stages[1].kwargs: CosineAnnealingLR takes no 'tmax' (it takes: T_max, eta_min, last_epoch)"),
    ({"optimizer": {"name": "NoSuchOptimizer"}}, "Unknown optimizer class: NoSuchOptimizer"),
    ({"precision": "fp8"}, "optimizer.precision must be one of"),
    ({"accumulate_steps": 0}, "optimizer.accumulate_steps must be a positive integer"),
    ({"momentum": 0.9}, "unknown keys ['momentum']"),
])
def test_the_check_finds_what_building_would_refuse(optimizer, message):
    if "optimizer" in optimizer:
        found = check_session_config({**session_with(), "optimizer": optimizer}).problems
    else:
        found = problems(**optimizer)
    assert len(found) == 1 and found[0].startswith("optimizer: Invalid optimizer config: ")
    assert message in found[0]


def test_building_says_the_same():
    with pytest.raises(ValueError) as built:
        OptimizerResource({"optimizer": {"name": "AdamW", "kwargs": {"lrr": 0.1}}})
    (checked,) = problems(optimizer={"name": "AdamW", "kwargs": {"lrr": 0.1}})
    assert checked == f"optimizer: {built.value}"


def test_legacy_keys_keep_their_message():
    with pytest.raises(ValueError, match="Legacy optimizer fields are no longer supported: learning_rate"):
        OptimizerResource({"optimizer": {"name": "SGD"}, "learning_rate": 0.1})


def test_a_class_taking_any_keyword_is_not_checked_by_name():
    from training_framework.components.builtin.optimization import _check_kwargs

    class Flexible:
        def __init__(self, params, **options):
            pass

    _check_kwargs(Flexible, {"anything": 1}, "optimizer.optimizer.kwargs", "params")


def test_the_settings_are_what_the_resource_keeps():
    config = {"optimizer": {"name": "SGD", "kwargs": {"lr": 0.5, "momentum": 0.9}},
              "accumulate_steps": 2, "precision": "bf16"}
    settings = parse_component_config(OptimizerResource, config).settings
    assert settings == OptimizerResource(config)._settings
    assert (settings.accumulate_steps, settings.precision) == (2, "bf16")
    assert settings.optimizer_spec == {"name": "SGD", "kwargs": {"lr": 0.5, "momentum": 0.9}}
