"""Config schemas of `trained_model` and `layer_inspector`: found by the
session check without building; whether the checkpoint exists is left to
building (it depends on the machine); a checkpoint's ignored keys are
dropped on restore."""

from __future__ import annotations

import pytest

from naavik.components.builtin.layer_inspection import LayerInspector
from naavik.components.builtin.model import TrainedModel
from naavik.components.config_schema import drop_unread_keys
from naavik.session import check_session_config


def analysis(tmp_path, **components):
    return {"session_type": "analysis",
            "session_config": {"rng_seed": 0, "sessions_dir": str(tmp_path), "max_iterations": 1,
                               "components_package": "naavik.components.builtin",
                               "device": "cpu"},
            "trained_model": {"model_checkpoint_path": str(tmp_path / "elsewhere" / "ckpt")},
            **components}


def problems(tmp_path, **components):
    return check_session_config(analysis(tmp_path, **components)).problems


def test_a_checkpoint_only_on_the_host_passes_the_check(tmp_path):
    assert problems(tmp_path, layer_inspector={"name_patterns": ["enc"]}) == []
    with pytest.raises(FileNotFoundError, match="does not exist"):        # building checks it
        TrainedModel({"model_checkpoint_path": str(tmp_path / "elsewhere" / "ckpt")})


@pytest.mark.parametrize("component, config, message", [
    ("trained_model", {}, "trained_model.model_checkpoint_path is required"),
    ("trained_model", {"model_checkpoint_path": "/x", "map_location": "cpu"}, "unknown keys ['map_location']"),
    ("trained_model", {"model_checkpoint_path": 3}, "must be a string or path-like object"),
    ("layer_inspector", {}, "requires at least one of 'name_patterns' or 'module_types'"),
    ("layer_inspector", {"name_patterns": ["("]}, "name_patterns contains an invalid regex"),
    ("layer_inspector", {"name_patterns": ["enc"], "always_call": "no"},
     "layer_inspector.always_call must be a boolean; got 'no'"),
    ("layer_inspector", {"name_patterns": ["enc"], "depth": 2}, "unknown keys ['depth']"),
    ("layer_inspector", {"module_types": ["torch.nn.NoSuchLayer"]}, "NoSuchLayer"),
])
def test_a_wrong_config_is_found_by_the_check(tmp_path, component, config, message):
    found = [item for item in problems(tmp_path, **{component: config}) if item.startswith(f"{component}: ")]
    assert len(found) == 1 and found[0].startswith(f"{component}: Invalid {component} config: ")
    assert message in found[0]


def test_building_says_the_same(tmp_path):
    with pytest.raises(TypeError) as built:
        LayerInspector({"name_patterns": ["enc"], "always_call": "no"})
    (checked,) = problems(tmp_path, layer_inspector={"name_patterns": ["enc"], "always_call": "no"})
    assert checked == f"layer_inspector: {built.value}"


def test_a_non_mapping_config_is_still_refused_as_such():
    with pytest.raises(TypeError, match="trained_model config must be a mapping"):
        TrainedModel(None)
    with pytest.raises(TypeError, match="layer_inspector config must be a mapping"):
        LayerInspector(None)


@pytest.mark.parametrize("component, name, config, stray", [
    (TrainedModel, "trained_model", {"model_checkpoint_path": "/x"}, "map_location"),
    (LayerInspector, "layer_inspector", {"name_patterns": ["enc"]}, "depth"),
])
def test_a_checkpoints_ignored_keys_are_dropped_on_restore(component, name, config, stray):
    with pytest.warns(UserWarning, match=f"{name}: the checkpoint's config has {stray}"):
        kept = drop_unread_keys(component, name, {**config, stray: 1})
    assert kept == config
