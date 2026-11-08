"""`module_part` freezing part of the part it holds: `frozen` and
`frozen_eval`, as `backbone_with_head` freezes its backbone.

The pretrained encoder is `Linear (0) -> BatchNorm (1) -> ReLU -> Dropout (3)
-> Linear (4)`; held as `module_part`'s part, its parameters are named
`module.0.weight`, `module.1.weight`, ...
"""

from __future__ import annotations

import pytest
import torch

from tests.components.builtin import test_fine_tuning as ft
from tests.test_utils import resource_named, stub_process_group
from training_framework.components.builtin import Checkpointer
from training_framework.session import TrainingSession


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    stub_process_group(monkeypatch)
    ft._register()


FIRST_BLOCK = ["module.0.*", "module.1.*"]


def _session(tmp_path, **part):
    path, _ = ft.pretrain_checkpoint(tmp_path / "pre")
    return ft.fine_tuning_session(
        tmp_path / "run", path, part={"submodule": "encoder", **part},
    )


def _part(session):
    return resource_named(session, "module_part")


def _requires_grad(session):
    return {
        name: parameter.requires_grad
        for name, parameter in _part(session).named_parameters()
    }


def test_matching_parameters_are_frozen_and_the_rest_train(tmp_path):
    session = _session(tmp_path, frozen=FIRST_BLOCK)
    before = ft.parameters(_part(session))

    ft.run(session)

    after = ft.parameters(_part(session))
    assert _requires_grad(session) == {
        "module.0.weight": False, "module.0.bias": False,
        "module.1.weight": False, "module.1.bias": False,
        "module.4.weight": True, "module.4.bias": True,
    }
    for name in before:
        changed = not torch.equal(before[name], after[name])
        assert changed is name.startswith("module.4."), name


def test_by_default_nothing_is_frozen(tmp_path):
    session = _session(tmp_path)

    assert all(_requires_grad(session).values())


def test_patterns_match_the_parts_own_names(tmp_path):
    # The names are `module.<...>`, so a pattern relative to the part matches
    # nothing, which is refused rather than freezing nothing.
    with pytest.raises(ValueError, match=r"module_part\.frozen"):
        _session(tmp_path, frozen=["0.*"])


def test_frozen_eval_keeps_a_frozen_batchnorm_still(tmp_path):
    session = _session(tmp_path, frozen=FIRST_BLOCK)
    norm = _part(session).module[1]
    statistics = norm.running_mean.clone()

    ft.run(session)

    assert not norm.training
    assert _part(session).module[4].training
    torch.testing.assert_close(norm.running_mean, statistics)


def test_without_frozen_eval_a_frozen_batchnorm_keeps_learning_statistics(tmp_path):
    session = _session(tmp_path, frozen=FIRST_BLOCK, frozen_eval=False)
    norm = _part(session).module[1]
    statistics = norm.running_mean.clone()

    ft.run(session)

    assert norm.training
    assert not torch.equal(norm.running_mean, statistics)


def test_freezing_everything_keeps_the_whole_part_in_eval(tmp_path):
    session = _session(tmp_path, frozen=["*"])

    ft.model_of(session).train()

    assert not any(_requires_grad(session).values())
    assert not _part(session).module.training


def test_frozen_eval_without_anything_frozen_is_refused(tmp_path):
    with pytest.raises(Exception, match="frozen_eval is true but nothing is frozen"):
        _session(tmp_path, frozen_eval=True)


def test_freezing_survives_a_restore(tmp_path):
    session = _session(tmp_path, frozen=FIRST_BLOCK)
    ft.run(session)
    saved = Checkpointer.save_checkpoint(session, tmp_path / "saved")

    restored = Checkpointer.load_checkpoint(saved)
    ft.model_of(restored).train()

    assert _requires_grad(restored) == _requires_grad(session)
    assert not _part(restored).module[1].training


def test_the_source_is_frozen_for_everything_that_holds_it(tmp_path):
    # `requires_grad` belongs to the parameter: a second part of the same
    # source, configured with nothing frozen, sees the same frozen weights.
    path, _ = ft.pretrain_checkpoint(tmp_path / "pre")
    config = ft.fine_tuning_config(
        tmp_path / "run", path,
        part={"submodule": "encoder", "frozen": FIRST_BLOCK},
        imports={"suffix": "pre"},
    )
    config["module_part#other"] = {
        "dependency_bindings": {"source": "ft_pretrain_model#pre"},
        "submodule": "encoder",
    }
    session = ft.quiet(TrainingSession(config))
    other = resource_named(session, "module_part#other")

    assert not other.module[0].weight.requires_grad
    assert other.module[4].weight.requires_grad


def test_freezing_a_whole_backbone_keeps_the_backbone_itself_in_eval(tmp_path):
    # Not only its children: a backbone whose own forward reads
    # `self.training` (functional dropout, say) must see eval too.
    path, _ = ft.pretrain_checkpoint(tmp_path / "pre")
    config = ft.without_import(ft.fine_tuning_config(
        tmp_path / "run", path,
        model={"frozen": ["*"]},
        bindings={"backbone": "ft_plain_backbone"},
        ft_plain_backbone={},
    ))
    model = ft.model_of(ft.quiet(TrainingSession(config)))

    model.train()

    assert not model.backbone.training
    assert model.head.training
