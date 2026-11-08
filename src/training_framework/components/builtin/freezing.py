"""Freezing a module's parameters by name, for every component that offers it.

A component that freezes part of a module it holds -- `backbone_with_head`
its backbone, `module_part` its part -- takes the same two options and gets
the same behaviour from here:

- `frozen`: glob patterns over the parameter names (`parameter_patterns`),
  each of which must match. Matching parameters get `requires_grad=False`
  when the component is built, before `ddp` wraps the model, so they get no
  gradient, cost no backward pass, and are never updated. Re-applied on
  every build, since `requires_grad` is not part of any state.
- `frozen_eval` (default: true when anything is frozen): every submodule
  whose parameters are all frozen stays in eval mode, so its BatchNorm
  statistics and dropout do not change what it computes.

`requires_grad` belongs to the parameter, so a module several components
hold is frozen for all of them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import nn

from training_framework.components.builtin.parameter_patterns import (
    check_patterns_match,
    matches_any,
    parameter_patterns,
)


def freeze_options(frozen: Any, frozen_eval: Any) -> tuple[tuple[str, ...], bool]:
    """The `frozen` patterns and `frozen_eval` flag a configuration gives,
    checked and with `frozen_eval`'s default filled in."""
    # Only an empty list means "nothing frozen": `false`, `0` or a key left
    # without a value is a mistake, not a request to train it all.
    if isinstance(frozen, (list, tuple)) and not frozen:
        patterns: tuple[str, ...] = ()
    else:
        patterns = tuple(parameter_patterns(frozen, "frozen"))
    if frozen_eval is not None and not isinstance(frozen_eval, bool):
        raise ValueError(f"frozen_eval must be true or false; got {frozen_eval!r}")
    if frozen_eval and not patterns:
        raise ValueError(
            "frozen_eval is true but nothing is frozen; list the frozen "
            "parameters under `frozen`"
        )
    if frozen_eval is None:
        # Frozen means frozen: BatchNorm statistics and dropout in a frozen
        # block change nothing about its weights, but they do change what it
        # computes.
        frozen_eval = bool(patterns)
    return patterns, frozen_eval


def freeze(module: nn.Module, patterns: Sequence[str], path: str) -> tuple[str, ...]:
    """Freeze the parameters of `module` whose names match `patterns`, and
    return the submodules of `module` this leaves with no trainable
    parameter, outermost only (`""` for `module` itself).

    `path` names the option in errors: a pattern matching nothing is one.
    """
    if not patterns:
        return ()
    named = list(module.named_parameters())
    check_patterns_match(patterns, [name for name, _ in named], path)
    frozen = set()
    for name, parameter in named:
        if matches_any(name, patterns):
            parameter.requires_grad_(False)
            frozen.add(id(parameter))

    def fully_frozen(candidate: nn.Module) -> bool:
        parameters = list(candidate.parameters())
        return bool(parameters) and all(
            id(parameter) in frozen for parameter in parameters
        )

    if fully_frozen(module):
        return ("",)
    names: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for child_name, child in parent.named_children():
            child_path = f"{prefix}{child_name}"
            if fully_frozen(child):
                names.append(child_path)
            else:
                visit(child, f"{child_path}.")

    visit(module, "")
    return tuple(names)


def keep_frozen_in_eval(module: nn.Module, names: Sequence[str]) -> None:
    """Put the submodules `freeze` returned back in eval mode; called from a
    holder's `train()` after the modes were set."""
    for name in names:
        module.get_submodule(name).eval()
