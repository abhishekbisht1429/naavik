"""Uniform configuration parsing for components that opt in.

A component sets ``config_schema`` to a dataclass; its fields validate the
*shape* of the configuration -- which keys are required, which are accepted,
and what they are coerced to -- while ``__post_init__`` validates *meaning*,
such as a head count dividing an embedding width.

Deliberately small. Nested specs whose entries are arbitrary user-supplied
values stay hand-parsed by the component that understands them.
"""

from collections.abc import Mapping, Sequence
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, get_origin


def _plain(value: Any) -> Any:
    """Return `value` with OmegaConf containers converted to dict/list.

    Keys are kept as they are. Turning them into strings here would make a
    mistyped key (a YAML `1:` where a name was meant) look valid to every
    schema, and fail only when the value is finally used.
    """
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    return value


def _coerce(value: Any, annotation: Any) -> Any:
    """Coerce a sequence to a tuple when the field is annotated as one."""
    if not isinstance(value, list):
        return value
    if isinstance(annotation, str):
        # `from __future__ import annotations` leaves the annotation a string.
        return tuple(value) if annotation.lstrip().startswith("tuple") else value
    if annotation is tuple or get_origin(annotation) is tuple:
        return tuple(value)
    return value


def _accepted_keys(schema: type) -> list[str]:
    return [field.name for field in fields(schema)]


def _check_no_reserved_fields(component_class: type, schema: type) -> None:
    """Refuse a schema field named like a component's wiring key: the session
    takes that key out of the entry as wiring, so the field could never be
    given, and would quietly keep its default."""
    # Imported here: `config` is the session-configuration module, which this
    # module otherwise has no need of.
    from training_framework.components.config import (
        DEPENDENCY_BINDINGS_KEY,
        LEGACY_DEPENDENCY_BINDINGS_KEY,
    )

    reserved = {DEPENDENCY_BINDINGS_KEY, LEGACY_DEPENDENCY_BINDINGS_KEY}
    clashing = sorted(reserved & set(_accepted_keys(schema)))
    if clashing:
        raise TypeError(
            f"{_name(component_class)} config_schema declares "
            f"{', '.join(repr(name) for name in clashing)}, which is reserved: "
            "in a component's entry it holds the component's own wiring "
            f"('{DEPENDENCY_BINDINGS_KEY}', formerly "
            f"'{LEGACY_DEPENDENCY_BINDINGS_KEY}') and never reaches the "
            "component. Rename the field."
        )


def parse_component_config(
        component_class: type,
        config: Mapping | None,
) -> Any:
    """Return `config` parsed into the component's ``config_schema``.

    Raises ``TypeError`` or ``ValueError`` -- whichever the underlying failure
    is -- wrapped in the ``Invalid <name> config: ...`` shape the framework
    uses elsewhere.
    """
    schema = component_class.config_schema
    if schema is None or not is_dataclass(schema):
        raise TypeError(
            f"{_name(component_class)} config_schema must be a dataclass; "
            f"got {schema!r}"
        )

    _check_no_reserved_fields(component_class, schema)

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise TypeError(f"{_name(component_class)} config must be a mapping")

    values = _plain(config)
    accepted = _accepted_keys(schema)

    # A schema may refuse keys with its own message before the generic
    # unknown-key one (for example, keys of a format it no longer reads).
    check_keys = getattr(schema, "check_keys", None)
    if check_keys is not None:
        try:
            check_keys(values)
        except (TypeError, ValueError) as error:
            raise type(error)(f"Invalid {_name(component_class)} config: {error}") from error

    unknown = sorted(set(values) - set(accepted), key=str)
    if unknown:
        raise ValueError(
            f"Invalid {_name(component_class)} config: unknown keys "
            f"{unknown}. Accepted keys: {accepted}"
        )

    missing = [
        field.name
        for field in fields(schema)
        if field.name not in values
        and field.default is MISSING
        and field.default_factory is MISSING  # type: ignore[misc]
    ]
    if missing:
        raise ValueError(
            f"Invalid {_name(component_class)} config: missing required keys "
            f"{sorted(missing)}"
        )

    annotations = {field.name: field.type for field in fields(schema)}
    arguments = {
        key: _coerce(value, annotations[key])
        for key, value in values.items()
    }

    try:
        return schema(**arguments)
    except (TypeError, ValueError) as error:
        raise type(error)(
            f"Invalid {_name(component_class)} config: {error}"
        ) from error


def _name(component_class: type) -> str:
    return getattr(component_class, "name", component_class.__name__)


def field_names(schema: type) -> tuple[str, ...]:
    """Return the configuration keys a schema accepts."""
    return tuple(_accepted_keys(schema))


def has_all_defaults(schema: type | None) -> bool:
    """Return whether every field of `schema` has a default.

    Such a component is fully configured by an empty mapping, so the session
    may build it without a top-level entry.
    """
    if schema is None or not is_dataclass(schema):
        return False
    return all(
        field.default is not MISSING
        or field.default_factory is not MISSING  # type: ignore[misc]
        for field in fields(schema)
    )
