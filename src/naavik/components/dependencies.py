"""What a component declares about the components it depends on.

`@requires_resource`, `@requires_hook` and `@requires_step` record each
dependency here, through `declare_dependency`, the one writer. A
declaration says, besides the name, whether the dependency is *local* -- a
part of this component that it alone is wired to, through its own
`dependency_bindings` entry, rather than a role shared by the session and
bound in the top-level `role_bindings` -- and what it is for, which a
missing-implementation message shows.

The names a class requires are still the public lists `required_resources`,
`required_hooks` and `required_steps`, which edges and injection read. The
writer keeps a class's list and its declarations together, so they cannot
disagree; a name in a list without a declaration (a list set by hand) reads
as a global dependency with no description.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from naavik.components.base import Component, Hook, Resource, Step
from naavik.components.config import DEPENDENCY_BINDINGS_KEY
from naavik.components.naming import validate_component_name


@dataclass(frozen=True)
class DependencyDeclaration:
    """One dependency, as the component that requires it declares it."""

    name: str
    category: type[Component]
    local: bool = False
    """Wired only in the consumer's own `dependency_bindings`, never by a
    top-level `role_bindings` entry."""
    description: str | None = None
    """What the dependency is for, shown when nothing implements it."""


@dataclass(frozen=True)
class _Category:
    """How one category of dependency is declared."""

    attribute: str
    """The public class list of the names required."""
    decorator: str
    consumers: tuple[type[Component], ...]
    """The component types that may declare it."""
    consumers_phrase: str
    """How an error names `consumers`, and what a class outside them is."""
    inherited: bool
    """Whether a class's first declaration starts from its parent's list.
    `required_steps` has never been: a subclass declaring a step of its own
    requires that step only, and keeps doing so."""


_CATEGORIES: dict[type[Component], _Category] = {
    Resource: _Category(
        "required_resources", "requires_resource", (Step, Hook, Resource),
        "Step, Hook, or Resource subclasses. '{name}' is neither.", True,
    ),
    Hook: _Category(
        "required_hooks", "requires_hook", (Step,),
        "Step subclasses. '{name}' is not a Step.", True,
    ),
    Step: _Category(
        "required_steps", "requires_step", (Step,),
        "Step subclasses. '{name}' is not a Step.", False,
    ),
}

_DECLARATIONS_ATTR = "_dependency_declarations"
"""Class attribute: `{category: {name: DependencyDeclaration}}`."""


def dependency_decorator(
        category: type[Component],
        name: str,
        *,
        local: bool,
        description: str | None,
):
    """Return the class decorator `@requires_<category>(name, ...)` is.

    The arguments are checked when the decorator is written, not when it is
    applied, so a mistake points at the line that made it.
    """
    spec = _CATEGORIES[category]
    validate_component_name(name, kind=f"Required {category.__name__} name")
    if not isinstance(local, bool):
        raise TypeError(
            f"@{spec.decorator}('{name}', local=...) must be true or false; "
            f"got {local!r}"
        )
    if description is not None and (
            not isinstance(description, str) or not description.strip()
    ):
        raise TypeError(
            f"@{spec.decorator}('{name}', description=...) must be a "
            f"non-empty string or None; got {description!r}"
        )
    declaration = DependencyDeclaration(name, category, local, description)

    def wrapper(cls):
        declare_dependency(cls, declaration)
        return cls
    return wrapper


def declare_dependency(cls: type, declaration: DependencyDeclaration) -> None:
    """Record that `cls` requires `declaration`: the one writer of both the
    class's public name list and its declarations."""
    spec = _CATEGORIES[declaration.category]
    if not issubclass(cls, spec.consumers):
        raise TypeError(
            f"@{spec.decorator} can only be applied to "
            + spec.consumers_phrase.format(name=cls.__name__)
        )
    own = cls.__dict__.get(_DECLARATIONS_ATTR)
    if own is None:
        own = {
            category: dict(declarations)
            for category, declarations in _inherited_declarations(cls).items()
        }
        setattr(cls, _DECLARATIONS_ATTR, own)
    if spec.attribute not in cls.__dict__:
        # The class's first dependency of this category: its list starts
        # from the parent's, or empty for a category that is not inherited.
        # A parent's declaration of a name the list no longer holds is not
        # read (`dependency_declaration`).
        setattr(
            cls,
            spec.attribute,
            list(getattr(cls, spec.attribute, ())) if spec.inherited else [],
        )

    declared = own.setdefault(declaration.category, {})
    previous = declared.get(declaration.name)
    if (
            previous is not None
            and previous != declaration
            and declaration.name in _own_names(cls, spec)
    ):
        raise ValueError(
            f"'{cls.__name__}' declares {declaration.category.__name__} "
            f"'{declaration.name}' twice, differently: {previous} and "
            f"{declaration}"
        )
    declared[declaration.name] = declaration
    getattr(cls, spec.attribute).append(declaration.name)
    _own_names(cls, spec).add(declaration.name)


def _inherited_declarations(cls: type) -> Mapping[type, Mapping[str, Any]]:
    for base in cls.__mro__[1:]:
        found = base.__dict__.get(_DECLARATIONS_ATTR)
        if found is not None:
            return found
    return {}


_OWN_NAMES_ATTR = "_dependency_declared_here"
"""Class attribute: the `(attribute, name)` pairs declared on this class
itself, so a subclass may redeclare a parent's dependency but one class may
not declare the same one two ways."""


def _own_names(cls: type, spec: _Category) -> set[str]:
    own = cls.__dict__.get(_OWN_NAMES_ATTR)
    if own is None:
        own = {}
        setattr(cls, _OWN_NAMES_ATTR, own)
    return own.setdefault(spec.attribute, set())


def dependency_declaration(
        component: Component | type[Component],
        name: str,
        category: type[Component] = Resource,
) -> DependencyDeclaration:
    """How `component` declares its `category` dependency `name`.

    A name it does not declare through the decorators -- one in a list set by
    hand -- reads as a global dependency with no description.
    """
    cls = component if isinstance(component, type) else type(component)
    found = _class_declarations(cls).get(category, {}).get(name)
    if found is not None and name in getattr(cls, _CATEGORIES[category].attribute, ()):
        return found
    return DependencyDeclaration(name, category)


def local_dependencies(component: Component | type[Component]) -> frozenset[str]:
    """The names `component` declares as local dependencies, any category."""
    cls = component if isinstance(component, type) else type(component)
    return frozenset(
        name
        for category, declarations in _class_declarations(cls).items()
        for name, declaration in declarations.items()
        if declaration.local
        and name in getattr(cls, _CATEGORIES[category].attribute, ())
    )


def _class_declarations(cls: type) -> Mapping[type, Mapping[str, DependencyDeclaration]]:
    for klass in cls.__mro__:
        found = klass.__dict__.get(_DECLARATIONS_ATTR)
        if found is not None:
            return found
    return {}


def local_role_deprecation(name: str, consumers: tuple[str, ...]) -> str:
    """The `deprecated` text of a `role()` declaration replaced by local
    dependencies of `consumers`: one wording for every such role."""
    listed = ", ".join(consumers)
    return (
        f"Role '{name}' is deprecated and will be removed in the next "
        f"release: it is now a local dependency of {listed}, declared by the "
        f"component itself. Wire it in that component's own entry, e.g. "
        f"{consumers[0]}: {{{DEPENDENCY_BINDINGS_KEY}: {{{name}: ...}}}}."
    )
