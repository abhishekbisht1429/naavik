"""Explain why a component lookup failed.

Error sites keep their existing headline (e.g. ``unmet prerequisite! ...``)
and append the indented ``Reason:``/``Fix:`` block built here, so callers can
tell apart a typo, a component registered only for another session type, a
category mismatch, and a registered component that is not active in the
session.
"""

from collections.abc import Collection, Mapping
from difflib import get_close_matches

from naavik.components.base import Component
from naavik.components.config import DEPENDENCY_BINDINGS_KEY
from naavik.components.dependencies import dependency_declaration
from naavik.components.registry import (
    _ROLE_DECORATOR_NAMES,
    RoleDeclaration,
    ScopeFinding,
    _SESSION_COMPONENT_REGISTRIES,
    _SESSION_ROLE_REGISTRIES,
    _SHARED_COMPONENT_REGISTRY,
    _SHARED_ROLE_REGISTRY,
    _component_type,
    _normalize_component_session_type,
)
from naavik.components.optional import OPTIONAL_COMPONENTS


def _category_name(component_class: type[Component]) -> str:
    try:
        return _component_type(component_class).__name__
    except TypeError:
        return component_class.__name__


def _registry_label(session_type: str | None) -> str:
    if session_type is None:
        return "shared registry"
    return f"'{session_type}' registry"


def _searched_registries(session_type: str | None) -> str:
    if session_type is None:
        return "the shared registry"
    return f"the shared registry or the '{session_type}' registry"


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _decorator_call(
        category: type[Component] | None,
        name: str,
        session_type: str | None,
) -> str:
    scope = f", session_type='{session_type}'" if session_type else ""
    decorators = (
        [_ROLE_DECORATOR_NAMES[category]]
        if category in _ROLE_DECORATOR_NAMES
        else ["resource", "hook", "step"]
    )
    return " / ".join(f"@{decorator}('{name}'{scope})" for decorator in decorators)


def _consumer_label(consumer: Component | type[Component]) -> str:
    consumer_class = consumer if isinstance(consumer, type) else type(consumer)
    name = getattr(consumer, "name", None)
    category = _category_name(consumer_class)
    if name:
        return f"{category} '{name}' ({consumer_class.__name__})"
    return consumer_class.__name__


def _suggestions(
        name: str,
        expected_type: type[Component] | None,
        session_type: str | None,
) -> list[str]:
    candidates = dict(_SHARED_COMPONENT_REGISTRY)
    if session_type is not None:
        candidates.update(_SESSION_COMPONENT_REGISTRIES.get(session_type, {}))
    names = [
        candidate
        for candidate, component_class in candidates.items()
        if expected_type is None or issubclass(component_class, expected_type)
    ]
    return get_close_matches(name, names, n=3, cutoff=0.6)


def explain_missing_component(
        requested_name: str,
        resolved_name: str,
        *,
        expected_type: type[Component] | None = None,
        session_type: str | None = None,
        consumer: Component | type[Component] | None = None,
        active_names: Collection[str] | None = None,
) -> str:
    """Return an indented Reason/Fix block, or "" if the lookup would succeed.

    ``active_names`` is the set of component names configured in the session;
    pass it only when "registered but not active" is a possible failure.
    """
    session_type = _normalize_component_session_type(session_type)
    expected = expected_type.__name__ if expected_type is not None else "component"
    lines: list[str] = []
    if consumer is not None:
        lines.append(f"Required by: {_consumer_label(consumer)}")
    if requested_name != resolved_name:
        lines.append(
            f"Binding: role_bindings maps '{requested_name}' to "
            f"'{resolved_name}'."
        )

    scoped = (
        _SESSION_COMPONENT_REGISTRIES.get(session_type, {})
        if session_type is not None
        else {}
    )
    shared_class = _SHARED_COMPONENT_REGISTRY.get(resolved_name)
    if resolved_name in scoped:
        found_class = scoped[resolved_name]
        found_in = _registry_label(session_type)
    elif shared_class is not None:
        found_class = shared_class
        found_in = _registry_label(None)
    else:
        found_class = None
        found_in = ""

    if found_class is not None:
        found_category = _category_name(found_class)
        if expected_type is not None and not issubclass(found_class, expected_type):
            reason = (
                f"'{resolved_name}' is registered in the {found_in} as "
                f"{_article(found_category)} {found_category} "
                f"({found_class.__name__}), not as {_article(expected)} "
                f"{expected}."
            )
            if (
                    resolved_name in scoped
                    and shared_class is not None
                    and issubclass(shared_class, expected_type)
            ):
                reason += (
                    f" It overrides the shared {expected} "
                    f"'{resolved_name}' ({shared_class.__name__}) for "
                    f"'{session_type}' sessions."
                )
            fix = (
                f"If {_article(expected)} {expected} is really needed, register "
                "one under another name and bind it via role_bindings: "
                f"{{'{requested_name}': '<implementation_name>'}}, or rename "
                "the conflicting component."
            )
        elif active_names is not None and resolved_name not in active_names:
            reason = (
                f"'{resolved_name}' is registered in the {found_in} as "
                f"{_article(found_category)} {found_category} "
                f"({found_class.__name__}) but is not active in this session."
            )
            fix = (
                f"Add a top-level '{resolved_name}' mapping to the session "
                f"config ('{resolved_name}: {{}}' if it needs no "
                "configuration), or depend on it from an active component."
            )
        else:
            return ""
    else:
        other_scopes = sorted(
            scope
            for scope, registry in _SESSION_COMPONENT_REGISTRIES.items()
            if scope != session_type and resolved_name in registry
        )
        if other_scopes:
            if expected_type is None:
                # Suggest the decorator matching how it is registered elsewhere.
                expected_type = _component_type(
                    _SESSION_COMPONENT_REGISTRIES[other_scopes[0]][resolved_name]
                )
                expected = expected_type.__name__
            registered_as = ", ".join(
                f"'{scope}' (as "
                f"{_category_name(_SESSION_COMPONENT_REGISTRIES[scope][resolved_name])})"
                for scope in other_scopes
            )
            this_session = (
                f"'{session_type}' sessions"
                if session_type is not None
                else "the shared registry"
            )
            reason = (
                f"'{resolved_name}' is not in {_searched_registries(session_type)}; "
                f"it is registered only for session type(s) {registered_as}, "
                f"so it is unavailable to {this_session}."
            )
            scoped_option = (
                f"register {_article(expected)} {expected} for this session "
                f"type with {_decorator_call(expected_type, resolved_name, session_type)}, "
                if session_type is not None
                else ""
            )
            fix = (
                "Use it from a "
                + " or ".join(f"'{scope}'" for scope in other_scopes)
                + f" session, {scoped_option}or register it as shared with "
                f"{_decorator_call(expected_type, resolved_name, None)}."
            )
        else:
            scoped_roles = (
                _SESSION_ROLE_REGISTRIES.get(session_type, {})
                if session_type is not None
                else {}
            )
            if resolved_name in scoped_roles:
                role, role_in = (
                    scoped_roles[resolved_name],
                    _registry_label(session_type),
                )
            elif resolved_name in _SHARED_ROLE_REGISTRY:
                role, role_in = (
                    _SHARED_ROLE_REGISTRY[resolved_name],
                    _registry_label(None),
                )
            else:
                role, role_in = None, ""
            role_scopes = sorted(
                scope
                for scope, roles in _SESSION_ROLE_REGISTRIES.items()
                if scope != session_type and resolved_name in roles
            )

            if role is not None:
                reason = (
                    f"'{resolved_name}' is declared as {_article(role.category.__name__)} "
                    f"{role.category.__name__} role in the {role_in}, but no "
                    "implementation is registered in "
                    f"{_searched_registries(session_type)}."
                )
            elif role_scopes:
                reason = (
                    f"'{resolved_name}' is declared as a role only for session "
                    f"type(s) {', '.join(repr(s) for s in role_scopes)}, and no "
                    "implementation is registered in "
                    f"{_searched_registries(session_type)}."
                )
            else:
                reason = (
                    f"No component named '{resolved_name}' is registered in "
                    f"{_searched_registries(session_type)}, or for any other "
                    "session type."
                )
            optional = OPTIONAL_COMPONENTS.get(resolved_name)
            if optional is not None and role is None and not role_scopes:
                lines.append(
                    f"Reason: '{resolved_name}' is an optional built-in; it "
                    f"is registered only once {optional.module} is imported."
                )
                lines.append(
                    f"Fix: Add 'import {optional.module}' to a module of "
                    "your session_config.components_package (or set "
                    f"components_package to {optional.module}). It needs "
                    f"`pip install training-framework[{optional.extra}]`."
                )
                return "\n".join(f"  {line}" for line in lines)
            scoped_hint = (
                f" (add session_type='{session_type}' to limit it to "
                f"'{session_type}' sessions)"
                if session_type is not None
                else ""
            )
            fix = (
                f"Register it with {_decorator_call(expected_type, resolved_name, None)}"
                f"{scoped_hint} and make sure its module is imported (for "
                "example through session_config.components_package)"
            )
            if role is not None or role_scopes:
                fix += (
                    ", or bind an existing implementation via "
                    f"role_bindings: {{'{requested_name}': "
                    "'<implementation_name>'}"
                )
            fix += "."
            matches = _suggestions(resolved_name, expected_type, session_type)
            if matches:
                fix += " Did you mean: " + ", ".join(
                    f"'{match}'" for match in matches
                ) + "?"

    lines.append(f"Reason: {reason}")
    lines.append(f"Fix: {fix}")
    return "\n".join(f"  {line}" for line in lines)


def with_explanation(headline: str, explanation: str) -> str:
    return f"{headline}\n{explanation}" if explanation else headline


def missing_dependency_message(
        *,
        category: type[Component],
        name: str,
        resolved_name: str,
        roles: Mapping[str, RoleDeclaration],
        consumer: Component | type[Component] | None = None,
) -> str | None:
    """The message for a dependency nothing implements, when something says
    what it should be: the consumer's own declaration of it (its
    `description`, or that it is local), else a declared `category` role of
    that name. None when neither does; the caller then reports an unmet
    prerequisite.
    """
    consumer_class = (
        None if consumer is None
        else consumer if isinstance(consumer, type) else type(consumer)
    )
    declared = (
        dependency_declaration(consumer_class, name, category)
        if consumer_class is not None else None
    )
    role = roles.get(resolved_name)
    if role is not None and role.category is not category:
        role = None
    described = declared is not None and (
        declared.local or declared.description is not None
    )
    if not described and role is None:
        return None

    description = (
        declared.description if described and declared.description is not None
        else role.description if role is not None
        else None
    )
    what = f"{category.__name__}: {description}" if description else category.__name__
    decorator = _ROLE_DECORATOR_NAMES[category]
    implement = (
        f"Implement a {category.__name__} subclass and register it via "
        f"@{decorator}('{resolved_name}', ...)"
    )
    if described and declared.local:
        owner = getattr(consumer_class, "name", None) or "<component>"
        return (
            f"Local dependency '{resolved_name}' ({what}) of "
            f"{consumer_class.__name__} has no implementation registered. "
            f"{implement}, or wire an existing implementation in the "
            f"component's own entry: {owner}: {{{DEPENDENCY_BINDINGS_KEY}: "
            f"{{{name}: <implementation_name>}}}}."
        )
    consumer_clause = (
        f" by {consumer_class.__name__}" if consumer_class is not None else ""
    )
    return (
        f"Role '{resolved_name}' ({what}) is required{consumer_clause} but has "
        f"no implementation registered. {implement}, or bind an existing "
        "implementation via role_bindings: "
        f"{{'{name}': '<implementation_name>'}}."
    )


def explain_scope_finding(
        finding: ScopeFinding,
        origin: str,
        *,
        from_import: bool = False,
) -> str:
    """The deprecation of a session-wide binding that reaches local
    dependencies, with the wiring that replaces it. `origin` says where the
    binding was written: the top-level bindings, or (`from_import`) an
    import's `role`, whose instance needs a name of its own to be wired."""
    one = len(finding.consumers) == 1
    fixes = "\n".join(
        f"    {consumer}: {{{DEPENDENCY_BINDINGS_KEY}: "
        f"{{{finding.role}: {finding.target}}}}}"
        for consumer in finding.consumers
    )
    return (
        f"'{finding.role}' is bound for the whole session ({origin}), but "
        f"{', '.join(finding.consumers)} take{'s' if one else ''} it as "
        f"{'a local dependency' if one else 'local dependencies'}. A "
        "session-wide binding of a local dependency is deprecated and still "
        "applies for now; wire it in the component's own entry instead:\n"
        f"{fixes}"
        + (
            "\n  The import's instance is reached that way only by a name of "
            "its own: drop the import's `role`, key the import by the "
            "source's instance name (its instances then get the suffix "
            "`imported`, or the import's `instance_name`), and name that "
            "instance above."
            if from_import else ""
        )
    )
