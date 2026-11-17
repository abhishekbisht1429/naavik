"""Check a session config without building the session.

`check_session_config` finds what building the session would refuse before
any component is constructed -- the session settings, the config layout,
every component name, the wiring, companions, singletons, and each planned
component's `config_schema` -- and says what it cannot check without
building: the order of the components' work (it needs them built),
configs a component reads itself in its constructor, imports (they restore
from checkpoints), and the device (it depends on the machine that runs).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

#: What a check never covers, said once with the result.
NOT_CHECKED = (
    "the order of the components' work (it needs them built)",
    "configs a component reads in its own constructor (no config_schema)",
    "the device (it depends on the machine that runs the session)",
)


@dataclass
class ConfigCheck:
    problems: list[str] = field(default_factory=list)
    #: Parts of this config that could not be checked, and why.
    notes: list[str] = field(default_factory=list)
    #: The components building the session would construct
    #: (`PlannedComponent`), when planning got that far.
    components: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _message(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}" if not str(error) else str(error)


def check_session_config(config: Mapping, session_type: str | None = None) -> ConfigCheck:
    """Check one session's config (an entry of a config file's `sessions`).
    `session_type` defaults to the entry's own, else training. Imports the
    session's components package, as building it would."""
    from training_framework.components.config_schema import parse_component_config
    from training_framework.session.components import (
        ImportsNotPlanned,
        SessionComponents,
        read_session_config,
    )
    from training_framework.session.config import (
        TRAINING_SESSION_TYPE,
        normalize_session_config,
        normalize_session_type,
    )
    from training_framework.session.registry import session_class_for_type
    from training_framework.util import import_all_modules

    result = ConfigCheck()
    if not isinstance(config, Mapping) or not isinstance(config.get("session_config"), Mapping):
        result.problems.append("a session config must be a mapping with a 'session_config' mapping")
        return result
    try:
        kind = normalize_session_type(session_type or config.get("session_type") or TRAINING_SESSION_TYPE)
        session_class = session_class_for_type(kind)
        settings = normalize_session_config(config["session_config"])
        import_all_modules(settings["components_package"])
        body = {key: value for key, value in config.items()
                if key not in ("session_type", "session_kwargs")}
        view = read_session_config(body, session_type=kind)
        components = SessionComponents(role_bindings=view.role_bindings,
                                       dependency_bindings=view.dependency_bindings,
                                       session_type=kind)
        plan = components.plan_from_config(
            body, default_configs=session_class._default_component_configs(), view=view)
    except ImportsNotPlanned as reason:
        result.notes.append(f"not checked: {reason}")
        return result
    except Exception as error:      # what building would raise, first problem first
        result.problems.append(_message(error))
        return result
    result.components = list(plan.components)
    for planned in plan.components:
        if planned.component_class.config_schema is None or planned.config is None:
            continue
        try:
            parse_component_config(planned.component_class, planned.config)
        except (TypeError, ValueError) as error:
            result.problems.append(f"{planned.name}: {_message(error)}")
    return result
