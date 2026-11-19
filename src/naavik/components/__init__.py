"""Component base classes, registration decorators, and built-in components."""

from naavik.components.base import (
    Component,
    ComponentDependencyError,
    ExtendableComponent,
    Hook,
    IterationHook,
    LifecycleHook,
    Resource,
    SessionHook,
    Stateful,
    StatefulIterationHook,
    StatefulLifecycleHook,
    StatefulLifeCycleHook,
    StatefulResource,
    StatefulSessionHook,
    StatefulStep,
    Step,
)
from naavik.components.config_schema import parse_component_config
from naavik.components.dependencies import (
    DependencyDeclaration,
    dependency_declaration,
    local_dependencies,
)
from naavik.components.epochs import EpochSource
from naavik.components.module import ModuleResource
from naavik.components.registry import (
    ANALYSIS_SESSION_TYPE,
    TRAINING_SESSION_TYPE,
    ComponentAliases,
    RoleBindings,
    RoleDeclaration,
    RoleRegistry,
    activates,
    component_registry,
    format_execution_graph,
    hook,
    rank_zero_only,
    reads,
    singleton,
    requires_hook,
    requires_resource,
    requires_step,
    resource,
    role,
    role_registry,
    step,
    topological_sort_of_components,
    wraps,
    writes,
)

__all__ = [
    "ANALYSIS_SESSION_TYPE",
    "TRAINING_SESSION_TYPE",
    "Component",
    "ComponentAliases",
    "ComponentDependencyError",
    "DependencyDeclaration",
    "EpochSource",
    "ExtendableComponent",
    "Hook",
    "IterationHook",
    "LifecycleHook",
    "ModuleResource",
    "parse_component_config",
    "Resource",
    "RoleBindings",
    "RoleDeclaration",
    "RoleRegistry",
    "SessionHook",
    "Stateful",
    "StatefulIterationHook",
    "StatefulLifecycleHook",
    "StatefulLifeCycleHook",
    "StatefulResource",
    "StatefulSessionHook",
    "StatefulStep",
    "Step",
    "activates",
    "component_registry",
    "dependency_declaration",
    "format_execution_graph",
    "hook",
    "local_dependencies",
    "rank_zero_only",
    "reads",
    "singleton",
    "requires_hook",
    "requires_resource",
    "requires_step",
    "resource",
    "role",
    "role_registry",
    "step",
    "topological_sort_of_components",
    "wraps",
    "writes",
]

from naavik.components import builtin as builtin


def __getattr__(name):
    if name == "ComponentBindings":
        import warnings

        warnings.warn(
            "ComponentBindings is deprecated; use RoleBindings",
            DeprecationWarning,
            stacklevel=2,
        )
        return RoleBindings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
