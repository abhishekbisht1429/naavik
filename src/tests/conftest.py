import os

import pytest

from naavik.components.registry import (
    _SESSION_COMPONENT_REGISTRIES,
    _SESSION_ROLE_REGISTRIES,
    _SHARED_COMPONENT_REGISTRY,
    _SHARED_ROLE_REGISTRY,
)
from naavik.session.registry import _SESSION_TYPE_REGISTRY


_DEFAULT_SHARED_COMPONENTS = dict(_SHARED_COMPONENT_REGISTRY)
_DEFAULT_SESSION_COMPONENTS = {
    session_type: dict(registry)
    for session_type, registry in _SESSION_COMPONENT_REGISTRIES.items()
}
_DEFAULT_SHARED_ROLES = dict(_SHARED_ROLE_REGISTRY)
_DEFAULT_SESSION_ROLES = {
    session_type: dict(registry)
    for session_type, registry in _SESSION_ROLE_REGISTRIES.items()
}
_DEFAULT_SESSION_TYPES = dict(_SESSION_TYPE_REGISTRY)


@pytest.fixture(autouse=True)
def reset_registries():
    _SHARED_COMPONENT_REGISTRY.clear()
    _SHARED_COMPONENT_REGISTRY.update(_DEFAULT_SHARED_COMPONENTS)

    for session_type in tuple(_SESSION_COMPONENT_REGISTRIES):
        if session_type not in _DEFAULT_SESSION_COMPONENTS:
            del _SESSION_COMPONENT_REGISTRIES[session_type]
    for session_type, defaults in _DEFAULT_SESSION_COMPONENTS.items():
        registry = _SESSION_COMPONENT_REGISTRIES.setdefault(session_type, {})
        registry.clear()
        registry.update(defaults)

    _SHARED_ROLE_REGISTRY.clear()
    _SHARED_ROLE_REGISTRY.update(_DEFAULT_SHARED_ROLES)

    for session_type in tuple(_SESSION_ROLE_REGISTRIES):
        if session_type not in _DEFAULT_SESSION_ROLES:
            del _SESSION_ROLE_REGISTRIES[session_type]
    for session_type, defaults in _DEFAULT_SESSION_ROLES.items():
        registry = _SESSION_ROLE_REGISTRIES.setdefault(session_type, {})
        registry.clear()
        registry.update(defaults)

    _SESSION_TYPE_REGISTRY.clear()
    _SESSION_TYPE_REGISTRY.update(_DEFAULT_SESSION_TYPES)


@pytest.fixture
def minimal_session_config_1(tmp_path):
    return {
        "session_config": {
            "max_iterations": 3,
            "sessions_dir": str(tmp_path / "sessions"),
            "device": "cpu",
            "rng_seed": 7,
            "components_package": "naavik.components.builtin",
        }
    }


@pytest.fixture
def minimal_session_config_2(tmp_path):
    return {
        "session_config": {
            "max_iterations": 2,
            "sessions_dir": str(tmp_path / "sessions"),
            "device": "cpu",
            "rng_seed": 11,
            "components_package": "naavik.components.builtin",
        }
    }


@pytest.fixture(autouse=True, scope="session")
def run_registry_out_of_the_users_home(tmp_path_factory):
    """Engine launches in tests are recorded in a temporary registry, never
    in the user's, and make no snapshot commits: test launches find their
    components inside this repository, whose working tree may hold the
    user's uncommitted work. Batches are not code-checked: the batch layer's
    tests use placeholder configs. Tests about any of these set their own
    values."""
    saved = {name: os.environ.get(name)
             for name in ("TRAINING_FRAMEWORK_RUNS", "TRAINING_FRAMEWORK_RUNS_SNAPSHOT",
                          "TRAINING_FRAMEWORK_CODE_CHECK")}
    os.environ["TRAINING_FRAMEWORK_RUNS"] = str(tmp_path_factory.mktemp("run-registry"))
    os.environ["TRAINING_FRAMEWORK_RUNS_SNAPSHOT"] = "0"
    os.environ["TRAINING_FRAMEWORK_CODE_CHECK"] = "0"
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
