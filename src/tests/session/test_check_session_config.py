"""`check_session_config` and `SessionComponents.plan_from_config`: what
building a session would refuse, found without constructing anything."""

from __future__ import annotations

import importlib
import sys

import pytest

from tests.session import check_components
from naavik.session import (
    SessionComponents,
    TrainingSession,
    check_session_config,
)
from naavik.session.components import read_session_config

PACKAGE = "tests.session.check_components"


@pytest.fixture(autouse=True)
def components():
    """Registered again for each test (the suite resets registrations)."""
    importlib.reload(sys.modules.get(PACKAGE) or importlib.import_module(PACKAGE))
    check_components.CONSTRUCTED.clear()
    yield
    check_components.CONSTRUCTED.clear()


def config(tmp_path, **components):
    data = {"session_config": {"rng_seed": 1, "sessions_dir": str(tmp_path),
                               "max_iterations": 1, "components_package": PACKAGE,
                               "show_execution_graph": False}}
    data.update(components)
    return data


def test_a_sound_config_passes_without_constructing_anything(tmp_path):
    result = check_session_config(config(tmp_path, needs_counted={}, counted={"size": 2}))
    assert result.ok and result.problems == [] and result.notes == []
    assert check_components.CONSTRUCTED == []


def test_the_plan_lists_what_building_would_construct_in_that_order(tmp_path):
    data = config(tmp_path, needs_counted={}, counted={"size": 2})
    view = read_session_config(data, session_type="training")
    components = SessionComponents(role_bindings=view.role_bindings,
                                   dependency_bindings=view.dependency_bindings,
                                   session_type="training")
    plan = components.plan_from_config(data, view=view)
    assert check_components.CONSTRUCTED == []
    planned = {item.name: item for item in plan.components}
    assert planned["counted"].config == {"size": 2} and planned["needs_counted"].config == {}
    assert plan.names.index("counted") < plan.names.index("needs_counted")   # prerequisite first
    TrainingSession(data)                                     # building constructs in that order
    built = [name for name in check_components.CONSTRUCTED]
    assert built == [name for name in plan.names if name in ("counted", "needs_counted")]


def test_a_misspelt_component_is_named(tmp_path):
    (problem,) = check_session_config(config(tmp_path, countd={})).problems
    assert "'countd'" in problem and "registered" in problem


def test_a_missing_prerequisite_is_found(tmp_path):
    data = config(tmp_path, needs_counted={"dependency_bindings": {"counted": "counted#other"}})
    (problem,) = check_session_config(data).problems
    assert "counted#other" in problem
    assert check_components.CONSTRUCTED == []


def test_a_config_schema_value_is_checked_and_named(tmp_path):
    (problem,) = check_session_config(config(tmp_path, counted={"size": 0})).problems
    assert problem.startswith("counted: ") and "size must be a positive integer" in problem
    assert check_components.CONSTRUCTED == []


def test_an_unknown_schema_key_is_refused(tmp_path):
    (problem,) = check_session_config(config(tmp_path, counted={"sise": 2})).problems
    assert problem.startswith("counted: ") and "sise" in problem


def test_bad_session_settings_are_found(tmp_path):
    data = config(tmp_path, counted={})
    del data["session_config"]["components_package"]
    (problem,) = check_session_config(data).problems
    assert "components_package" in problem


def test_imports_are_reported_as_not_checked(tmp_path):
    data = config(tmp_path, counted={},
                  import_components={"model": {"checkpoint": str(tmp_path / "ckpt")}})
    result = check_session_config(data)
    assert result.ok and len(result.notes) == 1 and "import_components" in result.notes[0]


def test_what_is_not_a_session_config_is_said_so():
    assert check_session_config("x").problems == [
        "a session config must be a mapping with a 'session_config' mapping"]


def test_a_fresh_process_imports_the_components_package_to_check(tmp_path):
    # As the batch evaluator runs it: nothing of the project imported yet.
    import json
    import os
    import subprocess
    from pathlib import Path

    src = Path(__file__).resolve().parents[2]
    program = ("import json, sys; from naavik.session import check_session_config; "
               "print(json.dumps(check_session_config(json.loads(sys.argv[1])).problems))")
    for data, expected in ((config(tmp_path, counted={"size": 2}), []),
                           (config(tmp_path, counted={"size": 0}), 1)):
        result = subprocess.run([sys.executable, "-c", program, json.dumps(data)],
                                capture_output=True, text=True, check=False,
                                env={**os.environ, "PYTHONPATH": str(src)})
        assert result.returncode == 0, result.stderr
        problems = json.loads(result.stdout.strip().splitlines()[-1])
        assert problems == expected if expected == [] else len(problems) == expected
