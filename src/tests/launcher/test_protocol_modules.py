"""Execution protocol 2 runs the framework as naavik (`naavik.launcher.task`,
`naavik.data`); a mission keeps running the modules its own plan's
protocol names, so missions launched before the rename (protocol 1) resume
with `training_framework.hpc.task`."""

from __future__ import annotations

import ast
import inspect
import json

import pytest

from naavik.launcher import DATA_MODULES, EXECUTION_PROTOCOL_VERSION, TASK_MODULES, task_module
from naavik.launcher.errors import HpcError
from tests.hpc.batch_repo import compile_repo, make_repo


def test_a_new_plan_records_protocol_2_and_runs_the_new_modules(tmp_path):
    from naavik.launcher.compile.render import plan_files

    compiled = compile_repo(make_repo(tmp_path))
    files = plan_files(compiled.plan, compiled.manifest.raw)
    assert EXECUTION_PROTOCOL_VERSION == 2 and json.loads(files["plan.json"])["protocol_version"] == 2
    script = next(text for name, text in files.items() if name.endswith("group-probes.sbatch"))
    assert "-m naavik.launcher.task" in script and "training_framework" not in script


@pytest.mark.parametrize("version, module", [(1, "training_framework.hpc.task"), (2, "naavik.launcher.task")])
def test_a_resume_runs_the_module_its_plans_protocol_names(version, module):
    from tests.hpc.services.test_resume import DIALECT, chosen_tasks, plan_json
    from naavik.launcher.services import resume as resuming

    plan = dict(plan_json("/b"), protocol_version=version)
    attempt = resuming.build_attempt(plan, {"attempts": []}, chosen_tasks(), DIALECT,
                                     resuming.ResourceChanges(walltime="2-00:00:00"))
    (group,) = attempt.groups
    assert f"-m {module}" in group.script


def test_an_unknown_protocol_is_named():
    with pytest.raises(HpcError, match="execution protocol 9, which this naavik does not know"):
        task_module(9)


def test_this_framework_runs_plans_of_both_protocols():
    from naavik.launcher.remote.task import SUPPORTED_PROTOCOL_VERSIONS

    assert SUPPORTED_PROTOCOL_VERSIONS == (1, 2) and DATA_MODULES[2] == "naavik.data"


def test_the_bootstraps_own_table_matches_the_launchers():
    # The bootstrap is standard-library Python bundled for the host; it
    # cannot import TASK_MODULES, so it carries a copy, which must agree.
    from naavik.launcher.remote import bootstrap_main

    tree = ast.parse(inspect.getsource(bootstrap_main.run_preflight))
    (table,) = [node for node in ast.walk(tree) if isinstance(node, ast.Dict)]
    assert ast.literal_eval(table) == TASK_MODULES
