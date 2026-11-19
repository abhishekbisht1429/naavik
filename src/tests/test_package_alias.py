"""`training_framework`, the earlier name of `naavik`: every old module path
is the same module object as its new one, so imports, configs'
`components_package`, legacy pickles, and the `python -m
training_framework...` commands earlier missions recorded keep working."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import naavik

SRC = str(Path(naavik.__file__).resolve().parents[1])


def run(*args, code=None):
    env = {**os.environ, "PYTHONPATH": SRC}
    command = [sys.executable, *(["-c", code] if code else args)]
    return subprocess.run(command, capture_output=True, text=True, env=env, check=False)


def test_the_old_name_is_the_new_package_and_its_modules():
    import training_framework
    import training_framework.session.training as old
    import naavik.session.training as new

    assert training_framework is naavik and old is new


def test_a_change_through_one_name_is_seen_through_the_other(monkeypatch):
    import training_framework.launcher.compile.compiler as old
    import naavik.launcher.compile.compiler as new

    monkeypatch.setattr(old, "FRAMEWORK_DISTRIBUTION", "patched")
    assert new.FRAMEWORK_DISTRIBUTION == "patched"


def test_the_earlier_hpc_paths_still_chain_through():
    import training_framework.hpc.compile.compiler as oldest
    import naavik.launcher.compile.compiler as new

    assert oldest is new


def test_one_warning_names_the_new_name():
    result = run(code="import warnings; warnings.simplefilter('always'); "
                      "import training_framework, training_framework.session, training_framework.runs")
    warnings = [line for line in result.stderr.splitlines() if "FutureWarning" in line]
    assert result.returncode == 0 and len(warnings) == 1
    assert "training_framework is now called naavik: import naavik instead" in warnings[0]


def test_a_legacy_pickle_of_old_module_paths_loads():
    from naavik.components.builtin.observability import Timer

    data = pickle.dumps(Timer({"call_every": 3}), protocol=2)
    old = data.replace(b"naavik.components", b"training_framework.components")
    assert old != data                                # the pickle names the old module
    restored = pickle.loads(old)
    assert isinstance(restored, Timer) and restored.call_every == 3


def test_a_config_naming_the_old_package_builds(tmp_path):
    from naavik.session import check_session_config

    config = {"session_type": "analysis",
              "session_config": {"rng_seed": 0, "sessions_dir": str(tmp_path), "max_iterations": 1,
                                 "components_package": "training_framework.components.builtin",
                                 "device": "cpu"},
              "trained_model": {"model_checkpoint_path": "/elsewhere/ckpt"}}
    assert check_session_config(config).problems == []


def test_the_recorded_run_commands_still_run():
    # Job scripts and plans of earlier missions run these.
    task = run("-m", "training_framework.hpc.task", "--help")
    assert task.returncode == 0 and "usage:" in task.stdout
    fetch = run("-m", "training_framework.data", "--help")     # a package: its __main__
    assert fetch.returncode == 0 and "--component" in fetch.stdout


def test_a_recorded_command_runs_with_the_modules_file_as_argv0():
    # `python -m` takes `sys.argv[0]` from the spec's origin; without it the
    # engine's argument parser fails in every job.
    import importlib.util

    origin = importlib.util.find_spec("training_framework.hpc.task").origin
    assert origin == str(Path(naavik.__file__).parent / "hpc" / "task.py")
    shown = run(code="import runpy, sys; sys.argv = ['x']; "
                     "sys.modules['__main__'].__dict__; "
                     "import training_framework; "
                     "spec = __import__('importlib.util').util.find_spec('training_framework.hpc.task'); "
                     "print(spec.origin)")
    assert shown.stdout.strip().endswith("naavik/hpc/task.py")


def test_a_recorded_command_does_not_warn_in_every_job_log():
    result = run("-m", "training_framework.hpc.task", "--help")
    assert result.returncode == 0 and "FutureWarning" not in result.stderr
