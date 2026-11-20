"""naavik's directories and variables under their current names, and the
earlier ones (`training-framework/`, `TRAINING_FRAMEWORK_*`) still read:
an old directory is used where it is, an old variable when the new one is
not set; jobs started before the rename are still followed and
cancelled."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from naavik import paths


@pytest.fixture
def xdg(tmp_path, monkeypatch):
    for kind in ("CONFIG", "STATE", "DATA", "CACHE"):
        monkeypatch.setenv(f"XDG_{kind}_HOME", str(tmp_path / kind.lower()))
    monkeypatch.setattr(paths, "_warned", set())
    return tmp_path


def test_a_fresh_machine_uses_the_new_name(xdg):
    assert paths.app_dir("state") == xdg / "state" / "naavik"


def test_an_existing_earlier_directory_is_used_where_it_is(xdg):
    earlier = xdg / "data" / "training-framework"
    earlier.mkdir(parents=True)
    with pytest.warns(FutureWarning, match=f"using {earlier}, naavik's directory under its earlier "
                                           f"name; rename it to {xdg / 'data' / 'naavik'}"):
        assert paths.app_dir("data") == earlier


def test_with_both_the_new_one_is_used_and_the_warning_says_so(xdg):
    (xdg / "config" / "training-framework").mkdir(parents=True)
    (xdg / "config" / "naavik").mkdir(parents=True)
    with pytest.warns(FutureWarning, match="is used and .*training-framework ignored"):
        assert paths.app_dir("config") == xdg / "config" / "naavik"


@pytest.fixture
def variables(monkeypatch):
    for name in ("NAAVIK_X", "TRAINING_FRAMEWORK_X"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(paths, "_warned", set())
    return monkeypatch


def test_a_variable_by_either_name(variables):
    assert paths.env("X") is None and paths.variable_name("X") == "NAAVIK_X"
    variables.setenv("TRAINING_FRAMEWORK_X", "old")
    with pytest.warns(FutureWarning, match="TRAINING_FRAMEWORK_X is the earlier name of NAAVIK_X"):
        assert paths.env("X") == "old"
    assert paths.variable_name("X") == "TRAINING_FRAMEWORK_X"     # messages name the one set
    variables.setenv("NAAVIK_X", "old")
    assert paths.env("X") == "old"                                # the same value: fine
    variables.setenv("NAAVIK_X", "new")
    with pytest.raises(ValueError, match="NAAVIK_X='new' and TRAINING_FRAMEWORK_X='old' are both set"):
        paths.env("X")


def test_the_run_registry_and_the_code_check_read_either_name(xdg, monkeypatch):
    from naavik.launcher.services.evaluation import enabled, switch_name
    from naavik.runs.registry import registry_dir

    for name in ("NAAVIK_RUNS", "TRAINING_FRAMEWORK_RUNS", "NAAVIK_CODE_CHECK", "TRAINING_FRAMEWORK_CODE_CHECK"):
        monkeypatch.delenv(name, raising=False)
    assert registry_dir() == xdg / "data" / "naavik" / "runs"
    monkeypatch.setenv("NAAVIK_RUNS", "off")
    assert registry_dir() is None
    monkeypatch.delenv("NAAVIK_RUNS")
    monkeypatch.setenv("TRAINING_FRAMEWORK_RUNS", str(xdg / "elsewhere"))
    assert registry_dir() == xdg / "elsewhere"
    monkeypatch.setenv("NAAVIK_CODE_CHECK", "0")
    assert not enabled() and switch_name() == "NAAVIK_CODE_CHECK"


# The host's runner state ---------------------------------------------------------------------


def write_job(state, ref, elements):
    jobs = state / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / f"{ref}.json").write_text(json.dumps({"elements": elements, "final": False}))


def test_jobs_under_either_name_are_followed_and_cancelled(xdg):
    from naavik.launcher.remote import host_runner

    current, earlier = (__import__("pathlib").Path(path) for path in host_runner.state_dirs())
    assert current == xdg / "state" / "naavik" / "host" / host_runner.socket.gethostname()
    ref = host_runner.job_ref(os.getpid())                          # a live "runner"
    write_job(earlier, ref, {"0": "running", "1": "pending"})       # started before the rename
    assert host_runner.state_dir() == str(earlier)                 # only the earlier one exists
    assert host_runner.live_states() == {ref: "running", f"{ref}_0": "running", f"{ref}_1": "pending"}
    current.mkdir(parents=True)                                     # a runner of the new name
    assert host_runner.state_dir() == str(current)
    assert ref in host_runner.live_states()                         # the old job is still seen
    assert host_runner.request_cancel(ref, [1]) == "cancelling 1 element(s)"
    assert (earlier / "jobs" / f"{ref}.json.cancel").read_text() == "1"   # where its runner reads it


def test_the_registry_script_sent_to_a_host_reads_either_name(xdg):
    from naavik.runs.sync import _REMOTE_REGISTRY

    def run(**variables):
        environment = {key: value for key, value in os.environ.items()
                       if key not in ("NAAVIK_RUNS", "TRAINING_FRAMEWORK_RUNS")}
        environment.update(variables)
        return subprocess.run([sys.executable, "-c", _REMOTE_REGISTRY], capture_output=True,
                              text=True, env=environment, check=True).stdout.strip()

    assert run() == str(xdg / "data" / "naavik" / "runs")
    (xdg / "data" / "training-framework").mkdir(parents=True)
    assert run() == str(xdg / "data" / "training-framework" / "runs")
    assert run(TRAINING_FRAMEWORK_RUNS="/r") == "/r" and run(NAAVIK_RUNS="off") == ""
