"""`naavik run CONFIG`: one config as a one-task batch built from the project
file, launched like any batch and followed to its end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tests.hpc.batch_repo import make_repo
from tests.hpc.test_end_to_end import COMPONENTS, environment, session  # noqa: F401 (fixture)
from tests.launcher.test_host_end_to_end import here  # noqa: F401 (fixture)
from training_framework.cli import main as naavik
from training_framework.launcher.config.schema import PROJECT_FILE, RunDefaults
from training_framework.launcher.errors import HpcError
from training_framework.launcher.services.follow import DETACHED, Follower
from training_framework.launcher.services.quick_run import parse_override, run_manifest
from training_framework.launcher.services.status import Row

RESOURCES = {"walltime": "00:10:00", "cpus": 2, "memory": "4G"}


@pytest.fixture(autouse=True)
def code_check_on(monkeypatch):
    monkeypatch.setenv("TRAINING_FRAMEWORK_CODE_CHECK", "1")


def project(tmp_path, run_defaults=None, with_project_file=True):
    files = {"my_project/components.py": COMPONENTS.read_text(),
             "configs/model/base.yaml": yaml.safe_dump({"sessions": [session()]})}
    shared = {"code": {"framework": "dependency", "paths": ["."]},
              "run_defaults": run_defaults or {"resources": RESOURCES}}
    if with_project_file:
        files[PROJECT_FILE] = yaml.safe_dump(shared)
    return make_repo(tmp_path, extra_files=files)


def run(root, targets, *extra):
    return naavik(["run", str(root / "configs" / "model" / "base.yaml"),
                   "--targets", str(targets), *extra])


def batch_dir_of(output: str) -> Path:
    return Path(output.split("submitted: ")[1].splitlines()[0])


# The manifest it builds -----------------------------------------------------------------------


@pytest.mark.parametrize("text, item", [
    ("session_config.max_iterations=5", {"path": "session_config.max_iterations", "value": 5}),
    ("optimizer:optimizer.kwargs.lr=3.0e-4",
     {"component": "optimizer", "path": "optimizer.kwargs.lr", "value": 3e-4}),
    # Read as config files are (YAML 1.1): a float needs its dot.
    ("optimizer:optimizer.kwargs.lr=3e-4",
     {"component": "optimizer", "path": "optimizer.kwargs.lr", "value": "3e-4"}),
    ("a.b=[1, 2]", {"path": "a.b", "value": [1, 2]}),
    ("a.b=", {"path": "a.b", "value": None}),
])
def test_override_forms(text, item):
    assert parse_override(text) == item


@pytest.mark.parametrize("text", ["no-equals", "=5", ":path=1", "model:=1", "a=[unclosed"])
def test_a_malformed_override_is_refused(text):
    with pytest.raises(HpcError, match="--override"):
        parse_override(text)


def test_the_options_override_run_defaults():
    defaults = RunDefaults({"gpus": 1, "walltime": "04:00:00", "memory": "8G"}, 2, None)
    manifest = run_manifest(config="configs/vit attn.yaml", project=PROJECT_FILE,
                            run_defaults=defaults, overrides=[], note="why", gpus=0,
                            walltime="00:05:00")
    group = manifest["groups"]["run"]
    assert group["resources"] == {"gpus": 0, "walltime": "00:05:00", "memory": "8G"}
    assert group["max_parallel"] == 2
    assert group["experiments"] == [{"name": "vit-attn", "overrides": []}]
    assert manifest["description"] == "why" and manifest["project"] == PROJECT_FILE


# Following ------------------------------------------------------------------------------------


class Script:
    """A task that is live for `live_polls` polls, then ends in `final`; its
    log grows by one line per poll."""

    def __init__(self, final, live_polls=2, interrupts=()):
        self.final, self.live_polls, self.interrupts = final, live_polls, list(interrupts)
        self.polls, self.cancels, self.out, self.said = 0, 0, b"", []
        self.log = b""

    def rows(self):
        self.polls += 1
        self.log += f"line {self.polls}\n".encode()
        live = self.polls <= self.live_polls and not self.cancels
        return [Row("setup", "", "done", ""),
                Row("run/x/000", "", "running" if live else self.final, "", live)]

    def sleep(self, _):
        if self.interrupts and self.interrupts.pop(0):
            raise KeyboardInterrupt

    def follower(self):
        return Follower(task="run/x/000", rows=self.rows, log=lambda: "log",
                        read=lambda path, offset: self.log[offset:],
                        cancel=self.cancel, write=self.write, say=self.said.append,
                        interval=0, sleep=self.sleep)

    def cancel(self):
        self.cancels += 1

    def write(self, chunk):
        self.out += chunk


@pytest.mark.parametrize("final, status", [("done", 0), ("failed (exit 3)", 3),
                                           ("failed (exit -9)", 1), ("interrupted", 1),
                                           ("ended without a record", 1)])
def test_a_followed_task_gives_its_exit_status_and_whole_log(final, status):
    script = Script(final)
    outcome = script.follower().run()
    assert (outcome.status, outcome.state) == (status, final)
    assert script.out == script.log                       # every byte, once, in order


def test_the_first_ctrl_c_cancels_and_keeps_following():
    script = Script("cancelled", live_polls=10, interrupts=[True])
    outcome = script.follower().run()
    assert script.cancels == 1 and outcome.cancelled and outcome.state == "cancelled"
    assert script.said == ["cancelling the task (Ctrl-C again to stop following)"]


def test_the_second_ctrl_c_detaches():
    script = Script("cancelled", live_polls=10, interrupts=[True, True])
    script.cancel = lambda: None                           # the task does not stop
    outcome = script.follower().run()
    assert outcome.status == DETACHED and outcome.state is None
    assert script.said[-1] == "detached; the task was asked to stop"


def test_an_unknown_task_is_an_error():
    follower = Follower(task="nope", rows=lambda: [], log=lambda: None, read=None,
                        cancel=None, write=None, say=None)
    with pytest.raises(HpcError, match="'nope' is not a task"):
        follower.run()


# End to end on this machine -------------------------------------------------------------------


def test_a_run_is_followed_to_its_end(tmp_path, here, capsys):
    root = project(tmp_path)
    assert run(root, here, "--yes", "--note", "baseline",
               "--override", "session_config.max_iterations=2") == 0
    out = capsys.readouterr().out
    assert "following run/base/000" in out
    assert "finished: exit 0 (ok)" in out                 # the task's log, followed
    assert out.rstrip().endswith("run/base/000: done")
    batch_dir = batch_dir_of(out)
    receipt = json.loads((batch_dir / "receipt.json").read_text())
    assert receipt["description"] == "baseline" and receipt["manifest"] == \
        "naavik run configs/model/base.yaml"
    config = yaml.safe_load((batch_dir / "tasks" / "run" / "base" / "000" / "config.yaml").read_text())
    assert config["sessions"][0]["session_config"]["max_iterations"] == 2


def test_the_note_reaches_the_runs_record(tmp_path, here, capsys):
    root = project(tmp_path)
    assert run(root, here, "--yes", "--note", "attention pooling") == 0
    capsys.readouterr()
    assert naavik(["runs", "list"]) == 0
    run_id = capsys.readouterr().out.splitlines()[1].split()[0]
    assert naavik(["runs", "show", run_id]) == 0
    assert "note     attention pooling" in capsys.readouterr().out


def test_a_failing_run_exits_with_its_status(tmp_path, here, capsys):
    # The optimizer's config is read only when it is built: the check passes,
    # the task fails.
    root = project(tmp_path)
    status = run(root, here, "--yes", "--override", "optimizer:optimizer.name=NoSuchOptimizer")
    out = capsys.readouterr().out
    assert status == 1 and "run/base/000: failed (exit 1)" in out


def test_detach_returns_once_launched(tmp_path, here, capsys):
    root = project(tmp_path)
    assert run(root, here, "--yes", "--detach") == 0
    out = capsys.readouterr().out
    assert "following" not in out and "follow: naavik logs " in out and "--task run/base/000" in out


def test_run_defaults_choose_the_target(tmp_path, here, capsys):
    targets = yaml.safe_load(here.read_text())
    targets["profiles"]["elsewhere"] = targets["profiles"]["cluster"]
    targets["default"] = "nowhere"                        # the file's default would fail
    here.write_text(yaml.safe_dump(targets))
    root = project(tmp_path, run_defaults={"resources": RESOURCES, "target": "elsewhere"})
    assert run(root, here, "--yes", "--detach") == 0


def test_without_a_project_file_it_is_named(tmp_path, here, capsys):
    root = project(tmp_path, with_project_file=False)
    assert run(root, here, "--yes") == 2
    assert f"naavik run needs a project file: {root.resolve() / PROJECT_FILE}" in \
        capsys.readouterr().err


def test_a_config_outside_the_repository_is_refused(tmp_path, here, capsys):
    root = project(tmp_path / "repo")
    outside = tmp_path / "elsewhere.yaml"
    outside.write_text("sessions: []\n")
    assert root.exists()
    assert naavik(["run", str(outside), "--targets", str(here), "--yes"]) == 2
    assert f"{tmp_path} is not inside a Git repository" in capsys.readouterr().err


def test_a_missing_config_is_named(tmp_path, here, capsys):
    root = project(tmp_path)
    assert naavik(["run", str(root / "configs" / "nope.yaml"), "--targets", str(here)]) == 2
    assert "nope.yaml does not exist" in capsys.readouterr().err


def test_the_code_check_refuses_a_bad_config_before_launch(tmp_path, here, capsys):
    root = project(tmp_path)
    assert run(root, here, "--yes", "--override", "role_bindings.model=integration_ddp_modell") == 2
    assert "run/base/000: " in capsys.readouterr().err
    assert not (tmp_path / "here" / "batches").exists()


def test_ctrl_c_cancels_the_running_task_and_waits_for_it(tmp_path, here, capsys, monkeypatch):
    import signal

    from training_framework.launcher.services import follow

    started = {"following": False, "sent": False}
    original = follow.Follower.run

    def run_and_mark(self):
        started["following"] = True
        return original(self)

    monkeypatch.setattr(follow.Follower, "run", run_and_mark)

    def interrupt(*_):
        if started["following"] and not started["sent"]:
            started["sent"] = True
            raise KeyboardInterrupt

    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 3.0, 0.5)
    try:
        root = project(tmp_path)
        status = run(root, here, "--yes", "--override", "session_config.max_iterations=10000000")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    captured = capsys.readouterr()
    assert started["sent"]
    assert "cancelling the task (Ctrl-C again to stop following)" in captured.err
    assert "cancelled run/base/000" in captured.out
    final = captured.out.rstrip().splitlines()[-1]
    assert final.startswith("run/base/000: ") and not final.endswith(": done"), final
    assert status != 0
