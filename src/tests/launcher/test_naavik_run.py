"""`naavik run`: the session of a session file, run by the engine here, in
this process: one run, recorded in the run registry, no mission. A session
continues with `--resume-session SESSION_DIR` (from its latest checkpoint,
said in the log) or `--resume-session-from-checkpoint CHECKPOINT`."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from tests.hpc.test_end_to_end import COMPONENTS, FRAMEWORK_SRC, session
from naavik.runs.registry import list_runs

pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="the session is a one-rank gloo DDP session",
)


@pytest.fixture
def project(tmp_path):
    """A project directory (no Git, no project file, no hosts file) with the
    integration components, a session file, and its own run registry."""
    package = tmp_path / "project" / "my_project"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "components.py").write_text(COMPONENTS.read_text())
    config = dict(session(max_iterations=2), checkpointer={"checkpoint_every": 1})
    (package / "config.yaml").write_text(yaml.safe_dump(config))
    return tmp_path / "project", tmp_path / "registry"


def naavik_run(project, *args):
    root, registry = project
    path = os.pathsep.join([str(FRAMEWORK_SRC), os.environ.get("PYTHONPATH", "")])
    env = dict(os.environ, NAAVIK_RUNS=str(registry), PYTHONPATH=path)
    for name in ("NAAVIK_RUN_CONTEXT", "TRAINING_FRAMEWORK_RUNS"):
        env.pop(name, None)
    return subprocess.run([sys.executable, "-m", "naavik.cli", "run", *args], cwd=root, env=env,
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=240)


def session_dir(result) -> Path:
    line = next(line for line in result.stdout.splitlines() if line.startswith("session: "))
    return Path(line.split()[1])


def test_a_session_file_runs_here_as_one_run(project):
    root, registry = project
    result = naavik_run(project, "my_project/config.yaml", "--note", "a direct run")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "python path: " in result.stdout and "(the current directory, as python -m)" in result.stdout
    assert "to continue it: naavik run --resume-session" in result.stdout
    (run,), _ = list_runs(registry)
    assert run.state == "done" and run.record["launch"]["note"] == "a direct run"
    assert run.record["launch"]["operation"] == "config" and "mission" not in run.record["launch"]
    (recorded,) = run.sessions
    assert [Path(source["path"]).name for source in recorded["config_sources"]] == ["config.yaml"]
    assert (root / session_dir(result)).is_dir()


def test_overrides_are_read_as_yaml_and_address_components(project):
    root, _ = project
    result = naavik_run(project, "my_project/config.yaml",
                        "--override", "session_config.max_iterations=1",
                        "--override", "optimizer:optimizer.kwargs.lr=3.0e-4",
                        "--override", "session_config.sessions_dir=./elsewhere")
    assert result.returncode == 0, result.stderr[-2000:]
    config = yaml.safe_load((root / session_dir(result) / "config.yaml").read_text())
    assert config["session_config"]["max_iterations"] == 1
    assert config["optimizer"]["optimizer"]["kwargs"]["lr"] == pytest.approx(3.0e-4)
    assert str(session_dir(result)).startswith("elsewhere")       # a direct run owns its sessions_dir


def test_an_override_through_a_group_is_refused_with_the_form_to_use(project):
    root, _ = project
    config = yaml.safe_load((root / "my_project" / "config.yaml").read_text())
    config["hooks"] = {"checkpointer": config.pop("checkpointer")}
    (root / "my_project" / "config.yaml").write_text(yaml.safe_dump(config))
    result = naavik_run(project, "my_project/config.yaml",
                        "--override", "hooks.checkpointer.checkpoint_every=2")
    assert result.returncode == 2
    assert "'hooks' is a component group" in result.stderr


def test_resume_session_continues_from_its_latest_checkpoint_and_says_which(project):
    root, registry = project
    first = naavik_run(project, "my_project/config.yaml")
    assert first.returncode == 0, first.stderr[-2000:]
    directory = session_dir(first)
    latest = max((root / directory / "checkpoints").iterdir())
    resumed = naavik_run(project, "--resume-session", str(directory))
    assert resumed.returncode == 0, resumed.stderr[-2000:]
    assert (f"resuming session {directory} from its latest checkpoint: "
            f"{directory / 'checkpoints' / latest.name} (iteration 2)") in resumed.stdout
    runs, _ = list_runs(registry)
    assert sorted(run.record["launch"]["operation"] for run in runs) == ["config", "resume"]


def test_resume_session_from_checkpoint_names_it(project):
    root, _ = project
    first = naavik_run(project, "my_project/config.yaml")
    checkpoint = min((root / session_dir(first) / "checkpoints").iterdir())
    resumed = naavik_run(project, "--resume-session-from-checkpoint", str(checkpoint))
    assert resumed.returncode == 0, resumed.stderr[-2000:]
    assert f"resuming from checkpoint {checkpoint}" in resumed.stdout


def test_a_checkpoint_given_to_resume_session_points_to_the_checkpoint_flag(project):
    root, _ = project
    first = naavik_run(project, "my_project/config.yaml")
    checkpoint = min((root / session_dir(first) / "checkpoints").iterdir())
    result = naavik_run(project, "--resume-session", str(checkpoint))
    assert result.returncode != 0
    assert "is a checkpoint: use --resume-session-from-checkpoint" in result.stderr


@pytest.mark.parametrize("args, message", [
    (("my_project/config.yaml", "--resume-session", "runs/x"), "not both"),
    ((), "naavik run needs CONFIG, --resume-session SESSION_DIR"),
    (("my_project/config.yaml", "--gpus", "2"), "--gpus belong to a mission: use naavik launch --session"),
    (("my_project/config.yaml", "--detach"), "--detach belong to a mission"),
])
def test_wrong_arguments_say_what_to_do(project, args, message):
    result = naavik_run(project, *args)
    assert result.returncode == 2 and message in result.stderr
