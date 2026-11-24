"""Every engine launch leaves a record: direct runs, failed builds, resumes
and extensions, multi-rank runs, batch tasks, and runs of uncommitted code.
Launched for real, in this process, with the integration components."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from naavik.engine import Configurator, TrainingEngine
from naavik.runs.recorder import ENV_RUN_DIRECTORY
from naavik.runs.registry import list_runs, parent_of, read_run

pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="the runs use gloo DDP",
)

COMPONENTS = "tests.integration.integration_training_components"
SRC = Path(__file__).resolve().parents[2]


def session(tmp_path, *, world_size=1, max_iterations=3, checkpoint_every=None, **changes):
    data = {
        "session_config": {
            "rng_seed": 7, "sessions_dir": str(tmp_path / "sessions"),
            "max_iterations": max_iterations, "device": "cpu",
            "components_package": COMPONENTS, "show_execution_graph": False,
        },
        "role_bindings": {"model": "integration_ddp_model", "dataset": "integration_dataset"},
        "integration_ddp_model": {"initial_weight": 0.0},
        "integration_dataset": {"dataset_size": 4},
        "ddp": {"world_size": world_size, "backend": "gloo", "master_addr": "127.0.0.1",
                "parallel_components": ["model", "dataset", "data_manager", "integration_data",
                                        "integration_train", "integration_loss", "optimizer",
                                        "integration_results"]},
        "data_manager": {"batch_size": world_size, "num_workers": 0, "pin_memory": False},
        "integration_data": {}, "integration_train": {}, "integration_loss": {},
        "optimizer": {"optimizer": {"name": "AdamW", "kwargs": {"lr": 0.1}}},
        "integration_results": {"output_dir": str(tmp_path / "results")},
    }
    if checkpoint_every:
        data["checkpointer"] = {"checkpoint_every": checkpoint_every}
    data.update(changes)
    return data


@pytest.fixture(autouse=True)
def components():
    """Registered again for each test (the suite resets registrations), as
    the integration tests do."""
    import importlib

    module = sys.modules.get(COMPONENTS)
    if module is None:
        importlib.import_module(COMPONENTS)
    else:
        importlib.reload(module)


@pytest.fixture
def registry(tmp_path, monkeypatch):
    root = tmp_path / "registry"
    monkeypatch.setenv("TRAINING_FRAMEWORK_RUNS", str(root))
    return root


def launch(monkeypatch, *argv) -> TrainingEngine:
    monkeypatch.setattr(sys, "argv", ["train", *argv])
    engine = TrainingEngine(Configurator())
    with engine:
        engine.start_session()
    return engine


def launch_config(tmp_path, monkeypatch, config, name="config.yaml") -> TrainingEngine:
    path = tmp_path / name
    path.write_text(yaml.safe_dump({"sessions": [config]}))
    return launch(monkeypatch, "--config", str(path))


def only_run(registry):
    runs, problems = list_runs(registry)
    assert not problems and len(runs) == 1, (runs, problems)
    return runs[0]


# A direct run --------------------------------------------------------------------------------


def test_a_direct_run_is_recorded_from_start_to_end(tmp_path, registry, monkeypatch, capsys):
    engine = launch_config(tmp_path, monkeypatch, session(tmp_path))
    run = only_run(registry)
    assert run.run_id == engine.run_id
    assert f"run {run.run_id} (naavik runs show {run.run_id})" in capsys.readouterr().err
    launch_record = run.record["launch"]
    assert launch_record["operation"] == "config" and launch_record["command"][1] == "--config"
    assert run.record["pid"] == os.getpid() and run.record["environment"]["torch"] == torch.__version__
    (recorded,) = run.sessions
    assert recorded["type"] == "training"
    assert Path(recorded["session_dir"]).parent == tmp_path / "sessions"
    assert (Path(recorded["session_dir"]) / "config.yaml").exists()
    config = run.config(registry)
    assert config["session_config"]["max_iterations"] == 3 and config["session_type"] == "training"
    assert recorded["config_sources"][0]["path"].endswith("config.yaml")
    assert run.finished["status"] == "ok" and run.finished["last_iteration"] == [3]
    assert run.state == "done"
    assert run.ranks[0]["device"] == "cpu" and run.ranks[0]["world_size"] == 1


def test_every_rank_records_its_device(tmp_path, registry, monkeypatch):
    launch_config(tmp_path, monkeypatch, session(tmp_path, world_size=2))
    run = only_run(registry)
    assert sorted(run.ranks) == [0, 1]
    assert {fact["backend"] for fact in run.ranks.values()} == {"gloo"}
    assert {fact["world_size"] for fact in run.ranks.values()} == {2}
    assert run.finished["last_iteration"] == [3, 3]


def test_a_launch_that_fails_while_building_is_recorded_as_failed(tmp_path, registry, monkeypatch):
    with pytest.raises(Exception):
        launch_config(tmp_path, monkeypatch, session(tmp_path, no_such_component={}))
    run = only_run(registry)
    assert run.sessions == [] and run.state == "failed"
    assert "no_such_component" in run.finished["error"]


def test_the_run_directory_reaches_workers_only_while_they_are_spawned(tmp_path, registry,
                                                                    monkeypatch):
    monkeypatch.delenv(ENV_RUN_DIRECTORY, raising=False)
    launch_config(tmp_path, monkeypatch, session(tmp_path))
    assert ENV_RUN_DIRECTORY not in os.environ


# Resume and extend ---------------------------------------------------------------------------


def test_a_resume_and_an_extension_continue_their_run(tmp_path, registry, monkeypatch):
    launch_config(tmp_path, monkeypatch, session(tmp_path, max_iterations=4, checkpoint_every=2))
    first = only_run(registry)
    checkpoints = sorted(Path(first.session_dirs[0]).rglob("manifest.json"))
    middle = str(checkpoints[0].parent)                      # iteration 2
    launch(monkeypatch, "--resume-session-from-checkpoint", middle)
    final = str(sorted(Path(first.session_dirs[0]).rglob("manifest.json"))[-1].parent)
    launch(monkeypatch, "--extend-session-from-checkpoint", final, "--override", "session_config.max_iterations=6")
    runs, _ = list_runs(registry)
    extension, resumed, original = runs                      # newest first
    assert resumed.record["launch"]["operation"] == "resume"
    assert resumed.record["launch"]["checkpoint"] == middle
    assert extension.record["launch"]["operation"] == "extend"
    assert parent_of(resumed, runs).run_id == original.run_id
    assert parent_of(extension, runs).run_id == resumed.run_id
    assert extension.finished["status"] == "ok"


# Never in the way ------------------------------------------------------------------------------


def test_an_unwritable_registry_is_one_warning_and_the_run_goes_on(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory")
    monkeypatch.setenv("TRAINING_FRAMEWORK_RUNS", str(blocker))
    engine = launch_config(tmp_path, monkeypatch, session(tmp_path))
    assert engine.run_id is None
    err = capsys.readouterr().err
    assert err.count("warning: run registry: cannot start a record") == 1
    assert "the run goes on unrecorded" in err


def test_a_registry_turned_off_records_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TRAINING_FRAMEWORK_RUNS", "off")
    engine = launch_config(tmp_path, monkeypatch, session(tmp_path))
    assert engine.run_id is None and "run registry" not in capsys.readouterr().err


# The code --------------------------------------------------------------------------------------


def test_a_run_of_uncommitted_code_keeps_a_snapshot_of_it(tmp_path, registry, monkeypatch):
    monkeypatch.setenv("TRAINING_FRAMEWORK_RUNS_SNAPSHOT", "1")
    project = tmp_path / "project"
    (project / "snapproject").mkdir(parents=True)
    (project / "snapproject" / "__init__.py").write_text("")
    # The integration components (registered once), from a package of this repository.
    source = f"from {COMPONENTS} import *  # noqa: F401,F403\n"
    (project / "snapproject" / "components.py").write_text(source)

    def git(*args):
        return subprocess.run(["git", *args], cwd=project, check=True, capture_output=True,
                              text=True).stdout.strip()

    git("init", "-q")
    git("add", "-A")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "one")
    head = git("rev-parse", "HEAD")
    (project / "snapproject" / "components.py").write_text(source + "\n# an uncommitted edit\n")
    monkeypatch.syspath_prepend(str(project))
    monkeypatch.setenv("PYTHONPATH", f"{project}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    config = session(tmp_path)
    config["session_config"]["components_package"] = "snapproject.components"
    launch_config(tmp_path, monkeypatch, config)

    code = only_run(registry).record["code"]
    assert code["repository"] == str(project) and code["base_sha"] == head and code["snapshot"]
    assert git("rev-parse", code["ref"]) == code["sha"] and code["ref"].endswith(only_run(registry).run_id)
    assert "# an uncommitted edit" in git("show", f"{code['sha']}:snapproject/components.py")
    assert git("rev-parse", "HEAD") == head                         # nothing committed
