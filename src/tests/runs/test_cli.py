"""`naavik runs list | show | diff | sync` and `naavik summarize --runs`, on
runs recorded by real engine launches."""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import pytest
import yaml

from tests.hpc.batch_repo import PROFILE
from tests.hpc.fake_cluster import fake_cluster  # noqa: F401 (fixture)
from tests.runs.test_engine_records import components, launch, launch_config, session  # noqa: F401
from training_framework.cli import main as naavik
from training_framework.runs.ids import new_run_id
from training_framework.runs.registry import RunWriter, list_runs

FAKE_PULL = r'''
import os, shutil, sys
args = sys.argv[1:]
source, target = args[-2].split(":", 1)[1], args[-1]
copied = 0
for directory, _, files in os.walk(source):
    for name in files:
        path = os.path.join(directory, name)
        destination = os.path.join(target, os.path.relpath(path, source))
        if os.path.exists(destination) and open(destination, "rb").read() == open(path, "rb").read():
            continue
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2(path, destination)
        copied += 1
print("Number of regular files transferred: %d" % copied)
'''


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    """Three launches: lr 0.1, lr 0.2, and one that fails while building."""
    registry = tmp_path / "registry"
    monkeypatch.setenv("TRAINING_FRAMEWORK_RUNS", str(registry))
    first = session(tmp_path)
    second = session(tmp_path)
    second["optimizer"]["optimizer"]["kwargs"]["lr"] = 0.2
    ids = [launch_config(tmp_path, monkeypatch, first, "a.yaml").run_id,
           launch_config(tmp_path, monkeypatch, second, "b.yaml").run_id]
    with pytest.raises(Exception):
        launch_config(tmp_path, monkeypatch, session(tmp_path, no_such_component={}), "c.yaml")
    ids.append(list_runs(registry)[0][0].run_id)
    return registry, ids


def rows(output: str) -> list[list[str]]:
    return [line.split() for line in output.splitlines()[1:] if line.strip()]


# list ---------------------------------------------------------------------------------------


def test_list_shows_every_run_newest_first(recorded, capsys):
    registry, (first, second, failed) = recorded
    capsys.readouterr()
    assert naavik(["runs", "list"]) == 0
    table = rows(capsys.readouterr().out)
    assert [row[0] for row in table] == [failed, second, first]
    states = {row[0]: row[4] for row in table}
    assert states == {failed: "failed", second: "done", first: "done"}
    assert table[1][-1].endswith("b.yaml") and table[2][-1].endswith("a.yaml")


@pytest.mark.parametrize("options, expected", [
    (["--state", "failed"], [2]),
    (["--where", "optimizer.optimizer.kwargs.lr=0.2"], [1]),
    (["--where", "optimizer.optimizer.kwargs.lr=0.1", "--where", "session_config.rng_seed=7"], [0]),
    (["--where", "no.such.key=1"], []),
    (["--since", "1h"], [2, 1, 0]),
    (["--host", "elsewhere"], []),
    (["--limit", "1"], [2]),
])
def test_list_selects_runs(recorded, capsys, options, expected):
    _, ids = recorded
    capsys.readouterr()
    assert naavik(["runs", "list", *options]) == 0
    output = capsys.readouterr().out
    assert [row[0] for row in rows(output)] == [ids[i] for i in expected] or \
        (not expected and output.startswith("no runs match"))


def test_list_as_json(recorded, capsys):
    _, ids = recorded
    capsys.readouterr()
    assert naavik(["runs", "list", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [item["run_id"] for item in data] == ids[::-1]
    assert data[1]["record"]["launch"]["operation"] == "config"


def test_a_turned_off_registry_is_an_error_unless_named(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TRAINING_FRAMEWORK_RUNS", "off")
    assert naavik(["runs", "list"]) == 2
    assert "turned off" in capsys.readouterr().err
    assert naavik(["runs", "list", "--registry", str(tmp_path)]) == 0


# show -----------------------------------------------------------------------------------------


def test_show_tells_a_runs_whole_story(recorded, capsys):
    registry, (first, _, _) = recorded
    capsys.readouterr()
    assert naavik(["runs", "show", first[:18].lower()]) == 0
    output = capsys.readouterr().out
    assert output.startswith(f"run {first} | done | started ")
    assert f"on {socket.gethostname()}" in output and "launch   config:" in output
    assert "rank 0   cpu | gloo world 1" in output
    (session_line,) = [line for line in output.splitlines() if line.startswith("session  0:")]
    config_path = session_line.split("(")[-1].rstrip(")")
    assert Path(config_path).exists()
    assert "from " in output and "a.yaml" in output
    assert "finished" in output and "| ok | last iteration [3]" in output


def test_show_names_a_failed_launchs_error(recorded, capsys):
    _, (_, _, failed) = recorded
    capsys.readouterr()
    assert naavik(["runs", "show", failed]) == 0
    output = capsys.readouterr().out
    assert "| failed |" in output and "! " in output and "no_such_component" in output


def test_show_of_a_resume_names_the_run_it_continues_and_its_checkpoints(tmp_path, monkeypatch, capsys):
    registry = tmp_path / "registry"
    monkeypatch.setenv("TRAINING_FRAMEWORK_RUNS", str(registry))
    original = launch_config(tmp_path, monkeypatch, session(tmp_path, max_iterations=4,
                                                             checkpoint_every=2)).run_id
    session_dir = list_runs(registry)[0][0].session_dirs[0]
    checkpoint = str(sorted(Path(session_dir).rglob("manifest.json"))[0].parent)
    resumed = launch(monkeypatch, "--resume-session", checkpoint).run_id
    capsys.readouterr()
    assert naavik(["runs", "show", resumed]) == 0
    output = capsys.readouterr().out
    assert f"continues {original} (done)" in output
    assert "checkpoint iteration 2:" in output and "checkpoint iteration 4:" in output


def test_show_of_an_unknown_or_ambiguous_run_is_an_error(recorded, capsys):
    _, ids = recorded
    capsys.readouterr()
    assert naavik(["runs", "show", "ZZZZ"]) == 2
    assert "no run with an id starting with 'ZZZZ'" in capsys.readouterr().err
    common = os.path.commonprefix(ids)
    assert naavik(["runs", "show", common]) == 2
    assert "3 runs" in capsys.readouterr().err


# diff -------------------------------------------------------------------------------------------


def test_diff_names_what_differs(recorded, capsys):
    _, (first, second, _) = recorded
    capsys.readouterr()
    assert naavik(["runs", "diff", first, second]) == 0
    output = capsys.readouterr().out
    assert "config  optimizer.optimizer.kwargs.lr: 0.1 | 0.2" in output
    assert "rng_seed" not in output and "code " not in output
    # The rendezvous port the engine picked differs by construction: named, not compared.
    assert "config  ddp.master_port" not in output
    assert "(not compared, they differ by construction: ddp.master_port" in output
    assert naavik(["runs", "diff", first, second, "--all-keys"]) == 0
    assert "config  ddp.master_port: " in capsys.readouterr().out


def test_diff_of_a_run_with_itself_finds_nothing(recorded, capsys):
    _, (first, _, _) = recorded
    capsys.readouterr()
    assert naavik(["runs", "diff", first, first]) == 0
    assert "no differences" in capsys.readouterr().out


# sync -----------------------------------------------------------------------------------------


@pytest.fixture
def two_machines(tmp_path, monkeypatch, fake_cluster):
    """A registry "on gpubox" (reached through the fake ssh, which runs here)
    and this machine's, and a target file naming gpubox and this machine."""
    remote = tmp_path / "gpubox-registry"
    monkeypatch.setenv("TRAINING_FRAMEWORK_RUNS", str(remote))    # what the fake ssh sees
    bin_dir = tmp_path / "pull-bin"
    bin_dir.mkdir()
    (bin_dir / "rsync").write_text(f"#!{sys.executable}\n{FAKE_PULL}")
    (bin_dir / "rsync").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    data = {"default": "gpubox", "targets": {
        "gpubox": dict(PROFILE["profiles"]["cluster"], ssh_host="gpubox"),
        "here": {**{k: v for k, v in PROFILE["profiles"]["cluster"].items() if k != "ssh_host"},
                 "local": True},
    }}
    targets = tmp_path / "targets.yaml"
    targets.write_text(yaml.safe_dump(data))
    return remote, tmp_path / "local-registry", targets


def _remote_run(remote: Path, finished: bool):
    from training_framework.launcher.remote.runtime import process_start_time

    writer = RunWriter.create(remote, {
        "run_id": new_run_id(), "host": "gpubox", "pid": os.getpid(),
        "process_start_time": process_start_time(os.getpid()), "launch": {"operation": "config"}})
    writer.store_config({"a": 1})
    if finished:
        writer.event("finished", status="ok")
    return writer


def test_sync_copies_a_targets_runs_and_their_later_events(two_machines, capsys):
    remote, local, targets = two_machines
    writer = _remote_run(remote, finished=False)
    assert naavik(["runs", "sync", "--targets", str(targets), "--registry", str(local)]) == 0
    # run.json and its config (no events yet: the run has not reached any).
    assert "gpubox: 2 file(s) copied from gpubox:" in capsys.readouterr().out
    (run,) = list_runs(local)[0]
    assert run.run_id == writer.run_id and run.state == "unknown"     # another machine's, unfinished
    writer.event("finished", status="ok")                             # it ends over there
    assert naavik(["runs", "sync", "--targets", str(targets), "--registry", str(local)]) == 0
    assert "gpubox: 1 file(s) copied" in capsys.readouterr().out
    assert list_runs(local)[0][0].state == "done"


def test_sync_never_deletes_and_skips_this_machine(two_machines, capsys):
    remote, local, targets = two_machines
    _remote_run(remote, finished=True)
    mine = RunWriter.create(local, {"run_id": new_run_id(), "host": socket.gethostname(),
                                    "launch": {"operation": "config"}})
    assert naavik(["runs", "sync", "--all-hosts", "--targets", str(targets),
                   "--registry", str(local)]) == 0
    output = capsys.readouterr().out
    assert "here: this machine" in output and "gpubox: " in output
    assert mine.run_id in {run.run_id for run in list_runs(local)[0]} and len(list_runs(local)[0]) == 2


def test_sync_from_a_target_without_runs_says_so(two_machines, capsys):
    _, local, targets = two_machines
    assert naavik(["runs", "sync", "--target", "gpubox", "--targets", str(targets),
                   "--registry", str(local)]) == 0
    assert "no runs recorded on gpubox yet" in capsys.readouterr().out


# summarize --runs ---------------------------------------------------------------------------------


def test_summarize_compares_recorded_runs_by_id(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TRAINING_FRAMEWORK_RUNS", str(tmp_path / "registry"))
    ids = []
    for lr in (0.1, 0.2):
        config = session(tmp_path, tensorboard={"serve": False},
                         tensorboard_summaries={"optimizer": True})
        config["optimizer"]["optimizer"]["kwargs"]["lr"] = lr
        ids.append(launch_config(tmp_path, monkeypatch, config, f"lr{lr}.yaml").run_id)
    capsys.readouterr()
    assert naavik(["summarize", "--runs", ids[0], "--runs", ids[1], "--no-seed-groups"]) == 0
    report = capsys.readouterr().out
    assert "optimizer.optimizer.kwargs.lr" in report and "0.2" in report


def test_summarize_needs_a_root_or_a_run(recorded, capsys):
    _, (_, _, failed) = recorded
    assert naavik(["summarize"]) == 2
    assert "give a ROOT directory or --runs RUN" in capsys.readouterr().err
    assert naavik(["summarize", "--runs", failed]) == 2
    assert "has no session (it failed while building)" in capsys.readouterr().err


def test_list_selects_a_batchs_tasks_and_names_them(recorded, capsys):
    registry, ids = recorded
    task = RunWriter.create(registry, {
        "run_id": new_run_id(), "host": socket.gethostname(), "pid": 0,
        "launch": {"operation": "config", "batch_id": "sweep-20261116T000000Z-abc",
                   "task": "g/e/002", "attempt": 2}})
    task.event("finished", status="ok")
    capsys.readouterr()
    assert naavik(["runs", "list", "--mission", "sweep-2026"]) == 0
    (row,) = rows(capsys.readouterr().out)
    assert row[0] == task.run_id and row[-4:] == ["sweep-20261116T000000Z-abc", "g/e/002",
                                                  "(attempt", "2)"]
