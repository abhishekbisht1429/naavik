"""A batch's note (`naavik run --note`, a manifest's `description`) is shown
in `batches`, `status`, `runs list`, and `summarize`, where some row has one;
listings without notes are as before. A note is shown, never compared."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from naavik.launcher.services.batches import BatchView, as_json, format_views
from naavik.reporting import discover_runs, summarize_runs
from naavik.reporting.render import render_json, render_markdown
from naavik.runs.ids import new_run_id
from naavik.runs.registry import RunWriter
from naavik.runs.views import table

from tests.reporting.helpers import config, losses, write_run


def view(batch_id, description=None):
    receipt = {"batch_id": batch_id, "batch_dir": "/b", "status": "submitted", "nodes": {},
               "profile": "here", "task_count": 1, "created": "2026-11-17T09:00:00+00:00"}
    if description:
        receipt["description"] = description
    return BatchView(receipt, "local")


# batches -------------------------------------------------------------------------------------


def test_batches_show_a_note_column_only_when_some_batch_has_one():
    plain = format_views([view("a"), view("b")], live=False, tasks=False, origins=False)
    assert "note" not in plain.splitlines()[0]
    noted = format_views([view("a", "attention pooling"), view("b")], live=False, tasks=False,
                         origins=False).splitlines()
    cells = [re.split(r"\s{2,}", line.strip()) for line in noted]
    header = cells[0]
    assert header[header.index("note") + 1] == "submit"
    assert cells[1][header.index("note")] == "attention pooling"
    assert cells[2][header.index("note")] == "-"


def test_batches_json_has_a_description_only_when_there_is_one():
    entries = as_json([view("a", "why"), view("b")], tasks=False)
    assert entries[0]["description"] == "why" and "description" not in entries[1]


# runs list -----------------------------------------------------------------------------------


def run(note=None):
    launch = {"operation": "new", "command": ["train.py"]}
    if note:
        launch["note"] = note
    record = {"run_id": new_run_id(), "host": "h", "started": "2026-11-17T09:00:00+00:00",
              "launch": launch, "code": None}
    return SimpleNamespace(run_id=record["run_id"], record=record, state="finished", events=[])


def test_runs_list_shows_notes_only_when_some_run_has_one():
    assert "note" not in table([run(), run()]).splitlines()[0]
    lines = table([run("baseline"), run()]).splitlines()
    assert lines[0].split()[5] == "note" and "baseline" in lines[1]


# summarize -----------------------------------------------------------------------------------


def task(root, index, *, seed, note=None):
    task_dir = root / "tasks" / "g" / "e" / f"{index:03d}"
    write_run(task_dir / "sessions", losses(2.0, 1.0),
              config(seed=seed, sessions_dir=str(task_dir / "sessions")))
    (task_dir / "variant.json").write_text("[]")
    (task_dir / "record.json").write_text(json.dumps({"exit_status": 0, **({"note": note} if note else {})}))


def test_summaries_show_notes_without_comparing_them(tmp_path):
    task(tmp_path, 0, seed=1, note="first try")
    task(tmp_path, 1, seed=2, note="second try")
    result = summarize_runs(discover_runs(tmp_path), ["loss"])
    assert len(result.groups) == 1                         # not a setting
    runs_table = render_markdown(result).split("## Runs")[1]
    header = [cell.strip() for cell in runs_table.strip().splitlines()[0].split("|")]
    assert header[header.index("attempts") + 1] == "note"
    assert "first try" in runs_table and "second try" in runs_table
    assert json.loads(render_json(result))["runs"][0]["note"] == "first try"


def test_summaries_without_notes_are_unchanged(tmp_path):
    task(tmp_path, 0, seed=1)
    result = summarize_runs(discover_runs(tmp_path), ["loss"])
    assert "note" not in render_markdown(result).split("## Runs")[1].splitlines()[2]
    assert "note" not in json.loads(render_json(result))["runs"][0]


def test_a_direct_runs_note_comes_from_its_run_record(tmp_path, monkeypatch):
    registry = tmp_path / "registry"
    monkeypatch.setenv("TRAINING_FRAMEWORK_RUNS", str(registry))
    session_dir = write_run(tmp_path / "runs", losses(1.0), config())
    writer = RunWriter.create(registry, {"run_id": new_run_id(), "code": None,
                                         "launch": {"note": "from the registry"}})
    writer.event("session", index=0, session_dir=str(session_dir))
    assert discover_runs(tmp_path / "runs").runs[0].note == "from the registry"


# End to end -----------------------------------------------------------------------------------

from tests.hpc.test_end_to_end import environment, session  # noqa: E402,F401 (fixture)
from tests.launcher.test_host_end_to_end import here, record, wait_for  # noqa: E402,F401
from tests.launcher.test_naavik_run import project  # noqa: E402
from naavik.cli import main as naavik_cli  # noqa: E402


def test_a_note_is_seen_everywhere(tmp_path, here, capsys):
    root = project(tmp_path)
    assert naavik_cli(["run", str(root / "configs" / "model" / "base.yaml"), "--targets", str(here),
                   "--yes", "--detach", "--note", "attention pooling"]) == 0
    output = capsys.readouterr().out
    batch_dir = Path(output.split("submitted: ")[1].splitlines()[0])
    batch_id = batch_dir.name
    wait_for(lambda: record(batch_dir, "run/base/000"), "the task's record")
    assert record(batch_dir, "run/base/000")["note"] == "attention pooling"
    assert naavik_cli(["status", batch_id]) == 0
    assert "note: attention pooling" in capsys.readouterr().out.splitlines()[1]
    assert naavik_cli(["batches"]) == 0
    listing = capsys.readouterr().out
    assert "note" in listing.splitlines()[0] and "attention pooling" in listing
    assert naavik_cli(["runs", "list"]) == 0
    assert "attention pooling" in capsys.readouterr().out
    (run,) = discover_runs(batch_dir / "tasks").runs
    assert run.note == "attention pooling"


@pytest.mark.parametrize("live", [False, True])
def test_an_unreachable_target_row_keeps_its_columns_beside_notes(live):
    from naavik.launcher.services.batches import UnreachableTarget

    lines = format_views([view("a", "why")], live=live, tasks=False, origins=True,
                         unreachable=[UnreachableTarget("far", "no route")]).splitlines()
    last = "now" if live else "submit"
    assert lines[2].index("(cannot list: no route)") == lines[0].rindex(last)   # the last column
