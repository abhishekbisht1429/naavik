"""`summarize --batch ID`: a batch read where it is -- its batch directory
when it ran on this machine, else its fetched copy -- never downloaded."""

from __future__ import annotations

import json

import pytest

from naavik.launcher.ports.targets import Target
from naavik.reporting.__main__ import main

from .helpers import config, losses, write_run


@pytest.fixture
def receipts(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    directory = tmp_path / "state" / "training-framework" / "submissions"
    directory.mkdir(parents=True)
    return directory


def batch(root, batch_id, seeds=(1, 2)):
    for index, seed in enumerate(seeds):
        task_dir = root / batch_id / "tasks" / "g" / "e" / f"{index:03d}"
        write_run(task_dir / "sessions", losses(2.0, 1.0 + seed / 10),
                  config(seed=seed, sessions_dir=str(task_dir / "sessions")))
        (task_dir / "variant.json").write_text("[]")
    return root / batch_id


def receipt(receipts, batch_id, batch_dir, target):
    data = {"batch_id": batch_id, "batch_dir": str(batch_dir), "status": "submitted",
            "nodes": {}, **target.receipt_fields()}
    (receipts / f"{batch_id}.json").write_text(json.dumps(data))
    return receipts / f"{batch_id}.json"


def test_a_batch_that_ran_here_is_read_where_it_is(tmp_path, receipts, capsys):
    batch_dir = batch(tmp_path / "here" / "batches", "sweep-1")
    receipt(receipts, "sweep-1", batch_dir, Target("local", None))
    assert main(["summarize", "--mission", "sweep-1", "--metric", "loss"]) == 0
    report = capsys.readouterr().out
    assert f"`{batch_dir}`" in report and "| g/e/000 |" in report.replace("tasks/", "")


def test_a_batch_from_elsewhere_is_read_from_its_fetched_copy(tmp_path, receipts, capsys):
    receipt(receipts, "sweep-2", "/scratch/me/batches/sweep-2", Target("ssh", "cluster"))
    copy = batch(tmp_path / "results", "sweep-2")
    assert main(["summarize", "--mission", "sweep-2", "--fetched", str(tmp_path / "results"),
                 "--metric", "loss"]) == 0
    assert f"`{copy}`" in capsys.readouterr().out


def test_a_batch_not_fetched_yet_is_not_downloaded(tmp_path, receipts, capsys):
    receipt(receipts, "sweep-3", "/scratch/me/batches/sweep-3", Target("ssh", "cluster"))
    assert main(["summarize", "--mission", "sweep-3", "--fetched", str(tmp_path / "results")]) == 2
    err = capsys.readouterr().err
    assert "--mission sweep-3 ran on" in err and "`naavik fetch sweep-3`" in err


def test_two_batches_compare_in_one_report(tmp_path, receipts, capsys):
    for name in ("sweep-a", "sweep-b"):
        receipt(receipts, name, batch(tmp_path / "here", name), Target("local", None))
    assert main(["summarize", "--mission", "sweep-a", "--mission", "sweep-b", "--metric", "loss"]) == 0
    report = capsys.readouterr().out
    assert "sweep-a/" in report and "sweep-b/" in report


def test_a_receipt_path_is_accepted(tmp_path, receipts, capsys):
    path = receipt(receipts, "sweep-4", batch(tmp_path / "here", "sweep-4"), Target("local", None))
    assert main(["summarize", "--mission", str(path), "--metric", "loss"]) == 0


@pytest.mark.parametrize("make, message", [
    (lambda tmp_path, receipts: None, "no receipt for it on this machine"),
    (lambda tmp_path, receipts: receipt(receipts, "gone", tmp_path / "nowhere", Target("local", None)),
     "its mission directory .* is gone"),
    (lambda tmp_path, receipts: (receipts / "gone.json").write_text("{not json"),
     "cannot be read"),
    (lambda tmp_path, receipts: (receipts / "gone.json").write_text(
        json.dumps({"batch_id": "gone", "status": "submitted", "nodes": {}})),
     "is missing or malformed: batch_dir"),
])
def test_what_cannot_be_read_is_said(tmp_path, receipts, capsys, make, message):
    import re

    make(tmp_path, receipts)
    assert main(["summarize", "--mission", "gone"]) == 2
    assert re.search(message, capsys.readouterr().err)
