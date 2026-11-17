"""Summaries compare the training config file batch tasks were made from
(when their batch used more than one), like their code."""

from __future__ import annotations

import json

from training_framework.reporting import discover_runs, summarize_runs

from .helpers import config, losses, write_run


def task(root, index, *, seed, config_file=None):
    task_dir = root / "tasks" / "g" / f"e{index}" / "000"
    write_run(task_dir / "sessions", losses(2.0, 1.0),
              config(seed=seed, sessions_dir=str(task_dir / "sessions")))
    (task_dir / "variant.json").write_text("[]")
    record = {"exit_status": 0}
    if config_file:
        record["config_file"] = config_file
    (task_dir / "record.json").write_text(json.dumps(record))


def test_runs_of_different_config_files_are_different_groups(tmp_path):
    for index, (seed, file) in enumerate([(1, "a.yaml"), (2, "a.yaml"), (1, "b.yaml"), (2, "b.yaml")]):
        task(tmp_path, index, seed=seed, config_file=file)
    result = summarize_runs(discover_runs(tmp_path), ["loss"])
    assert "config" in [column.name for column in result.columns]
    assert sorted(sorted(group.members) for group in result.groups) == [
        ["g/e0/000", "g/e1/000"], ["g/e2/000", "g/e3/000"]]


def test_without_config_files_the_report_is_unchanged(tmp_path):
    task(tmp_path, 0, seed=1)
    task(tmp_path, 1, seed=2)
    result = summarize_runs(discover_runs(tmp_path), ["loss"])
    assert "config" not in result.shared and len(result.groups) == 1
