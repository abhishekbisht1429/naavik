"""Summaries of an analysis mission's runs (`from_mission`): each is compared
by what trained the checkpoint it analysed (the source experiment's varied
values and the checkpoint's iteration), not by the checkpoint's path; the
source's training seed is a seed, so analyses of runs that differ only in it
are one group."""

from __future__ import annotations

import json

from naavik.reporting import discover_runs, summarize_runs
from naavik.reporting.render import render_json, render_markdown

from naavik.reporting.summary import DEFAULT_IGNORED_KEYS

from .helpers import config, losses, write_run

LR = {"component": "optimizer", "path": ["optimizer", "kwargs", "lr"]}
SEED = {"path": ["session_config", "rng_seed"]}
INTO = {"component": "trained_model", "path": ["model_checkpoint_path"]}


def analysis(root, index, *, lr, seed, iteration=6000, into=INTO, source_experiment=None):
    """An analysis run of the checkpoint of training run (lr, seed)."""
    task_dir = root / "tasks" / "eval" / "test" / f"{index:03d}"
    checkpoint = f"/m/train/tasks/probe/lr/{index:03d}/sessions/s/checkpoints/{iteration}"
    write_run(task_dir / "sessions", losses(2.0 - lr, 1.0 - lr),
              config(seed=0, sessions_dir=str(task_dir / "sessions"),
                     trained_model={"model_checkpoint_path": checkpoint}))
    (task_dir / "variant.json").write_text("[]")
    source = {"mission": "train-20261118T090000Z-9e04101571a2",
              "experiment": source_experiment or f"probe/lr/{index:03d}",
              "checkpoint": checkpoint, "iteration": iteration,
              "variant": [{"address": LR, "value": lr}, {"address": SEED, "value": seed}]}
    if into is not None:
        source["into"] = into
    (task_dir / "record.json").write_text(json.dumps({"exit_status": 0, "source": source}))


def sweep(root, iterations=(6000, 6000, 6000, 6000)):
    for index, ((lr, seed), iteration) in enumerate(zip(
            [(0.001, 41), (0.001, 42), (0.01, 41), (0.01, 42)], iterations)):
        analysis(root, index, lr=lr, seed=seed, iteration=iteration)


def members(result):
    return sorted(sorted(group.members) for group in result.groups)


def test_analyses_of_one_training_setting_are_one_group_over_its_seeds(tmp_path):
    sweep(tmp_path)
    result = summarize_runs(discover_runs(tmp_path), ["loss"])
    assert members(result) == [["eval/test/000", "eval/test/001"], ["eval/test/002", "eval/test/003"]]
    columns = {column.name: column for column in result.columns}
    assert columns["source optimizer:optimizer.kwargs.lr"].seed is False
    assert columns["source session_config.rng_seed"].seed is True
    # The checkpoint's path is not a setting: what trained it is.
    assert not [name for name in columns if "model_checkpoint_path" in name]
    assert result.shared["source iteration"] == 6000
    assert {group.hyperparameters["source optimizer:optimizer.kwargs.lr"] for group in result.groups} == {0.001, 0.01}
    # Each run's seeds: its own (the same in every analysis) and its source's.
    assert sorted(group.seeds for group in result.groups) == [
        [f"session_config.rng_seed=0, source session_config.rng_seed={seed}" for seed in (41, 42)]] * 2


def test_checkpoints_of_different_iterations_are_never_one_group(tmp_path):
    sweep(tmp_path, iterations=(3000, 6000, 6000, 6000))
    result = summarize_runs(discover_runs(tmp_path), ["loss"])
    assert members(result) == [["eval/test/000"], ["eval/test/001"], ["eval/test/002", "eval/test/003"]]
    assert "source iteration" in [column.name for column in result.columns]


def test_the_source_experiment_is_shown_not_compared(tmp_path):
    sweep(tmp_path)
    result = summarize_runs(discover_runs(tmp_path), ["loss"])
    assert "source experiment" not in [column.name for column in result.columns]
    table = render_markdown(result)
    header = next(line for line in table.splitlines() if line.startswith("| run"))
    assert "| source experiment |" in header
    assert "probe/lr/002" in table
    runs = json.loads(render_json(result))["runs"]
    assert {run["source"]["experiment"] for run in runs} == {f"probe/lr/00{i}" for i in range(4)}


def test_without_into_the_checkpoint_path_is_compared_and_said(tmp_path):
    analysis(tmp_path, 0, lr=0.001, seed=41, into=None)
    analysis(tmp_path, 1, lr=0.001, seed=42, into=None)
    result = summarize_runs(discover_runs(tmp_path), ["loss"])
    assert members(result) == [["eval/test/000"], ["eval/test/001"]]
    assert any("does not say where its checkpoint was written" in p for p in result.problems)


def test_a_source_setting_can_be_ignored_and_seed_groups_turned_off(tmp_path):
    sweep(tmp_path, iterations=(3000, 6000, 6000, 6000))
    ignored = summarize_runs(discover_runs(tmp_path), ["loss"],
                             ignored_keys=(*DEFAULT_IGNORED_KEYS, "source iteration"))
    assert members(ignored) == [["eval/test/000", "eval/test/001"], ["eval/test/002", "eval/test/003"]]
    assert summarize_runs(discover_runs(tmp_path), ["loss"], group_by_seed=False).groups is None
