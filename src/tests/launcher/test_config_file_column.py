"""The experiment table names each experiment's session config file (a 6th
column) when the mission uses more than one, and `status` / `missions
--experiments` show it first in the values; tables of 5 columns read as
before."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.hpc.batch_repo import compile_repo, make_repo
from tests.launcher.test_config_files_axis import FILES, VARY, compile_mission, mission
from naavik.launcher.compile.render import plan_files
from naavik.launcher.services.batches import as_json
from naavik.launcher.services.status import BatchFiles, task_rows

FIVE = "index\ttask\tvariant\tconfig\tconfig_hash"
LR = json.dumps([{"address": {"component": "optimizer", "path": ["optimizer", "kwargs", "lr"]},
                  "value": 0.001}])


def table_of(compiled, job):
    return plan_files(compiled.plan, compiled.manifest.raw)[f"groups/{job}/tasks.tsv"].splitlines()


def test_a_mission_of_several_files_names_each_experiments_file(tmp_path):
    lines = table_of(compile_mission(tmp_path, mission(vary=VARY[:1])), "arch")
    assert lines[0] == FIVE + "\tconfig_file"
    assert [line.split("\t")[5] for line in lines[1:]] == [FILES[0]] * 2 + [FILES[1]] * 2
    # The job scripts' column (the experiment's config.yaml) is where it was.
    assert lines[1].split("\t")[3] == "tasks/arch/heads/000/config.yaml"


def test_a_mission_of_one_file_keeps_its_table(tmp_path):
    lines = table_of(compile_repo(make_repo(tmp_path)), "probes")
    assert lines[0] == FIVE and all(len(line.split("\t")) == 5 for line in lines)


def rows(table):
    receipt = {"nodes": {"group-g": {"status": "submitted", "job_ref": "5"}}}
    return task_rows(receipt, BatchFiles({}, {"g": table}), {"5_0": "running", "5_1": "pending"})


def test_status_shows_the_file_first_in_the_values():
    table = (f"{FIVE}\tconfig_file\n"
             f"0\tg/e/000\t{LR}\ttasks/g/e/000/config.yaml\th\tconfigs/a.yaml\n"
             f"1\tg/e/001\t[]\ttasks/g/e/001/config.yaml\th\tconfigs/b.yaml\n")
    found = {row.name: row for row in rows(table)}
    assert found["g/e/000"].values == "config=configs/a.yaml optimizer.kwargs.lr=0.001"
    assert found["g/e/001"].values == "config=configs/b.yaml"
    assert (found["g/e/000"].config_file, found["g/e/001"].config_file) == ("configs/a.yaml", "configs/b.yaml")


def test_a_table_of_five_columns_reads_as_before():
    table = f"{FIVE}\n0\tg/e/000\t{LR}\ttasks/g/e/000/config.yaml\th\n"
    (row,) = rows(table)
    assert row.values == "optimizer.kwargs.lr=0.001" and row.config_file is None


@pytest.mark.parametrize("config_file", [None, "configs/a.yaml"])
def test_json_carries_the_file_only_when_there_is_one(config_file):
    from naavik.launcher.services.status import Row

    view = SimpleNamespace(batch_id="m", receipt={}, origin="here", now="running", problem=None,
                           rows=[Row("g/e/000", "", "running", "job 5_0", True, config_file)])
    (entry,) = as_json([view], tasks=True)
    (task,) = entry["tasks"]
    assert ("config_file" in task) == (config_file is not None)
    assert task.get("config_file") == config_file
