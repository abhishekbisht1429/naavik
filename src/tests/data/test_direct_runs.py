"""Direct engine runs ensure declared data that asks for a download, once and
under a lease, before any session is built."""

from __future__ import annotations

import importlib
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.data import data_components
from naavik.data.direct import ensure_downloads
from naavik.engine import TrainingEngine

PACKAGE = "tests.data.data_components"


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    importlib.reload(data_components)
    monkeypatch.delenv("NAAVIK_RUN_CONTEXT", raising=False)


def session(tmp_path, **fake_data):
    return {"session_config": {"rng_seed": 1, "sessions_dir": str(tmp_path / "runs"),
                               "max_iterations": 1, "components_package": PACKAGE,
                               "show_execution_graph": False},
            "fake_data": {"root": str(tmp_path / "data"), **fake_data}}


def fetches(tmp_path) -> int:
    path = tmp_path / "fetches"
    return len(path.read_text().split()) if path.exists() else 0


def test_a_download_is_ensured_once(tmp_path):
    config = session(tmp_path, download=True)
    assert ensure_downloads([(config, None)], notify=lambda _: None) == [str(tmp_path / "data")]
    ensure_downloads([(config, None)], notify=lambda _: None)
    assert fetches(tmp_path) == 1 and (tmp_path / "data" / "data.bin").exists()


def test_without_download_nothing_is_fetched(tmp_path):
    assert ensure_downloads([(session(tmp_path), None)], notify=lambda _: None) == []
    assert fetches(tmp_path) == 0


def test_a_batch_task_leaves_it_to_its_target(tmp_path, monkeypatch):
    monkeypatch.setenv("NAAVIK_RUN_CONTEXT", "{}")
    assert ensure_downloads([(session(tmp_path, download=True), None)], notify=lambda _: None) == []
    assert fetches(tmp_path) == 0


def test_a_config_that_does_not_plan_is_left_to_building(tmp_path):
    config = session(tmp_path, download=True, colour="red")
    assert ensure_downloads([(config, None)], notify=lambda _: None) == []


def test_two_sessions_on_one_path_fetch_once(tmp_path):
    config = session(tmp_path, download=True)
    assert ensure_downloads([(config, None), (config, None)], notify=lambda _: None) == \
        [str(tmp_path / "data")]
    assert fetches(tmp_path) == 1


def _run(tmp_path):
    ensure_downloads([(session(Path(tmp_path), download=True, fetch_seconds=1.0), None)],
                     notify=lambda _: None)


def test_concurrent_runs_fetch_once(tmp_path):
    context = multiprocessing.get_context("fork")
    workers = [context.Process(target=_run, args=(str(tmp_path),)) for _ in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(60)
        assert worker.exitcode == 0
    assert fetches(tmp_path) == 1


def test_the_engine_ensures_before_it_builds(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(TrainingEngine, "register_session",
                        lambda self, config, **_: seen.append(
                            (tmp_path / "data" / "data.bin").exists()))
    configurator = SimpleNamespace(
        mode="new", session_configs=(session(tmp_path, download=True),),
        process_timeout_on_join=10.0, heartbeat_timeout=30.0, stop_sync_grace_period=0.01,
        stop_sync_poll_interval=0.005, debug=False, checkpoint_path=None, new_max_iters=None)
    with TrainingEngine(configurator):
        pass
    assert seen == [True] and fetches(tmp_path) == 1


def test_only_a_true_download_counts():
    from naavik.data.direct import asks_for_download

    assert asks_for_download({"download": True})
    for value in ("false", "yes", 1, None):
        assert not asks_for_download({"download": value})       # a component without a schema
