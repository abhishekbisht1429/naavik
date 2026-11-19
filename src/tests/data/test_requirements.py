"""Data requirements: declared by components from config alone, obtained
once by `ensure` (lease + ready marker keyed by content, never by code)."""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import pytest

import naavik
from naavik.data import (
    DataError,
    DataRequirement,
    ensure,
    requirement_key,
    requirements_of,
)
from naavik.launcher.remote.lease import KeyMismatch

from .fake_datasets import ByHand, Fetchable, NotDeclared

SRC = Path(naavik.__file__).resolve().parents[1]


def located(cls, root, **config):
    (item,) = requirements_of(cls, {"root": str(root), **config})
    return item


def fetches(root) -> int:
    path = Path(root).parent / "fetches"
    return len(path.read_text().split()) if path.exists() else 0


# Declaring ------------------------------------------------------------------------------------


def test_a_component_without_a_declaration_is_not_managed():
    assert requirements_of(NotDeclared, {"root": "/d"}) == ()


def test_the_location_is_the_config_field_and_subpath():
    class Split(Fetchable):
        @classmethod
        def data_requirements(cls, config):
            return (DataRequirement("split", "root", "1", key={"s": "a"}, subpath="a"),)

    item = located(Split, "/d/x")
    assert (item.path, item.location) == ("/d/x", "/d/x/a")
    assert located(Fetchable, "/d/x/").location == "/d/x"


@pytest.mark.parametrize("declared, message", [
    ([DataRequirement("a", "root", "1")], "must return a tuple of DataRequirement"),
    ((DataRequirement("a", "root", "1"), DataRequirement("a", "root", "1")), "'a' twice"),
    ((DataRequirement("a", "version", "1"),), "config field 'version'"),
    ((DataRequirement("a", "nowhere", "1"),), "holds None, not a path"),
])
def test_a_wrong_declaration_is_refused(declared, message):
    class Wrong(Fetchable):
        @classmethod
        def data_requirements(cls, config):
            return declared

    with pytest.raises(DataError, match=message):
        requirements_of(Wrong, {"root": "/d", "version": ""})


def test_a_declaration_without_fetch_and_verify_is_refused():
    class Half:
        @classmethod
        def data_requirements(cls, config):
            return ()

    with pytest.raises(DataError, match="but not fetch_data, verify_data"):
        requirements_of(Half, {})


@pytest.mark.parametrize("fields, message", [
    ({"name": "has space"}, "name"),
    ({"version": ""}, "version must be a non-empty string"),
    ({"subpath": "/abs"}, "subpath must be a relative path"),
    ({"subpath": "a/../.."}, "subpath must be a relative path"),
    ({"key": {"a": object()}}, "key must be JSON"),
    ({"size_hint": -1}, "size_hint"),
])
def test_a_bad_requirement_is_refused(fields, message):
    with pytest.raises(DataError, match=message):
        DataRequirement(**{"name": "a", "path": "root", "version": "1", **fields})


def test_the_content_key_is_name_version_and_key_only():
    base = DataRequirement("a", "root", "1", key={"split": "x"})
    assert requirement_key(base) == requirement_key(DataRequirement("a", "dir", "1", key={"split": "x"},
                                                                    size_hint=5))
    for other in (DataRequirement("b", "root", "1", key={"split": "x"}),
                  DataRequirement("a", "root", "2", key={"split": "x"}),
                  DataRequirement("a", "root", "1", key={"split": "y"})):
        assert requirement_key(other) != requirement_key(base)


# Obtaining ------------------------------------------------------------------------------------


def test_ensure_fetches_once_then_finds_it_ready(tmp_path):
    root = tmp_path / "data"
    item = located(Fetchable, root)
    assert ensure(Fetchable, item, notify=lambda _: None) == "fetched"
    assert ensure(Fetchable, item, notify=lambda _: None) == "ready"
    assert fetches(root) == 1 and (root / "data.bin").exists()


def test_a_fetch_that_fails_verification_leaves_no_marker(tmp_path):
    root = tmp_path / "data"
    item = located(Fetchable, root, broken=True)
    with pytest.raises(DataError, match="fetched, but it is not right: data.bin is missing"):
        ensure(Fetchable, item, notify=lambda _: None)
    assert not list(tmp_path.glob("*.tf-hpc-lock"))           # the lease was released
    with pytest.raises(DataError):
        ensure(Fetchable, item, notify=lambda _: None)        # tried again, not trusted
    assert fetches(root) == 2


def test_ready_data_is_used_without_taking_a_lease(tmp_path):
    # A ready location may be on a shared, read-only filesystem.
    from naavik.launcher.remote.runtime import StdlibFilesystem

    class NoLocks(StdlibFilesystem):
        def mkdir_atomic(self, path):
            raise PermissionError(path)

    item = located(Fetchable, tmp_path / "data")
    ensure(Fetchable, item, notify=lambda _: None)
    assert ensure(Fetchable, item, notify=lambda _: None, filesystem=NoLocks()) == "ready"


def test_new_content_at_a_ready_location_is_refused_not_overwritten(tmp_path):
    root = tmp_path / "data"
    ensure(Fetchable, located(Fetchable, root), notify=lambda _: None)
    with pytest.raises(KeyMismatch, match="is ready with key"):
        ensure(Fetchable, located(Fetchable, root, version="2"), notify=lambda _: None)


def test_data_obtained_by_hand_is_verified_never_fetched(tmp_path):
    root = tmp_path / "data"
    with pytest.raises(DataError, match="cannot be fetched .*: data.bin is missing"):
        ensure(ByHand, located(ByHand, root), notify=lambda _: None)
    root.mkdir()
    (root / "data.bin").write_text("placed")
    assert ensure(ByHand, located(ByHand, root), notify=lambda _: None) == "present"
    assert fetches(root) == 0


def _ensure_in_process(root):
    item = located(Fetchable, root, fetch_seconds=1.0)
    print(ensure(Fetchable, item, notify=lambda _: None))


def test_concurrent_runs_fetch_once(tmp_path):
    root = tmp_path / "data"
    context = multiprocessing.get_context("fork")
    workers = [context.Process(target=_ensure_in_process, args=(str(root),)) for _ in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(60)
        assert worker.exitcode == 0
    assert fetches(root) == 1


# The command the target runs ------------------------------------------------------------------


def command(action, root, requirement="fake", component="tests.data.fake_datasets.Fetchable"):
    return subprocess.run(
        [sys.executable, "-m", "training_framework.data", action, "--component", component,
         "--config", json.dumps({"root": str(root)}), "--requirement", requirement],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(SRC)})


def test_the_command_fetches_and_verifies(tmp_path):
    root = tmp_path / "data"
    assert command("verify", root).returncode == 1
    result = command("fetch", root)
    assert result.returncode == 0 and "fake at " in result.stdout and ": fetched" in result.stdout
    assert command("fetch", root).stdout.rstrip().endswith(": ready")
    assert command("verify", root).returncode == 0


def test_the_command_names_what_is_wrong(tmp_path):
    result = command("fetch", tmp_path, requirement="other")
    assert result.returncode == 2 and "has no data requirement 'other' (it has: fake)" in result.stdout
    result = command("fetch", tmp_path, component="tests.data.fake_datasets.Nope")
    assert result.returncode == 2 and "error:" in result.stdout
