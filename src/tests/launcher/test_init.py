"""`naavik init`: the project file and a host for this machine, written once
and explicitly, so `naavik run` works right after it. Never overwrites; asks
before adding the host to an existing hosts file (and never without a
terminal); writes nothing it would not read back."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import naavik
from tests.hpc.test_end_to_end import COMPONENTS, environment, session  # noqa: F401 (fixture)
from naavik.cli import main as naavik_cli
from naavik.launcher.config.schema import parse_profiles, parse_project
from naavik.launcher.errors import HpcError
from naavik.launcher.services import init as initializing

SRC = Path(naavik.__file__).resolve().parents[1]


def _repo(root: Path, layout: str = "root") -> Path:
    """A Git repository holding a package at its root or in `src/`."""
    package = root / ("src" if layout == "src" else ".") / "my_project"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


@pytest.fixture
def here(tmp_path, monkeypatch):
    """A repository as the working directory, isolated XDG directories, and
    naavik counted as installed (this suite may run from a checkout)."""
    for kind in ("CONFIG", "DATA", "STATE", "CACHE"):
        monkeypatch.setenv(f"XDG_{kind}_HOME", str(tmp_path / "xdg" / kind.lower()))
    root = _repo(tmp_path / "repo")
    monkeypatch.chdir(root)
    monkeypatch.setattr(initializing, "installed_version", lambda: "0.5.0")
    return root, tmp_path / "xdg" / "config" / "naavik" / "hosts.yaml"


def _init(*args) -> int:
    return naavik_cli(["init", *args])


# A new setup ------------------------------------------------------------------------------


def test_init_writes_the_project_file_and_a_host_for_this_machine(here, capsys):
    root, hosts = here
    assert _init() == 0
    out = capsys.readouterr().out
    assert f"wrote {root / 'naavik-project.yaml'} (commit it)" in out
    assert f"wrote {hosts} (host 'here')" in out and "next: naavik run <config.yaml>" in out
    project = parse_project(yaml.safe_load((root / "naavik-project.yaml").read_text()), "p")
    assert project.mission_file_keys["code"] == {"framework_from": "installed",
                                                 "python_paths": ["."]}
    assert project.run_defaults.target == "here" and project.run_defaults.resources == {}
    host = parse_profiles(yaml.safe_load(hosts.read_text()), None, str(hosts))
    assert host.name == "here" and host.local and host.scheduler == "host"
    assert host.scheduler_settings.gpus == "all"
    data = yaml.safe_load(hosts.read_text())["hosts"]["here"]
    assert data["run_code_from"] == "working_tree"
    assert data["python_environment"] == {"use_existing": "current"}
    assert all(Path(path).is_absolute() for path in data["directories"].values())


def test_a_repository_without_a_commit_is_told_to_make_one(here, capsys):
    assert _init() == 0
    assert "note: the repository has no commit yet; naavik run needs one" in \
        capsys.readouterr().out


def test_a_dry_run_prints_both_files_and_writes_nothing(here, capsys):
    root, hosts = here
    assert _init("--dry-run") == 0
    out = capsys.readouterr().out
    assert "framework_from: installed" in out and "default_host: here" in out
    assert not (root / "naavik-project.yaml").exists() and not hosts.exists()


@pytest.mark.parametrize("written, gpus", [("all", "all"), ("2", 2), ("0,1", (0, 1)),
                                           ("none", ())])
def test_gpus_sets_the_pool(here, written, gpus):
    _, hosts = here
    assert _init("--gpus", written) == 0
    assert parse_profiles(yaml.safe_load(hosts.read_text()), None, "h").scheduler_settings.gpus \
        == gpus


def test_another_host_name_and_hosts_file(here, tmp_path, capsys):
    root, _ = here
    hosts = tmp_path / "elsewhere" / "hosts.yaml"
    assert _init("--host", "box", "--hosts-file", str(hosts)) == 0
    assert f"next: naavik run <config.yaml> --hosts-file {hosts}" in capsys.readouterr().out
    assert parse_profiles(yaml.safe_load(hosts.read_text()), None, "h").name == "box"
    assert yaml.safe_load((root / "naavik-project.yaml").read_text()) \
        ["quick_run_defaults"]["host"] == "box"


# What is detected -------------------------------------------------------------------------


def test_a_src_layout_puts_src_on_the_path(tmp_path, monkeypatch, capsys):
    for kind in ("CONFIG", "DATA"):
        monkeypatch.setenv(f"XDG_{kind}_HOME", str(tmp_path / "xdg" / kind.lower()))
    root = _repo(tmp_path / "repo", layout="src")
    (root / "tests").mkdir()
    (root / "tests" / "__init__.py").write_text("")       # does not decide the layout
    monkeypatch.chdir(root)
    monkeypatch.setattr(initializing, "installed_version", lambda: "0.5.0")
    assert _init() == 0
    assert "python_paths: ['src'] (src/ holds the package(s) my_project)" in \
        capsys.readouterr().out


def test_an_unclear_layout_asks_for_python_paths(here, capsys):
    root, hosts = here
    (root / "src" / "other").mkdir(parents=True)
    (root / "src" / "other" / "__init__.py").write_text("")
    assert _init() == 2
    assert "pass --python-paths" in capsys.readouterr().err
    assert not hosts.exists()
    assert _init("--python-paths", "src,.") == 0
    assert yaml.safe_load((root / "naavik-project.yaml").read_text())["code"]["python_paths"] \
        == ["src", "."]


def test_the_naavik_repository_runs_its_own_framework():
    detected = initializing.detect(SRC.parent, None)
    assert detected.framework_from == "this_repository" and detected.python_paths == ("src",)


def test_without_naavik_installed_init_says_to_install_it(tmp_path, monkeypatch):
    monkeypatch.setattr(initializing, "installed_version", lambda: None)
    with pytest.raises(HpcError, match="pip install naavik"):
        initializing.detect(_repo(tmp_path / "repo"), None)


# Existing files ---------------------------------------------------------------------------


def test_an_existing_project_file_is_left_as_is_and_its_host_named(here, capsys):
    root, _ = here
    (root / "naavik-project.yaml").write_text("code: {framework_from: installed, python_paths: [.]}\n")
    assert _init() == 0
    out = capsys.readouterr().out
    assert "exists; left as is (it does not name host 'here'" in out
    assert (root / "naavik-project.yaml").read_text().startswith("code: {")


EXISTING = """\
# my hosts
default_host: box
hosts:
    # the GPU box
    box:
        connection: {ssh: box}
        scheduler: {kind: builtin, gpus: [0]}
        directories: {missions: /m, python_environments: /e, datasets: /d}
        python_environment: {use_existing: /opt/env}
"""


def test_an_existing_hosts_file_is_not_changed_without_a_terminal(here, capsys):
    _, hosts = here
    hosts.parent.mkdir(parents=True)
    hosts.write_text(EXISTING)
    assert _init() == 0
    out = capsys.readouterr().out
    assert "not asked (no terminal); pass --yes to add it" in out
    assert "here:" in out and hosts.read_text() == EXISTING


def test_yes_adds_the_host_keeping_comments_indentation_and_a_backup(here, capsys):
    _, hosts = here
    hosts.parent.mkdir(parents=True)
    hosts.write_text(EXISTING)
    assert _init("--yes") == 0
    text = hosts.read_text()
    assert "# my hosts" in text and "    # the GPU box" in text
    assert "\n    here:" in text and "\n        connection: local" in text
    data = yaml.safe_load(text)
    assert set(data["hosts"]) == {"box", "here"} and data["default_host"] == "box"
    assert yaml.safe_load(EXISTING)["hosts"]["box"] == data["hosts"]["box"]
    assert hosts.with_name("hosts.yaml.bak").read_text() == EXISTING
    assert parse_profiles(data, "here", "h").local


def test_the_answer_decides_on_a_terminal(here, monkeypatch, capsys):
    _, hosts = here
    hosts.parent.mkdir(parents=True)
    hosts.write_text(EXISTING)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda question: "n")
    assert _init() == 0 and hosts.read_text() == EXISTING
    monkeypatch.setattr("builtins.input", lambda question: "y")
    assert _init() == 0 and "here" in yaml.safe_load(hosts.read_text())["hosts"]


def test_a_host_of_that_name_already_there_stops_init(here, capsys):
    root, hosts = here
    hosts.parent.mkdir(parents=True)
    hosts.write_text(EXISTING)
    assert _init("--host", "box", "--yes") == 2
    assert "already has a host named 'box'" in capsys.readouterr().err
    assert hosts.read_text() == EXISTING and not (root / "naavik-project.yaml").exists()


def test_inline_hosts_are_not_edited_and_the_block_is_printed(here, capsys):
    _, hosts = here
    hosts.parent.mkdir(parents=True)
    inline = "hosts: {box: {connection: local}}\n"
    hosts.write_text(inline)
    assert _init("--yes") == 0
    out = capsys.readouterr().out
    assert "written inline" in out and "here:" in out and hosts.read_text() == inline


# Right after init --------------------------------------------------------------------------


def test_naavik_run_works_right_after_init(tmp_path, environment):
    """In the environment the run uses (`use_existing: current`): init's
    interpreter is the experiment's."""
    root = _repo(tmp_path / "repo")
    package = root / "my_project"
    (package / "components.py").write_text(COMPONENTS.read_text())
    (package / "config.yaml").write_text(yaml.safe_dump(session()))
    # As the first-run guide says: a first commit (the working tree runs as a
    # snapshot on it); init's files stay uncommitted.
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "first"],
                   cwd=root, check=True)
    env = dict(os.environ, **{f"XDG_{kind}_HOME": str(tmp_path / "xdg" / kind.lower())
                              for kind in ("CONFIG", "DATA", "STATE", "CACHE")})
    env.pop("PYTHONPATH", None)
    python = str(environment / "bin" / "python")
    init = subprocess.run([python, "-m", "naavik.cli", "init"], cwd=root, env=env,
                          capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert init.returncode == 0, init.stderr
    run = subprocess.run([python, "-m", "naavik.cli", "run", "my_project/config.yaml"],
                         cwd=root, env=env, capture_output=True, text=True,
                         stdin=subprocess.DEVNULL, timeout=240)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    assert "run/config/000: done" in run.stdout
