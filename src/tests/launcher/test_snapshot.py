"""`worktree: snapshot`: a batch runs the working tree as it is, made into a
commit on HEAD, without the user committing (or their index, branch, or
files changing)."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.hpc.batch_repo import compile_repo, make_repo, manifest, profile
from tests.hpc.test_end_to_end import environment, session  # noqa: F401 (fixture)
from tests.launcher.test_host_end_to_end import here, launch, record, repo, wait_for  # noqa: F401
from naavik.cli import main as naavik_cli
from naavik.launcher.errors import PlanError
from naavik.launcher.ports.repository import SNAPSHOT_REFS


def git(root, *args) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True).stdout


def _snapshot_profile(**changes):
    return profile(worktree="snapshot", **changes)


def _user_state(root: Path) -> tuple:
    """What a snapshot must leave as it was: HEAD, the branch, the index,
    the status, and the files."""
    index = Path(git(root, "rev-parse", "--git-path", "index").strip())
    index = index if index.is_absolute() else root / index
    files = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*")
             if path.is_file() and ".git" not in path.parts}
    return (git(root, "rev-parse", "HEAD"), git(root, "symbolic-ref", "HEAD"),
            hashlib.sha256(index.read_bytes()).hexdigest(), git(root, "status", "--porcelain"),
            files, git(root, "for-each-ref"))


def test_a_clean_tree_runs_head_itself(tmp_path):
    root = make_repo(tmp_path)
    plan = compile_repo(root, profile_data=_snapshot_profile()).plan
    assert plan.sha == git(root, "rev-parse", "HEAD").strip()
    assert plan.base_sha is None and "base_sha" not in plan.as_json()
    assert not any("snapshot" in note for note in plan.notes)


def test_changed_and_untracked_files_run_without_a_commit(tmp_path):
    root = make_repo(tmp_path)
    head = git(root, "rev-parse", "HEAD").strip()
    (root / "my_project" / "__init__.py").write_text("CHANGED = True\n")
    (root / "my_project" / "new.py").write_text("NEW = True\n")
    (root / ".gitignore").write_text("ignored.bin\n")
    (root / "ignored.bin").write_bytes(b"x" * 100)
    before = _user_state(root)

    plan = compile_repo(root, profile_data=_snapshot_profile()).plan
    assert plan.sha != head and plan.base_sha == head
    assert git(root, "show", f"{plan.sha}:my_project/__init__.py") == "CHANGED = True\n"
    assert git(root, "show", f"{plan.sha}:my_project/new.py") == "NEW = True\n"
    assert "ignored.bin" not in git(root, "ls-tree", "-r", "--name-only", plan.sha)
    assert git(root, "rev-parse", f"{plan.sha}^").strip() == head
    (note,) = [note for note in plan.notes if "snapshot" in note]
    assert f"on {head[:12]}: 1 modified, 2 untracked file(s)" in note   # new.py and .gitignore
    assert plan.as_json()["base_sha"] == head and plan.as_json()["snapshot"] is True
    assert _user_state(root) == before                                   # nothing of theirs moved


def test_the_same_tree_gives_the_same_snapshot(tmp_path):
    root = make_repo(tmp_path)
    (root / "my_project" / "__init__.py").write_text("CHANGED = True\n")
    first = compile_repo(root, profile_data=_snapshot_profile()).plan
    second = compile_repo(root, profile_data=_snapshot_profile()).plan
    assert first.sha == second.sha and first.canonical() == second.canonical()
    (root / "my_project" / "__init__.py").write_text("CHANGED = 2\n")
    assert compile_repo(root, profile_data=_snapshot_profile()).plan.sha != first.sha


def test_an_uncommitted_manifest_and_config_are_taken_too(tmp_path):
    root = make_repo(tmp_path)
    data = yaml.safe_load((root / "batch.yaml").read_text())
    data["name"] = "edited"
    (root / "batch.yaml").write_text(yaml.safe_dump(data))
    plan = compile_repo(root, profile_data=_snapshot_profile()).plan
    assert plan.batch_id.startswith("edited-")


def test_untracked_files_over_the_limit_are_refused_by_name(tmp_path):
    root = make_repo(tmp_path)
    (root / "data.bin").write_bytes(b"x" * 2048)
    (root / "small.txt").write_text("x")
    with pytest.raises(PlanError, match=r"exceed working_tree_untracked_limit \(1\.0 KiB\).*data\.bin \(2\.0 KiB\)"):
        compile_repo(root, profile_data=_snapshot_profile(snapshot_limit="1K"))


@pytest.mark.parametrize("changes, message", [
    ({"source": {"strategy": "bundle", "ref": "main"}}, "code_transfer.commit 'main' cannot be used"),
    ({"source": {"strategy": "fetch", "url": "https://example.org/x.git"}},
     "code_transfer.method 'git_clone' cannot stage a working tree"),
])
def test_a_snapshot_is_staged_only_from_head_by_bundle(tmp_path, changes, message):
    root = make_repo(tmp_path, manifest=manifest(**changes))
    git(root, "branch", "-f", "main")
    with pytest.raises(PlanError, match=message):
        compile_repo(root, profile_data=_snapshot_profile())


def test_commit_mode_still_refuses_a_dirty_tree_and_says_how_to_snapshot(tmp_path):
    root = make_repo(tmp_path)
    (root / "my_project" / "__init__.py").write_text("CHANGED = True\n")
    with pytest.raises(PlanError, match="run_code_from: working_tree"):
        compile_repo(root)


def test_a_changed_submodule_is_refused(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    git(library, "init", "-q")
    (library / "lib.py").write_text("A = 1\n")
    git(library, "add", "-A")
    git(library, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "one")
    root = make_repo(tmp_path / "project")
    git(root, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(library), "lib")
    git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "submodule")
    compile_repo(root, profile_data=_snapshot_profile())                    # unchanged: fine
    (root / "lib" / "lib.py").write_text("A = 2\n")
    with pytest.raises(PlanError, match="submodule changes: lib has uncommitted changes"):
        compile_repo(root, profile_data=_snapshot_profile())


def test_an_unknown_worktree_mode_is_refused():
    from naavik.launcher.config.schema import parse_profiles

    with pytest.raises(PlanError, match="run_code_from must be one of commit, working_tree"):
        parse_profiles(profile(worktree="stash"), None, "targets.yaml")


# launch ---------------------------------------------------------------------------------


def test_validate_and_render_leave_no_ref(tmp_path, monkeypatch, capsys):
    root = make_repo(tmp_path)
    (root / "my_project" / "__init__.py").write_text("CHANGED = True\n")
    targets = tmp_path / "targets.yaml"
    targets.write_text(yaml.safe_dump(_snapshot_profile()))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert naavik_cli(["validate", str(root / "batch.yaml"), "--targets", str(targets)]) == 0
    assert "note: snapshot of the working tree on" in capsys.readouterr().out
    assert naavik_cli(["render", str(root / "batch.yaml"), "--targets", str(targets),
                   "--output", str(tmp_path / "preview")]) == 0
    assert SNAPSHOT_REFS not in git(root, "for-each-ref")


def test_a_launched_snapshot_runs_the_uncommitted_code_and_is_kept(tmp_path, here, capsys):
    targets = yaml.safe_load(here.read_text())
    targets["profiles"]["cluster"]["worktree"] = "snapshot"
    here.write_text(yaml.safe_dump(targets))
    root = repo(tmp_path, session(), [1])
    components = root / "my_project" / "components.py"
    components.write_text(components.read_text() + "\nprint('SNAPSHOT-MARKER', flush=True)\n")
    head = git(root, "rev-parse", "HEAD").strip()

    batch_dir, receipt = launch(root, here, capsys)
    ref = f"{SNAPSHOT_REFS}/{batch_dir.name}"
    snapshot = git(root, "rev-parse", ref).strip()
    assert receipt["base_sha"] == head and receipt["snapshot_ref"] == ref
    assert receipt["sha"] == snapshot != head
    assert git(root, "diff", "--name-only", head, ref).split() == ["my_project/components.py"]
    wait_for(lambda: record(batch_dir, "cpu/seeds/000"), "the task's record")
    task_record = record(batch_dir, "cpu/seeds/000")
    assert task_record["exit_status"] == 0
    assert task_record["sha"] == snapshot and task_record["base_sha"] == head
    log = next((batch_dir / "logs" / "cpu").glob("*_0.out")).read_text()
    assert "SNAPSHOT-MARKER" in log
    assert git(root, "rev-parse", "HEAD").strip() == head                # nothing committed


def test_a_tracked_file_that_gitignore_now_matches_stays_in_the_snapshot(tmp_path):
    # `git add -A` alone skips ignored files; tracked ones are kept because
    # the snapshot starts from the user's index.
    root = make_repo(tmp_path)
    (root / ".gitignore").write_text("configs/\n")
    plan = compile_repo(root, profile_data=_snapshot_profile()).plan
    assert "configs/train.yaml" in git(root, "ls-tree", "-r", "--name-only", plan.sha)


def test_a_snapshot_carries_heads_author_committer_and_dates(tmp_path):
    root = make_repo(tmp_path)
    (root / "my_project" / "__init__.py").write_text("CHANGED = True\n")
    plan = compile_repo(root, profile_data=_snapshot_profile())
    people = "%an <%ae> %ad | %cn <%ce> %cd"
    assert git(root, "log", "-1", f"--format={people}", plan.plan.sha) == \
        git(root, "log", "-1", f"--format={people}", "HEAD")
