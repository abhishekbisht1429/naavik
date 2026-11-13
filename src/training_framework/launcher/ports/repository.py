"""`RepositoryReader`: everything the HPC layer reads from Git.

Files are read at the resolved commit, never from the worktree.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from training_framework.launcher.errors import HpcError


#: Where `launch` keeps each snapshot batch's commit, so it stays reachable.
SNAPSHOT_REFS = "refs/naavik/snapshots"


@dataclass(frozen=True)
class Snapshot:
    """The working tree as a commit. `snapshot` False: the tree is clean and
    `sha` is HEAD itself."""

    sha: str
    base: str
    modified: int = 0
    untracked: int = 0
    untracked_bytes: int = 0
    snapshot: bool = False

    def note(self) -> str:
        return (f"snapshot of the working tree on {self.base[:12]}: {self.modified} modified, "
                f"{self.untracked} untracked file(s) ({_human_size(self.untracked_bytes)}); "
                f"kept as {SNAPSHOT_REFS}/<batch-id> by launch")


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024:
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


class RepositoryReader(Protocol):
    root: Path

    def resolve(self, ref: str) -> str: ...

    def is_clean(self) -> bool: ...

    def snapshot(self, limit: int) -> Snapshot: ...

    def read(self, sha: str, path: str) -> bytes | None: ...

    def exists(self, sha: str, path: str) -> bool: ...

    def object_id(self, sha: str, path: str) -> str | None: ...

    def is_directory(self, sha: str, path: str) -> bool: ...


class GitRepository:
    """`RepositoryReader` over the `git` executable."""

    def __init__(self, start: Path):
        result = self._git(["rev-parse", "--show-toplevel"], cwd=start)
        if result.returncode != 0:
            raise HpcError(f"{start} is not inside a Git repository")
        self.root = Path(result.stdout.decode().strip())

    @staticmethod
    def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, check=False,
        )

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        return self._git(args, self.root)

    def resolve(self, ref: str) -> str:
        result = self._run(["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"])
        if result.returncode != 0:
            raise HpcError(f"cannot resolve '{ref}' to a commit")
        return result.stdout.decode().strip()

    def is_clean(self) -> bool:
        result = self._run(["status", "--porcelain", "--untracked-files=normal"])
        if result.returncode != 0:
            raise HpcError(f"git status failed: {result.stderr.decode().strip()}")
        return not result.stdout.strip()

    def read(self, sha: str, path: str) -> bytes | None:
        result = self._run(["show", f"{sha}:{path}"])
        return result.stdout if result.returncode == 0 else None

    @staticmethod
    def _spec(sha: str, path: str) -> str:
        # Git names the root tree `<sha>:`, not `<sha>:.`.
        path = path.strip("/")
        return f"{sha}:" if path in ("", ".") else f"{sha}:{path}"

    def exists(self, sha: str, path: str) -> bool:
        return self._run(["cat-file", "-e", self._spec(sha, path)]).returncode == 0

    def is_directory(self, sha: str, path: str) -> bool:
        result = self._run(["cat-file", "-t", self._spec(sha, path)])
        return result.returncode == 0 and result.stdout.decode().strip() == "tree"

    def object_id(self, sha: str, path: str) -> str | None:
        """Git's content hash of a file or directory at `sha`."""
        result = self._run(["rev-parse", "--verify", "--quiet", self._spec(sha, path)])
        return result.stdout.decode().strip() if result.returncode == 0 else None

    def _must(self, args: list[str], what: str, env: dict | None = None) -> str:
        result = subprocess.run(["git", *args], cwd=self.root, capture_output=True, check=False,
                                env=None if env is None else {**os.environ, **env})
        if result.returncode != 0:
            raise HpcError(f"{what} failed: {result.stderr.decode().strip()}")
        return result.stdout.decode()

    def snapshot(self, limit: int) -> Snapshot:
        """The working tree as a commit on HEAD: tracked changes and untracked
        files that `.gitignore` does not exclude. Made with a temporary index,
        so the user's index, branch, and files are untouched; author,
        committer, and dates are HEAD's, so the same tree on the same HEAD
        always gives the same commit. Refused when the untracked files exceed
        `limit` bytes, or a submodule has changes (only its recorded commit
        would be taken)."""
        head = self.resolve("HEAD")
        problems = self._submodule_changes()
        if problems:
            raise HpcError("a snapshot cannot take submodule changes: " + "; ".join(problems)
                           + " (commit them, or use worktree: commit)")
        untracked = [name for name in self._must(
            ["ls-files", "--others", "--exclude-standard", "-z"], "git ls-files").split("\0") if name]
        sizes = {name: os.path.getsize(self.root / name) for name in untracked
                 if (self.root / name).is_file()}
        total = sum(sizes.values())
        if total > limit:
            largest = sorted(sizes.items(), key=lambda item: -item[1])[:5]
            raise HpcError(
                f"untracked files ({_human_size(total)}) exceed snapshot_limit "
                f"({_human_size(limit)}); add them to .gitignore or raise the limit. Largest: "
                + ", ".join(f"{name} ({_human_size(size)})" for name, size in largest))
        index = self._must(["rev-parse", "--git-path", "index"], "git rev-parse").strip()
        index = index if os.path.isabs(index) else str(self.root / index)
        with tempfile.TemporaryDirectory() as directory:
            temporary = os.path.join(directory, "index")
            if os.path.exists(index):
                shutil.copyfile(index, temporary)
            env = {"GIT_INDEX_FILE": temporary}
            self._must(["add", "-A"], "git add (temporary index)", env)
            tree = self._must(["write-tree"], "git write-tree", env).strip()
            changed = [name for name in self._must(
                ["diff-index", "--cached", "--name-only", "-z", "HEAD"], "git diff-index",
                env).split("\0") if name]
        if tree == self._must(["rev-parse", f"{head}^{{tree}}"], "git rev-parse").strip():
            return Snapshot(head, head)
        fields = self._must(["log", "-1", "--format=%an%x00%ae%x00%ad%x00%cn%x00%ce%x00%cd",
                             "--date=raw", head], "git log").rstrip("\n").split("\0")
        identity = dict(zip(("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_AUTHOR_DATE",
                             "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL", "GIT_COMMITTER_DATE"),
                            fields))
        sha = self._must(["commit-tree", tree, "-p", head, "-m",
                          f"naavik: snapshot of the working tree on {head[:12]}"],
                         "git commit-tree", identity).strip()
        new = set(untracked)
        return Snapshot(sha, head, modified=len([name for name in changed if name not in new]),
                        untracked=len(new), untracked_bytes=total, snapshot=True)

    def _submodule_changes(self) -> list[str]:
        """Submodules whose checkout differs from what the tree records."""
        if not (self.root / ".gitmodules").exists():
            return []
        found: dict[str, str] = {}
        paths = set()
        # Lines: <flag><sha> <path> [(<describe>)]; flag `+` = another commit.
        for line in self._must(["submodule", "status", "--recursive"],
                               "git submodule status").splitlines():
            parts = line[1:].split()
            if len(parts) < 2:
                continue
            paths.add(parts[1])
            if line.startswith("+"):
                found[parts[1]] = "is at another commit than recorded"
        for line in self._must(["status", "--porcelain", "--ignore-submodules=none"],
                               "git status").splitlines():
            path = line[3:]
            if path in paths and path not in found:
                found[path] = "has uncommitted changes"
        return [f"{path} {reason}" for path, reason in found.items()]

    def keep_snapshot(self, sha: str, batch_id: str) -> str:
        """Keep a snapshot batch's commit reachable; returns the ref."""
        ref = f"{SNAPSHOT_REFS}/{batch_id}"
        self._must(["update-ref", ref, sha], "git update-ref")
        return ref

    @staticmethod
    def bundle_ref(sha: str) -> str:
        return f"refs/hpc/stage-{sha}"

    def create_bundle(self, sha: str, path: str) -> None:
        """A bundle holding `sha` (`git bundle` needs a ref, so a temporary
        one is created and always deleted)."""
        ref = self.bundle_ref(sha)
        result = self._run(["update-ref", ref, sha])
        if result.returncode != 0:
            raise HpcError(f"git update-ref failed: {result.stderr.decode().strip()}")
        try:
            result = self._run(["bundle", "create", "--quiet", path, ref])
            if result.returncode != 0:
                raise HpcError(f"git bundle failed: {result.stderr.decode().strip()}")
        finally:
            self._run(["update-ref", "-d", ref])

