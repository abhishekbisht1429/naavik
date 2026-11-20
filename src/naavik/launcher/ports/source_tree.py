"""A batch's exact tree, exported once per commit for checks that run its
code on this machine (never the working tree, which may differ).

`$XDG_CACHE_HOME/naavik/trees/<sha>`: written into a temporary
directory and renamed into place, so a check running at the same time never
sees a half-written tree, and a commit is exported at most once.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from naavik.launcher.ports.local import LocalSourceStager


def trees_directory() -> Path:
    from naavik.paths import app_dir

    return app_dir("cache") / "trees"


def source_tree(repository_root: Path, sha: str) -> Path:
    """The tree of commit `sha` of the repository at `repository_root`."""
    final = trees_directory() / sha
    if final.is_dir():
        return final
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=final.parent, prefix=f".{sha[:12]}-"))
    try:
        target = staging / "tree"
        LocalSourceStager(Path(repository_root)).stage(sha, str(target))
        try:
            os.rename(target, final)
        except OSError:
            if not final.is_dir():         # not another export that won the race
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return final
