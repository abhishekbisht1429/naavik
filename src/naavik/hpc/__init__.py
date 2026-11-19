"""Slurm support for the launcher (`naavik.launcher`).

Everything that does not depend on a scheduler (manifests, the compiler,
plans, ports, receipts, status, resume, fetch, the remote task wrapper) lives
in `naavik.launcher`. This package holds only what is specific
to Slurm clusters: the dialect (`slurm`) and the submitter and job queries
(`slurm_submitter`). Two things stay here for compatibility although they are
not Slurm-specific: `task` (the remote task entry point, whose module name
is part of the execution protocol) and `__main__` (`python -m
naavik.hpc`). The standard-library job probe (`SlurmJobProbe`) stays
in `launcher/remote/runtime.py`, which the bootstrap bundles as one
standard-library file.

The module paths of earlier versions (`naavik.hpc.compile...`,
`naavik.hpc.services...`, ...) still import: each resolves to
the same module object in `naavik.launcher`, so code that
imports or monkeypatches them keeps working. They are deprecated.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

from naavik.launcher import EXECUTION_PROTOCOL_VERSION
from naavik.launcher.errors import HpcError, PlanError

__all__ = ["EXECUTION_PROTOCOL_VERSION", "HpcError", "PlanError"]

_PREFIX = __name__ + "."
#: Top-level names that moved to the launcher unchanged.
_MOVED = frozenset({
    "compile", "config", "errors", "framework", "names", "ports", "remote",
    "scheduler", "services",
})
#: Modules that moved to a different place (old name below this package ->
#: new full name).
_RENAMED = {
    "scheduler.slurm": "naavik.hpc.slurm",
    "scheduler.slurm_submitter": "naavik.hpc.slurm_submitter",
}


def _new_name(fullname: str) -> str | None:
    if not fullname.startswith(_PREFIX):
        return None
    relative = fullname[len(_PREFIX):]
    if relative in _RENAMED:
        return _RENAMED[relative]
    if relative.split(".", 1)[0] in _MOVED:
        return "naavik.launcher." + relative
    return None


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Imports an old module path as the module at its new path (the same
    object, so attributes set through either name are seen by both)."""

    def find_spec(self, fullname, path=None, target=None):
        new = _new_name(fullname)
        if new is None or importlib.util.find_spec(new) is None:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        new = _new_name(spec.name)
        if spec.name.count(".") == 2:      # warn once per old top-level name
            warnings.warn(
                f"{spec.name} moved to {new}; the old path is deprecated",
                DeprecationWarning, stacklevel=2,
            )
        return importlib.import_module(new)

    def exec_module(self, module):
        """Already executed under its new name."""


if not any(isinstance(finder, _Alias) for finder in sys.meta_path):
    sys.meta_path.insert(0, _Alias())
