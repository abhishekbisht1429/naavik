"""`training_framework` is the earlier name of `naavik`; deprecated.

Importing `training_framework` gives the `naavik` package itself, and every
`training_framework.X` resolves to the same module object as `naavik.X`, so
code that imports or monkeypatches either name sees the other's changes.
What keeps working through it:
- imports and `components_package` strings in configs;
- legacy single-file checkpoints, which are pickles of module paths;
- `python -m training_framework...` commands that earlier missions recorded
  in their job scripts and plans.

One `FutureWarning` per process says so.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

_OLD = "training_framework"
_NEW = "naavik"


def _new_name(fullname: str) -> str | None:
    if fullname.startswith(_OLD + "."):
        return _NEW + fullname[len(_OLD):]
    return None


class _EarlierName(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Imports `training_framework.X` as the module `naavik.X` (the same
    object). It also lends `runpy` the new module's code, so `python -m
    training_framework.X` runs `naavik.X` (a package runs its `__main__`)."""

    @staticmethod
    def _spec_of(fullname):
        new = _new_name(fullname)
        return None if new is None else importlib.util.find_spec(new)

    def find_spec(self, fullname, path=None, target=None):
        new = self._spec_of(fullname)
        if new is None:
            return None
        # The new module's file as origin: `python -m` sets `sys.argv[0]`
        # (and `__file__`) from it.
        return importlib.util.spec_from_loader(fullname, self, origin=new.origin)

    def create_module(self, spec):
        return importlib.import_module(_new_name(spec.name))

    def exec_module(self, module):
        """Already executed under its new name."""

    def is_package(self, fullname):
        spec = self._spec_of(fullname)
        return spec is not None and spec.submodule_search_locations is not None

    def get_code(self, fullname):
        spec = self._spec_of(fullname)
        return spec.loader.get_code(spec.name)


if not any(isinstance(finder, _EarlierName) for finder in sys.meta_path):
    sys.meta_path.insert(0, _EarlierName())

def _run_as_a_command() -> bool:
    """Whether this import is `python -m training_framework...`: a command a
    job script recorded, which no one edits; warning there would only put
    a line in every job's log."""
    frame = sys._getframe(1)
    while frame is not None:
        name = frame.f_code.co_filename
        if name == "<frozen runpy>" or name.endswith(("/runpy.py", "\\runpy.py")):
            return True
        frame = frame.f_back
    return False


if not _run_as_a_command():
    warnings.warn(
        "training_framework is now called naavik: import naavik instead "
        "(the earlier name is deprecated)",
        FutureWarning,
        stacklevel=2,
    )

# `import training_framework` gives `naavik` itself.
sys.modules[__name__] = importlib.import_module(_NEW)
