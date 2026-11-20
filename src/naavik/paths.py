"""Where naavik keeps its files, and the environment variables it reads,
under their current names and their earlier ones (`training-framework`,
`TRAINING_FRAMEWORK_*`).

Directories are `naavik/` under each XDG base (config, state, data, cache).
One that exists only under its earlier name, `training-framework/`, is used
where it is, with one warning: nothing is moved or copied, so hosts files,
receipts, run records, and caches are never lost. With both present, the
`naavik/` one is used and the warning says which was ignored.

Variables are `NAAVIK_<NAME>`. `TRAINING_FRAMEWORK_<NAME>` is read when the
new one is not set, with one warning; both set to different values is an
error.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

APP = "naavik"
EARLIER_APP = "training-framework"
VARIABLE_PREFIX = "NAAVIK_"
EARLIER_VARIABLE_PREFIX = "TRAINING_FRAMEWORK_"

#: XDG kind -> (variable, default under the home directory).
_BASES = {
    "config": ("XDG_CONFIG_HOME", (".config",)),
    "state": ("XDG_STATE_HOME", (".local", "state")),
    "data": ("XDG_DATA_HOME", (".local", "share")),
    "cache": ("XDG_CACHE_HOME", (".cache",)),
}
_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        warnings.warn(message, FutureWarning, stacklevel=3)


def xdg_base(kind: str) -> Path:
    variable, default = _BASES[kind]
    return Path(os.environ.get(variable) or os.path.join(os.path.expanduser("~"), *default))


def app_dir(kind: str) -> Path:
    """naavik's directory under the XDG base `kind`: `naavik/`, or the
    earlier `training-framework/` when only that one exists."""
    base = xdg_base(kind)
    current, earlier = base / APP, base / EARLIER_APP
    if earlier.is_dir() and not current.exists():
        _warn_once(str(earlier), f"using {earlier}, naavik's directory under its earlier name; "
                                 f"rename it to {current}")
        return earlier
    if earlier.is_dir():
        _warn_once(str(earlier), f"both {current} and {earlier} (the earlier name) exist; "
                                 f"{current} is used and {earlier} ignored: move what you need")
    return current


def app_dir_candidates(kind: str) -> tuple[Path, Path]:
    """Both names of the directory, current first, for reading what either
    holds (jobs started before the rename)."""
    base = xdg_base(kind)
    return base / APP, base / EARLIER_APP


def variable_name(name: str) -> str:
    """The variable a message should name: the one set (current or
    earlier), else the current one."""
    if os.environ.get(VARIABLE_PREFIX + name) is None and \
            os.environ.get(EARLIER_VARIABLE_PREFIX + name) is not None:
        return EARLIER_VARIABLE_PREFIX + name
    return VARIABLE_PREFIX + name


def env(name: str) -> str | None:
    """The value of `NAAVIK_<name>`, else of `TRAINING_FRAMEWORK_<name>`
    (with one warning); None when neither is set."""
    current, earlier = VARIABLE_PREFIX + name, EARLIER_VARIABLE_PREFIX + name
    value, old = os.environ.get(current), os.environ.get(earlier)
    if value is not None and old is not None and value != old:
        raise ValueError(f"{current}={value!r} and {earlier}={old!r} are both set: they are one "
                         f"setting under its current and earlier name; unset {earlier}")
    if value is None and old is not None:
        _warn_once(earlier, f"{earlier} is the earlier name of {current}; set {current} instead")
        return old
    return value
