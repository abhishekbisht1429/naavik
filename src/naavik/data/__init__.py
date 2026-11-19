"""Data a component needs, declared by the component, obtained in one place.

A dataset component declares, from its config alone (nothing is built):

```python
@classmethod
def data_requirements(cls, config) -> tuple[DataRequirement, ...]   # config: parsed schema
@classmethod
def fetch_data(cls, config, requirement, path) -> None              # path: the config field's value
@classmethod
def verify_data(cls, config, requirement, path) -> list[str]        # problems; cheap
```

A component that declares nothing is not managed (it reads what is there,
as before). `ensure` obtains a requirement: a ready marker keyed by the
requirement's content key means done; otherwise, under a lease on the
location, it fetches, verifies, and writes the marker. The target's
prepare step (`python -m naavik.data fetch`) and direct engine
runs use it alike, so two runs never fetch the same data twice or at once.
Nothing is ever deleted.

The content key is the requirement's name, version, and `key` -- never the
code or the environment: a new commit does not fetch again.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from naavik.launcher.names import check_name, hash_value

__all__ = ["DataError", "DataRequirement", "Located", "ensure", "requirement_key",
           "requirements_of", "verify"]


class DataError(Exception):
    """A requirement that is declared wrongly, cannot be obtained, or fails
    its verification."""


@dataclass(frozen=True)
class DataRequirement:
    #: What the target's `data:` modes name ("cifar10").
    name: str
    #: The config field holding where the data lives ("root").
    path: str
    #: Changed whenever the content changes.
    version: str
    #: Anything else that identifies the content (iNaturalist: its split).
    key: Mapping = field(default_factory=dict)
    #: Where, under the field's path, this requirement's content is ("" =
    #: the path itself). Requirements of different keys under one path must
    #: own different subpaths (their markers and leases are there).
    subpath: str = ""
    #: False: the data is obtained by hand (`upload` or `present` it).
    fetchable: bool = True
    #: Bytes, roughly; shown before an upload or fetch.
    size_hint: int | None = None

    def __post_init__(self):
        if problem := check_name(self.name, "a data requirement's name"):
            raise DataError(problem)
        for what in ("path", "version"):
            value = getattr(self, what)
            if not isinstance(value, str) or not value:
                raise DataError(f"data requirement '{self.name}': {what} must be a non-empty "
                                f"string; got {value!r}")
        if not isinstance(self.key, Mapping):
            raise DataError(f"data requirement '{self.name}': key must be a mapping")
        try:
            json.dumps(dict(self.key), sort_keys=True)
        except (TypeError, ValueError) as error:
            raise DataError(f"data requirement '{self.name}': key must be JSON ({error})") from None
        if not isinstance(self.subpath, str) or os.path.isabs(self.subpath) or ".." in \
                self.subpath.split("/"):
            raise DataError(f"data requirement '{self.name}': subpath must be a relative path "
                            f"inside {self.path}; got {self.subpath!r}")
        if not isinstance(self.fetchable, bool):
            raise DataError(f"data requirement '{self.name}': fetchable must be a boolean")
        if self.size_hint is not None and (isinstance(self.size_hint, bool)
                                           or not isinstance(self.size_hint, int)
                                           or self.size_hint < 0):
            raise DataError(f"data requirement '{self.name}': size_hint must be a byte count")

    def as_json(self) -> dict:
        return {"name": self.name, "path": self.path, "version": self.version,
                "key": dict(self.key), "subpath": self.subpath,
                "fetchable": self.fetchable, "size_hint": self.size_hint}


def requirement_key(requirement: DataRequirement) -> str:
    """The content key: equal for the same content, whatever code asks."""
    return hash_value({"data_requirement": requirement.name, "version": requirement.version,
                       "key": dict(requirement.key)})


@dataclass(frozen=True)
class Located:
    """A requirement of one component config, with where it is."""

    requirement: DataRequirement
    #: The value of `requirement.path` in the config.
    path: str
    #: Where its content (marker, lease) is: `path`/`subpath`.
    location: str
    #: The parsed config (`config_schema`) or the mapping.
    config: Any


def declares_data(component_class: type) -> bool:
    return callable(getattr(component_class, "data_requirements", None))


def requirements_of(component_class: type, config: Mapping | None) -> tuple[Located, ...]:
    """The requirements of `component_class` with `config`; () when it
    declares none. Raises DataError for a wrong declaration."""
    if not declares_data(component_class):
        return ()
    name = getattr(component_class, "__qualname__", repr(component_class))
    missing = [method for method in ("fetch_data", "verify_data")
               if not callable(getattr(component_class, method, None))]
    if missing:
        raise DataError(f"{name} declares data_requirements but not {', '.join(missing)}")
    parsed = config
    if getattr(component_class, "config_schema", None) is not None:
        from naavik.components.config_schema import parse_component_config

        parsed = parse_component_config(component_class, config or {})
    declared = component_class.data_requirements(parsed)
    if not isinstance(declared, tuple) or not all(isinstance(item, DataRequirement)
                                                  for item in declared):
        raise DataError(f"{name}.data_requirements must return a tuple of DataRequirement; "
                        f"got {declared!r}")
    found, seen = [], set()
    for requirement in declared:
        if requirement.name in seen:
            raise DataError(f"{name} declares data requirement '{requirement.name}' twice")
        seen.add(requirement.name)
        path = _field(parsed, requirement.path)
        if not isinstance(path, str) or not path:
            raise DataError(f"{name}: data requirement '{requirement.name}' is at config field "
                            f"'{requirement.path}', which holds {path!r}, not a path")
        location = os.path.join(path, requirement.subpath) if requirement.subpath else path
        found.append(Located(requirement, path, location.rstrip("/") or "/", parsed))
    return tuple(found)


def _field(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def verify(component_class: type, located: Located) -> list[str]:
    problems = component_class.verify_data(located.config, located.requirement, located.path)
    if not isinstance(problems, list) or not all(isinstance(p, str) for p in problems):
        raise DataError(f"{component_class.__qualname__}.verify_data must return a list of "
                        f"strings; got {problems!r}")
    return problems


def ensure(component_class: type, located: Located, *, notify: Callable[[str], None] = print,
           scheduler: str | None = None, filesystem=None, lease_held: bool = False) -> str:
    """Make `located` ready; returns what happened: `ready` (already),
    `fetched`, or `present` (obtained by hand and verified). `scheduler`:
    the batch scheduler this runs under (its job owns the lease), or None
    for a process of its own. `lease_held`: the caller holds the lease on
    the location already (a batch's prepare step), so none is taken."""
    from naavik.launcher.remote.lease import Lease, check_marker, write_marker
    from naavik.launcher.remote.runtime import (
        ProcessProbe,
        StdlibFilesystem,
        current_owner,
        current_process_owner,
        owner_probe,
    )

    requirement, location = located.requirement, located.location
    label = f"{requirement.name} {requirement.version} at {location}"
    filesystem = filesystem or StdlibFilesystem()
    key = requirement_key(requirement)
    if not requirement.fetchable:
        # Obtained by hand: verified each time (markers are for what we fetch).
        problems = verify(component_class, located)
        if problems:
            raise DataError(f"{label} cannot be fetched ({_by_hand(requirement)}) and is not "
                            "there: " + "; ".join(problems))
        return "present"
    if check_marker(filesystem, location, key):
        return "ready"
    if lease_held:
        return _fetch(component_class, located, label, key, filesystem, write_marker, notify)
    owner = current_owner(scheduler) if scheduler else current_process_owner()
    probe = owner_probe(scheduler) if scheduler else ProcessProbe()
    lease = Lease(filesystem, probe, location, owner, notify=notify)
    lease.acquire()
    try:
        if check_marker(filesystem, location, key):
            notify(f"{label}: fetched meanwhile by another run")
            return "ready"
        return _fetch(component_class, located, label, key, filesystem, write_marker, notify)
    finally:
        lease.release()


def _fetch(component_class, located, label, key, filesystem, write_marker, notify) -> str:
    notify(f"{label}: fetching")
    component_class.fetch_data(located.config, located.requirement, located.path)
    problems = verify(component_class, located)
    if problems:
        raise DataError(f"{label}: fetched, but it is not right: " + "; ".join(problems))
    write_marker(filesystem, located.location, key)
    notify(f"{label}: ready")
    return "fetched"


def _by_hand(requirement: DataRequirement) -> str:
    return "it is obtained by hand: place it there, or upload it"
