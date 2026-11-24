"""`naavik init`: the two files `naavik run` needs, written once and
explicitly, for this machine.

- `naavik-project.yaml` at the repository root: where the framework comes
  from (`code.framework_from`), what goes on `PYTHONPATH`
  (`code.python_paths`), and the host `naavik run` uses
  (`quick_run_defaults.host`). No `per_experiment`: on this machine's
  builtin scheduler an experiment has no time, CPU, or memory limit, and 0
  GPUs unless `naavik run --gpus N`.
- A host for this machine in the hosts file: `connection: local`, the
  builtin scheduler with the GPU pool, `run_code_from: working_tree`, the
  environment running naavik, and absolute directories.

Nothing is inferred later: what is detected here (the framework, the python
paths) is written down, and the files are read back with the parsers
`naavik run` uses. Files are never overwritten; a host is added to an
existing hosts file only as text under `hosts:` (comments and layout kept),
and only when the result parses to the same hosts plus the new one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

import naavik
from naavik.distribution import installed_version
from naavik.launcher.config.schema import PROJECT_FILE, parse_profiles, parse_project
from naavik.launcher.errors import HpcError

DEFAULT_HOST = "here"
#: Directories at the repository root that hold code but do not decide its
#: layout (a src-layout project often has a `tests` package at the root).
_NOT_LAYOUT = frozenset({"tests", "test", "docs", "examples"})
_HOSTS_LINE = re.compile(r"\Ahosts:[ \t]*(#.*)?\Z")
_HOSTS_INLINE = re.compile(r"\Ahosts:[ \t]*[\[{]")


@dataclass(frozen=True)
class Detected:
    framework_from: str
    #: Why: shown to the user.
    framework_reason: str
    python_paths: tuple[str, ...]
    python_paths_reason: str


def detect(root: Path, python_paths: tuple[str, ...] | None) -> Detected:
    """Where the framework comes from, and the python paths (`python_paths`
    when given), each with the reason."""
    running = Path(naavik.__file__).resolve()
    if root.resolve() in running.parents:
        framework, why = "this_repository", "this repository is naavik (the one running)"
    else:
        version = installed_version()
        if version is None:
            raise HpcError(
                "naavik is not installed in this environment, and this repository is not "
                "naavik itself; experiments run in this environment "
                "(python_environment.use_existing: current), so install it first: "
                "pip install naavik")
        framework, why = "installed", f"naavik {version} is installed in this environment"
    if python_paths is not None:
        return Detected(framework, why, python_paths, "given with --python-paths")
    at_root = _packages(root, skip=_NOT_LAYOUT | {"src"})
    in_src = _packages(root / "src")
    if in_src and not at_root:
        paths, reason = ("src",), f"src/ holds the package(s) {', '.join(in_src)}"
    elif at_root and not in_src:
        paths, reason = (".",), f"the repository root holds the package(s) {', '.join(at_root)}"
    else:
        where = ("both the repository root and src/ hold packages" if at_root
                 else "neither the repository root nor src/ holds a package")
        raise HpcError(f"cannot tell what goes on PYTHONPATH: {where}; pass "
                       "--python-paths (e.g. --python-paths . or --python-paths src)")
    return Detected(framework, why, paths, reason)


def _packages(directory: Path, skip=frozenset()) -> list[str]:
    if not directory.is_dir():
        return []
    return sorted(child.name for child in directory.iterdir()
                  if child.is_dir() and not child.name.startswith(".")
                  and child.name not in skip and (child / "__init__.py").is_file())


# What is written -----------------------------------------------------------------------


def project_text(detected: Detected, host: str) -> str:
    paths = ", ".join(f'"{path}"' for path in detected.python_paths)
    text = (
        f"# Written by `naavik init` on {date.today().isoformat()}; edit freely.\n"
        "code:\n"
        f"  framework_from: {detected.framework_from}"
        f"{'  # naavik comes from the environment' if detected.framework_from == 'installed' else ''}\n"
        f"  python_paths: [{paths}]   # put on PYTHONPATH for experiments\n"
        "quick_run_defaults:\n"
        f"  host: {host}   # the host `naavik run` uses without --host\n"
        "  # per_experiment: {gpus: 1, walltime: \"02:00:00\"}   # none: 0 GPUs, no limits\n"
    )
    parse_project(yaml.safe_load(text), PROJECT_FILE)     # never write what run refuses
    return text


def gpus_text(gpus) -> str:
    """`scheduler.gpus` as YAML: `all`, a count, or `[0, 1]`."""
    if isinstance(gpus, (list, tuple)):
        return "[" + ", ".join(str(index) for index in gpus) + "]"
    return str(gpus)


def host_lines(name: str, gpus, directories: dict[str, str], unit: int) -> list[str]:
    """The host's entry, as lines of text indented by `unit` spaces per level
    (its key at one level)."""
    one, two, three = " " * unit, " " * (2 * unit), " " * (3 * unit)
    return [
        f"{one}{name}:   # this machine, added by `naavik init` on {date.today().isoformat()}",
        f"{two}connection: local",
        f"{two}scheduler: {{kind: builtin, gpus: {gpus_text(gpus)}}}"
        "   # GPUs: all | a count | [indices]",
        f"{two}run_code_from: working_tree   # uncommitted work runs; a snapshot keeps what ran",
        f"{two}python_environment: {{use_existing: current}}   # the environment running naavik",
        f"{two}directories:",
        *(f"{three}{key}: {value}" for key, value in directories.items()),
    ]


def directories_under(base: Path) -> dict[str, str]:
    return {"missions": str(base / "missions"),
            "python_environments": str(base / "envs"),
            "datasets": str(base / "datasets")}


def new_hosts_text(name: str, gpus, directories: dict[str, str]) -> str:
    text = "\n".join([
        f"# Written by `naavik init` on {date.today().isoformat()}; edit freely.",
        f"default_host: {name}",
        "hosts:",
        *host_lines(name, gpus, directories, 2),
    ]) + "\n"
    check_host(text, name, "hosts.yaml")
    return text


def check_host(text: str, name: str, source: str) -> None:
    """Refuse a hosts file in which host `name` does not parse."""
    parse_profiles(yaml.safe_load(text), name, source)


@dataclass(frozen=True)
class Insertion:
    #: The file with the host added, or None when it cannot be done safely.
    text: str | None
    #: Why not (None when `text` is set).
    reason: str | None = None


def insert_host(existing: str, name: str, gpus, directories: dict[str, str],
                source: str) -> Insertion:
    """`existing` with host `name` added right under its `hosts:` line, as
    text, indented like the entries there; only if the result parses to the
    same file plus that host."""
    try:
        before = yaml.safe_load(existing) or {}
    except yaml.YAMLError as error:
        return Insertion(None, f"{source} is not valid YAML ({error})")
    hosts = before.get("hosts") if isinstance(before, dict) else None
    if isinstance(hosts, dict) and name in hosts:
        raise HpcError(f"{source} already has a host named '{name}'; pick another name "
                       "with --host")
    lines = existing.splitlines()
    index = next((i for i, line in enumerate(lines) if _HOSTS_LINE.match(line)), None)
    if index is None:
        inline = any(_HOSTS_INLINE.match(line) for line in lines)
        return Insertion(None, f"its hosts are written inline (hosts: {{...}})" if inline
                         else "it has no `hosts:` mapping (it may use the earlier names)")
    unit = _child_indent(lines[index + 1:])
    if unit is None:
        return Insertion(None, "its hosts are indented with tabs")
    block = host_lines(name, gpus, directories, unit)
    text = "\n".join(lines[:index + 1] + block + lines[index + 1:]) + "\n"
    after = yaml.safe_load(text)
    unchanged = ({**after, "hosts": {k: v for k, v in after["hosts"].items() if k != name}}
                 == {**before, "hosts": hosts or {}})
    if not unchanged or name not in after["hosts"]:
        return Insertion(None, "adding the host as text would change its other entries")
    check_host(text, name, source)
    return Insertion(text)


def _child_indent(lines: list[str]) -> int | None:
    """The indentation of the first entry under `hosts:` (2 when it has
    none); None for tabs."""
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        leading = line[:len(line) - len(line.lstrip())]
        if "\t" in leading:
            return None
        return len(leading) or 2
    return 2


def parse_gpus(text: str):
    """`--gpus`: `all`, `none`, a count, or indices `0,1`."""
    value = text.strip()
    if value == "all":
        return "all"
    if value == "none":
        return []
    if value.isdigit():
        return int(value)
    parts = [part.strip() for part in value.split(",")]
    if parts and all(part.isdigit() for part in parts):
        return [int(part) for part in parts]
    raise HpcError(f"--gpus must be all, none, a count, or GPU indices such as 0,1; got {text!r}")
