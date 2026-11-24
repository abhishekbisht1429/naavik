"""`naavik runs sync`: copy the runs recorded on targets into this registry.

Records are written once (or only appended, by their one writer), and ids
are unique, so a copy never conflicts with a run of this machine: files of
another machine's runs are only ever written here by sync. Nothing here is
deleted. A target on this machine shares this registry: nothing to copy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

#: Prints the registry directory as `registry_dir` computes it, run with the
#: target's `python3` (standard library only).
_REMOTE_REGISTRY = r"""
import os
configured = os.environ.get("NAAVIK_RUNS")
if configured is None:
    configured = os.environ.get("TRAINING_FRAMEWORK_RUNS")   # the earlier name
if configured == "off":
    print("")
elif configured:
    print(configured)
else:
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    current, earlier = os.path.join(base, "naavik"), os.path.join(base, "training-framework")
    used = earlier if os.path.isdir(earlier) and not os.path.exists(current) else current
    print(os.path.join(used, "runs"))
"""


def _target_names(args) -> tuple[Path, list[str]]:
    """The hosts file and the hosts to sync: `--all-hosts`, the `--host`s,
    else `default_host`. A file in the earlier names (`targets` /
    `profiles`, `default`) is read as the current one, as `parse_profiles`
    reads it."""
    from naavik.launcher.__main__ import default_profiles_path
    from naavik.launcher.config.names import hosts_file_from_earlier

    path = args.profiles or default_profiles_path()
    if not path.exists():
        raise _error(f"hosts file {path} does not exist")
    raw, _, problems = hosts_file_from_earlier(yaml.safe_load(path.read_text()) or {}, str(path))
    if problems:
        raise _error("; ".join(problems))
    entries = raw.get("hosts") if isinstance(raw, dict) else None
    if not isinstance(entries, dict) or not entries:
        raise _error(f"{path} has no hosts")
    if args.all_targets:
        return path, sorted(entries)
    if args.profile:
        return path, list(args.profile)
    if raw.get("default_host"):
        return path, [raw["default_host"]]
    raise _error(f"{path} has no default_host; pass --host or --all-hosts")


def _error(message: str):
    from naavik.runs.__main__ import RunsError

    return RunsError(message)


def sync_targets(args, root: Path) -> int:
    from naavik.launcher.config.schema import parse_profiles
    from naavik.launcher.errors import HpcError
    from naavik.launcher.ports.targets import ports_for, target_of_profile

    path, names = _target_names(args)
    raw = yaml.safe_load(path.read_text())
    failed = 0
    for name in names:
        try:
            target = target_of_profile(parse_profiles(raw, name, str(path)))
            if target.is_local:
                print(f"{name}: this machine; its runs are in this registry already")
                continue
            ports = ports_for(target)
            found = ports.executor.run(["python3", "-c", _REMOTE_REGISTRY])
            remote = found.stdout.strip()
            if found.status != 0 or not remote:
                reason = found.stderr.strip() or "its registry is turned off"
                raise HpcError(f"cannot find {target}'s run registry ({reason})")
            if not ports.filesystem.exists(remote):
                print(f"{name}: no runs recorded on {target} yet ({remote})")
                continue
            copied = ports.transfer.pull_tree(remote, str(root))
            print(f"{name}: {copied} file(s) copied from {target}:{remote}")
        except HpcError as error:
            failed += 1
            print(f"error: {name}: {error}", file=sys.stderr)
    return 1 if failed else 0
