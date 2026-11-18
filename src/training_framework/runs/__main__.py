"""`naavik runs` (also `python -m training_framework.runs`): every recorded
run on this machine, and those synced from targets."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from training_framework.runs import views
from training_framework.runs.registry import find_run, list_runs, registry_dir

EXIT_ERROR = 2


class RunsError(Exception):
    pass


def _parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    def registry_option(command):
        command.add_argument("--registry", type=Path,
                             help="the registry to read (default: $TRAINING_FRAMEWORK_RUNS, "
                                  "else $XDG_DATA_HOME/training-framework/runs)")

    listing = commands.add_parser("list", allow_abbrev=False, help="the recorded runs, newest first")
    listing.add_argument("--since", metavar="AGE", help="only runs started within AGE: 30m, 12h, 7d, 2w")
    listing.add_argument("--state", action="append", default=[],
                         help="only runs in this state (repeatable): done, failed, interrupted, "
                              "running, 'ended without a record', unknown")
    listing.add_argument("--host", action="append", default=[], help="only runs on this host (repeatable)")
    listing.add_argument("--mission", dest="batch",
                         help="only experiments of missions whose id contains MISSION")
    listing.add_argument("--where", action="append", default=[], metavar="KEY=VALUE",
                         help="only runs whose (first session's) config has KEY equal to VALUE "
                              "(dotted path; VALUE read as YAML; repeatable)")
    listing.add_argument("--limit", type=int, default=50, help="at most LIMIT runs (default 50; 0: all)")
    listing.add_argument("--json", action="store_true", help="print JSON instead of a table")
    registry_option(listing)

    show = commands.add_parser("show", allow_abbrev=False, help="one run in detail")
    show.add_argument("run", help="a run id, or a unique beginning of one")
    show.add_argument("--json", action="store_true", help="print the record as JSON")
    registry_option(show)

    diff = commands.add_parser("diff", allow_abbrev=False, help="what differs between two runs")
    diff.add_argument("a", help="a run id (or a unique beginning)")
    diff.add_argument("b", help="another run id (or a unique beginning)")
    diff.add_argument("--all-keys", action="store_true",
                      help="also compare the config keys that differ between runs by construction "
                           "(sessions_dir, the rendezvous address and port)")
    registry_option(diff)

    sync = commands.add_parser("sync", allow_abbrev=False, help="copy the runs recorded on hosts into this registry")
    which = sync.add_mutually_exclusive_group()
    which.add_argument("--host", "--target", "--profile", dest="profile", action="append", default=[],
                       help="this host (repeatable; default: the hosts file's default_host)")
    which.add_argument("--all-hosts", dest="all_targets", action="store_true",
                       help="every host of the hosts file")
    sync.add_argument("--hosts-file", "--targets", "--profiles", dest="profiles", type=Path,
                      help="the hosts file (default: $XDG_CONFIG_HOME/training-framework/hosts.yaml)")
    registry_option(sync)
    return parser


def _root(args) -> Path:
    root = args.registry or registry_dir()
    if root is None:
        raise RunsError("the run registry is turned off (TRAINING_FRAMEWORK_RUNS=off); "
                        "pass --registry")
    return root


def _where(text: str) -> tuple[str, object]:
    key, separator, value = text.partition("=")
    if not separator or not key:
        raise RunsError(f"--where {text!r} must be KEY=VALUE")
    return key.strip(), yaml.safe_load(value)


def _list(args) -> int:
    from training_framework.launcher.services.batches import parse_age

    root = _root(args)
    runs, problems = list_runs(root)
    since = None
    if args.since:
        try:
            since = datetime.now(timezone.utc) - parse_age(args.since)
        except ValueError as error:
            raise RunsError(str(error)) from None
    conditions = [_where(item) for item in args.where]
    chosen = []
    for run in runs:
        if since is not None and views.started(run) < since:
            continue
        if args.state and run.state not in args.state:
            continue
        if args.host and run.record.get("host") not in args.host:
            continue
        if args.batch and args.batch not in str((run.record.get("launch") or {}).get("batch_id") or ""):
            continue
        try:
            if any(views.config_value(run, root, key) != value for key, value in conditions):
                continue
        except KeyError:
            continue
        chosen.append(run)
    if args.limit:
        chosen = chosen[:args.limit]
    if args.json:
        print(json.dumps([views.as_json(run) for run in chosen], indent=2, default=str))
    elif chosen:
        print(views.table(chosen))
    else:
        print(f"no runs{' match' if runs else ''} (registry {root})")
    for problem in problems:
        print(f"problem: {problem}", file=sys.stderr)
    return 0


def _find(root: Path, prefix: str):
    try:
        return find_run(root, prefix)
    except LookupError as error:
        raise RunsError(str(error)) from None


def _show(args) -> int:
    root = _root(args)
    run = _find(root, args.run)
    if args.json:
        print(json.dumps(views.as_json(run), indent=2, default=str))
        return 0
    runs, _ = list_runs(root)
    print(views.detail(run, runs, root))
    return 0


def _diff(args) -> int:
    root = _root(args)
    from training_framework.reporting.summary import DEFAULT_IGNORED_KEYS

    ignored = () if args.all_keys else DEFAULT_IGNORED_KEYS
    print(views.difference(_find(root, args.a), _find(root, args.b), root, ignored))
    return 0


def _sync(args) -> int:
    from training_framework.runs.sync import sync_targets

    return sync_targets(args, _root(args))


def main(argv: list[str] | None = None, prog: str = "python -m training_framework.runs") -> int:
    args = _parser(prog).parse_args(argv)
    try:
        return {"list": _list, "show": _show, "diff": _diff, "sync": _sync}[args.command](args)
    except RunsError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
