"""`python -m training_framework.data fetch|verify --component DOTTED --config JSON
--requirement NAME [--scheduler S]`: obtain or check one data requirement of
one component config (the target's prepare step runs `fetch`).

The location is the config's own (the batch has already rewritten it for
the target), so the command and the config can never disagree.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from training_framework.data import DataError, ensure, requirements_of, verify


def _log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[data {stamp}] {message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training_framework.data")
    parser.add_argument("action", choices=("fetch", "verify"))
    parser.add_argument("--component", required=True, help="the component class, dotted")
    parser.add_argument("--config", required=True, help="its config entry, as JSON")
    parser.add_argument("--requirement", required=True, help="the requirement's name")
    parser.add_argument("--scheduler", help="the batch scheduler this runs under")
    args = parser.parse_args(argv)
    from training_framework.components.importing import import_object

    try:
        component_class = import_object(args.component, "--component")
        config = json.loads(args.config)
        found = {located.requirement.name: located
                 for located in requirements_of(component_class, config)}
        if args.requirement not in found:
            raise DataError(f"{args.component} has no data requirement '{args.requirement}' "
                            f"(it has: {', '.join(sorted(found)) or 'none'})")
        located = found[args.requirement]
        if args.action == "verify":
            problems = verify(component_class, located)
            for problem in problems:
                _log(f"{located.location}: {problem}")
            if problems:
                return 1
            _log(f"{located.location}: ok")
            return 0
        outcome = ensure(component_class, located, notify=_log, scheduler=args.scheduler)
        _log(f"{args.requirement} at {located.location}: {outcome}")
        return 0
    except (DataError, ValueError, TypeError, ImportError) as error:
        _log(f"error: {error}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
