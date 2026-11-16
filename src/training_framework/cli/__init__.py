"""`naavik` (also `navik`, `python -m training_framework.cli`): one command
for launching experiments and reading their results.

Subcommands are those of the launcher (`launch`, `validate`, `status`,
`batches`, `resume`, `fetch`, ...), `summarize` (run summaries,
`training_framework.reporting`), and `runs` (the run registry,
`training_framework.runs`). The earlier entry points keep working:
`python -m training_framework.hpc ...` and
`python -m training_framework.reporting summarize ...`.
"""

from __future__ import annotations

import sys

PROG = "naavik"

#: Subcommands served by another program than the launcher: module, and
#: whether that program's own commands follow the subcommand (`naavik runs
#: list` -> `list`) or the subcommand is one of them (`summarize`).
_OTHER_PROGRAMS = {
    "summarize": ("training_framework.reporting.__main__", False),
    "runs": ("training_framework.runs.__main__", True),
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in _OTHER_PROGRAMS:
        import importlib

        module, own_commands = _OTHER_PROGRAMS[argv[0]]
        program = importlib.import_module(module)
        if own_commands:
            return program.main(argv[1:], prog=f"{PROG} {argv[0]}")
        return program.main(argv, prog=PROG)
    from training_framework.launcher.__main__ import main as launcher_main

    return launcher_main(argv, prog=PROG)
