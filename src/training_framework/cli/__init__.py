"""`naavik` (also `navik`, `python -m training_framework.cli`): one command
for launching experiments and reading their results.

Subcommands are those of the launcher (`launch`, `validate`, `status`,
`batches`, `resume`, `fetch`, ...) plus `summarize` (run summaries,
`training_framework.reporting`). The earlier entry points keep working:
`python -m training_framework.hpc ...` and
`python -m training_framework.reporting summarize ...`.
"""

from __future__ import annotations

import sys

PROG = "naavik"

#: Subcommands served by another program than the launcher, by module.
_OTHER_PROGRAMS = {
    "summarize": "training_framework.reporting.__main__",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in _OTHER_PROGRAMS:
        import importlib

        program = importlib.import_module(_OTHER_PROGRAMS[argv[0]])
        return program.main(argv, prog=PROG)
    from training_framework.launcher.__main__ import main as launcher_main

    return launcher_main(argv, prog=PROG)
