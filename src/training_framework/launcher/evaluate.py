"""`python -m training_framework.launcher.evaluate INPUT OUTPUT`: check a
batch's task configs with the batch's own code.

Run by `validate`, `render`, and `launch` in a separate process whose
`PYTHONPATH` is the batch's exported tree (`launcher/ports/source_tree.py`),
so the code checked is the code that will run. INPUT is JSON `{"tasks":
[{"name", "config"}]}` (a task's config file: `{"sessions": [...]}`);
OUTPUT receives `{"framework": {...}, "tasks": {name: {"problems": [...],
"notes": [...]}}}`. Output goes to a file: what the project's imports print
cannot corrupt it.
"""

from __future__ import annotations

import json
import sys


def evaluate(tasks: list[dict]) -> dict:
    from training_framework.runs.facts import framework_facts
    from training_framework.session import check_session_config

    results = {}
    for task in tasks:
        problems, notes = [], []
        sessions = (task.get("config") or {}).get("sessions") or []
        for index, session in enumerate(sessions):
            where = f"session {index}: " if len(sessions) > 1 else ""
            check = check_session_config(session)
            problems += [where + problem for problem in check.problems]
            notes += [where + note for note in check.notes]
        results[task["name"]] = {"problems": problems, "notes": notes}
    return {"framework": framework_facts(), "tasks": results}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: python -m training_framework.launcher.evaluate INPUT OUTPUT", file=sys.stderr)
        return 2
    with open(argv[0]) as handle:
        tasks = json.load(handle)["tasks"]
    result = evaluate(tasks)
    with open(argv[1], "w") as handle:
        json.dump(result, handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
