"""`python -m training_framework.launcher.evaluate INPUT OUTPUT`: check a
batch's task configs with the batch's own code.

Run by `validate`, `render`, and `launch` in a separate process whose
`PYTHONPATH` is the batch's exported tree (`launcher/ports/source_tree.py`),
so the code checked is the code that will run. INPUT is JSON `{"tasks":
[{"name", "config"}]}` (a task's config file: `{"sessions": [...]}`);
OUTPUT receives `{"framework": {...}, "tasks": {name: {"problems": [...],
"notes": [...], "requirements": [...]}}}`, a requirement being the data a
planned component declares (`training_framework.data`): `{"session",
"component", "class", "requirement", "path", "location"}`. Output goes to
a file: what the project's imports print cannot corrupt it.
"""

from __future__ import annotations

import json
import sys


def evaluate(tasks: list[dict]) -> dict:
    from training_framework.runs.facts import framework_facts
    from training_framework.session import check_session_config

    results = {}
    for task in tasks:
        problems, notes, requirements = [], [], []
        sessions = (task.get("config") or {}).get("sessions") or []
        for index, session in enumerate(sessions):
            where = f"session {index}: " if len(sessions) > 1 else ""
            check = check_session_config(session)
            problems += [where + problem for problem in check.problems]
            notes += [where + note for note in check.notes]
            if check.problems:
                continue
            found, data_problems = _requirements(check.components, index)
            requirements += found
            problems += [where + problem for problem in data_problems]
        results[task["name"]] = {"problems": problems, "notes": notes,
                                 "requirements": requirements}
    return {"framework": framework_facts(), "tasks": results}


def _requirements(components, session: int) -> tuple[list[dict], list[str]]:
    """The data each planned component declares."""
    from training_framework.data import DataError, requirements_of

    found, problems = [], []
    for planned in components:
        cls = planned.component_class
        try:
            located = requirements_of(cls, planned.config)
        except (DataError, TypeError, ValueError) as error:
            problems.append(f"{planned.name}: its data requirements: {error}")
            continue
        for item in located:
            found.append({"session": session, "component": planned.name,
                          "class": f"{cls.__module__}.{cls.__qualname__}",
                          "requirement": item.requirement.as_json(),
                          "path": item.path, "location": item.location})
    return found, problems


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
