"""The code check of `validate`, `render`, and `launch`: every task config of
a compiled batch checked by the batch's own code (`launcher/evaluate.py`),
in a separate process on the batch's exported tree.

Which interpreter: a target on this machine that uses an existing
environment runs the tasks with it, so it checks them too; otherwise the
interpreter running `naavik`. The framework that checks must be the one the
target will run: from the tree in checkout mode; in dependency mode with a
managed environment, the pinned version and (for a `git+...@<sha>` pin) the
pinned commit.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from training_framework.launcher.errors import HpcError

_FULL_SHA = re.compile(r"@([0-9a-f]{40})\b")
#: `TRAINING_FRAMEWORK_CODE_CHECK=0` skips the check (for tests of the batch
#: layer, whose configs are placeholders); a skip is warned about and
#: recorded in the receipt, never silent.
ENV_SWITCH = "TRAINING_FRAMEWORK_CODE_CHECK"


def enabled() -> bool:
    return os.environ.get(ENV_SWITCH, "1") != "0"


def _checkout_version(location: str | None) -> str | None:
    """The version a framework checkout declares (`pyproject.toml`)."""
    import tomllib

    for parent in Path(location or "/").parents:
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            try:
                project = tomllib.loads(candidate.read_text()).get("project", {})
            except (OSError, ValueError):
                return None
            return project.get("version") if project.get("name") == "training-framework" else None
    return None


@dataclass
class Evaluation:
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _interpreter(profile) -> str:
    if profile.local and profile.env.existing:
        return f"{profile.env.existing}/bin/python"
    return sys.executable


def _pythonpath(tree: Path, manifest, interpreter: str) -> list[str]:
    paths = [str(tree / path) for path in manifest.code_paths]
    if manifest.code_framework == "dependency" and interpreter == sys.executable:
        # The framework checking is this process's own, wherever it lives
        # (installed, or a checkout on PYTHONPATH); its identity is compared.
        import training_framework

        paths.append(str(Path(training_framework.__file__).resolve().parents[1]))
    return paths


def framework_mismatch(manifest, plan, repository, framework: dict, tree: Path) -> str | None:
    """Why the framework that checked is not the one the target will run."""
    if manifest.code_framework == "checkout":
        location = Path(framework.get("location") or "/")
        return None if location.is_relative_to(tree.resolve()) else (
            f"the check used the framework at {location}, not the batch's own checkout "
            f"({tree}); is it in code.paths?")
    if not plan.environment.get("managed"):
        return None                        # an existing environment is the target's own
    from training_framework.launcher.config.requirements import parse_requirements

    raw = repository.read(plan.sha, manifest.requirements) if manifest.requirements else None
    if raw is None:
        return None
    pins = parse_requirements(raw.decode(), manifest.requirements).pins_named("training-framework")
    if len(pins) != 1:
        return None                        # the compiler reports this already
    pin, mine = pins[0], framework
    commit = _FULL_SHA.search(pin.url or "")
    if commit is not None:
        if mine.get("sha") != commit.group(1) or mine.get("dirty"):
            return (f"the target runs training-framework at commit {commit.group(1)[:12]}, but the "
                    f"framework here is {(mine.get('sha') or 'not a Git checkout')[:12]}"
                    f"{' with uncommitted changes' if mine.get('dirty') else ''} "
                    f"({mine.get('location')}); install the pinned framework here "
                    "(guide 10, step 5)")
        return None
    if pin.version is not None and mine.get("version") != pin.version:
        if mine.get("version") is None and _checkout_version(mine.get("location")) == pin.version:
            return None                    # a checkout of the pinned release (noted by the caller)
        return (f"the target runs training-framework {pin.version}, but the framework here is "
                f"{mine.get('version') or 'not an installed release'} ({mine.get('location')}); "
                "install the pinned framework here (guide 10, step 5)")
    return None


def checkout_note(manifest, plan, framework: dict) -> str | None:
    """A version pin checked with a checkout that declares that version."""
    if manifest.code_framework != "dependency" or not plan.environment.get("managed"):
        return None
    if framework.get("version") is None and framework.get("sha"):
        version = _checkout_version(framework.get("location"))
        if version:
            return (f"the configs were checked with a training-framework checkout "
                    f"({framework['sha'][:12]}{', with uncommitted changes' if framework.get('dirty') else ''}) "
                    f"that declares {version}; the target installs the pinned release")
    return None


def evaluate_batch(plan, manifest, repository, profile) -> Evaluation:
    """Check every task of `plan` with the batch's code."""
    from training_framework.launcher.ports.source_tree import source_tree

    result = Evaluation()
    tree = source_tree(repository.root, plan.sha)
    interpreter = _interpreter(profile)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(_pythonpath(tree, manifest, interpreter))}
    with tempfile.TemporaryDirectory() as directory:
        given, answer = Path(directory, "tasks.json"), Path(directory, "result.json")
        given.write_text(json.dumps({"tasks": [{"name": task.name, "config": task.config}
                                               for task in plan.tasks]}, default=str))
        try:
            completed = subprocess.run(
                [interpreter, "-m", "training_framework.launcher.evaluate", str(given), str(answer)],
                capture_output=True, text=True, cwd=str(tree), env=env, check=False)
        except OSError as error:
            raise HpcError(f"cannot run the code check with {interpreter}: {error}") from None
        if completed.returncode != 0 or not answer.exists():
            tail = "\n".join(completed.stderr.strip().splitlines()[-15:])
            result.problems.append(f"the code check could not run ({interpreter}, exit "
                                   f"{completed.returncode}):\n{tail}")
            return result
        found = json.loads(answer.read_text())
    mismatch = framework_mismatch(manifest, plan, repository, found["framework"], tree)
    if mismatch:
        result.problems.append(mismatch)
    elif note := checkout_note(manifest, plan, found["framework"]):
        result.notes.append(note)
    seen_notes = set()
    for task in plan.tasks:
        outcome = found["tasks"].get(task.name, {})
        result.problems += [f"{task.name}: {problem}" for problem in outcome.get("problems", [])]
        for note in outcome.get("notes", []):
            if note not in seen_notes:
                seen_notes.add(note)
                result.notes.append(f"{task.name}: {note}")
    return result
