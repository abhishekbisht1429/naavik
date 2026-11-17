"""What `naavik runs` prints: a table of runs, one run in detail, two runs
side by side. Pure functions of the records (and the registry's stored
configs); the command line only selects and prints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from training_framework.runs.ids import run_id_time
from training_framework.runs.registry import Run, parent_of


def started(run: Run) -> datetime:
    return run_id_time(run.run_id)


def _local(moment: datetime) -> str:
    return moment.astimezone().strftime("%Y-%m-%d %H:%M")


def code_text(code: dict | None) -> str:
    if not code:
        return "-"
    if not code.get("sha"):
        return "?" if not code.get("error") else "? (see show)"
    text = code["sha"][:10]
    if code.get("snapshot"):
        text += " (snapshot)"
    elif code.get("dirty"):
        text += " (dirty)"
    return text


def what_text(run: Run) -> str:
    """What was run: a batch task, else the config file or checkpoint."""
    launch = run.record.get("launch") or {}
    if launch.get("batch_id"):
        attempt = launch.get("attempt") or 1
        return f"{launch['batch_id']} {launch.get('task')}" + (f" (attempt {attempt})" if attempt > 1 else "")
    if launch.get("checkpoint"):
        return f"{launch.get('operation')} {launch['checkpoint']}"
    command = launch.get("command") or []
    if "--config" in command and command.index("--config") + 1 < len(command):
        return command[command.index("--config") + 1]
    return " ".join(command[1:]) or "-"


def table(runs: list[Run]) -> str:
    notes = any((run.record.get("launch") or {}).get("note") for run in runs)
    rows = [("run", "started", "host", "state", "code") + (("note",) if notes else ()) + ("what",)]
    for run in runs:
        note = ((run.record.get("launch") or {}).get("note") or "-",) if notes else ()
        rows.append((run.run_id, _local(started(run)), run.record.get("host") or "?",
                     run.state, code_text(run.record.get("code"))) + note + (what_text(run),))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]) - 1)]
    return "\n".join("  ".join(cell.ljust(width) for cell, width in zip(row, widths)) + "  " + row[-1]
                     for row in rows)


def as_json(run: Run) -> dict:
    return {"run_id": run.run_id, "state": run.state, "record": run.record,
            "events": run.events, "ranks": {str(k): v for k, v in run.ranks.items()},
            "problems": run.problems}


# One run ------------------------------------------------------------------------------


def checkpoints(run: Run) -> list[tuple[str, Any]]:
    """(path, iteration) of every complete checkpoint under the run's
    session directories (derived, not recorded)."""
    found = []
    for directory in run.session_dirs:
        for manifest in sorted(Path(directory).rglob("manifest.json")):
            try:
                iteration = json.loads(manifest.read_text()).get("iteration")
            except (OSError, ValueError):
                iteration = "?"
            found.append((str(manifest.parent), iteration))
    return found


def detail(run: Run, runs: list[Run], root: Path) -> str:
    record = run.record
    launch = record.get("launch") or {}
    lines = [f"run {run.run_id} | {run.state} | started {_local(started(run))} on "
             f"{record.get('host')} (pid {record.get('pid')}, {record.get('user')})"]
    parent = parent_of(run, runs)
    lines.append(f"launch   {launch.get('operation')}: {' '.join(launch.get('command') or [])}")
    if launch.get("batch_id"):
        lines.append(f"batch    {launch['batch_id']} | task {launch.get('task')} | attempt "
                     f"{launch.get('attempt')} | {launch.get('scheduler')}"
                     + (" | restarted" if launch.get("restarted") else ""))
    if launch.get("note"):
        lines.append(f"note     {launch['note']}")
    if parent is not None:
        lines.append(f"continues {parent.run_id} ({parent.state})")
    code = record.get("code")
    if code:
        text = f"code     {code.get('sha') or '?'}"
        if code.get("snapshot"):
            text += f" = snapshot on {code.get('base_sha')} ({code.get('ref')})"
        if code.get("repository"):
            text += f" | {code['repository']}"
        lines.append(text)
        if code.get("error"):
            lines.append(f"         ! {code['error']}")
    else:
        lines.append("code     - (not in a Git repository)")
    env = record.get("environment") or {}
    lines.append(f"env      python {env.get('python')}, torch {env.get('torch')}, CUDA "
                 f"{env.get('cuda')}, packages {(env.get('packages_sha256') or '')[:12]}")
    framework = env.get("framework")
    if isinstance(framework, dict):
        text = f"framework {framework.get('version') or '(no installed version)'} | {framework.get('location')}"
        if framework.get("sha"):
            text += f" | git {framework['sha'][:10]}" + (" (dirty)" if framework.get("dirty") else "")
        lines.append(text)
    for rank, facts in sorted(run.ranks.items()):
        device = facts.get("device")
        if facts.get("name"):
            device += f" {facts['name']} ({facts.get('uuid')})"
        lines.append(f"rank {rank}   {device} | {facts.get('backend') or '-'} world "
                     f"{facts.get('world_size')} | CUDA_VISIBLE_DEVICES="
                     f"{facts.get('cuda_visible_devices')}")
    for session in run.sessions:
        lines.append(f"session  {session.get('index')}: {session.get('type')} | "
                     f"{session.get('session_dir')} | config {str(session.get('config_sha256'))[:12]}"
                     f" ({root}/configs/{session.get('config_sha256')}.yaml)")
        for source in session.get("config_sources") or []:
            path = source.get("path") if isinstance(source, dict) else source
            lines.append(f"           from {path}")
    for path, iteration in checkpoints(run):
        lines.append(f"checkpoint iteration {iteration}: {path}")
    finished = run.finished
    if finished:
        lines.append(f"finished {finished.get('at')} | {finished.get('status')} | last iteration "
                     f"{finished.get('last_iteration')}")
        if finished.get("error"):
            lines.append(f"         ! {finished['error']}")
    for problem in run.problems:
        lines.append(f"problem  {problem}")
    return "\n".join(lines)


# Two runs ------------------------------------------------------------------------------


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flat = {}
        for key, item in value.items():
            flat.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return flat or {prefix: {}}
    return {prefix: value}


def config_value(run: Run, root: Path, key: str) -> Any:
    """`key` (dotted) of the run's first session config; KeyError if absent."""
    flat = flatten(run.config(root) or {})
    if key not in flat:
        raise KeyError(key)
    return flat[key]


def _ignored(key: str, ignored) -> bool:
    return any(key == item or key.startswith(f"{item}.") for item in ignored)


def difference(a: Run, b: Run, root: Path, ignored=()) -> str:
    """What differs, leaving out config keys in `ignored` (a key and
    everything under it): those that differ between runs by construction."""
    def text(value):
        return yaml.safe_dump(value, default_flow_style=True).strip() if isinstance(value, (dict, list)) \
            else json.dumps(value) if not isinstance(value, str) else value

    lines = [f"        {a.run_id} | {b.run_id}"]
    code_a, code_b = a.record.get("code") or {}, b.record.get("code") or {}
    if code_a.get("sha") != code_b.get("sha"):
        lines.append(f"code    {code_text(code_a)} | {code_text(code_b)}")
        if code_a.get("sha") and code_b.get("sha"):
            lines.append(f"        git diff {code_a['sha'][:12]} {code_b['sha'][:12]}")
    config_a, config_b = flatten(a.config(root) or {}), flatten(b.config(root) or {})
    skipped = set()
    for key in sorted(set(config_a) | set(config_b)):
        left, right = config_a.get(key, "(absent)"), config_b.get(key, "(absent)")
        if left != right and _ignored(key, ignored):
            skipped.add(key)
        elif left != right:
            lines.append(f"config  {key}: {text(left)} | {text(right)}")
    env_a, env_b = a.record.get("environment") or {}, b.record.get("environment") or {}
    for key in sorted(set(env_a) | set(env_b)):
        if env_a.get(key) != env_b.get(key):
            lines.append(f"env     {key}: {env_a.get(key)} | {env_b.get(key)}")
    for rank in sorted(set(a.ranks) | set(b.ranks)):
        left = (a.ranks.get(rank) or {}).get("name") or (a.ranks.get(rank) or {}).get("device")
        right = (b.ranks.get(rank) or {}).get("name") or (b.ranks.get(rank) or {}).get("device")
        if left != right:
            lines.append(f"rank {rank}  {left} | {right}")
    if len(lines) == 1:
        lines.append("no differences in code, config, environment, or devices")
    if skipped:
        lines.append(f"(not compared, they differ by construction: {', '.join(sorted(skipped))}; "
                     "--all-keys compares them)")
    return "\n".join(lines)
