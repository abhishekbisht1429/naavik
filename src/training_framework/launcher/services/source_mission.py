"""The `checkpoint_facts` port: another mission's complete checkpoints, for
a recipe's `from_mission`.

Found from the source's receipt on this machine:
- the source ran on the host this mission runs on: its checkpoints are
  listed there, read-only, in one command (`find_checkpoints`, as `resume`
  and `fetch` do);
- the source ran elsewhere and this mission runs on this machine: its copy
  brought back by `naavik fetch` (`<fetched>/<mission-id>`) is read;
- otherwise it is refused, naming both hosts: checkpoints are never copied
  between hosts.
"""

from __future__ import annotations

import os
from pathlib import Path

from training_framework.launcher.compile.compiler import SourceCheckpoint, SourceMission
from training_framework.launcher.errors import HpcError

#: Where `naavik fetch` puts a mission by default (`--dest`), relative to here.
FETCHED = "hpc-results"


def read_source_mission(mission_id: str, profile, *, receipts_dir=None, fetched=FETCHED,
                        require_tools=None) -> SourceMission:
    """The source mission's experiments and their complete checkpoints;
    every problem is returned, never raised (the compiler names the recipe)."""
    try:
        return _read(mission_id, profile, receipts_dir, fetched, require_tools)
    except HpcError as error:
        return SourceMission(problems=[str(error)])


def _read(mission_id, profile, receipts_dir, fetched, require_tools):
    from training_framework.launcher.ports.local import LocalExecutor
    from training_framework.launcher.ports.targets import (
        ports_for,
        target_of_profile,
        target_of_receipt,
    )
    from training_framework.launcher.services.checkpoints import find_checkpoints
    from training_framework.launcher.services.resume import task_index
    from training_framework.launcher.services.status import fetch_batches, missing_files

    receipt = _receipt(mission_id, receipts_dir)
    if not receipt.get("staged"):
        raise HpcError(f"{mission_id} never reached its host ({receipt.get('status')}); it has "
                       "no checkpoints")
    source = target_of_receipt(receipt)
    if source is None:
        raise HpcError(f"the receipt of {mission_id} does not record its host (it was written "
                       "before receipts did); launch the training again, or fetch it and "
                       "analyse it on this machine")
    host = target_of_profile(profile)
    if source == host:
        if require_tools is not None:
            require_tools(*host.tools(transfer=False))
        executor = ports_for(host).executor
        mission_dir = receipt["batch_dir"]
        files = fetch_batches(executor, [mission_dir]).get(mission_dir)
        if problem := missing_files(receipt, files):
            raise HpcError(f"{problem} on {host}")
        names = list(task_index(files.tables))
        where = f"on host '{profile.name}' (read-only)"
    elif host.is_local:
        copy = os.path.abspath(os.path.join(fetched, mission_id))
        if not os.path.isdir(copy):
            raise HpcError(
                f"{mission_id} ran on {source}, and this mission runs on this machine: fetch "
                f"it here first (`naavik fetch {mission_id}`; add --checkpoints all for "
                f"checkpoint: all) (looked at {copy})")
        names = list(task_index(_tables(copy)))
        if not names:
            raise HpcError(f"the fetched copy {copy} has no experiment tables (groups/*/tasks.tsv)")
        executor, mission_dir = LocalExecutor(), copy
        where = f"from its fetched copy {copy}"
    else:
        raise HpcError(
            f"{mission_id} ran on {source}, but this mission runs on host '{profile.name}' "
            f"({host}): an analysis runs where its checkpoints are. Launch it on {source}, "
            f"or fetch {mission_id} and launch it on this machine")
    found, _ = find_checkpoints(executor, mission_dir, names)
    return SourceMission(
        {name: tuple(SourceCheckpoint(checkpoint.path, checkpoint.iteration)
                     for checkpoint in found.get(name, []) if checkpoint.iteration is not None)
         for name in names},
        [], where, _variants(executor, mission_dir, mission_id))


def _variants(executor, mission_dir, mission_id) -> dict[str, tuple[dict, ...]]:
    """Each experiment's varied values, from the mission's plan."""
    import json

    result = executor.run(["cat", f"{mission_dir}/plan.json"])
    try:
        plan = json.loads(result.stdout) if result.status == 0 else None
        return {task["name"]: tuple(task.get("variant", ())) for task in plan["tasks"]}
    except (ValueError, TypeError, KeyError):
        raise HpcError(f"the plan of {mission_id} ({mission_dir}/plan.json) cannot be read"
                       + (f": {result.stderr.strip()}" if result.stderr.strip() else "")) from None


def _receipt(mission_id, receipts_dir) -> dict:
    import json

    from training_framework.launcher.services.receipt import default_receipts_dir
    from training_framework.launcher.services.status import receipt_problem

    path = Path(receipts_dir or default_receipts_dir()) / f"{mission_id}.json"
    if not path.is_file():
        raise HpcError(f"no receipt for {mission_id} on this machine (looked at {path}); "
                       "name a mission launched from here")
    try:
        receipt = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise HpcError(f"the receipt {path} cannot be read ({error})") from None
    if problem := receipt_problem(receipt):
        raise HpcError(f"the receipt {path} is not valid ({problem})")
    return receipt


def _tables(copy: str) -> dict[str, str]:
    """{job: its experiment table} of a fetched copy."""
    tables = {}
    for path in sorted(Path(copy, "groups").glob("*/tasks.tsv")):
        tables[path.parent.name] = path.read_text()
    return tables
