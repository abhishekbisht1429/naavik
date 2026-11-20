"""The run registry: one record of every engine launch on this machine.

    $NAAVIK_RUNS (default $XDG_DATA_HOME/naavik/runs)/
      <run-id>/run.json        written once, when the run starts
      <run-id>/events.jsonl    appended by the launching process only
      <run-id>/rank-<r>.json   written once by each worker: its device
      configs/<sha256>.yaml    resolved session configs, stored once

Every file has one writer, records are never rewritten, and run ids are
unique, so the registries of several machines merge by copying (no
conflicts). What is already on disk elsewhere (checkpoints, which run a
resume continues) is derived when shown, not recorded twice. Recording never
fails a run: an error becomes one warning and a note in the record.
`NAAVIK_RUNS=off` turns the registry off (`TRAINING_FRAMEWORK_RUNS`, the
earlier name, is read too).
"""

from naavik.runs.ids import new_run_id
from naavik.runs.registry import Run, RunWriter, list_runs, read_run, registry_dir

__all__ = ["Run", "RunWriter", "list_runs", "new_run_id", "read_run", "registry_dir"]
