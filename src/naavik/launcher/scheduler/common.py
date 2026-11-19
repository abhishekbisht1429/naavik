"""Job-script pieces shared by every scheduler's dialect."""

from __future__ import annotations

import shlex


def task_body(batch_dir: str, group: str, command: list[str], index_variable: str,
              resume_table: str | None = None) -> list[str]:
    """The shell lines that run one element of a group's task set: find its
    config (and, for a later attempt, its checkpoint) by the element's index
    in `$<index_variable>`, run the task in the background, turn the
    scheduler's advance signal (USR1) into a cooperative stop (SIGINT), and
    exit 75 when stopped that way. Every scheduler's job script uses it, so
    the stop protocol and records are the same everywhere."""
    table = shlex.quote(f"{batch_dir}/groups/{group}/tasks.tsv")
    run = shlex.join(command)
    for name in ("config", "record", "checkpoint"):
        run = run.replace(f"'{{{name}}}'", f'"${name}"').replace(f"{{{name}}}", f'"${name}"')
    lookup = []
    if resume_table:
        lookup = [
            f"checkpoint=$(awk -F'\\t' -v i=\"${index_variable}\" "
            f"'NR > 1 && $1 == i {{print $3}}' {shlex.quote(resume_table)})",
            f'[ -n "$checkpoint" ] || {{ echo "no checkpoint for array index ${index_variable}" >&2; exit 2; }}',
        ]
    return [
        f"config={shlex.quote(batch_dir)}/$(awk -F'\\t' -v i=\"${index_variable}\" "
        f"'NR > 1 && $1 == i {{print $4}}' {table})",
        'record="$(dirname "$config")/record.json"',
        *lookup,
        "interrupted=0",
        "child=''",
        'log() { echo "[hpc $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }',
        'on_signal() { interrupted=1; log "received the scheduler advance signal; stopping the task cooperatively"; '
        '[ -n "$child" ] && kill -INT "$child" 2>/dev/null || true; }',
        "trap on_signal USR1",
        "set +e",
        f"{run} &",
        "child=$!",
        'while true; do wait "$child"; status=$?; kill -0 "$child" 2>/dev/null || break; done',
        'log "task process exited with status $status"',
        'if [ "$interrupted" -eq 1 ]; then exit 75; fi',
        'exit "$status"',
    ]
