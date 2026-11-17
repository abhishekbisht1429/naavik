# Every Run Is Recorded

[← Docs](../README.md) · [Project README](../../README.md)

Every launch of the engine leaves a record: a direct run from your entry
point or your IDE, a batch task on any target, a resume, an extension, and a
launch that fails while its config is being built. The record says what ran
(the config, the code, the environment), where (the host, every rank's
device), and how it ended. `naavik runs` lists, shows, and compares them.

```text
$ python -m my_project.train --config configs/vit.yaml
run 01M7KMT4A2T9AK0R89QKNFF9NG (naavik runs show 01M7KMT4A2T9AK0R89QKNFF9NG)
...
```

That line (on standard error) is the only change to a launch's output.
Recording never fails a run: if the registry cannot be written (a full disk,
a read-only directory), you get one `warning: run registry: ...` line and
the run goes on.

## Finding runs: `naavik runs list`

```text
$ naavik runs list
run                         started           host    state  code                what
01M7KMT4A2T9AK0R89QKNFF9NG  2026-11-17 02:19  gpubox  done   9e04101571 (dirty)  configs/lr0.2.yaml
01M7KMT17KHKWYGDJW7ZMTC15B  2026-11-17 02:19  gpubox  done   9e04101571 (dirty)  configs/lr0.1.yaml
```

Newest first, at most 50 (`--limit N`, `0` for all). `what` is the config
file, the checkpoint a resume or extension started from, or a batch task
(`<batch-id> <task>`). `code` is the commit; `(snapshot)` when the working
tree was taken as a snapshot commit, `(dirty)` when it had changes and no
snapshot was kept (see [The code](#the-code)).

| Option | Lists |
|---|---|
| `--since 30m` / `12h` / `7d` / `2w` | runs started within that time |
| `--state S` | runs in this state (repeatable) |
| `--host H` | runs on this host (repeatable) |
| `--batch TEXT` | tasks of batches whose id contains `TEXT` |
| `--where KEY=VALUE` | runs whose config has `KEY` (a dotted path, e.g. `optimizer.optimizer.kwargs.lr`) equal to `VALUE` (read as YAML, so `0.2` is a number); repeatable, all must hold |
| `--json` | the records as JSON |
| `--registry PATH` | another registry than this machine's |

A run's **state** is worked out when it is listed, never stored:

| State | Meaning |
|---|---|
| `done`, `failed`, `interrupted` | how the launch ended (`interrupted`: Ctrl-C, or a scheduler's stop) |
| `running` | not ended, and its process still runs on this machine |
| `ended without a record` | not ended, and its process is gone: killed, out of memory, or the machine restarted |
| `unknown` | not ended, recorded on another machine: run `naavik runs sync` again |

## One run: `naavik runs show`

Any unique beginning of the id will do:

```text
$ naavik runs show 01M7KMT4A2T9
run 01M7KMT4A2T9AK0R89QKNFF9NG | done | started 2026-11-17 02:19 on gpubox (pid 2679978, me)
launch   config: train --config configs/lr0.2.yaml
code     9e041015717d2f0d331d67f57359eff86b0e4248 | /home/me/my-project
env      python 3.13.14, torch 2.13.0+cu130, CUDA 13.0, packages 86a9ba0db444
framework (no installed version) | /home/me/training-framework/src/training_framework | git 9e04101571 (dirty)
rank 0   cpu | gloo world 1 | CUDA_VISIBLE_DEVICES=2
session  0: training | runs/session_20261117_021910 | config c45d78bab46b (~/.local/share/training-framework/runs/configs/c45d78...yaml)
           from configs/lr0.2.yaml
checkpoint iteration 2: runs/session_20261117_021910/checkpoints/20261117_021912_234657749
checkpoint iteration 4: runs/session_20261117_021910/checkpoints/20261117_021912_256008841
finished 2026-11-17T02:19:13.025363+00:00 | ok | last iteration [4]
```

- `launch`: how it was started (`config`, `resume`, or `extend`) and the
  command. A batch task adds its batch, task, attempt, and scheduler.
- `continues <run>`: for a resume or an extension, the earlier run of the
  same session (the latest one whose session directory it restored).
- `env`: Python, torch, CUDA, and a hash of the installed packages (equal
  hashes, same packages). `framework`: where the framework was imported
  from, its version only if that is the installed copy, and its Git commit
  when it is a checkout.
- `rank N`: each worker's device as it saw it: for a GPU, its name and
  UUID (the one `nvidia-smi` and `CUDA_VISIBLE_DEVICES` use).
- `session`: the session directory, the resolved config stored in the
  registry, and the files it was built from (with includes).
- `checkpoint`: every complete checkpoint in the session directory, found
  when shown.
- `finished`: the outcome, each rank's last iteration, and the error.

`--json` prints the whole record.

## Two runs: `naavik runs diff`

```text
$ naavik runs diff 01M7KMT17KHK 01M7KMT4A2T9
        01M7KMT17KHKWYGDJW7ZMTC15B | 01M7KMT4A2T9AK0R89QKNFF9NG
config  optimizer.optimizer.kwargs.lr: 0.1 | 0.2
(not compared, they differ by construction: ddp.master_port; --all-keys compares them)
```

Code (with the `git diff` to run when the commits differ), every differing
config key, the environment, and each rank's device. Like
[`summarize`](12-comparing-runs.md#which-settings-are-compared), it leaves
out the keys that differ between runs by construction (`sessions_dir`, the
rendezvous address and port); `--all-keys` compares them too.

To compare runs' metrics, give their ids to `summarize`:

```bash
naavik summarize --runs 01M7KMT17KHK --runs 01M7KMT4A2T9 --metric train/loss:min
```

## The code

A direct run records the Git repository holding its components package
(`session_config.components_package`; else the working directory):

- a clean working tree: its HEAD;
- uncommitted changes: the working tree as a commit on HEAD, made with a
  temporary index (your index, branch, and files are untouched) and kept as
  `refs/naavik/runs/<run-id>`, so `git diff <HEAD> refs/naavik/runs/<run-id>`
  shows exactly what ran. The same tree on the same HEAD is the same commit.
  This is the [snapshot](08-hpc-batches.md#uncommitted-work-worktree-snapshot)
  batches use, with the same limit on untracked files (50 MB);
- outside a Git repository: nothing.

With `TRAINING_FRAMEWORK_RUNS_SNAPSHOT=0`, or when untracked files exceed
the limit, no snapshot is kept: the record says the tree was dirty and why.
A batch task records its batch's commit (and snapshot base) instead.

`naavik summarize` uses these records to tell runs on different code apart
(see [Comparing runs](12-comparing-runs.md#which-settings-are-compared)).
A batch launched with a description (`naavik run --note TEXT`, or the
manifest's `description`) keeps it in each task's record; `naavik runs
show` prints it as `note`.

## Across machines

Each machine has its own registry. A batch task records on the machine it
runs on; copy those runs here with:

```bash
naavik runs sync                   # the target file's default target
naavik runs sync --target gpubox   # repeatable
naavik runs sync --all-targets
```

`sync` copies over SSH (rsync) what is new or changed (a run's events keep
arriving until it ends) and deletes nothing. A target on this machine
(`local: true`) shares this registry; there is nothing to copy. Ids are
unique and every record has one writer, so copies never conflict.

## Where records live

```
$TRAINING_FRAMEWORK_RUNS (default ~/.local/share/training-framework/runs)/
  <run-id>/run.json        written once, when the run starts
  <run-id>/events.jsonl    its sessions and its end, added by the launching process
  <run-id>/rank-<r>.json   each worker's device, written once
  configs/<sha256>.yaml    resolved configs, each stored once
```

| Variable | Effect |
|---|---|
| `TRAINING_FRAMEWORK_RUNS=PATH` | keep the registry in `PATH` |
| `TRAINING_FRAMEWORK_RUNS=off` | record nothing (`naavik runs` then needs `--registry`) |
| `TRAINING_FRAMEWORK_RUNS_SNAPSHOT=0` | never make snapshot commits for runs of a dirty tree |

Run ids are 26 characters (time of start, then random) that sort by start
time. Records are never rewritten, so removing a run is deleting its
directory; stored configs shared with other runs stay.

---

**See also:** [Running batches](08-hpc-batches.md) for batch tasks and
targets; [Comparing runs](12-comparing-runs.md) for metrics.
