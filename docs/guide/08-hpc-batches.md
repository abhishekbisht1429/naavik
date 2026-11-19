# Running missions: here, over SSH, or on a cluster

The launcher (`naavik`) turns one session config into a **mission**: a set
of experiments, launched on a **host**: this machine, a machine you reach
over SSH, or a Slurm cluster. A mission file groups its experiments into
**jobs** (experiments that need the same resources run as one scheduler job:
a Slurm job array, or one runner elsewhere); a job's **recipes** say which
experiments to make; each **experiment** is a single-node run of the engine
(with DDP inside the node if configured). Everything is decided on your
workstation; the host only checks and executes. The same mission file runs
on every host: pick one with `--host`.

```bash
pip install 'training-framework[hpc]'     # adds `packaging`, used on the workstation
naavik validate missions/sweep.yaml
naavik render   missions/sweep.yaml --output /tmp/preview
naavik launch   missions/sweep.yaml
naavik run      configs/model/base.yaml   # one config as a mission of one experiment, followed
naavik missions --live        # all your missions and what they are doing
naavik status   <mission-id>  # one mission, experiment by experiment
naavik resume   <mission-id>  # continue unfinished experiments from their checkpoints
naavik fetch    <mission-id>  # download results, logs, latest checkpoints
naavik logs     <mission-id> <experiment> --follow   # an experiment's log
naavik cancel   <mission-id> [--experiment <name>]   # stop jobs, or some experiments
```

Every key of the hosts file, mission files, and the project file is
described in the [configuration reference](../reference/hpc-configuration.md).

**Earlier names still work.** `naavik` (also `navik`) is installed with the
package; `python -m training_framework.hpc <command>` runs the same
commands. Files and flags written with earlier names (`hpc.yaml` or
`targets.yaml`, `--target`, `submit`, `batches`, a mission file's `groups`,
`matrix`, ...) are still read; each launch warns once and lists every
`old -> new` to rename by hand ([earlier names](../reference/hpc-configuration.md#earlier-names)).
A mission file was called an *experiment file*, and before that a
*manifest*; checkpoints have a `manifest.json` of their own, which is a
different thing.

Missions always run one exact tree. By default it is a commit you made, so
uncommitted or untracked files are an error; with `run_code_from:
working_tree` it is your working tree as it is (see
[Uncommitted work](#uncommitted-work-run_code_from-working_tree)).

For a first run on Big Red 200, follow the staged checklist in
[First runs on Big Red 200](09-hpc-big-red-200.md).

## 1. The hosts file (your machines, on your machine)

`~/.config/training-framework/hosts.yaml` (or `--hosts-file PATH`) holds
everything specific to the machines you run on, so the committed mission
file runs unchanged anywhere. Pick a host with `--host NAME`, else
`default_host`.

```yaml
default_host: bigred200
hosts:
  bigred200:
    connection: {ssh: bigred200}          # an alias from ~/.ssh/config
    scheduler:
      kind: slurm
      sbatch_options: {account: r00000}
      partitions: {gpu_jobs: gpu, cpu_jobs: general}
      max_array_size: 1000                # the site's MaxArraySize
    directories:
      missions: /N/scratch/me/naavik/missions
      python_environments: /N/scratch/me/naavik/envs
      datasets: /N/scratch/me/naavik/datasets
    shell_commands_before:                # shell lines, per phase
      everything: ["module load python/3.12"]
      experiments: ["module load cudatoolkit"]
    python_environment:
      interpreter_from: site              # site | uv | conda
      site_interpreter: ...               # printed by show-site-interpreter (below)
    environment_build: {on: compute_node} # or login_node, when compute nodes lack internet
    data_preparation: {on: compute_node}
    local_to_host_paths: {/home/me/datasets: /N/project/mylab/datasets}
    confirm_uploads_over: 20G
```

SSH uses your own OpenSSH configuration, agent, and control connections; no
credentials are stored and MFA is never automated (`BatchMode=yes` fails fast
instead of prompting). Open a control connection first if your site needs MFA.

For a managed environment built from the site's Python, record that
interpreter's identity once and paste the output into the host:

```bash
naavik show-site-interpreter
```

If the site later changes its Python, `launch` stops with the new block to
paste; the new identity gives a new, separate environment.

### Hosts: this machine, an SSH machine, or a cluster

A host says how it is reached and what schedules the jobs there:

```yaml
default_host: here
hosts:
  here:                          # this machine
    connection: local
    scheduler: {kind: builtin, gpus: [4, 5]}   # the GPUs experiments may use
    run_code_from: working_tree  # run uncommitted work (below)
    directories: {missions: /data/me/naavik/missions, python_environments: /data/me/naavik/envs, datasets: /data/me/naavik/datasets}
    python_environment: {use_existing: current}   # the environment running naavik (e.g. the IDE's interpreter)
  gpubox:                        # a machine reached over SSH
    connection: {ssh: gpubox}
    scheduler: {kind: builtin, gpus: [0, 1, 2, 3]}
    directories: {...}
    python_environment: {use_existing: /opt/envs/train}
  bigred200:                     # a Slurm cluster, as above
    connection: {ssh: bigred200}
    scheduler: {kind: slurm, ...}
    directories: {...}
```

- **How it is reached:** `connection: {ssh: ...}` (over SSH and rsync) or
  `connection: local` (this machine: no `ssh` or `rsync` needed, nothing is
  copied to fetch).
- **What runs the jobs:** `kind: slurm`, or `kind: builtin` for a machine
  without a scheduler. With `builtin`, each job is a small runner process
  started detached on the host (it outlives your SSH connection). It runs a
  job's experiments at most `max_parallel_experiments` at a time, gives each
  experiment its GPUs from the `scheduler.gpus` pool, sends the same early
  stop signal before the walltime that Slurm does, and kills an experiment
  at the walltime.
- **GPUs on a builtin host:** an experiment sees only its own GPUs
  (`CUDA_VISIBLE_DEVICES` holds their UUIDs, numbered from 0 inside the
  experiment, so use `device: cuda` in the config); an experiment of a job
  without GPUs sees none. The pool's GPUs are shared by every mission on the
  machine and never given to two experiments at once. With `when_gpus_busy:
  wait` (default) a GPU someone else is using is skipped until it is free:
  one with a compute process `nvidia-smi` lists, or with memory in use
  (other containers' processes are not listed, but their memory is);
  `when_gpus_busy: use_anyway` ignores other users.
- **Not enforced by `builtin`:** `cpus` and `memory` are recorded, not
  limited.
- **Before staging,** `launch` checks the host: `python3` 3.8 or newer for
  the runner and, with a GPU pool, every pool GPU in `nvidia-smi`.

Results, logs, `status`, `resume`, `cancel`, and `fetch` are the same on
every host. A mission that ran on this machine is already here: `fetch`
prints its directory (`--dest` copies it).

### Uncommitted work: `run_code_from: working_tree`

With `run_code_from: working_tree` on a host, `validate`, `render`, and
`launch` take your working tree as it is: tracked changes, and untracked
files that `.gitignore` does not exclude (the mission file and configs too).
They become a commit on HEAD, made with a temporary index, so your index,
branch, and files are untouched; the same tree on the same HEAD always gives
the same commit. Before confirming, `launch` prints `note: snapshot of the
working tree on <HEAD>: N modified, M untracked file(s)`.

`launch` keeps the commit as `refs/naavik/snapshots/<mission-id>`, so what
ran stays in your repository: `git diff <HEAD> refs/naavik/snapshots/<mission-id>`.
`validate` and `render` keep nothing. Plans, receipts, and experiment
records carry `base_sha` (the HEAD it was taken on) besides `sha`.

Refused, with the reason: untracked files above
`working_tree_untracked_limit` (default `50M`; the error names the largest,
so a stray dataset never goes into Git), changed submodules, a
`code_transfer.commit` other than HEAD, and `code_transfer.method:
git_clone`.

## 2. The mission file (committed)

```yaml
format_version: 1
name: vit-fine-tuning
code_transfer: {method: git_bundle, commit: HEAD}   # or git_clone, with clone_url:
code:
  framework_from: installed             # this_repository when this repo *is* training-framework
  python_paths: ["."]                   # PYTHONPATH entries in the checkout
python_packages:
  requirements_file: requirements/hpc.txt   # fully pinned (see section 4)
  python_version: "3.12"                # optional pin; required for uv/conda
base:
  config_file: configs/fine-tuning.yaml
  session_index: 0
jobs:
  probes:
    per_experiment: {walltime: "02:00:00", gpus: 1, cpus: 8, memory: 64G}
    max_parallel_experiments: 4
    recipes:
      - name: frozen-backbone
        overrides:
          - {component: backbone_with_head, path: frozen, value: ["module.*"]}
        vary:
          - {component: optimizer, path: optimizer.kwargs.lr, values: [0.001, 0.0001]}
          - {path: session_config.rng_seed, values: [41, 42]}
```

- An address is `{component?, path}`: a component's entry wherever it is
  listed (top level or a `resources`/`hooks`/`steps` group), or a path in the
  session. Use a list path when a key contains a dot.
- Each experiment is: the base config, then the recipe's `overrides`, then
  one combination of its `vary` values (the last entry varies fastest). The
  example gives four experiments, `probes/frozen-backbone/000` to `003`.
- A recipe may have its own session config, for a change of structure
  (other components or wiring) rather than of values:
  `{name: attnpool, config_file: configs/model/vit-attnpool.yaml, vary: [...]}`.
  Its overrides and `vary` are checked against its own session (the same
  `base.session_index`). When any recipe sets `config_file:`, every
  experiment records the file it was made from, and `summarize` compares it
  as `config`, so two structures are never one seed group.
- To compare several structures on the same sweep, one recipe can list them:
  `config_files: [configs/model/vit-base.yaml, configs/model/vit-attnpool.yaml]`.
  Each file gets every combination of the recipe's `vary` values (the file
  is the slower axis), and its `overrides` and `vary` are checked against
  every file; an error ends with `(in <file>)`. With three files and the
  four combinations above, the recipe is twelve experiments. `status`
  shows each experiment's file.
- `per_experiment` resources are portable; partitions, QoS, and account come
  from the host. Scheduler-specific options go under `scheduler_options:
  {slurm: {...}}`.
- `ddp.world_size` must be set explicitly: with `nccl` it must equal `gpus`;
  with `gloo` it must fit in `cpus`.
- `base.engine_options` accepts `--heartbeat-timeout`,
  `--stop-sync-grace-period`, `--stop-sync-poll-interval`, and
  `--process_timeout_on_join`; config changes belong in overrides.
- `note` (optional) says why the mission runs; it is kept in the receipt and
  in each experiment's run record.

### Analysing another mission's checkpoints: `from_mission`

An [analysis session](07-analysis-sessions.md) evaluates one trained
checkpoint. To evaluate every checkpoint a training mission wrote, a recipe
takes its experiments from that mission:

```yaml
# missions/eval-arch.yaml
format_version: 1
name: eval-arch
project_file: naavik-project.yaml
base: {config_file: configs/analysis/test-set.yaml}   # session_type: analysis
jobs:
  eval:
    per_experiment: {walltime: "00:30:00", gpus: 1, cpus: 8, memory: 32G}
    recipes:
      - name: test-set
        from_mission:
          id: arch-20261118T090000Z-9e04101571a2   # the training mission
          experiments: ["short/*"]                 # optional: names or globs
          checkpoint: latest                       # latest (default) | all
          # into: {component: trained_model, path: model_checkpoint_path}   (the default)
```

```text
$ naavik validate missions/eval-arch.yaml --host workstation
note: listed the checkpoints of arch-20261118T090000Z-9e04101571a2 on host 'workstation' (read-only)
note: eval/test-set: short/base/003 of arch-... has no complete checkpoint; left out
ok: 6 experiment(s) in 1 job(s); commit ...
```

- **One experiment per checkpoint**, with the checkpoint's path written at
  `into`. `checkpoint: latest` takes each source experiment's newest
  complete checkpoint; `all` takes every one. It combines with `overrides`
  and `vary`: each checkpoint with every combination.
- **Chosen when the mission is compiled.** The plan names each
  experiment's exact checkpoint, so `validate` shows what will run, and
  `resume` continues with the same checkpoints. A training mission still
  running gives the checkpoints written so far.
- **Where the source is read.** On the host the analysis runs on, if the
  training ran there. For a mission on this machine, a training mission
  from another host is read from its fetched copy, `./hpc-results/<id>`
  (`naavik fetch <id>`; add `--checkpoints all` for `checkpoint: all`).
  A source on another host is refused, naming both: checkpoints are never
  copied between hosts. The source's receipt must be on this machine.
- **Refused:** a session that is not an analysis session, an `into` that
  `overrides` or `vary` also sets, a glob that matches no experiment, and
  a choice left with no complete checkpoint.
- **Each experiment records its source** (mission, experiment, checkpoint,
  iteration, and what the source experiment varied). `summarize` compares
  the results by those training settings, so analyses of training runs
  that differ only in their seed are one group
  ([Comparing runs](12-comparing-runs.md#analysis-missions-compared-by-their-training)).

### Sharing settings: the project file

`code_transfer`, `code`, `python_packages`, and `data_sources` are usually
the same for every mission of a project. Put them once in
`naavik-project.yaml` at the root of the repository, and name it from each
mission file:

```yaml
# naavik-project.yaml
code: {framework_from: installed, python_paths: ["."]}
python_packages: {requirements_file: requirements/hpc.txt, python_version: "3.12"}
data_sources: [...]
quick_run_defaults:           # what `naavik run` uses (see below)
  per_experiment: {walltime: "04:00:00", gpus: 1, cpus: 8, memory: 64G}
  max_parallel_experiments: 1
  host: workstation
```

```yaml
# missions/sweep.yaml
format_version: 1
name: sweep
project_file: naavik-project.yaml
base: {config_file: configs/fine-tuning.yaml}
jobs: {...}
```

The project file is read from the mission's commit, like the mission file,
and must be identical at `code_transfer.commit`. A key set in both files is
an error (each setting has one place). The staged `mission.yaml` is the two
merged, in the current names (missions before this release staged
`experiments.yaml` or `manifest.yaml`), and the mission's hash of its
mission file (`manifest_hash`) covers both. `quick_run_defaults` is only
read by `naavik run`.

## 3. Data

Data reaches a host in one of two ways. **Declared data**: a dataset
component says what it needs (the built-in datasets do; see
[Image datasets](../reference/datasets.md#its-data-fetched-once)), and the
host says how it gets there. **Data sources**: config values you list in
the mission file's `data_sources`. A source wins on the address it names,
so existing mission files keep working.

### Declared data: `dataset_delivery` on the host

```yaml
# hosts.yaml
hosts:
  cluster:
    ...
    dataset_delivery:
      default: download_on_host     # download_on_host | copy_from_here | already_on_host
      imagenet: already_on_host     # by requirement name
      flowers102: copy_from_here
```

- `download_on_host`: downloaded on the host, once, by a data preparation
  step before the experiments (on the login node or as a job:
  `data_preparation.on`, with `data_preparation.allocation`), at the
  config's path mapped through `local_to_host_paths`. The data is keyed by
  what it is (name, version, split), never by the commit: a new commit or
  working tree finds it ready. Data obtained by hand (ImageNet, Stanford
  Cars) cannot be downloaded; `validate` says to use `copy_from_here` or
  `already_on_host`.
- `copy_from_here`: the content-keyed copy below, of the config's path on
  this machine. When the data is not here yet, `validate` says so and
  `launch` (or `naavik run`) downloads it here first.
- `already_on_host`: already on the host at the config's path (mapped
  through `local_to_host_paths`); `launch` checks that it exists.

Two missions (or a mission and a direct run on the same machine) never
download the same data at once: downloading takes a lease on the location,
and a ready marker next to it (`<path>.tf-hpc-ready`) records what is
there. A location that holds other content (an older version) is refused,
never overwritten: remove it, or use a new path.

A host without `dataset_delivery` manages nothing: experiments read what is
at their configs' paths, as before, and `validate` notes which requirements
it left alone. Managed data needs the code check (it is what finds the
requirements), so `TRAINING_FRAMEWORK_CODE_CHECK=0` is refused on such a
host.

### Data sources

Only config values listed in `data_sources` are treated as data sources:

```yaml
data_sources:
  - name: flowers
    address: {component: flowers102, path: root}
    copy_from_here: {}                 # the value is a local path: copy it once
  - name: inat
    address: {component: inaturalist, path: root}
    already_on_host: {}                # already on the host (via local_to_host_paths)
  - name: places
    address: {component: places, path: root}
    prepare_on_host:                   # built or downloaded on the host
      command: ["python", "-m", "my_project.prepare_places", "{path}"]
      content_version: "1"
      rerun_when_changed: ["my_project/prepare_places.py"]   # optional: re-prepare only when this changes
      allocation: {walltime: "04:00:00", cpus: 4, memory: 16G}
```

`validate --suggest-data` prints `copy_from_here: {}` entries for absolute
local paths found in your configs. Copies are keyed by content and stored
once (under `directories.datasets`, or the `local_to_host_paths` target);
later missions reuse them. A ready artifact is never modified. Keep
`download: false` for copied or prepared datasets; ImageNet must be copied
or placed on the host. Sources whose values are the same local path (for
example the train and val components of one dataset) are one copy, counted
once.

## 4. Environments

With `python_environment.use_existing` on the host, experiments use your own
environment as is. Otherwise a managed environment is built once per
distinct input (pinned requirements, kind, interpreter identity, shell
commands) by an environment build step and shared by later missions. The
requirements file must pin everything, for example:

```bash
uv pip compile pyproject.toml --extra vision -o requirements/hpc.txt
```

[Setting up a project](10-hpc-projects.md) gives the full steps, for this
repository and for projects that depend on it.

With `framework_from: installed` it must pin `training-framework` exactly;
with `this_repository` it must not contain it. Your project's code is never
installed: experiments see it through `code.python_paths`.

Experiments, preflight, and data preparation commands run with the
environment's `bin` first on `PATH`, so `python` in a preparation command is
the environment's interpreter. For conda environments that need activation,
set `python_environment.activate_with_conda_run: true`: every command then
runs under `conda run -p <env>`.

## 5. Checking a mission: `validate` and `render`

```bash
naavik validate missions/sweep.yaml [--suggest-data]
naavik render   missions/sweep.yaml --output /tmp/preview
```

`validate` runs the same compiler as `launch`, entirely on your machine: it
never uploads or submits, and never connects to the host, except to list the
checkpoints a recipe's [`from_mission`](#analysing-another-missions-checkpoints-from_mission)
names (read-only; it says so in a note). It builds the whole plan
and reports every problem at once, each with the file and key it concerns,
instead of stopping at the first. In order, it checks:

1. **The host**: every key strictly (unknown keys, types, values), and the
   Slurm settings (no options naavik owns, no control characters).
2. **The repository**: the working tree is clean (no uncommitted or
   untracked files), the mission file is committed and identical at `HEAD`
   and `code_transfer.commit`, and the commit resolves to one. Everything
   after this is read from that commit, not from your working files.
3. **The mission file**: every key strictly; host settings (`account`,
   `partition`, ...) are refused; `code.python_paths` exist in the commit.
4. **The environment**: the interpreter (`site_interpreter` or
   `python_version`) against `requires-python`; a valid
   `interpreter_from`/`kind` pair; the requirements file fully pinned, with
   the framework rule for `framework_from`; the environment key.
5. **The experiments**: every job, recipe, and combination of `vary` values
   becomes an experiment config; every override and `vary` address must
   exist; two identical experiments are an error.
6. **Topology**: `ddp.world_size` against `gpus` (`nccl`) or `cpus`
   (`gloo`); only the allowed engine options.
7. **Data**: every source's address resolves in every experiment;
   `copy_from_here` paths exist and are allowed, and are hashed for their key
   (see below); `already_on_host` and `prepare_on_host` paths go through
   `local_to_host_paths`. Absolute local paths that are not data sources
   give a note; `--suggest-data` prints entries for them.
8. **Job scripts**: every Slurm script is rendered (resources, partitions,
   array sizes against `max_array_size`, dependencies, the early-stop
   signal), so mistakes `sbatch` would reject surface here.
9. **Every experiment's config, with the mission's own code**: the commit
   (or working tree) that will run is exported once into
   `~/.cache/training-framework/trees/<commit>`, and a separate Python
   process checks each experiment's config there, building nothing: session
   settings, every component name (a misspelling is refused, naming the
   experiment), wiring, companions, singletons, and every `config_schema`
   value. Your working tree is never used (unless the host runs it), so a fix
   you have not committed does not hide a broken commit.

The check runs with the host's environment when the host is this machine
with `python_environment.use_existing`, else with the interpreter running
`naavik`; its framework must be the one the host will run. With
`framework_from: this_repository` it is the mission's own (it must be in
`code.python_paths`). With a managed environment, a `git+...@<commit>` pin
needs that exact, clean framework commit here; a `==X.Y.Z` pin needs that
installed version, or a framework checkout whose `pyproject.toml` says
`X.Y.Z` (with a note). Otherwise `validate` says what to install.

What it cannot check without building the session is printed once as a
`note:`: `import_components` (it reads checkpoints), the dataflow and graph
order (they need built components), and configs that only their
constructor reads (your own components without a `config_schema`; every
built-in has one).
Those still fail when the experiment starts. Values handed to torch (an
optimizer's `lr`, a scheduler's `T_max`) are checked by torch, at startup.

`TRAINING_FRAMEWORK_CODE_CHECK=0` skips this check. It is never silent:
`validate`, `render`, and `launch` print a warning, and the receipt records
`code_check: skipped` (`done` otherwise).

It prints any `note:` lines, then `ok: N experiment(s) in J job(s); commit
...; environment ...` and exits 0, or `error: N problems:` with the list and
exits 2.

Hashing a `copy_from_here` source reads every file the first time; later
runs only compare each file's size and timestamps with a cache
(`~/.cache/training-framework/hashes.sqlite3`) and read only what changed.
For large datasets the progress line says which is happening:
`checking ...: 120,000/1,281,167 files, 14.2 GiB (0 hashed, 120,000
unchanged since the last run)`. An interrupted run keeps what it hashed.
Give a huge, fixed dataset a `content_version` to skip hashing entirely.

What `validate` cannot know, because it stays offline, `launch` checks
next: the site interpreter's identity over SSH, which environments and data
are already on the host, that `already_on_host` paths exist (preflight), and
whether Slurm accepts the account, partition, and limits.

**Part of a mission file.** `--job JOB` and `--recipe JOB/RECIPE` (each
repeatable, on `validate`, `render`, and `launch`) make only those recipes
into experiments, for example the short ones here and the long ones on a
cluster:

```bash
naavik launch missions/arch.yaml --job short --host workstation
naavik launch missions/arch.yaml --recipe long/full --host cluster
```

Experiments keep the names and numbers the whole mission file gives them,
and the config files they record are the whole file's, so the parts compare
as one mission. The receipt records the `selection`.

`render` runs the same checks and writes the mission directory it would
stage (job scripts, experiment configs, `plan.json`, `tasks.tsv`) to
`--output`, which must be empty or missing, so you can read exactly what
would be sent.

## 6. What `launch` does

1. Checks that `ssh` and `rsync` are installed (for an SSH host), then
   compiles the mission (the same checks as `validate`).
2. Checks the site interpreter's identity, then which environment, copies,
   and prepared data are already ready.
3. Claims a new mission directory, stages the exact commit (a Git bundle),
   and writes the plan files.
4. Shows a summary (experiments, environment, copies and their size) and
   asks for confirmation (`--yes` to skip; required when not interactive).
5. Copies pending data, runs login-node steps, preflights a ready
   environment, then submits the jobs: environment build -> data
   preparation -> one array per job. A failed job cancels its dependants.

Each step prints a timestamped progress line (staging, login-node steps,
every submitted job). Copies show rsync's live progress on a terminal and,
elsewhere (an IDE console, a log), a line every 30 seconds; the last line
says how many files were sent and how many were already there.

**Interrupted copies resume.** A copy goes to a fixed location and is marked
ready only when it completes, so nothing ever runs on partial data. Run
`launch` again: rsync skips the files already there and continues the
partial one. While a copy is incomplete, `<location>.tf-hpc-pending` records
which content it is; if the local data has changed since, `launch` refuses
to resume (rsync never deletes at the destination, so files you removed
would stay) and says what to delete.

**Ctrl-C** during `launch` is recorded in the receipt. Jobs already
submitted are cancelled, so no partial mission runs. If an `sbatch` was in
flight, the receipt says `needs reconciliation` with the `squeue` query to
check it; otherwise `failed`.

The receipt (`~/.local/state/training-framework/submissions/<mission-id>.json`,
mirrored as `receipt.json` in the mission directory) lists every job
submitted. If a submission's outcome is uncertain (a dropped connection),
nothing is retried; the receipt says `needs reconciliation` and prints the
`squeue --name=...` query that finds the job.

### One config, followed: `naavik run`

```bash
naavik run configs/model/vit-attnpool.yaml \
    --override optimizer:optimizer.kwargs.lr=3.0e-4 --note "attention pooling"
```

`naavik run CONFIG` runs one config as a mission of one experiment, built
from the project file at the root of the repository holding `CONFIG` (it is
required): job `run`, one recipe named after the file
(`run/vit-attnpool/000`), with `quick_run_defaults.per_experiment` (`--gpus`
and `--walltime` override them). The host is `--host`, else
`quick_run_defaults.host`, else the hosts file's `default_host`. Everything
else is `launch`: the same checks, the same staging, a receipt, a run record.
It asks only when the copies exceed `confirm_uploads_over` (`--yes` skips
that too).

- `--override PATH=VALUE` sets a path in the session;
  `--override component:PATH=VALUE` a path inside that component's entry,
  wherever it is listed (the mission file's [addresses](#2-the-mission-file-committed)).
  `VALUE` is YAML, read like a config file: `3e-4` is the *string*
  `"3e-4"`, `3.0e-4` the number. Repeatable.
- `--note TEXT` is the mission's `note`: kept in the receipt and in the run
  record (`naavik runs show` prints it).

It then **follows the experiment**: its log as it grows, then a last line
such as `run/vit-attnpool/000: done`, and exits with the experiment's exit
status (`1` when it ended without one: cancelled, no record). This works the
same on every host. **Ctrl-C** cancels the experiment (as `naavik cancel
--experiment`) and keeps following until it stops (mid-training it saves and
ends `interrupted`); a **second Ctrl-C** stops following, and prints the
`naavik logs <mission-id> <experiment> --follow` command to resume.
`--detach` returns once the experiment is launched.

With `run_code_from: working_tree` on the host, `naavik run` runs your
working tree as it is: no commit is needed for a quick try, and the snapshot
keeps exactly what ran.

## 7. Keeping track of your missions

Every `launch` leaves a **receipt**: a small JSON file describing the
mission (when and from where it was launched, its host, commit, experiment
count) and every job it submitted, with its job ID. Receipts live in
`~/.local/state/training-framework/submissions/<mission-id>.json` (or
`$XDG_STATE_HOME/...`), and a copy is kept in the mission directory on the
host as `receipt.json`. Two commands read them; neither changes anything.

| Command | Shows | Connects to the host |
|---|---|---|
| `missions` | every mission, one line each | no |
| `missions --live` | ... plus what each mission is doing now | yes, once per host |
| `missions --experiments` | ... plus every experiment of every mission | yes, once per host |
| `status <mission-id>` | one mission in detail | yes |

### All your missions: `missions`

```bash
naavik missions
```

```
mission                                 launched          host       experiments  submit
ssl-dino-20261015T091200Z-3f2a1c9e4b7d  2026-10-15 09:12  bigred200  4            submitted
vit-probes-20261014T220100Z-9a8b7c6d5e  2026-10-14 22:01  bigred200  24           submitted
vit-probes-20261014T214000Z-9a8b7c6d5e  2026-10-14 21:40  bigred200  24           staged (declined at confirmation)
imagenet-20261013T014700Z-1c2d3e4f5a6b  2026-10-13 01:47  bigred200  8            failed (rsync not found on this machine's PATH; ...)
```

Newest first, from the receipts on this machine; it works offline and needs
no host settings. The `submit` column is how the launch itself ended:

| `submit` | Meaning |
|---|---|
| `submitted` | every job was accepted by the scheduler |
| `staged (...)` | the mission was prepared but not submitted (declined, or over `confirm_uploads_over`) |
| `failed (...)` | the launch stopped with an error; jobs already submitted were cancelled |
| `needs reconciliation (...)` | an `sbatch` had an uncertain outcome; check with the `squeue --name` query in the receipt |
| `submitting` | `launch` was killed before it could record an outcome |

### What they are doing now: `--live`

```bash
naavik missions --live
```

```
mission                                 launched          host       experiments  submit     now
ssl-dino-20261015T091200Z-3f2a1c9e4b7d  2026-10-15 09:12  bigred200  4            submitted  running: 2 running, 2 pending
vit-probes-20261014T220100Z-9a8b7c6d5e  2026-10-14 22:01  bigred200  24           submitted  finished: 20 done, 3 interrupted, 1 failed
vit-probes-20261014T214000Z-9a8b7c6d5e  2026-10-14 21:40  bigred200  24           staged     not submitted
```

`--live` makes one SSH connection per host, whatever the number of missions:
one query of the scheduler for all your jobs, and one command that reads
every listed mission's experiment records. The `now` column summarizes the
experiments:

- `running: ...` / `queued: ...`: some experiments are still in the
  scheduler's queue (`queued` when every one of them is still waiting to
  start). Any Slurm state other than a final one (completed, failed,
  cancelled, timeout, out of memory, node failure, preempted, ...) counts as
  still in the queue, including held, suspended, and requeued jobs;
- `finished: ...`: none is left in the queue; the counts say how they ended;
- `(setup failed)`, `(prepare-... failed)`: a step the experiments depended
  on failed, so they were cancelled;
- `needs reconciliation: ...`: an `sbatch` had an uncertain outcome, so a
  job may exist that the receipt does not know; check with the `squeue
  --name` query in the receipt (`status` prints it) before launching again;
- `not submitted`, `submit failed`: nothing ran.

If a host cannot be reached over SSH (a missing or unreadable key, an empty
agent, no open control connection for two-factor login), every command that
connects says so with SSH's own reason and what to check, e.g. `error:
cannot reach bigred200 over SSH: me@h: Permission denied (publickey).`, and
exits non-zero. `missions --live` checks the connection even when this
machine has no receipts; a plain `missions` never connects, and an empty
list notes that it shows only missions launched from this machine.

When a mission's state cannot be known, `now` says why instead of guessing:
`? (cannot read <host>: ...)` for a host that cannot be reached, `? (mission
directory ... not found ...)` when the mission directory or an experiment
table is gone (deleted, purged from scratch, or the receipt points at
another host; it adds when the mission's jobs are still in the queue), or
`? (...receipt...)` for a damaged receipt. Other missions are unaffected.

### Every experiment: `--experiments`

```bash
naavik missions --experiments --since 1d
```

```
mission                                launched          host       experiments  submit     now
listing-20261015T012123Z-97dfad756615  2026-10-15 01:21  bigred200  3            submitted  running: 2 running, 1 done
    experiment     values                     state    detail
    cpu/seeds/000  session_config.rng_seed=1  done     /N/scratch/.../tasks/cpu/seeds/000/sessions/session_...
    cpu/seeds/001  session_config.rng_seed=2  running  job 8565732_1
    cpu/seeds/002  session_config.rng_seed=3  running  job 8565732_2
```

Each mission is followed by its steps (`setup`, `prepare-...`) and
experiments: the experiment's name (`<job>/<recipe>/<NNN>`), its varied
values, its state, and a detail. `--experiments` implies `--live`.
Experiment states:

| State | Meaning | Detail |
|---|---|---|
| `pending`, `running` (and other Slurm states) | in the scheduler's queue | the job or array element, e.g. `job 4711_3` |
| `done` | exited 0 | its run directory |
| `failed (exit N)` | exited non-zero | the last line of the error |
| `interrupted` | stopped by the early-stop signal before the time limit, or by `scancel --signal=USR1` | |
| `ended without a record` | the job ended without writing its record: killed for memory or time, or the node failed | the log to read |
| `cancelled` | cancelled before it ran (for example, its environment build failed) | |
| `built`, `already ready`, `prepared`, `ran on login` | (steps) the environment or data step's outcome | its location |

### Choosing what to list

| Option | Lists |
|---|---|
| `--since 30m` / `12h` / `7d` / `2w` | missions launched within that time |
| `--name TEXT` | missions whose id contains `TEXT` (ids start with the mission file's `name`) |
| `--active` | only missions with queued or running experiments (implies `--live`) |
| `--limit N` | the newest `N` |
| `--json` | the same information as JSON (with `--experiments`, every experiment), for scripts and notebooks |
| `--receipts DIR` | receipts from another directory |

For example, `missions --active` answers "what is running right now", and
`missions --experiments --name vit-probes --limit 1` shows the latest sweep
experiment by experiment.

### Missions launched from other machines: `--remote`

Receipts are per machine: a mission launched from your laptop is not in the
list on a workstation. `--remote` also reads the receipts kept in the
mission directories on the host (under its `directories.missions`) and adds
the missions that are missing here, with a `from` column naming the machine
that launched them:

```bash
naavik missions --remote --live
```

`--remote` needs a host (`--host`, or the hosts file's `default_host`). So
do `--live` and `--experiments` for receipts written before receipts
recorded their host; newer receipts carry it.

### Every host at once: `--all-hosts`

```text
$ naavik missions --all-hosts
mission                     launched          host         from          experiments  note               submit
arch-20261118T0900Z-...     2026-11-18 09:00  workstation  this machine  6            attn pool vs base  submitted
arch-20261118T0905Z-...     2026-11-18 09:05  cluster      laptop        8            attn pool vs base  submitted
?                                             lab-box                                                    (cannot list: cannot reach lab-box over SSH: ...)
```

Every host of the hosts file, one connection each: the missions in its
`directories.missions` beside the receipts here (each mission once), with
the `from` column of `--remote`. A host that cannot be listed (out of reach,
a mistake in its settings) is one `?` row giving the reason and an `error:`
line; the others are listed all the same, and the exit status is 1. It
combines with `--live`, `--experiments`, and `--json`, not with `--host` or
`--remote`.

The `note` column shows a mission's note (`naavik run --note`, a mission
file's `note`) when some mission has one; `status` prints it as a `note:`
line.

### One mission in detail: `status`

```bash
naavik status <mission-id>      # or the receipt's path
```

```
mission vit-fine-tuning-20261011T030035Z-2926f55be79e: submitted | /N/scratch/...
experiment         values                 state            detail
setup                                     built            /N/scratch/me/naavik/envs/ea3c...
probes/frozen/000  lr=0.001 rng_seed=41   done             .../sessions/session_...
probes/frozen/001  lr=0.001 rng_seed=42   running          job 4711_1
probes/frozen/002  lr=0.0001 rng_seed=41  failed (exit 1)  ValueError: ...
```

The same rows as `missions --experiments`, for one mission, with its host
and directory. `status` connects to the host recorded in the receipt,
whatever your default host is; the hosts file is read only for receipts
written before receipts recorded their host, and a `--host` that points at a
different one is refused rather than queried. A mission whose directory is
gone is an error, not a list of experiments without records. (Values are
shortened above; both print each varied value's path, such as
`optimizer.kwargs.lr=0.001`.) In a mission that uses more than one session
config file, the values start with the experiment's file,
`config=configs/model/vit-attnpool.yaml ...`, and `--json` gives it as
`config_file`.

### Logs

```bash
naavik logs <mission-id> probes/frozen/001             # the latest attempt's log
naavik logs <mission-id> probes/frozen/001 --attempt 1
naavik logs <mission-id> setup --follow                # a step; Ctrl-C stops following
```

An experiment's log is `<mission-dir>/logs/<job>/<job id>_<index>.out` on
the host (a step's `<mission-dir>/logs/<step>/<job id>.out`); `logs` finds
it from the receipt and reads it where the mission ran.

Every log line written by naavik on the host starts with `[hpc ...]` (`[hpc
setup ...]`, `[hpc prepare ...]`) and a UTC time. An experiment's log opens
with its name, varied values, commit, host, and GPUs, and closes with
`finished: exit N (ok|failed|interrupted)` and the record's path; the
wrapper adds when the scheduler's advance signal arrived and the
experiment's exit status. Environment build and data preparation logs show
each step, whom they are waiting for when another job holds the lease, and,
on failure, a single `FAILED:` line.

### Cancelling: `cancel`

```bash
naavik cancel <mission-id>                     # every job still queued or running
naavik cancel <mission-id> --experiment probes/frozen/003 --experiment probes/frozen/004
```

`cancel` lists what it will stop and asks (`--yes` skips the question; it is
required when not interactive). It acts only on jobs the scheduler still
lists, and says which had already ended. Running experiments get the
scheduler's stop (Slurm: `scancel`; a builtin runner sends the early stop
signal first, so the experiment stops cooperatively and records
`interrupted`, then kills what is left). The receipt records what was
cancelled: `status` shows an experiment cancelled with `--experiment` as
`cancelled`, even while its job runs on for the others.

## 8. Results

```
<directories.missions>/<mission-id>/
  groups/<job>/tasks.tsv          # array index -> experiment, varied values, config
  tasks/<job>/<recipe>/<NNN>/
    config.yaml  variant.json  record.json
    sessions/session_<timestamp>/ # checkpoints, events, ...
  logs/<job>/<job id>_<index>.out
```

(The directory layout keeps its earlier names, `groups/` and `tasks/`, so
missions launched before the renames read the same.)

`record.json` ties each result to its commit, config, environment, and exit
status. To work with results on your machine, `fetch` them (section 11). In
configs that run on a cluster set the `tensorboard` component's `serve:
false`, then view a whole mission at once:

```bash
ssh -L 6006:localhost:6006 bigred200
tensorboard --logdir <mission dir>/tasks --port 6006
```

## 9. Interrupts and time limits

Each array job asks the scheduler for an early signal
(`process_timeout_on_join` plus the job's `stop_grace_period`, default 120s,
before the time limit). The experiment wrapper turns it into a cooperative
stop of the engine; the experiment records `interrupted` and exits non-zero.
The latest periodic checkpoint is kept. Slurm's own SIGTERM at the limit
stops everything without that grace, so leave enough `stop_grace_period` for
one iteration.

## 10. Resuming unfinished experiments

An experiment stopped by its time limit (`interrupted`) or killed (`ended
without a record`, e.g. out of memory or time) can continue from its latest
checkpoint:

```bash
naavik resume <mission-id> --walltime 2-00:00:00
```

```
resume vit-probes-20261014T220100Z-9a8b7c6d5e as attempt 2: 3 experiment(s)
  probes/frozen/001   interrupted             from iteration 35640/89124  tasks/probes/frozen/001/sessions/.../checkpoints/...
  ...
  job probes: walltime 2-00:00:00, 1 GPU(s), 8 CPU(s), 64G
Resume? [y/N]
```

What it does:

1. Picks the mission's experiments that are `interrupted` or `ended without
   a record`. Failed ones are left out (resuming repeats their error); add
   `--include-failed` after fixing the cause, or name experiments with
   `--experiment <name>` (repeatable; a `cancelled` one too). A finished
   experiment is never resumed. One with no checkpoint (it never got that
   far, or never ran) is skipped, unless `--restart` is given: it then runs
   again from its config as the new attempt (`restart from its config (no
   checkpoint)` in the summary; its record says `restarted`).
2. Finds each experiment's newest **complete** checkpoint (one with its
   `manifest.json`; a checkpoint still being written is skipped) under its
   sessions directory, by iteration. Experiments that cannot resume are
   listed with the reason: no `checkpointer` in the config, no checkpoint
   written yet, a checkpoint already at the run's total iterations, or an
   explicit `checkpointer.checkpoints_dir` (not searched; pass
   `--checkpoint <experiment>=<path>`). The total is the one the checkpoint's
   manifest records (`total_iterations`), which counts extensions and a
   length given in `epochs`. A checkpoint written before manifests recorded
   it falls back to the config's `max_iterations`, or to `?` when the config
   gives `epochs`.
3. Checks the host can still run them: the mission's source checkout and
   environment are there, and its framework supports resuming (missions
   whose environment predates `resume` are refused: launch a new mission).
   It refuses while any job of the mission is still queued or running.
4. Submits one job array per job with exactly those experiments, as
   **attempt 2** (then 3, ...), in the same mission directory, with the same
   environment, experiment wrapper (early-stop signal, logs), and scheduler
   settings: account, QoS, partitions, and `sbatch_options` come from the
   mission's own plan, so editing or switching hosts afterwards does not
   change them, and no hosts file is needed to resume. A submission that
   fails part-way cancels only that attempt's jobs, and its experiments stay
   resumable. Each experiment runs the engine with `--resume-session
   <checkpoint>`: weights, optimizer, learning-rate schedule, RNG, and data
   position continue exactly, in the same session directory.

Only the attempt's resources may change: `--walltime`, `--cpus`,
`--memory`, `--max-parallel`. The session config and the number of GPUs
stay as they were (a resumed session continues unchanged). `--dry-run`
shows the summary and the job scripts without submitting; `--yes` skips the
question (required when not interactive).

Afterwards `status` and `missions --experiments` follow each experiment's
latest attempt (`running  job 150_0 (attempt 2)`, then `done ... (attempt
2)`), and `missions` shows `submitted (attempt 2)`. The experiment's
`record.json` describes the latest attempt (`attempt`, `resumed_from`) and
keeps the earlier ones under `previous_attempts`; the receipt lists every
attempt (`attempts`), with its jobs, experiments, checkpoints, and
resources, under the same transactional rules as `launch`.

To lose less work when an experiment is stopped, checkpoint often enough
(`checkpointer.checkpoint_every`): the stop keeps the latest periodic
checkpoint.

## 11. Getting results back

Every experiment also has a [run record](13-run-registry.md) on the machine
it ran on (its `record.json` names it as `run_id`); `naavik runs sync`
brings those records here, without the run files.

```bash
naavik fetch <mission-id>            # into ./hpc-results/<mission-id>/
```

```
fetch vit-probes-20261014T220100Z-9a8b7c6d5e: 2 experiment(s) from bigred200 into hpc-results/vit-probes-...
  probes/frozen/000              done                     iteration 6000
  probes/frozen/001              done                     iteration 6000
50 file(s), 1.2 GiB; already here: 0 file(s), 0 B
fetched 50 of 50 file(s) (1.2 GiB); 0 were already here
checkpoints: 2 verified
view: tensorboard --logdir hpc-results/vit-probes-.../tasks
```

For each selected experiment, `fetch` downloads its `record.json`,
`config.yaml`, and `variant.json`; its run directory without checkpoints
(TensorBoard events, dumped configs, component outputs); its **latest
complete checkpoint** (the newest with a `manifest.json`, across resumed
attempts); and the job log of every attempt. The mission's `plan.json`,
`receipt.json`, and experiment tables come too. Files keep their paths
relative to the mission directory, under `<dest>/<mission-id>/`.

| Option | Effect |
|---|---|
| `--dest DIR` | download into `DIR/<mission-id>/` (default `./hpc-results`) |
| `--state done` (default) / `finished` / `all` | successful experiments only; any experiment that ended (interrupted and failed ones too, for debugging); also experiments still queued or running (their files may be mid-write) |
| `--experiment NAME` | these experiments, whatever their state (repeatable) |
| `--checkpoints latest` (default) / `all` / `none` | which checkpoints of each experiment |
| `--no-logs` | skip the job logs |
| `--include GLOB` | also files matching a shell glob relative to the mission directory (e.g. an explicit `checkpoints_dir` inside it) |
| `--dry-run` | show the experiments, checkpoints, and sizes; download nothing |
| `--yes` | do not ask; `fetch` asks only before downloading more than 1 GiB |

It runs one `rsync` for the whole mission, resumable: running `fetch` again
(say, after more experiments finished, or after an interruption) copies only
new or changed files. Each fetched checkpoint is verified against its
manifest (the SHA-256 of every component file); a mismatch is reported and
`fetch` exits non-zero. `fetched.json` in the mission's local directory
records what was fetched, from which host, and each checkpoint's iteration
and verification. Nothing on the host changes.

Fetched checkpoints load locally like any other
(`Checkpointer.load_checkpoint(path)`, `load_component`, analysis
sessions), with your project's components importable. Paths stored inside
the files (configs' `sessions_dir`, records' `run_directories`) still name
the host; `fetched.json` maps the mission to its local directory. Resuming a
fetched checkpoint locally with `--resume-session` is not supported: the
checkpoint's session would write to its session directory on the host.

To compare the fetched experiments, their varied values and TensorBoard
scalars, with runs that differ only in their seed grouped into mean ± std:

```bash
naavik summarize --mission <mission-id> --metric train/loss:min
```

See [Comparing runs](12-comparing-runs.md).

## Limitations

Single-node experiments only; schedulers: Slurm and `builtin`; `builtin`
does not limit CPUs or memory; resuming cannot change the config or the
number of GPUs, and missions launched before `resume` existed cannot be
resumed; components that bind fixed ports (other than `tensorboard` with
`serve: false`) fail when experiments share a node; explicitly configured
component output directories are not redirected per experiment.
