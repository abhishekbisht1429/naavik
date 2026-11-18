# HPC configuration reference

Every key of the two files the [HPC batch layer](../guide/08-hpc-batches.md)
reads. Both are strict: an unknown key, a wrong type, or a key in the wrong
file is an error that `validate` reports with its location; nothing is
silently ignored.

- The **profile** holds everything specific to one cluster and one user. It
  lives on your machine and is never committed.
- The **experiment file** holds the experiments (earlier called the
  *manifest*; not a checkpoint's `manifest.json`). It is committed with your code
  and runs unchanged on any cluster that has a profile.

Sizes are an integer with an optional `K`, `M`, `G`, or `T` suffix (powers
of 1024): `64G`. Walltimes are `[D-]HH:MM:SS`: `02:00:00`, `1-12:00:00`.
Durations are an integer with `s`, `m`, or `h`: `120s`. Names (batch,
groups, experiments, data sources) use letters, digits, `_`, and `-` only,
since they become directory and job names.

## The target file

`~/.config/training-framework/targets.yaml` (or `$XDG_CONFIG_HOME/...`, or
`--targets PATH`).

```yaml
default: bigred200          # used when --target is not given
targets:
  bigred200: {...}          # one entry per cluster/account
```

| Key | Required | Meaning |
|---|---|---|
| `default` | no | Name of the profile to use without `--target`. With neither, commands stop. |
| `targets` | yes | Mapping of profile name to profile: one entry per target. `profiles` is its earlier name and is still read (a file may not have both). |

### Profile keys

| Key | Required | Default | Meaning |
|---|---|---|---|
| `ssh_host` | this or `local` | | Host passed to `ssh` and `rsync`. Usually an alias from `~/.ssh/config`, which supplies user, key, and connection sharing. |
| `local` | this or `ssh_host` | | `true`: the target is this machine; commands run here, without `ssh` or `rsync`. Any other value is refused. |
| `roots` | yes | | Absolute directories on the target; see [roots](#roots). |
| `scheduler` | yes | | What runs the jobs: `slurm`, or `host` for a machine without a scheduler. |
| `slurm` | with `scheduler: slurm` | | Slurm settings; see [slurm](#slurm). A block for a scheduler the profile does not use is an error. |
| `host` | with `scheduler: host` | | Host scheduler settings (`host: {}` without GPUs); see [host](#host). |
| `worktree` | no | `commit` | What a batch runs: `commit` (HEAD; the working tree must be clean) or `snapshot` (the working tree as it is, made into a commit on HEAD); see the guide's [Uncommitted work](../guide/08-hpc-batches.md#uncommitted-work-worktree-snapshot). |
| `snapshot_limit` | no | `50M` | With `worktree: snapshot`: most bytes of untracked files a snapshot may take into Git; above it the snapshot is refused, naming the largest files. |
| `hooks` | no | none | Shell lines run before commands; see [hooks](#hooks). |
| `env` | no | managed venv from the site Python | Where tasks' Python environment comes from; see [env](#env). |
| `setup_location` | no | `job` | Where a managed environment is built: `job` (a Slurm job the task arrays depend on) or `login` (on the login node, during `submit`). Use `login` when compute nodes have no internet. |
| `prepare_location` | no | `job` | Where `prepare` data steps run: `job` or `login`, as above. |
| `path_map` | no | none | Local path prefix → cluster path prefix; see [path_map](#path_map). |
| `upload_limit` | no | `20G` | Total size of new uploads above which `launch` asks for explicit confirmation (`--yes` also accepts it). |
| `setup_resources` | no | `{walltime: "01:00:00", cpus: 4, memory: 16G}` | Resources of the setup job (`setup_location: job`); same keys as [resources](#resources). Make `memory` fit the CPU partition's nodes. Also the resources of the jobs that fetch declared data. |
| `data_modes` | no | not managed | How declared data reaches this target: `{default: MODE, <requirement name>: MODE}`, MODE one of `fetch`, `upload`, `present` (`default` defaults to `fetch`); see the guide's [Declared data](../guide/08-hpc-batches.md#declared-data-data_modes-in-the-profile). Without it, declared data is not managed. |

#### roots

All three are required, absolute paths on the cluster.

| Key | Holds |
|---|---|
| `batches` | One directory per submitted batch: source checkout, plan, task configs, records, runs, logs. |
| `envs` | Managed environments, one directory per environment key, shared by batches. Prefer a filesystem that is not purged. |
| `data` | Uploaded datasets, stored by content key and reused by later batches (unless `path_map` places them). |

#### slurm

| Key | Required | Meaning |
|---|---|---|
| `defaults` | no | `#SBATCH` options added to **every** job of every batch (setup, prepare, task arrays): `{account: r00000, qos: normal}` renders `#SBATCH --account=r00000` and `#SBATCH --qos=normal`. Values are strings or integers. Options the HPC layer sets itself (time, cpus, memory, GPUs, partition, array, dependency, output, signal, job name, ...) are refused; `account` and `qos` are allowed here and only here. A group's `scheduler_options` with the same key wins for that group. |
| `partitions` | no* | `{gpu: <name>, cpu: <name>}`: the partition for jobs that request GPUs and for those that do not. *A batch with a GPU group needs `gpu`; one with a CPU job (including setup and prepare jobs) needs `cpu`. |
| `max_array_size` | yes | The cluster's `MaxArraySize` (`scontrol show config`). A group with more tasks is refused at validation instead of by `sbatch`. |

Control characters (such as newlines) are refused in every Slurm value, so
no value can add an extra `#SBATCH` line.

#### host

| Key | Default | Meaning |
|---|---|---|
| `gpus` | `[]` | GPU indices (as `nvidia-smi` numbers them) that jobs may use, in the order they are tried. A group may not request more GPUs than the pool has. Each GPU is given to one task at a time across every batch on the machine. |
| `busy_gpus` | `wait` | `wait`: skip a pool GPU someone else is using (a compute process `nvidia-smi` lists, or 256 MiB or more of memory in use, which also shows other containers' processes) until it is free; `use`: only the pool decides. |

The host scheduler takes no `scheduler_options`. It runs a runner with the
target's `python3` (3.8 or newer, checked by `launch`), keeps its job state
under `$XDG_STATE_HOME/training-framework/host/<hostname>/` on the target,
and does not limit a task's CPUs or memory.

#### hooks

Lists of shell lines, run in order before the command of each phase, in the
job script or the remote shell (`module load ...`, `export ...`). They are
part of the environment key (setup) or prepare key, so changing them gives
a new environment or prepared dataset.

| Key | Runs before |
|---|---|
| `common` | everything below |
| `setup` | building a managed environment, and `probe-python` |
| `prepare` | `prepare` data steps |
| `task` | each training task |

#### env

| Key | Default | Meaning |
|---|---|---|
| `existing` | | Absolute path of an environment you manage yourself on the target. Tasks use `<existing>/bin/python`; nothing is built and the other keys below (except `conda_run`) do not apply. In `dependency` mode it must contain `training-framework`. With `local: true`, `current` means the environment of the interpreter running `naavik` (e.g. your IDE's project interpreter), resolved into the plan. |
| `python_source` | `site` | Where a managed environment's interpreter comes from: `site` (`python3` after the hooks, e.g. a module), `uv` (`uv python install <environment.python>`), or `conda` (conda's base Python). `uv` and `conda` need `environment.python` in the experiment file. |
| `site_python_identity` | | Required with `python_source: site`: the site interpreter's `{implementation, version, executable, platform}`, printed by `naavik probe-python`. `submit` checks it against the cluster and stops with the new block if the site changed it. |
| `backend` | `venv` | Kind of managed environment: `venv` (from the interpreter above) or `conda` (`conda create -p <env> -c conda-forge python=<environment.python>`). A `conda` environment installs its own Python, so it needs `python_source: conda` (or `uv`) and `environment.python`; with `site` it is refused. |
| `conda_packages` | `[]` | With `backend: conda`: extra conda packages for `conda create` (e.g. `["cudatoolkit=11.8"]`); part of the environment key. |
| `conda_run` | `false` | Run every command through `conda run --no-capture-output -p <env>` (for conda environments that need activation). Needs a conda environment. |
| `bootstrap_python` | | Path of the interpreter that runs the setup bootstrap (and is the base of a `venv`), instead of the one chosen by `python_source`. For sites whose default Python lacks `venv`/`ensurepip`. |

All environments put their `bin` first on `PATH` for tasks, preflight, and
prepare commands.

#### path_map

```yaml
path_map:
  /home/me/datasets: /N/project/mylab/datasets
```

Absolute local prefixes to absolute cluster prefixes; the longest matching
prefix wins. Used to translate config values of [data sources](#data): a
`cluster` source's local path becomes its cluster path, a `prepare` source
is prepared at the mapped path, and an `upload` goes to the mapped path
instead of `roots.data/<key>/...`.

## The experiment file

A YAML file committed in your repository; its path is the argument of
`validate`, `render`, and `launch`. Site keys (`ssh_host`, `roots`,
`account`, `partition`, `qos`, `profile`, ...) are refused here.

| Key | Required | Default | Meaning |
|---|---|---|---|
| `version` | yes | | Experiment file format version; must be `1`. |
| `name` | yes | | Batch name; batch IDs are `<name>-<UTC time>-<short commit>`. |
| `source` | no | bundle of `HEAD` | Which commit runs and how it reaches the cluster; see [source](#source). |
| `code` | yes | | How tasks find the framework and your code; see [code](#code). |
| `environment` | no | | Requirements for a managed environment; see [environment](#environment). |
| `training` | yes | | The training config and engine options; see [training](#training). |
| `data` | no | | Config values that are datasets; see [data](#data). |
| `groups` | yes | | Groups of experiments sharing resources; see [groups](#groups). |
| `project` | no | | A project file (a path in the repository) adding `source`, `code`, `environment`, and `data`; see [The project file](#the-project-file). |
| `description` | no | | Why the batch runs; kept in the receipt and each task's run record (`naavik run --note`). |

`code` is required in the experiment file or its project file.

### The project file

`naavik-project.yaml` at the repository root (any path, when an experiment file
names it with `project:`): what a project's batches share. Read from the
batch's commit; it must be identical at `source.ref`. A key in both the
experiment file and the project file is an error.

| Key | Meaning |
|---|---|
| `source`, `code`, `environment`, `data` | As in the experiment file. |
| `run_defaults.resources` | [resources](#resources) of a `naavik run` task (`--gpus`, `--walltime` override). |
| `run_defaults.max_parallel` | `max_parallel` of a `naavik run` group (a positive integer). |
| `run_defaults.target` | The target `naavik run` uses when no `--target` is given. |

### source

| Key | Default | Meaning |
|---|---|---|
| `strategy` | `bundle` | `bundle`: a Git bundle of the commit is copied over SSH (no Git credentials needed on the cluster). `fetch`: the cluster clones `url` itself. |
| `ref` | `HEAD` | The commit to run (branch, tag, or SHA), resolved on your machine. The experiment file must be identical at `HEAD` and at `ref`. |
| `url` | | Repository URL the cluster clones from; required with `fetch`, refused with `bundle`. |

The working tree must be clean: uncommitted or untracked files are an
error.

### code

| Key | Required | Meaning |
|---|---|---|
| `framework` | yes | `checkout`: this repository *is* training-framework; it runs from the staged commit and the requirements must not contain it. `dependency`: training-framework is installed in the environment: with a managed environment the requirements must pin it exactly once; with `env.existing` the environment must already contain it (no requirements file is needed). |
| `paths` | yes | Relative directories of the repository put on `PYTHONPATH` for tasks (e.g. `["."]`, `["src"]`). Your code is never pip-installed. |

### environment

Used only for managed environments (no `env.existing`).

| Key | Meaning |
|---|---|
| `requirements` | Repository path of a fully pinned requirements file; required for a managed environment. It is installed with `pip install --no-deps`, so it must list every package, transitive ones included (generate it with `uv pip compile`; see [Setting up a project for HPC batches](../guide/10-hpc-projects.md)). Each line is `name==X.Y.Z`, or a direct reference with a full 40-character commit SHA or an archive hash; the only options are `--index-url`, `--extra-index-url`, `--find-links`, and `--hash=` after a requirement. |
| `python` | Python version `"X.Y"` or `"X.Y.Z"`, quoted (YAML reads `3.12` as a number). Required with `python_source: uv` or `conda`; with `site` it must match the identity. It must satisfy `requires-python` in your `pyproject.toml`. |

### training

| Key | Default | Meaning |
|---|---|---|
| `config` | | Repository path of the training config (the usual `sessions:` file). Required. Its [includes](../guide/04-configuration.md#sharing-configuration-between-files) are read at the batch's commit, by paths that stay inside the repository; each task's rendered config has them merged. |
| `session` | `0` | Index of the session in `sessions` that the tasks run. |
| `engine_options` | `[]` | Engine command-line options passed to every task. Only `--heartbeat-timeout`, `--stop-sync-grace-period`, `--stop-sync-poll-interval`, and `--process_timeout_on_join` are accepted; config changes belong in overrides. |

### data

```yaml
data:
  sources:
    - name: flowers
      target: {component: flowers102, path: root}
      upload: {}
```

Only config values listed here are treated as data; every other path is
left as written. Each source has:

| Key | Meaning |
|---|---|
| `name` | Name of the source, unique in the experiment file. |
| `target` | [Address](#addresses) of the config value holding the path. |
| exactly one of `upload`, `cluster`, `prepare` | What kind of data it is (below). |

**`upload`**: the value is an absolute path on your machine; it is copied
once with rsync and reused by every batch with the same content.

| Key | Meaning |
|---|---|
| `version` | Optional. Key the upload by the source's `name` and this string instead of hashing its content (for very large datasets that never change). Change the string to upload again; renaming the source also gives a new key, so it uploads again too. |

Paths inside the repository, containing it, `/`, or your home directory are
refused.

**`cluster`**: the data is already on the cluster.

| Key | Meaning |
|---|---|
| `path` | Optional absolute cluster path. Without it the value is translated by `path_map` (or used as is). It must exist; preflight checks it. |

**`prepare`**: the data is built or downloaded on the cluster, once, before
the tasks run, at the value's (mapped) path.

| Key | Meaning |
|---|---|
| `command` | Required. Argument list run from the staged source; `{path}` is replaced by the target path (`{{`/`}}` for literal braces). It must create the path and be safe to run again. |
| `version` | Required. Change it to prepare again. |
| `inputs` | Optional repository paths; the step is redone only when these change. Without it, any new commit prepares again. |
| `resources` | [Resources](#resources) of the prepare job; required with `prepare_location: job`, refused with `login`. |

### groups

```yaml
groups:
  probes:
    resources: {walltime: "02:00:00", gpus: 1, cpus: 8, memory: 64G}
    max_parallel: 4
    experiments: [...]
```

A mapping of group name to group. Each group becomes one job array; all its
tasks get the same resources.

| Key | Required | Default | Meaning |
|---|---|---|---|
| `resources` | yes | | [Resources](#resources) of each task. |
| `experiments` | yes | | Non-empty list of [experiments](#experiments). |
| `max_parallel` | no | no limit | Most tasks of the group running at once (Slurm's `%N` array throttle). |
| `stop_grace` | no | `120s` | Time for the last iteration and checkpoint when the task is stopped before its walltime; the early signal is sent `process_timeout_on_join + stop_grace` before the limit. |
| `scheduler_options` | no | | Extra scheduler options for this group only: `{slurm: {constraint: a100}}` renders `--constraint=a100` (every option is written `--key=value`). The same rules as `slurm.defaults` apply, except that `account` and `qos` are refused (they belong in the profile); a key here overrides the profile's default. |

#### resources

Used by groups, `prepare` sources, and the profile's `setup_resources`.

| Key | Required | Default | Meaning |
|---|---|---|---|
| `walltime` | yes | | Time limit `[D-]HH:MM:SS`. |
| `memory` | yes | | Memory per job (one node), e.g. `64G`. |
| `cpus` | yes | | CPU cores per task. |
| `gpus` | no | `0` | GPUs per task on one node. With `nccl`, `ddp.world_size` must equal it; with `gloo`, `world_size` must fit in `cpus`. |

`partition`, `account`, and `qos` are refused here: they come from the
profile.

#### experiments

| Key | Required | Meaning |
|---|---|---|
| `name` | yes | Unique within the group; tasks are `<group>/<name>/000`, `001`, ... |
| `overrides` | no | List of `{component?, path, value}`: values set for every task of the experiment. |
| `matrix` | no | List of `{component?, path, values}`: one task per combination of the values; the last entry varies fastest. Without a matrix the experiment is one task. |
| `config` | no | Its own training config (a repository path; default `training.config`), for a change of structure. Its session is `training.session`, and its overrides and matrix are checked against it. When any experiment sets it, every task records its file (`config_file`), compared as `config` by `summarize`. |

Each task's config is: its training config (with its includes merged),
then the experiment's `overrides`, then its combination of matrix values.
A `${...}` interpolation brought in by an override or matrix value is
refused, as in the training config itself.

### Addresses

`{component?, path}` names one value in the selected session of the
training config.

- With `component`, `path` is inside that component's entry, wherever it is
  listed (top level, or inside a `resources`, `hooks`, or `steps` group).
- Without it, `path` starts at the session (`session_config.rng_seed`).
- `path` is a dotted string (`optimizer.kwargs.lr`) or a list of keys
  (`["optimizer", "kwargs", "lr"]`); use the list form when a key contains a
  dot.
- `session_config.sessions_dir` is owned by the HPC layer (each task gets its
  own) and cannot be addressed.
