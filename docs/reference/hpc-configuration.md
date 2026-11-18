# Configuration reference: hosts, missions, projects

Every key of the three files the [mission launcher](../guide/08-hpc-batches.md)
reads. All are strict: an unknown key, a wrong type, or a key in the wrong
file is an error that `validate` reports with its location; nothing is
silently ignored.

- The **hosts file** (`hosts.yaml`) holds everything specific to the
  machines you run on and to you. It lives on your machine and is never
  committed.
- A **mission file** holds the experiments of a mission. It is committed with
  your code and runs unchanged on any host.
- The **project file** (`naavik-project.yaml`) holds what a project's
  mission files share. It is committed too.

Files written with [earlier names](#earlier-names) are still read; each
launch warns once and lists every `old -> new`, to rename by hand.

Sizes are an integer with an optional `K`, `M`, `G`, or `T` suffix (powers
of 1024): `64G`. Walltimes are `[D-]HH:MM:SS`: `02:00:00`, `1-12:00:00`.
Durations are an integer with `s`, `m`, or `h`: `120s`. Names (missions,
jobs, recipes, data sources) use letters, digits, `_`, and `-` only, since
they become directory and job names.

## The hosts file

`~/.config/training-framework/hosts.yaml` (or `$XDG_CONFIG_HOME/...`, or
`--hosts-file PATH`).

```yaml
default_host: bigred200       # used when --host is not given
hosts:
  bigred200: {...}            # one entry per machine or cluster account
```

| Key | Required | Meaning |
|---|---|---|
| `default_host` | no | The host to use without `--host`. With neither, commands stop. |
| `hosts` | yes | Mapping of host name to its settings. |

### Host keys

| Key | Required | Default | Meaning |
|---|---|---|---|
| `connection` | yes | | How the host is reached: `local` (this machine; commands run here, without `ssh` or `rsync`) or `{ssh: <name>}` (a host name or an alias from `~/.ssh/config`, which supplies user, key, and connection sharing). |
| `scheduler` | yes | | What runs the jobs, and its settings: `{kind: slurm, ...}` (see [Slurm](#scheduler-kind-slurm)) or `{kind: builtin, ...}` (naavik's own, for a machine without a scheduler; see [builtin](#scheduler-kind-builtin)). |
| `directories` | yes | | Absolute directories on the host; see [directories](#directories). |
| `run_code_from` | no | `commit` | What a mission runs: `commit` (HEAD; the working tree must be clean) or `working_tree` (the working tree as it is, made into a commit on HEAD); see the guide's [Uncommitted work](../guide/08-hpc-batches.md#uncommitted-work-run_code_from-working_tree). |
| `working_tree_untracked_limit` | no | `50M` | With `run_code_from: working_tree`: most bytes of untracked files it may take into Git; above it the launch is refused, naming the largest files. |
| `shell_commands_before` | no | none | Shell lines run before commands; see [shell_commands_before](#shell_commands_before). |
| `python_environment` | no | managed venv from the site Python | Where experiments' Python environment comes from; see [python_environment](#python_environment). |
| `environment_build` | no | `{on: compute_node}` | Where a managed environment is built and with what: `on: compute_node` (a job the experiments depend on) or `login_node` (on the login node, during `launch`; use it when compute nodes have no internet), and `allocation` ([resources](#per_experiment); default `{walltime: "01:00:00", cpus: 4, memory: 16G}`; make `memory` fit the CPU partition's nodes). |
| `data_preparation` | no | `{on: compute_node}` | The same for data preparation steps (`prepare_on_host` sources, and fetches of declared datasets): `on` and `allocation`. |
| `local_to_host_paths` | no | none | Local path prefix → path prefix on the host; see [local_to_host_paths](#local_to_host_paths). |
| `confirm_uploads_over` | no | `20G` | Total size of new uploads above which `launch` asks for explicit confirmation (`--yes` also accepts it). |
| `dataset_delivery` | no | not managed | How declared datasets reach this host: `{default: MODE, <requirement name>: MODE}`, MODE one of `download_on_host`, `copy_from_here`, `already_on_host` (`default` defaults to `download_on_host`); see the guide's [Declared data](../guide/08-hpc-batches.md#declared-data-dataset_delivery-on-the-host). Without it, declared data is not managed. |

#### directories

All three are required, absolute paths on the host.

| Key | Holds |
|---|---|
| `missions` | One directory per launched mission: source checkout, plan, experiment configs, records, runs, logs. |
| `python_environments` | Managed environments, one directory per environment key, shared by missions. Prefer a filesystem that is not purged. |
| `datasets` | Uploaded datasets, stored by content key and reused by later missions (unless `local_to_host_paths` places them). |

#### scheduler: {kind: slurm}

| Key | Required | Meaning |
|---|---|---|
| `sbatch_options` | no | `#SBATCH` options added to **every** job of every mission (environment build, data preparation, experiment arrays): `{account: r00000, qos: normal}` renders `#SBATCH --account=r00000` and `#SBATCH --qos=normal`. Values are strings or integers. Options naavik sets itself (time, cpus, memory, GPUs, partition, array, dependency, output, signal, job name, ...) are refused; `account` and `qos` are allowed here and only here. A job's `scheduler_options` with the same key wins for that job. |
| `partitions` | no* | `{gpu_jobs: <name>, cpu_jobs: <name>}`: the partition for jobs that request GPUs and for those that do not. *A mission with a GPU job needs `gpu_jobs`; one with a CPU job (including the environment build and data preparation) needs `cpu_jobs`. |
| `max_array_size` | yes | The cluster's `MaxArraySize` (`scontrol show config`). A job with more experiments is refused at validation instead of by `sbatch`. |

Control characters (such as newlines) are refused in every Slurm value, so
no value can add an extra `#SBATCH` line.

#### scheduler: {kind: builtin}

| Key | Default | Meaning |
|---|---|---|
| `gpus` | `[]` | GPU indices (as `nvidia-smi` numbers them) that experiments may use, in the order they are tried. A job may not request more GPUs than the pool has. Each GPU is given to one experiment at a time across every mission on the machine. |
| `when_gpus_busy` | `wait` | `wait`: skip a pool GPU someone else is using (a compute process `nvidia-smi` lists, or 256 MiB or more of memory in use, which also shows other containers' processes) until it is free; `use_anyway`: only the pool decides. |

The builtin scheduler takes no `scheduler_options`. It runs a runner with the
host's `python3` (3.8 or newer, checked by `launch`), keeps its job state
under `$XDG_STATE_HOME/training-framework/host/<hostname>/` on the host,
and does not limit an experiment's CPUs or memory.

#### shell_commands_before

Lists of shell lines, run in order before the command of each phase, in the
job script or the remote shell (`module load ...`, `export ...`). They are
part of the environment key (environment build) or the preparation key, so
changing them gives a new environment or prepared dataset.

| Key | Runs before |
|---|---|
| `everything` | everything below |
| `environment_build` | building a managed environment, and `show-site-interpreter` |
| `data_preparation` | data preparation steps |
| `experiments` | each experiment |

#### python_environment

| Key | Default | Meaning |
|---|---|---|
| `use_existing` | | Absolute path of an environment you manage yourself on the host. Experiments use `<use_existing>/bin/python`; nothing is built and the other keys below (except `activate_with_conda_run`) do not apply. With `framework_from: installed` it must contain `training-framework`. With `connection: local`, `current` means the environment of the interpreter running `naavik` (e.g. your IDE's project interpreter), resolved into the plan. |
| `interpreter_from` | `site` | Where a managed environment's interpreter comes from: `site` (`python3` after the shell commands, e.g. a module), `uv` (`uv python install <python_version>`), or `conda` (conda's base Python). `uv` and `conda` need `python_packages.python_version` in the mission or project file. |
| `site_interpreter` | | Required with `interpreter_from: site`: the site interpreter's `{implementation, version, executable, platform}`, printed by `naavik show-site-interpreter`. `launch` checks it against the host and stops with the new block if the site changed it. |
| `kind` | `venv` | Kind of managed environment: `venv` (from the interpreter above) or `conda` (`conda create -p <env> -c conda-forge python=<python_version>`). A `conda` environment installs its own Python, so it needs `interpreter_from: conda` (or `uv`) and `python_version`; with `site` it is refused. |
| `conda_packages` | `[]` | With `kind: conda`: extra conda packages for `conda create` (e.g. `["cudatoolkit=11.8"]`); part of the environment key. |
| `activate_with_conda_run` | `false` | Run every command through `conda run --no-capture-output -p <env>` (for conda environments that need activation). Needs a conda environment. |
| `bootstrap_interpreter` | | Path of the interpreter that runs the environment build's bootstrap (and is the base of a `venv`), instead of the one chosen by `interpreter_from`. For sites whose default Python lacks `venv`/`ensurepip`. |

All environments put their `bin` first on `PATH` for experiments, preflight,
and data preparation.

#### local_to_host_paths

```yaml
local_to_host_paths:
  /home/me/datasets: /N/project/mylab/datasets
```

Absolute local prefixes to absolute prefixes on the host; the longest
matching prefix wins. Used to translate config values of [data
sources](#data_sources): an `already_on_host` source's local path becomes its
path on the host, a `prepare_on_host` source is prepared at the mapped path,
and a `copy_from_here` goes to the mapped path instead of
`directories.datasets/<key>/...`.

## Mission files

A YAML file committed in your repository; its path is the argument of
`validate`, `render`, and `launch`. Host settings (`connection`,
`directories`, `account`, `partition`, `qos`, ...) are refused here.

| Key | Required | Default | Meaning |
|---|---|---|---|
| `format_version` | yes | | Mission file format version; must be `1`. |
| `name` | yes | | Mission name; mission ids are `<name>-<UTC time>-<short commit>`. |
| `note` | no | | Why the mission runs; kept in the receipt and each experiment's run record (`naavik run --note`). |
| `project_file` | no | | A project file (a path in the repository) adding `code_transfer`, `code`, `python_packages`, and `data_sources`; see [The project file](#the-project-file). |
| `code_transfer` | no | bundle of `HEAD` | Which commit runs and how it reaches the host; see [code_transfer](#code_transfer). |
| `code` | yes | | How experiments find the framework and your code; see [code](#code). |
| `python_packages` | no | | What a managed environment installs; see [python_packages](#python_packages). |
| `base` | yes | | The session config the experiments start from, and engine options; see [base](#base). |
| `data_sources` | no | | Config values that are datasets; see [data_sources](#data_sources). |
| `jobs` | yes | | Experiments grouped by what each needs; see [jobs](#jobs). |

`code` is required in the mission file or its project file.

### The project file

`naavik-project.yaml` at the repository root (any path, when a mission file
names it with `project_file:`): what a project's missions share. Read from
the mission's commit; it must be identical at `code_transfer.commit`. A key in
both the mission file and the project file is an error.

| Key | Meaning |
|---|---|
| `code_transfer`, `code`, `python_packages`, `data_sources` | As in a mission file. |
| `quick_run_defaults.per_experiment` | [Resources](#per_experiment) of a `naavik run` experiment (`--gpus`, `--walltime` override). |
| `quick_run_defaults.max_parallel_experiments` | For `naavik run` (a positive integer). |
| `quick_run_defaults.host` | The host `naavik run` uses when no `--host` is given. |

### code_transfer

| Key | Default | Meaning |
|---|---|---|
| `method` | `git_bundle` | `git_bundle`: a Git bundle of the commit is copied over SSH (no Git credentials needed on the host). `git_clone`: the host clones `clone_url` itself. |
| `commit` | `HEAD` | The commit to run (branch, tag, or SHA), resolved on your machine. The mission file must be identical at `HEAD` and at this commit. |
| `clone_url` | | Repository URL the host clones from; required with `git_clone`, refused with `git_bundle`. |

The working tree must be clean (uncommitted or untracked files are an error),
unless the host has `run_code_from: working_tree`.

### code

| Key | Required | Meaning |
|---|---|---|
| `framework_from` | yes | `this_repository`: this repository *is* training-framework; it runs from the staged commit and the requirements must not contain it. `installed`: training-framework is installed in the environment: with a managed environment the requirements must pin it exactly once; with `python_environment.use_existing` the environment must already contain it (no requirements file is needed). |
| `python_paths` | yes | Relative directories of the repository put on `PYTHONPATH` for experiments (e.g. `["."]`, `["src"]`). Your code is never pip-installed. |

### python_packages

Used only for managed environments (no `python_environment.use_existing`).

| Key | Meaning |
|---|---|
| `requirements_file` | Repository path of a fully pinned requirements file; required for a managed environment. It is installed with `pip install --no-deps`, so it must list every package, transitive ones included (generate it with `uv pip compile`; see [Setting up a project](../guide/10-hpc-projects.md)). Each line is `name==X.Y.Z`, or a direct reference with a full 40-character commit SHA or an archive hash; the only options are `--index-url`, `--extra-index-url`, `--find-links`, and `--hash=` after a requirement. |
| `python_version` | Python version `"X.Y"` or `"X.Y.Z"`, quoted (YAML reads `3.12` as a number). Required with `interpreter_from: uv` or `conda`; with `site` it must match the identity. It must satisfy `requires-python` in your `pyproject.toml`. |

### base

| Key | Default | Meaning |
|---|---|---|
| `config_file` | | Repository path of the session config (the usual `sessions:` file). Required. Its [includes](../guide/04-configuration.md#sharing-configuration-between-files) are read at the mission's commit, by paths that stay inside the repository; each experiment's rendered config has them merged. A recipe may name its own (`config_file`). |
| `session_index` | `0` | Index of the session in `sessions` that the experiments run. |
| `engine_options` | `[]` | Engine command-line options passed to every experiment. Only `--heartbeat-timeout`, `--stop-sync-grace-period`, `--stop-sync-poll-interval`, and `--process_timeout_on_join` are accepted; config changes belong in overrides. |

### data_sources

```yaml
data_sources:
  - name: flowers
    address: {component: flowers102, path: root}
    copy_from_here: {}
```

Only config values listed here are treated as data sources (datasets that
[declare their data](datasets.md#its-data-fetched-once) need no entry); every
other path is left as written. Each source has:

| Key | Meaning |
|---|---|
| `name` | Name of the source, unique in the mission file. |
| `address` | [Address](#addresses) of the config value holding the path. |
| exactly one of `copy_from_here`, `already_on_host`, `prepare_on_host` | How the data gets there (below). |

**`copy_from_here`**: the value is an absolute path on your machine; it is
copied once with rsync and reused by every mission with the same content.

| Key | Meaning |
|---|---|
| `content_version` | Optional. Key the copy by the source's `name` and this string instead of hashing its content (for very large datasets that never change). Change the string to copy again; renaming the source also gives a new key. |

Paths inside the repository, containing it, `/`, or your home directory are
refused.

**`already_on_host`**: the data is already on the host.

| Key | Meaning |
|---|---|
| `path` | Optional absolute path on the host. Without it the value is translated by `local_to_host_paths` (or used as is). It must exist; preflight checks it. |

**`prepare_on_host`**: the data is built or downloaded on the host, once,
before the experiments run, at the value's (mapped) path.

| Key | Meaning |
|---|---|
| `command` | Required. Argument list run from the staged source; `{path}` is replaced by the path (`{{`/`}}` for literal braces). It must create the path and be safe to run again. |
| `content_version` | Required. Change it to prepare again. |
| `rerun_when_changed` | Optional repository paths; the step is redone only when these change. Without it, any new commit prepares again. |
| `allocation` | [Resources](#per_experiment) of the preparation job; required with `data_preparation.on: compute_node`, refused with `login_node`. |

### jobs

```yaml
jobs:
  short:
    per_experiment: {walltime: "02:00:00", gpus: 1, cpus: 8, memory: 64G}
    max_parallel_experiments: 4
    recipes: [...]
```

A mapping of job name to job. Each job becomes one scheduler job (a Slurm
job array, or one runner of the builtin scheduler); all its experiments get
the same resources.

| Key | Required | Default | Meaning |
|---|---|---|---|
| `per_experiment` | yes | | [Resources](#per_experiment) of each experiment. |
| `recipes` | yes | | Non-empty list of [recipes](#recipes). |
| `max_parallel_experiments` | no | no limit | Most experiments of the job running at once (Slurm's `%N` array throttle). |
| `stop_grace_period` | no | `120s` | Time for the last iteration and checkpoint when the experiment is stopped before its walltime; the early signal is sent `process_timeout_on_join + stop_grace_period` before the limit. |
| `scheduler_options` | no | | Extra scheduler options for this job only: `{slurm: {constraint: a100}}` renders `--constraint=a100` (every option is written `--key=value`). The same rules as `sbatch_options` apply, except that `account` and `qos` are refused (they belong in `hosts.yaml`); a key here overrides the host's. |

#### per_experiment

Used by jobs, `prepare_on_host` sources (`allocation`), and the host's
`environment_build` / `data_preparation` (`allocation`).

| Key | Required | Default | Meaning |
|---|---|---|---|
| `walltime` | yes | | Time limit `[D-]HH:MM:SS`. |
| `memory` | yes | | Memory per job (one node), e.g. `64G`. |
| `cpus` | yes | | CPU cores per experiment. |
| `gpus` | no | `0` | GPUs per experiment on one node. With `nccl`, `ddp.world_size` must equal it; with `gloo`, `world_size` must fit in `cpus`. |

`partition`, `account`, and `qos` are refused here: they come from the host.

#### recipes

| Key | Required | Meaning |
|---|---|---|
| `name` | yes | Unique within the job; experiments are `<job>/<name>/000`, `001`, ... |
| `config_file` | no | Its own session config (a repository path; default `base.config_file`), for a change of structure. Its session is `base.session_index`, and its overrides and `vary` are checked against it. When any recipe sets it, every experiment records its file (`config_file`), compared as `config` by `summarize`. |
| `overrides` | no | List of `{component?, path, value}`: values set for every experiment of the recipe. |
| `vary` | no | List of `{component?, path, values}`: one experiment per combination of the values; the last entry varies fastest. Without it the recipe is one experiment. |

Each experiment's config is: its session config (with its includes merged),
then the recipe's `overrides`, then its combination of `vary` values. A
`${...}` interpolation brought in by an override or a varied value is
refused, as in the session config itself.

### Addresses

`{component?, path}` names one value in the selected session of the
session config.

- With `component`, `path` is inside that component's entry, wherever it is
  listed (top level, or inside a `resources`, `hooks`, or `steps` group).
- Without it, `path` starts at the session (`session_config.rng_seed`).
- `path` is a dotted string (`optimizer.kwargs.lr`) or a list of keys
  (`["optimizer", "kwargs", "lr"]`); use the list form when a key contains a
  dot.
- `session_config.sessions_dir` is owned by naavik (each experiment gets its
  own) and cannot be addressed.

## Earlier names

Files written with these names are still read, with one warning per file
listing what to rename. Rename them by hand; an earlier name beside its
current one is an error.

**Files and commands**

| Earlier | Current |
|---|---|
| `hpc.yaml`, `targets.yaml` | `hosts.yaml` |
| manifest, experiment file | mission file |
| batch | mission |
| task | experiment |
| `naavik batches`, `submit`, `probe-python` | `naavik missions`, `launch`, `show-site-interpreter` |
| `--target`, `--profile` / `--targets`, `--profiles` | `--host` / `--hosts-file` |
| `--task` (resume, fetch, cancel), `missions --tasks` | `--experiment`, `missions --experiments` |

**The hosts file**

| Earlier | Current |
|---|---|
| `default` | `default_host` |
| `targets` / `profiles` | `hosts` |
| `ssh_host: X` / `local: true` | `connection: {ssh: X}` / `connection: local` |
| `scheduler: slurm` + `slurm: {...}` | `scheduler: {kind: slurm, ...}` |
| `scheduler: host` + `host: {...}` | `scheduler: {kind: builtin, ...}` |
| `slurm.defaults`, `slurm.partitions {gpu, cpu}` | `sbatch_options`, `partitions {gpu_jobs, cpu_jobs}` |
| `host.busy_gpus: wait\|use` | `when_gpus_busy: wait\|use_anyway` |
| `roots {batches, envs, data}` | `directories {missions, python_environments, datasets}` |
| `hooks {common, setup, prepare, task}` | `shell_commands_before {everything, environment_build, data_preparation, experiments}` |
| `env {existing, python_source, site_python_identity, backend, conda_packages, conda_run, bootstrap_python}` | `python_environment {use_existing, interpreter_from, site_interpreter, kind, conda_packages, activate_with_conda_run, bootstrap_interpreter}` |
| `setup_location: job\|login` + `setup_resources` | `environment_build {on: compute_node\|login_node, allocation}` |
| `prepare_location: job\|login` | `data_preparation {on: compute_node\|login_node, allocation}` |
| `path_map` | `local_to_host_paths` |
| `upload_limit` | `confirm_uploads_over` |
| `worktree: commit\|snapshot`, `snapshot_limit` | `run_code_from: commit\|working_tree`, `working_tree_untracked_limit` |
| `data_modes: fetch\|upload\|present` | `dataset_delivery: download_on_host\|copy_from_here\|already_on_host` |

**Mission and project files**

| Earlier | Current |
|---|---|
| `version`, `description`, `project` | `format_version`, `note`, `project_file` |
| `source {strategy: bundle\|fetch, ref, url}` | `code_transfer {method: git_bundle\|git_clone, commit, clone_url}` |
| `code {framework: dependency\|checkout, paths}` | `code {framework_from: installed\|this_repository, python_paths}` |
| `environment {requirements, python}` | `python_packages {requirements_file, python_version}` |
| `training {config, session, engine_options}` | `base {config_file, session_index, engine_options}` |
| `data: {sources: [...]}` | `data_sources: [...]` |
| source `target`, `upload`, `cluster`, `prepare` | `address`, `copy_from_here`, `already_on_host`, `prepare_on_host` |
| `version`, `inputs`, `resources` (in a source) | `content_version`, `rerun_when_changed`, `allocation` |
| `groups {G: {resources, max_parallel, stop_grace, experiments}}` | `jobs {J: {per_experiment, max_parallel_experiments, stop_grace_period, recipes}}` |
| experiment `config`, `matrix` | recipe `config_file`, `vary` |
| `run_defaults {resources, max_parallel, target}` | `quick_run_defaults {per_experiment, max_parallel_experiments, host}` |
