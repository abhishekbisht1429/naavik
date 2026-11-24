# Mission files and the project file

[← Docs](../README.md)

Every key of the two committed files the [mission
launcher](../guide/08-missions.md) reads:

- A **mission file** holds the experiments of a mission. It is committed with
  your code and runs unchanged on any host.
- The **project file** (`naavik-project.yaml`) holds what a project's
  mission files share. It is committed too.

What is specific to a machine (how it is reached, its scheduler, its
directories) belongs in the [hosts file](hosts-file.md), which is not
committed. Both files are strict: an unknown key, a wrong type, or a key in
the wrong file is an error that `validate` reports with its location;
nothing is silently ignored. Files written with [earlier
names](earlier-names.md#mission-and-project-files) are still read; each
launch warns once and lists every `old -> new`, to rename by hand.

## Values

Sizes are an integer with an optional `K`, `M`, `G`, or `T` suffix (powers
of 1024): `64G`. Walltimes are `[D-]HH:MM:SS`: `02:00:00`, `1-12:00:00`.
Durations are an integer with `s`, `m`, or `h`: `120s`. Names (missions,
jobs, recipes, data sources) use letters, digits, `_`, and `-` only, since
they become directory and job names.

## Mission files

A YAML file committed in your repository; its path is the argument of
`validate`, `render`, and `launch`. Host settings (`connection`,
`directories`, `account`, `partition`, `qos`, ...) are refused here.

| Key | Required | Default | Meaning |
|---|---|---|---|
| `format_version` | yes | | Mission file format version; must be `1`. |
| `name` | yes | | Mission name; mission ids are `<name>-<UTC time>-<short commit>`. |
| `note` | no | | Why the mission runs; kept in the receipt and each experiment's run record (`naavik launch --session --note`). |
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
| `quick_run_defaults.per_experiment` | [Resources](#per_experiment) of a `naavik launch --session` experiment (`--gpus`, `--walltime` override). |
| `quick_run_defaults.max_parallel_experiments` | For `naavik launch --session` (a positive integer). |
| `quick_run_defaults.host` | The host `naavik launch --session` uses when no `--host` is given. |

### code_transfer

| Key | Default | Meaning |
|---|---|---|
| `method` | `git_bundle` | `git_bundle`: a Git bundle of the commit is copied to the host (no Git credentials needed there). `git_clone`: the host clones `clone_url` itself. |
| `commit` | `HEAD` | The commit to run (branch, tag, or SHA), resolved on your machine. The mission file must be identical at `HEAD` and at this commit. |
| `clone_url` | | Repository URL the host clones from; required with `git_clone`, refused with `git_bundle`. |

The working tree must be clean (uncommitted or untracked files are an error),
unless the host has [`run_code_from: working_tree`](hosts-file.md#host-keys).

### code

| Key | Required | Meaning |
|---|---|---|
| `framework_from` | yes | `this_repository`: this repository *is* naavik; it runs from the staged commit and the requirements must not contain it. `installed`: naavik is installed in the environment: with a managed environment the requirements must pin it exactly once; with [`python_environment.use_existing`](hosts-file.md#python_environment) the environment must already contain it (no requirements file is needed). |
| `python_paths` | yes | Relative directories of the repository put on `PYTHONPATH` for experiments (e.g. `["."]`, `["src"]`). Your code is never pip-installed. |

### python_packages

Used only for managed environments (no `python_environment.use_existing` on
the host).

| Key | Meaning |
|---|---|
| `requirements_file` | Repository path of a fully pinned requirements file; required for a managed environment. It is installed with `pip install --no-deps`, so it must list every package, transitive ones included (generate it with `uv pip compile`; see [Setting up a project](../guide/10-mission-projects.md)). Each line is `name==X.Y.Z`, or a direct reference with a full 40-character commit SHA or an archive hash; the only options are `--index-url`, `--extra-index-url`, `--find-links`, and `--hash=` after a requirement. |
| `python_version` | Python version `"X.Y"` or `"X.Y.Z"`, quoted (YAML reads `3.12` as a number). Required with `interpreter_from: uv` or `conda`; with `site` it must match the identity. It must satisfy `requires-python` in your `pyproject.toml`. |

### base

| Key | Default | Meaning |
|---|---|---|
| `config_file` | | Repository path of the session config: a file that is one session. Required. Its [includes](../guide/04-configuration.md#sharing-configuration-between-files) are read at the mission's commit, by paths that stay inside the repository; each experiment's rendered config has them merged. A recipe may name its own (`config_file`). |
| `session_index` | | No longer needed: a config file is one session. `0` is accepted with a note to remove it; any other value is an error. |
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
| `path` | Optional absolute path on the host. Without it the value is translated by the host's [`local_to_host_paths`](hosts-file.md#local_to_host_paths) (or used as is). It must exist; preflight checks it. |

**`prepare_on_host`**: the data is built or downloaded on the host, once,
before the experiments run, at the value's (mapped) path.

| Key | Meaning |
|---|---|
| `command` | Required. Argument list run from the staged source; `{path}` is replaced by the path (`{{`/`}}` for literal braces). It must create the path and be safe to run again. |
| `content_version` | Required. Change it to prepare again. |
| `rerun_when_changed` | Optional repository paths; the step is redone only when these change. Without it, any new commit prepares again. |
| `allocation` | [Resources](#per_experiment) of the preparation job; required with the host's [`data_preparation.on: compute_node`](hosts-file.md#where-setup-steps-run), refused with `login_node`. |

### jobs

```yaml
jobs:
  short:
    per_experiment: {walltime: "02:00:00", gpus: 1, cpus: 8, memory: 64G}
    max_parallel_experiments: 4
    recipes: [...]
```

A mapping of job name to job. Each job becomes one scheduler job (one runner
of the [builtin scheduler](hosts-file.md#scheduler-builtin), or a [Slurm](hosts-file.md#scheduler-slurm)
job array); all its experiments get the same allocation (`per_experiment`: GPUs, time, ...).

| Key | Required | Default | Meaning |
|---|---|---|---|
| `per_experiment` | yes | | [Resources](#per_experiment) of each experiment. |
| `recipes` | yes | | Non-empty list of [recipes](#recipes). |
| `max_parallel_experiments` | no | no limit | Most experiments of the job running at once (on Slurm, the `%N` array throttle). |
| `stop_grace_period` | no | `120s` | Time for the last iteration and checkpoint when the experiment is stopped before its walltime; the early signal is sent `process_timeout_on_join + stop_grace_period` before the limit. |
| `scheduler_options` | no | | Extra scheduler options for this job only, which tie the job to that scheduler: `{slurm: {constraint: a100}}` renders `--constraint=a100` (every option is written `--key=value`). The same rules as [`sbatch_options`](hosts-file.md#scheduler-slurm) apply, except that `account` and `qos` are refused (they belong in the hosts file); a key here overrides the host's. The builtin scheduler takes none. |

#### per_experiment

Used by jobs, `prepare_on_host` sources (`allocation`), and the host's
[`environment_build` / `data_preparation`](hosts-file.md#where-setup-steps-run)
(`allocation`).

| Key | Required | Default | Meaning |
|---|---|---|---|
| `walltime` | on Slurm | no limit | Time limit `[D-]HH:MM:SS`. Left out (builtin hosts only), the experiment runs until it ends. |
| `memory` | on Slurm | no limit | Memory per experiment (on one node), e.g. `64G`. The builtin scheduler never limits it: left out, the experiment uses what the machine has; given, it is recorded with a note. |
| `cpus` | on Slurm | no limit | CPU cores per experiment. As `memory` on the builtin scheduler. |
| `gpus` | no | `0` | GPUs per experiment on one node, taken from the host's pool ([`scheduler.gpus`](hosts-file.md#scheduler-builtin) on a builtin host). With `nccl`, `ddp.world_size` must equal it; with `gloo` and `cpus` given, `world_size` must fit in `cpus`. |

A Slurm host refuses a job without `walltime`, `cpus`, or `memory`, naming
them; a mission file that leaves them out runs on builtin hosts only.

`partition`, `account`, and `qos` are refused here: they come from the host.

#### recipes

| Key | Required | Meaning |
|---|---|---|
| `name` | yes | Unique within the job; experiments are `<job>/<name>/000`, `001`, ... |
| `config_file` | no | Its own session config (a repository path; default `base.config_file`), for a change of structure. Its overrides and `vary` are checked against it. When any recipe sets it, every experiment records its file (`config_file`), compared as `config` by `summarize`. Not with `config_files`. |
| `config_files` | no | Two or more session configs, distinct: one more axis, each file with every `vary` combination. Every override and `vary` address is checked against each file (a problem ends with `(in <file>)`). Each experiment records its own file, and `status` shows it (`config=<file>`). Not with `config_file`. |
| `overrides` | no | List of `{component?, path, value}`: values set for every experiment of the recipe. |
| `vary` | no | List of `{component?, path, values}`: one experiment per combination of the values; the last entry varies fastest. Without it the recipe is one experiment. |
| `from_mission` | no | Take the recipe's experiments from another mission's checkpoints: one experiment per chosen checkpoint (below). Its session must be an analysis session. |

Each experiment's config is: its session config (with its includes merged;
with `config_files`, one of them), then the recipe's `overrides`, then its
checkpoint (with `from_mission`), then its combination of `vary` values.
The axes, slowest first: the checkpoint, the config file, then `vary` (its
last entry fastest). A `${...}` interpolation brought in
by an override or a varied value is refused, as in the session config
itself.

#### from_mission

| Key | Required | Default | Meaning |
|---|---|---|---|
| `id` | yes | | The mission whose checkpoints are analysed, as `naavik missions` lists it. Its receipt must be on this machine. |
| `experiments` | no | all | Names or globs of its experiments (`short/*`). A glob that matches none is an error. |
| `checkpoint` | no | `latest` | `latest`: each experiment's newest complete checkpoint; `all`: every complete one. |
| `into` | no | `{component: trained_model, path: model_checkpoint_path}` | The address the checkpoint's path is written at. It must not also be set by `overrides` or `vary`. |

The checkpoints are chosen when the mission is compiled (`validate`,
`render`, `launch`), so the plan names each experiment's exact checkpoint
and a resume reuses it. The checkpoint is the slowest axis: each checkpoint
with every combination of `vary` values. An experiment without a complete
checkpoint is left out with a note; if none is left, it is an error.

The source is read where the analysis runs: on the same host (read-only;
`validate` connects for it and says so), or, for a mission on this
machine, from the source's copy fetched into `./hpc-results/<id>`
(`naavik fetch`, with `--checkpoints all` for `checkpoint: all`). A source
on another host is refused.

Each experiment records its `source`: `{mission, experiment, checkpoint,
iteration, into, variant}` (`variant`: the source experiment's varied
values), in the plan and in its `record.json`.

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
  own, `<mission dir>/tasks/<experiment>/sessions`) and cannot be addressed;
  the session file's value is not used, and `launch` says so in a note.
