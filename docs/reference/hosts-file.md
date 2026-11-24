# The hosts file

[← Docs](../README.md)

The hosts file describes the machines you run on: this machine, a machine
you reach over SSH, or a Slurm cluster. It holds everything specific to
those machines and to you, so the committed [mission files](mission-files.md)
run unchanged on any host. It lives on your machine and is never committed.

These commands read it:

- `naavik validate`, `render`, `launch`, and the commands that follow a
  mission (`missions`, `status`, `logs`, `cancel`, `resume`, `fetch`); see
  [Running missions](../guide/08-missions.md).
- `naavik launch --session`, which launches one session file as a mission of one experiment.
- `naavik runs sync`, which copies the runs recorded on hosts into your
  [run registry](../guide/13-run-registry.md).

The file is strict: an unknown key, a wrong type, or a key that belongs in a
mission file is an error that `validate` reports with its location. Sizes,
walltimes, and durations are written as in [mission
files](mission-files.md#values). Files written with [earlier
names](earlier-names.md#the-hosts-file) are still read.

## Where it is

`~/.config/naavik/hosts.yaml` (or `$XDG_CONFIG_HOME/naavik/hosts.yaml`), or
the file given with `--hosts-file PATH`.

```yaml
default_host: here            # used when --host is not given
hosts:
  here:                       # this machine
    connection: local
    scheduler: {kind: builtin, gpus: [0, 1]}
    directories: {missions: /data/me/naavik/missions, python_environments: /data/me/naavik/envs, datasets: /data/me/naavik/datasets}
    python_environment: {use_existing: current}
  bigred200: {...}            # one entry per machine or cluster account
```

| Key | Required | Meaning |
|---|---|---|
| `default_host` | no | The host to use without `--host`. With neither, commands stop. |
| `hosts` | yes | Mapping of host name to its settings. |

## Host keys

| Key | Required | Default | Meaning |
|---|---|---|---|
| `connection` | yes | | How the host is reached: `local` (this machine; commands run here, without `ssh` or `rsync`) or `{ssh: <name>}` (a host name or an alias from `~/.ssh/config`, which supplies user, key, and connection sharing). |
| `scheduler` | yes | | What runs the jobs, and its settings: `{kind: builtin, ...}` (naavik's own, for a machine without a scheduler; see [builtin](#scheduler-builtin)) or `{kind: slurm, ...}` (see [slurm](#scheduler-slurm)). |
| `directories` | yes | | Absolute directories on the host; see [directories](#directories). |
| `run_code_from` | no | `commit` | What a mission runs: `commit` (HEAD; the working tree must be clean) or `working_tree` (the working tree as it is, made into a commit on HEAD); see the guide's [Uncommitted work](../guide/08-missions.md#uncommitted-work-run_code_from-working_tree). |
| `working_tree_untracked_limit` | no | `50M` | With `run_code_from: working_tree`: most bytes of untracked files it may take into Git; above it the launch is refused, naming the largest files. |
| `shell_commands_before` | no | none | Shell lines run before commands; see [shell_commands_before](#shell_commands_before). |
| `python_environment` | no | managed venv from the site Python | Where experiments' Python environment comes from; see [python_environment](#python_environment). |
| `environment_build` | no | `{on: compute_node}` | Where a managed environment is built; see [Where setup steps run](#where-setup-steps-run). |
| `data_preparation` | no | `{on: compute_node}` | The same for data preparation steps (`prepare_on_host` sources, and fetches of declared datasets). |
| `local_to_host_paths` | no | none | Local path prefix → path prefix on the host; see [local_to_host_paths](#local_to_host_paths). |
| `confirm_uploads_over` | no | `20G` | Total size of new uploads above which `launch` asks for explicit confirmation (`--yes` also accepts it). |
| `dataset_delivery` | no | not managed | How declared datasets reach this host: `{default: MODE, <requirement name>: MODE}`, MODE one of `download_on_host`, `copy_from_here`, `already_on_host` (`default` defaults to `download_on_host`); see the guide's [Declared data](../guide/08-missions.md#declared-data-dataset_delivery-on-the-host). Without it, declared data is not managed. |

### directories

All three are required, absolute paths on the host.

| Key | Holds |
|---|---|
| `missions` | One directory per launched mission: source checkout, plan, experiment configs, records, runs, logs. |
| `python_environments` | Managed environments, one directory per environment key, shared by missions. Prefer a filesystem that is not purged. |
| `datasets` | Uploaded datasets, stored by content key and reused by later missions (unless `local_to_host_paths` places them). |

### Schedulers

Each job of a mission becomes one scheduler job, whichever scheduler the
host uses. Results, logs, `status`, `resume`, `cancel`, and `fetch` work the
same on every host.

#### Scheduler: builtin

`scheduler: {kind: builtin, ...}`: naavik's own scheduler, for a machine
without one (this machine, or a machine over SSH). Each job is a runner
process started detached on the host. The runner decides what runs when: at
most `max_parallel_experiments` experiments of the job at a time, each with
the GPUs its mission file asks for, taken from the pool when they are free.
It sends the early stop signal before the walltime and kills an experiment
at the walltime; an experiment without a walltime runs until it ends.

| Key | Default | Meaning |
|---|---|---|
| `gpus` | `all` | The GPU pool, which is also the limit on GPUs in use at once by all your experiments on the machine, in every mission: `all` (every GPU of the machine), a count (`4`: any 4 at once, whichever are free), or a list of `nvidia-smi` indices (`[0, 1]`: only those). Which GPUs an experiment gets is decided when it starts. Each GPU is given to one experiment at a time. `[]` or `0` allows none. |
| `when_gpus_busy` | `wait` | `wait`: skip a pool GPU someone else is using (a compute process `nvidia-smi` lists, or 256 MiB or more of memory in use, which also shows other containers' processes) until it is free; `use_anyway`: only the pool decides. |

How many GPUs each experiment gets is the mission file's
[`per_experiment.gpus`](mission-files.md#per_experiment) (default 0). A
request the pool can never satisfy is refused before anything is staged:
by `validate` for a count or a list, by `launch` (which counts the machine's
GPUs) for `all`. Should one still reach the runner, every experiment of the
job fails at once with the reason in its log. An experiment waiting for GPUs
says why, in `naavik status` and while `naavik launch --session` follows it:

```text
run/vit/000  pending  job h4711.9_0: waiting for GPUs: needs 4; 2 free now (limit 4; 2 held by your experiments)
```

The builtin scheduler takes no `scheduler_options`. It runs a runner with the
host's `python3` (3.8 or newer, checked by `launch`), keeps its job state
under `$XDG_STATE_HOME/naavik/host/<hostname>/` on the host, and never limits
an experiment's CPUs or memory: `per_experiment.cpus` and `memory` may be
left out (the experiment uses what the machine has), and when given they are
recorded, with a note saying they are not enforced. `nvidia-smi` is needed
only when a job asks for GPUs or the pool lists indices.

#### Scheduler: slurm

`scheduler: {kind: slurm, ...}`: a Slurm cluster. Each job is a Slurm job
array.

| Key | Required | Meaning |
|---|---|---|
| `sbatch_options` | no | `#SBATCH` options added to **every** job of every mission (environment build, data preparation, experiment arrays): `{account: r00000, qos: normal}` renders `#SBATCH --account=r00000` and `#SBATCH --qos=normal`. Values are strings or integers. Options naavik sets itself (time, cpus, memory, GPUs, partition, array, dependency, output, signal, job name, ...) are refused; `account` and `qos` are allowed here and only here. A job's `scheduler_options` with the same key wins for that job. |
| `partitions` | no* | `{gpu_jobs: <name>, cpu_jobs: <name>}`: the partition for jobs that request GPUs and for those that do not. *A mission with a GPU job needs `gpu_jobs`; one with a CPU job (including the environment build and data preparation) needs `cpu_jobs`. |
| `max_array_size` | yes | The cluster's `MaxArraySize` (`scontrol show config`). A job with more experiments is refused at validation instead of by `sbatch`. |

Control characters (such as newlines) are refused in every Slurm value, so
no value can add an extra `#SBATCH` line.

### Where setup steps run

`environment_build` and `data_preparation` take the same two keys:

| Key | Default | Meaning |
|---|---|---|
| `on` | `compute_node` | `compute_node`: a job of the host's scheduler, which the experiments depend on. `login_node`: run directly on the host during `launch` (on a cluster, the login node; use it when compute nodes have no internet). |
| `allocation` | `{walltime: "01:00:00", cpus: 4, memory: 16G}` | [Resources](mission-files.md#per_experiment) of that job, with `on: compute_node`. On a cluster, make `memory` fit the CPU partition's nodes. |

A data preparation step on `login_node` needs the environment built on
`login_node` too, unless it is already built. A `prepare_on_host` source
gives its own `allocation`; see [data_sources](mission-files.md#data_sources).

### shell_commands_before

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

### python_environment

| Key | Default | Meaning |
|---|---|---|
| `use_existing` | | Absolute path of an environment you manage yourself on the host. Experiments use `<use_existing>/bin/python`; nothing is built and the other keys below (except `activate_with_conda_run`) do not apply. With `framework_from: installed` it must contain `naavik`. With `connection: local`, `current` means the environment of the interpreter running `naavik` (e.g. your IDE's project interpreter), resolved into the plan. |
| `interpreter_from` | `site` | Where a managed environment's interpreter comes from: `site` (`python3` after the shell commands, e.g. a module), `uv` (`uv python install <python_version>`), or `conda` (conda's base Python). `uv` and `conda` need `python_packages.python_version` in the mission or project file. |
| `site_interpreter` | | Required with `interpreter_from: site`: the site interpreter's `{implementation, version, executable, platform}`, printed by `naavik show-site-interpreter`. `launch` checks it against the host and stops with the new block if the site changed it. |
| `kind` | `venv` | Kind of managed environment: `venv` (from the interpreter above) or `conda` (`conda create -p <env> -c conda-forge python=<python_version>`). A `conda` environment installs its own Python, so it needs `interpreter_from: conda` (or `uv`) and `python_version`; with `site` it is refused. |
| `conda_packages` | `[]` | With `kind: conda`: extra conda packages for `conda create` (e.g. `["cudatoolkit=11.8"]`); part of the environment key. |
| `activate_with_conda_run` | `false` | Run every command through `conda run --no-capture-output -p <env>` (for conda environments that need activation). Needs a conda environment. |
| `bootstrap_interpreter` | | Path of the interpreter that runs the environment build's bootstrap (and is the base of a `venv`), instead of the one chosen by `interpreter_from`. For sites whose default Python lacks `venv`/`ensurepip`. |

All environments put their `bin` first on `PATH` for experiments, preflight,
and data preparation.

### local_to_host_paths

```yaml
local_to_host_paths:
  /home/me/datasets: /N/project/mylab/datasets
```

Absolute local prefixes to absolute prefixes on the host; the longest
matching prefix wins. Used to translate config values of [data
sources](mission-files.md#data_sources): an `already_on_host` source's local
path becomes its path on the host, a `prepare_on_host` source is prepared at
the mapped path, and a `copy_from_here` goes to the mapped path instead of
`directories.datasets/<key>/...`.
