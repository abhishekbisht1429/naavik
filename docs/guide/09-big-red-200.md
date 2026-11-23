# First runs on Big Red 200

A checklist for bringing up [missions](08-missions.md) on Indiana
University's Big Red 200, one new risk per stage. Site details (account,
partitions, modules, limits) change; the commands in step 1 read the current
values from the cluster instead of assuming them.

Experiments need a Python environment on the cluster. There are two ways to
get one, and the stages below use both:

- **Existing**: an environment you made yourself; the host points at it
  (`python_environment.use_existing`). Nothing is built. Quickest for a
  first run.
- **Managed**: naavik builds an environment from your pinned requirements
  file, once, and every later mission with the same inputs reuses it.
  Reproducible and no manual setup; see [section 5](#5-managed-environments-building-on-the-cluster).

## 0. On your workstation

```bash
pip install "naavik @ git+ssh://git@github.com/abhishekbisht1429/naavik.git@dev"
# or, from a clone: pip install -e .
naavik --help         # validate, render, launch, status, show-site-interpreter
```

## 1. SSH and site facts (once)

Big Red 200 uses Duo two-factor authentication. naavik never prompts
(`BatchMode=yes`), so keep a control connection open in `~/.ssh/config`:

```
Host bigred200
    HostName bigred200.uits.iu.edu
    User <your-iu-username>
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
```

```bash
ssh bigred200 true                              # answer Duo once
ssh -o BatchMode=yes bigred200 hostname         # must work without a prompt
```

Read the facts the host needs on the cluster:

```bash
ssh bigred200
sinfo -s                                         # partition names (general, gpu)
sacctmgr -nP show assoc user=$USER format=account,partition,qos   # your accounts
scontrol show config | grep -i MaxArraySize
module avail python                              # the Python module to load
which rsync; echo /N/scratch/$USER               # rsync, scratch path
```

### Finding your Slurm account

`--account` (`-A`) names the allocation a job is charged to; it is set once
on the host (`scheduler.sbatch_options.account`) and added to every job. At
IU, Slurm accounts come from RT Projects (projects.rt.iu.edu): a PI's
project with a Big Red 200 allocation, which you must be added to. Names
usually look like `r00123`.

If `sacctmgr` above prints nothing, either your user has no account yet or
the site hides accounting data from users. These tell you which:

```bash
sacctmgr show user $USER withassoc format=user,defaultaccount,account,partition,qos
sshare -U -u $USER                                # accounts you belong to
sacct -u $USER -S 2025-01-01 -X --format=jobid,account,partition | tail   # accounts of past jobs
sbatch --test-only -p general -t 00:05:00 --wrap="true"                 # valid without --account?
sbatch --test-only -p general -A <candidate> -t 00:05:00 --wrap="true"  # valid with this account?
```

`--test-only` checks a job without submitting it.

- An account shown by `sshare` or `sacct`: put it in
  `scheduler.sbatch_options.account`.
- The test without `-A` succeeds: you have a default account, and
  `sbatch_options: {}` works; setting the account explicitly is still safer.
- `Invalid account or account/partition combination specified`: you need an
  account. Check RT Projects, ask your PI to add you, or ask UITS Research
  Technologies (rtech@iu.edu) for the account name.

After setting it, `render` the mission and check that the job script has
`#SBATCH --account=...`.

Scratch is purged after a period without access; keep results you need
elsewhere, and prefer a non-purged location (Slate, `/N/slate/$USER`, if you
have it) for managed environments, since a purge can remove files from an
environment that is still marked ready.

## 2. The host

`~/.config/naavik/hosts.yaml` (every key is described in the
[hosts file reference](../reference/hosts-file.md)). Everything
except `python_environment` is the same for both kinds of environment:

```yaml
default_host: bigred200
hosts:
  bigred200:
    connection: {ssh: bigred200}
    scheduler:
      kind: slurm
      sbatch_options: {account: <your-account>}
      partitions: {gpu_jobs: gpu, cpu_jobs: general}
      max_array_size: <MaxArraySize>
    directories:
      missions: /N/scratch/<user>/naavik/missions
      python_environments: /N/slate/<user>/naavik/envs   # managed environments (or scratch)
      datasets: /N/scratch/<user>/naavik/datasets
    shell_commands_before:
      everything: ["module load python/<version>"]       # from `module avail python`
    python_environment:
      use_existing: /N/u/<user>/BigRed200/envs/tf        # stages A-C; section 5 replaces this
    environment_build: {on: login_node}
    data_preparation: {on: login_node}
```

An existing environment needs torch and the framework's dependencies. With
`framework_from: this_repository` the framework itself comes from the
staged commit; with `installed` the environment must have it installed.

## 3. A smoke-test mission

A small committed repository: this repository itself (`framework_from:
this_repository`), or your project (`installed`). In the session config use
a small `max_iterations`, set `ddp.world_size` explicitly, and set
`tensorboard.serve: false` if the config has a `tensorboard` component.

```yaml
format_version: 1
name: br200-smoke
code: {framework_from: this_repository, python_paths: ["src", "."]}   # or installed + your project
base: {config_file: configs/smoke.yaml}
jobs:
  cpu:
    per_experiment: {walltime: "00:15:00", cpus: 2, memory: 8G}
    recipes:
      - name: seeds
        vary: [{path: session_config.rng_seed, values: [1, 2]}]
```

Commit everything, then check the generated scripts before launching:

```bash
naavik validate missions/smoke.yaml
naavik render missions/smoke.yaml --output /tmp/br200-preview
less /tmp/br200-preview/groups/cpu/group-cpu.sbatch   # account, partition, module lines
```

## 4. Stages

Go on only when a stage passes. Stages A-C use the existing environment;
D repeats them with a managed one.

**A. CPU smoke test** (SSH, staging, `sbatch`, arrays, logs, `status`):

```bash
naavik launch missions/smoke.yaml
naavik status <mission-id>          # pending -> running -> done
naavik missions --experiments       # every mission and experiment at a glance
ssh bigred200 'cat <mission-dir>/logs/cpu/*.out'
ssh bigred200 'cat <mission-dir>/tasks/cpu/seeds/000/record.json'
```

Passes when both experiments are `done`, each log has `task cpu/seeds/00N |
...` and `finished: exit 0 (ok)`, and each record names a run directory.

**B. GPUs.** Add a job with `per_experiment: {walltime: "00:20:00", gpus: 1,
cpus: 8, memory: 32G}` (with `nccl`, `ddp.world_size: 1`). The log header
shows `GPUs 0` instead of `not set`. Then `gpus: 2` with `world_size: 2` for
DDP inside a node.

**C. Cooperative stop.** One experiment with a large `max_iterations`, then:

```bash
scancel --signal=USR1 --batch <jobid>_0
```

The log shows `received the scheduler advance signal...` and `finished:
exit 75 (interrupted)`; `status` shows `interrupted`. Continue it with
`naavik resume <mission-id> --walltime 01:00:00`
(see [Resuming unfinished experiments](08-missions.md#10-resuming-unfinished-experiments)).
For the real time limit: `walltime: "00:05:00"` with `stop_grace_period: 60s`.

**D. A managed environment.** Follow [section 5](#5-managed-environments-building-on-the-cluster):
build on the login node (D1), reuse it (D2), then try a build in a job (D3),
and repeat stages A and B on it.

**E. Data.** Add a `copy_from_here: {}` source pointing at a small local
directory. `launch` shows rsync's progress; a second mission skips the copy.

## 5. Managed environments (building on the cluster)

### Choose the interpreter

`python_environment.interpreter_from` says where the environment's Python
comes from; `python_environment.kind` (`venv`, default, or `conda`) says
what kind of environment is built from it. A `venv` uses that interpreter
itself; a `conda` environment installs its own Python at
`python_packages.python_version`, so it is refused with `interpreter_from:
site`. Usual pairs: `site` + `venv` (the default), `uv` + `venv`, and
`conda` + `conda`.

| `interpreter_from` | Python from | Needs |
|---|---|---|
| `site` (default) | the site module (`module load python/...` in `shell_commands_before.everything`) | `python_environment.site_interpreter`, from `show-site-interpreter` |
| `uv` | `uv python install <python_version>` | `uv` on `PATH` (via a shell command), `python_packages.python_version` in the mission or project file |
| `conda` | conda's base interpreter | conda on `PATH` (via a shell command, e.g. `module load conda`), `python_version` |

Start with `site`. Record the interpreter's identity once and paste the
printed block into the host (it replaces `use_existing`):

```bash
naavik show-site-interpreter
```

```yaml
    python_environment:
      interpreter_from: site
      site_interpreter: {implementation: cpython, version: "3.12.x",
                         executable: /N/soft/.../python3.12, platform: linux-x86_64}
```

If the site later changes that Python, `launch` stops and prints the new
block; pasting it gives a new, separate environment. The interpreter must
satisfy your project's `requires-python`, and `python_version` (if set) must
match it.

### Write the pinned requirements file

Step by step, including installing uv and a project that depends on
naavik: [Setting up a project](10-mission-projects.md).

The mission file (or the project file) names a fully pinned file:

```yaml
python_packages:
  requirements_file: requirements/missions.txt
  python_version: "3.12"                 # optional with site; required with uv/conda
```

Every line is `name==X.Y.Z` or a direct reference with a full 40-character
commit SHA (or an archive hash); the only option lines allowed are
`--index-url`, `--extra-index-url`, `--find-links`, and `--hash` after a
requirement. Anything else is reported with its line number by `validate`.

- **`framework_from: this_repository`** (this repository): the file must
  *not* contain `naavik`; it comes from the staged commit.

  ```bash
  uv pip compile pyproject.toml --extra vision -o requirements/missions.txt
  ```

- **`framework_from: installed`** (your project): the file must pin
  `naavik` exactly once. From git, that is a direct reference
  to a full commit SHA, which `uv pip compile` writes when your project
  depends on the git URL:

  ```
  naavik @ git+https://github.com/abhishekbisht1429/naavik.git@<40-character sha>
  ```

  `git rev-parse dev` prints the full SHA. A private repository needs its
  own access from Big Red 200 (a deploy key or token).

- **CUDA torch**: add the PyTorch index to the file, e.g.
  `--extra-index-url https://download.pytorch.org/whl/cu124` (or compile
  with `--emit-index-url`), and pin `torch` to that index's version.

Commit the requirements file with the mission file.

### D1. Build on the login node

With `environment_build: {on: login_node}`, `launch` builds the environment
on the login node before submitting (installing torch takes several
minutes), then preflights it:

```bash
naavik launch missions/smoke.yaml
```

`launch` prints `environment: will be built` and `building the environment
on the login node`. The build's record is at
`<mission-dir>/nodes/setup/record.json` (`status: built`), and `status`
shows a `setup` row. Then repeat stages A and B.

### D2. Reuse

Launch the mission again (or another mission with the same requirements):
`launch` prints `environment: ready` and builds nothing. The environment
lives at `<directories.python_environments>/<key>`; the key is a hash of the
pinned requirements, the kind, the interpreter's identity, and the
environment build's shell commands. Changing any of them gives a new
environment next to the old one; old ones are never deleted automatically.
`launch --fresh-env` builds a private environment for one mission.

### D3. Build in a job

Login nodes are shared; a large build is better in a job. Switch to
`environment_build.on: compute_node` and give the build job its allocation:

```yaml
    environment_build:
      on: compute_node
      allocation: {walltime: "01:00:00", cpus: 4, memory: 16G}
```

Change a pin (or use `--fresh-env`) so there is something to build. The
mission now starts with a `setup` job; the arrays wait for it (`afterok`)
and are cancelled if it fails. Its log is `<mission-dir>/logs/setup/<job>.out`:

```
[hpc setup ...] creating the venv environment at ...
[hpc setup ...] installing the pinned requirements from requirements/missions.txt (this can take several minutes)
[hpc setup ...] checking installed dependencies (pip check)
[hpc setup ...] running preflight in the new environment
[hpc setup ...] preflight ok: Python 3.12.x | naavik ...
[hpc setup ...] environment ready
```

A pip network error in that log means compute nodes cannot reach the
package index: keep `on: login_node`. Two missions that need the same new
environment at once build it once: the second build job logs `waiting for
the lease on ... held by ...` and then `another job finished building the
environment meanwhile`.

## 6. When something goes wrong

- `error: cannot reach bigred200 over SSH: ...`: SSH could not log in; the
  end of the line is SSH's reason (`Permission denied (publickey)`, `bad
  permissions` on the key file, `Connection timed out`). Check that `ssh
  bigred200 true` works in a terminal without a prompt, re-open the Duo
  control connection if it expired (`ssh bigred200 true`), and keep the key
  private (`chmod 600 ~/.ssh/id_*`).
- `launch` prints the mission directory and the receipt path on failure;
  the receipt is `~/.local/state/naavik/submissions/<mission-id>.json`.
- `sbatch` refusing with `Invalid account or account/partition combination
  specified`: the host's account is missing, wrong, or not allowed on that
  partition; see [Finding your Slurm account](#finding-your-slurm-account).
  Nothing ran, and jobs submitted before it were cancelled.
- `needs reconciliation` prints a `squeue --name=...` query: run it before
  launching again.
- Logs: `<mission-dir>/logs/<job>/<job id>_<index>.out` and `logs/setup/`.
- `status` shows `ended without a record` (with the log's path) for a job
  killed before it could write its record: out of memory, or the time limit
  without enough `stop_grace_period`.
- A failed environment build ends its log with `FAILED: ...`; the pip output
  above it says why. It leaves no ready marker, so the next launch rebuilds.
- Clean up: `scancel -u $USER --name=<mission-id>.group-<job>`, then remove
  the mission directory. Environments under `directories.python_environments`
  are shared by missions; remove one only when no mission uses it.

Not yet verified on Big Red 200 itself: its `squeue` output as read by
`status`, its module names, and network access from compute nodes. Stages A
and D settle them.
