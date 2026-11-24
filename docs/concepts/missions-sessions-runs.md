# Missions, Experiments, Sessions, and Runs

[← Docs](../README.md) · [Project README](../../README.md)

Naavik uses a handful of words with exact meanings. This page defines them,
shows how they nest, and says where each one lives and which command shows
it. The guides use these words only in these senses.

## How they nest

```text
mission                one `naavik launch`, on one host
 └─ job                experiments with the same resources: one scheduler job
     └─ recipe         makes experiments: overrides, vary, config files, from_mission
         └─ experiment one session file with its values: <job>/<recipe>/<NNN>
             └─ attempt   1, then 2, 3, ... with each `naavik resume`
                 └─ run      the session in progress: a run id and a run record
                     └─ session   training or analysis, in its session directory
                          └─ checkpoints
```

**A session and its runs are a program and its processes.** The session is
what exists: its directory, its state, its checkpoints, whether anything is
running or not. A run is the session in progress, from the moment the
engine starts or continues it until it stops, ends, or fails, just as a
process is a program running. Two facts follow:

- **A session outlives its runs.** A resume or an extension is a new run (a
  new run id) that continues the *same* session, in the same session
  directory. One session can have several runs; `naavik summarize` compares
  sessions, `naavik runs` lists runs.
- **A run need not belong to a mission.** `naavik run CONFIG` runs a
  session here, in its own process: a run, with a record, but no mission,
  experiment, or attempt. So does running the engine from a program of your
  own ([Running the engine from Python](../reference/engine.md)).

## The terms

**Session.** What the framework runs: a training or analysis workflow built
from components, with its iteration count, its RNG seed, and its state. A
session is described by a **session file** (YAML: `session_config` and its
components, see [Configuration](../guide/04-configuration.md)) and lives in
a **session directory**, `session_<timestamp>/`, holding its `config.yaml`,
its logs and TensorBoard events, and its checkpoints. Its **session type**
(`training`, `analysis`) decides which components it may use.

**Component.** A piece of a session: a **resource** (state shared by the
session: a model, a dataset, the optimizer), a **step** (work done every
iteration), or a **hook** (work at fixed points of the lifecycle). Components
are registered from your **components package** and wired by **role**
(`model`, `dataset`, ...). See
[Resources, hooks, and steps](../guide/01-resources-hooks-steps.md).

**Checkpoint.** A session's saved state at one iteration, in its session
directory; a resume or an extension starts from one. See
[Checkpoints, resume, and extend](../guide/05-checkpoints-and-resume.md).

**Run.** A session in progress: the engine builds a new session (or
restores one from a checkpoint) and runs it until it ends, fails, or is
stopped. `naavik run CONFIG` starts a session as a run; `naavik run
--resume-session SESSION_DIR` continues one as another run, from its latest
checkpoint. Every run gets a **run id** and leaves a **run record** (what
ran, where, on which devices, how it ended) in the **run registry** of the
machine it ran on. See [Every run is recorded](../guide/13-run-registry.md).
(A program of your own can register several sessions with one engine; that
launch is then one run of all of them.)

**Experiment.** One session file with the values its recipe sets, run on a
host: named `<job>/<recipe>/<NNN>` (`run/config/000` for `naavik launch
--session`).
Each experiment has its own directory, and so its own session.

**Attempt.** One try at an experiment. The first launch is attempt 1;
`naavik resume` continues unfinished experiments from their latest
checkpoint as attempt 2, 3, .... Each attempt is one run, of the same
session.

**Recipe** and **job.** A mission file groups experiments into **jobs**:
experiments that need the same resources (GPUs, time) and run as one
scheduler job. A job's **recipes** say which experiments to make: values to
set (`overrides`), values to sweep (`vary`: one experiment per combination),
other session files (`config_file`, `config_files`), or another mission's
checkpoints to analyse (`from_mission`).

**Mission.** A set of experiments launched together on one host with
`naavik launch`: from a mission file, or from one session file (`naavik
launch --session`: a mission of one experiment). It has a **mission id**,
`<name>-<UTC time>-<short commit>`; a **mission directory** on the host
(the code that runs, the **plan**, each experiment's config, records, logs,
and sessions); and a **receipt** on your machine (what was launched, where,
and the job references). The code a mission runs is one exact commit, or a
**snapshot** of your working tree kept as a Git ref.

**Host** and **scheduler.** A **host** is a machine you run on, described in
your **hosts file**: how it is reached (**connection**: `local` or SSH) and
what runs its jobs (**scheduler**). The **builtin scheduler** is naavik's
own, for a machine without one: a **runner** process per job that gives
each experiment its GPUs from the host's **GPU pool** (`scheduler.gpus`) and
stops it at its walltime. On a cluster the scheduler is Slurm. See
[the hosts file](../reference/hosts-file.md).

## The files you write

| File | What it says | Committed? | Reference |
|---|---|---|---|
| session file (`config.yaml`) | one session: `session_config` and its components | yes | [Configuration](../guide/04-configuration.md) |
| mission file (`missions/sweep.yaml`) | the experiments of a mission: jobs, recipes, resources | yes | [mission files](../reference/mission-files.md) |
| project file (`naavik-project.yaml`) | what a project's missions share: where the framework comes from, `PYTHONPATH`, defaults for `naavik launch --session` | yes | [the project file](../reference/mission-files.md#the-project-file) |
| hosts file (`~/.config/naavik/hosts.yaml`) | your machines: connection, scheduler, directories | no | [the hosts file](../reference/hosts-file.md) |

`naavik init` writes the project file and a host for this machine. `naavik
run` needs neither: only the session file.

## Two examples, end to end

**A run, here.**

```bash
naavik run my_project/config.yaml
```

starts the session as one run:

| Term | Here |
|---|---|
| run | `01M85JJPEMVQKQ75WZD31G4K78` |
| session directory | `./runs/session_20261124_012629/` |
| checkpoints | `./runs/session_20261124_012629/checkpoints/<time>/` |

Stopped with Ctrl-C, the session is still there, with its checkpoints. Then

```bash
naavik run --resume-session ./runs/session_20261124_012629
```

is a new run, `01M85JK36DR7B0DZ54E4M4G397`, of the same session: the log
says `resuming session ./runs/session_20261124_012629 from its latest
checkpoint: ./runs/session_20261124_012629/checkpoints/20261124_012635_454345652
(iteration 20000)`, and its record says it continues
`01M85JJPEMVQKQ75WZD31G4K78`.

**A mission.**

```bash
naavik launch --session my_project/config.yaml
```

makes, in order:

| Term | Here |
|---|---|
| mission | `config-20261124T004953Z-18e63472adeb` (the file's name, the time, the commit) |
| job, recipe | `run`, `config` |
| experiment | `run/config/000` |
| attempt | 1 |
| run | `01M85GFTEYNDFWT721KWR9QE8X` |
| session directory | `<mission dir>/tasks/run/config/000/sessions/session_20261124_004957/` |
| checkpoints | `.../session_20261124_004957/checkpoints/<time>/` |

Cancelled mid-way, the experiment is `interrupted`. Then

```bash
naavik resume config-20261124T004953Z-18e63472adeb
```

is attempt 2: a new run, `01M85GH8T97JQEX412MHDH11Z4`, whose record says it
continues `01M85GFTEYNDFWT721KWR9QE8X`, writing into the same
`session_20261124_004957/` from its latest checkpoint. `naavik runs list`
shows two runs; `naavik summarize`, which compares sessions, has one row for
it (when the session writes TensorBoard scalars, as reports need).

## Where each lives, and what shows it

| Term | Lives in | Shown by |
|---|---|---|
| mission | its mission directory on the host; its receipt on your machine | `naavik missions`, `naavik status` |
| experiment, attempt | `<mission dir>/tasks/<experiment>/` (its config, `record.json`, `sessions/`) and `logs/` | `naavik status`, `naavik logs` |
| run | the run registry of the machine that ran it (`~/.local/share/naavik/runs`) | `naavik runs list`, `show`, `diff`; `naavik runs sync` copies |
| session | its session directory | `naavik summarize` |
| checkpoint | `<session dir>/checkpoints/` | `naavik runs show`, `naavik resume` |

`naavik fetch` copies a mission's results from its host to your machine.

## Earlier names

Some of these had other names before: a mission was a *batch*, an
experiment a *task*, a job a *group*, a recipe an *experiment*, a mission
file a *manifest* or *experiment file*, a host a *target* or *profile*. Files
written with them are still read; see [Earlier names](../reference/earlier-names.md).
Mission directories still hold a `tasks/` directory, and logs say `task`,
under the earlier name.

---

**See also:** [Architecture and process model](architecture.md) for what
happens inside one run;
[Running missions](../guide/08-missions.md) for missions in practice.
