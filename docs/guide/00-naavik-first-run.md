# Your first local run with Naavik

[← Docs](../README.md) · [Project README](../../README.md)

This guide runs one small session on your own computer, continues it after
an interruption, and then launches it as a mission. It uses the counter
example from the project README; no GPU or cluster is needed. The words
**session**, **run**, and **mission** have exact meanings in naavik; see
[Missions, experiments, sessions, and runs](../concepts/missions-sessions-runs.md).

## Before you start

You need Python 3.12 or newer and an environment with the framework
installed. From a clone of this repository, install it into your active
environment:

```bash
python -m pip install -e .
```

Create the `my_project` package, component module, and `config.yaml` described
in [the README quick start](../../README.md#quick-start), in a new directory,
and run the commands below from that directory.

The example's `components_package: my_project.components` tells the framework
where to find the counter, increment step, and progress hook. Its configuration
runs five CPU iterations and writes a checkpoint.

## Run the session

```bash
naavik run my_project/config.yaml --note "first local run"
```

`naavik run` runs the session of the file with the engine, here, in this
process: one **run**. The progress hook prints counter values; around them,
naavik prints the run id and the session directory:

```text
run 01M85JJBA6X3CFG24YZYBPJDTM (naavik runs show 01M85JJBA6X3CFG24YZYBPJDTM)
session: ./sessions/session_20261124_012617
...
session: ./sessions/session_20261124_012617 (to continue it: naavik run --resume-session ./sessions/session_20261124_012617)
```

The **session** is what stays: its directory holds its `config.yaml`, its
logs, and its checkpoints. The run was the session in progress.

## Continue it

A session stopped before its end (Ctrl-C, a crash) continues from its
latest checkpoint as a new run of the same session:

```bash
naavik run --resume-session ./sessions/session_20261124_012617
```

The log names the checkpoint it took, before anything is built:

```text
resuming session ./sessions/session_20261124_012617 from its latest checkpoint: ./sessions/session_20261124_012617/checkpoints/<time> (iteration 5)
```

`--resume-session-from-checkpoint CHECKPOINT` continues from another
checkpoint instead. See [Resume](05-checkpoints-and-resume.md#resume).

## Find the runs

```bash
naavik runs list
naavik runs show RUN_ID
```

Replace `RUN_ID` with the id, or a unique prefix, from the list. The record
includes the configuration, the code, the outcome, and the session's
directory and checkpoints; a resume's record says which run it continues.

## Launch it as a mission

A **mission** runs sessions as experiments on a host: this machine, one
over SSH, or a cluster, followed with `naavik status`, resumed with `naavik
resume`, swept over values with a mission file. Missions run one exact
commit (or a snapshot of your working tree on top of one), so make the
directory a Git repository with a first commit (Git needs your name and
email for it):

```bash
git init
git add my_project
git commit -m "first commit"
```

Then describe the project and this machine, once:

```bash
naavik init
```

It prints what it detected and writes two files:

- `naavik-project.yaml` at the repository root (commit it): the installed
  framework (`code.framework_from: installed`), your package on the path
  (`code.python_paths: ["."]`), and the host `naavik launch --session` uses
  (`quick_run_defaults.host: here`). With no `per_experiment`, an
  experiment gets no GPUs and has no time, CPU, or memory limit on this
  machine.
- A host named `here` in `~/.config/naavik/hosts.yaml`: this machine
  (`connection: local`), naavik's builtin scheduler with every GPU of the
  machine as its pool (`gpus: all`), your working tree as it is
  (`run_code_from: working_tree`: no commit needed for each edit; a snapshot
  keeps what ran), the environment running `naavik`
  (`use_existing: current`, which must stay available), and absolute
  directories under `~/.local/share/naavik/`.

`naavik init` never overwrites a file. If you already have a hosts file, it
shows the host and asks before adding it (keeping your comments and a
`.bak` copy); without a terminal it only prints the host to add. `--dry-run`
prints both files and writes nothing; `--host`, `--hosts-file`, `--gpus`,
and `--python-paths` change what it writes. Every key is described in the
references for [the hosts file](../reference/hosts-file.md) and
[mission files](../reference/mission-files.md).

Now launch the session file as a mission of one experiment:

```bash
naavik launch --session my_project/config.yaml --note "first mission"
```

Naavik creates a mission containing one experiment, checks and stages it,
then follows its log. The final line should look like:

```text
run/config/000: done
```

The launch also prints the mission directory: its receipt, experiment
files, logs, and the experiment's session. `naavik missions` lists your
missions and `naavik status MISSION_ID` shows their experiments.

Next, learn how to [run missions on other hosts](08-missions.md),
[compare sessions](12-comparing-sessions.md), or [build your own components](01-resources-hooks-steps.md).
