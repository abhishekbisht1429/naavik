# Your first local run with Naavik

[← Docs](../README.md) · [Project README](../../README.md)

This guide runs one small experiment on your own computer, follows its log,
and shows you where to find the result. It uses the counter example from the
project README; no GPU or cluster is needed.

## Before you start

You need Git, Python 3.12 or newer, and an environment with the framework
installed (the launcher comes with it). From a clone of this repository,
install it into your active environment:

```bash
python -m pip install -e .
```

Create the `my_project` package, component module, and `config.yaml` described
in [the README quick start](../../README.md#quick-start). Put the package in a
new directory and run the commands below from that directory. The directory
must be a Git repository; initialize it and create an initial commit if it is not already one:

```bash
git init
git add my_project
git commit -m "Add first experiment"
```

Git needs a configured author name and email to make that initial commit.

The example's `components_package: my_project.components` tells the framework
where to find the counter, increment step, and progress hook. Its configuration
runs five CPU iterations and writes a checkpoint.

## Describe the project and this machine

From your new project repository root:

```bash
naavik init
```

It prints what it detected and writes two files:

- `naavik-project.yaml` at the repository root (commit it): the installed
  framework (`code.framework_from: installed`), your package on the path
  (`code.python_paths: ["."]`), and the host `naavik run` uses
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

## Run the experiment

From your new project repository root:

```bash
naavik run my_project/config.yaml --note "first local run"
```

Naavik creates a **mission** containing one **experiment**, checks and stages
it, then follows its log. The progress hook prints counter values during
the run. The final line should look like:

```text
run/config/000: done
```

The launch also prints the submitted mission directory. It contains the
receipt, experiment files, and logs; the run record identifies the session
and checkpoint locations. A local mission is already on this machine, so
fetching it is optional.

## Find the run

```bash
naavik runs list
naavik runs show RUN_ID
```

Replace `RUN_ID` with the ID, or a unique prefix, from the list. The record
includes the configuration, code, outcome, and session locations. To see the
mission and its experiments, use `naavik missions` and
`naavik status MISSION_ID`.

Next, learn how to [run missions on other hosts](08-missions.md),
[compare results](12-comparing-sessions.md), or [build your own components](01-resources-hooks-steps.md).
