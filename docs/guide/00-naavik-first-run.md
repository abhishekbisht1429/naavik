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

## Describe the project and local host

Create `naavik-project.yaml` at the root of your new project repository:

```yaml
code:
  framework_from: installed
  python_paths: ["."]
quick_run_defaults:
  per_experiment: {gpus: 0}
```

This tells Naavik to use the installed framework and make your project package
importable. The resource settings describe one experiment: here, no GPUs. On
this machine's built-in scheduler, an experiment has no time, CPU, or memory
limit unless you set `walltime` (CPUs and memory are never limited there).

Create a host file outside the project. These commands use absolute paths
under your home directory for the local mission files and data:

```bash
mkdir -p "$HOME/naavik"
cat > "$HOME/naavik/hosts.yaml" <<EOF_HOSTS
default_host: here
hosts:
  here:
    connection: local
    scheduler: {kind: builtin}         # gpus: all, the default
    run_code_from: working_tree
    python_environment: {use_existing: current}
    directories:
      missions: $HOME/naavik/missions
      python_environments: $HOME/naavik/envs
      datasets: $HOME/naavik/datasets
EOF_HOSTS
```

This example uses a separate hosts file so it does not change your existing
Naavik host configuration. `working_tree` takes a snapshot of your current project, so you do
not need to commit each edit. The current environment must remain available
for the local worker.

## Run the experiment

From your new project repository root:

```bash
naavik run my_project/config.yaml --hosts-file "$HOME/naavik/hosts.yaml" --note "first local run"
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
[compare results](12-comparing-runs.md), or [build your own components](01-resources-hooks-steps.md).
