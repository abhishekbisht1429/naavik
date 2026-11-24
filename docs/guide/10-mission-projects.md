# Setting up a project for missions

How to prepare a repository for [missions](08-missions.md): where the
framework comes from, and how to produce the pinned requirements file a
managed environment is built from. Every key used here is described in the
references for [mission files](../reference/mission-files.md) and [the hosts
file](../reference/hosts-file.md).

## Where the framework comes from

| `code.framework_from` | When | The framework comes from | The requirements file |
|---|---|---|---|
| `this_repository` | the repository *is* naavik | the staged commit (`code.python_paths: ["src"]`) | must **not** contain `naavik` |
| `installed` | your project uses naavik as a dependency | the environment on the host | must pin `naavik` exactly once |

Either way your own code is never pip-installed: `code.python_paths` puts it
on `PYTHONPATH` from the staged commit, so code changes do not rebuild the
environment. Only dependency changes do.

## Why a pinned requirements file

`pyproject.toml` says which versions your code is *compatible* with
(`numpy>=2.4`); a managed environment needs the *exact* versions to install.
The requirements file is that resolution, done once on your machine and
committed:

- The environment's key hashes it, so the same file always means the same
  packages, and missions with the same file share one environment.
- It is installed with `pip install --no-deps`: pip resolves nothing on the
  host, so the file must list every package, indirect dependencies
  included.
- `validate` checks it before anything reaches the host: every line is
  `name==X.Y.Z` or a direct reference with a full 40-character commit SHA or
  an archive hash, and the only options are `--index-url`,
  `--extra-index-url`, `--find-links`, and `--hash=`.

You do not need it with `python_environment.use_existing` (an environment
you manage yourself).

## Install uv

[uv](https://docs.astral.sh/uv/) resolves `pyproject.toml` for another
platform and Python version, which is what the host needs. Any one of:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # standalone, into ~/.local/bin
pip install uv
conda install -c conda-forge uv
```

```bash
uv --version
```

## The host's Python version

Resolve for the Python that runs on the host:

```bash
ssh <cluster> 'module load python/<version> && python3 --version'
# or, with the host in hosts.yaml: naavik show-site-interpreter
```

The examples below use 3.12; use what the host reports. With
`interpreter_from: site` it must match the host's `site_interpreter`; with
`uv` or `conda` it is `python_packages.python_version`.

## `this_repository`: naavik itself

```bash
cd naavik
mkdir -p requirements
uv pip compile pyproject.toml --extra vision \
    --python-version 3.12 \
    --python-platform linux \
    -o requirements/missions.txt
grep -i '^naavik' requirements/missions.txt    # must print nothing
```

`uv pip compile pyproject.toml` writes the dependencies only, not the
project itself. Leave out `--extra vision` if no mission uses the image
datasets. `--python-platform linux` resolves for Linux x86_64 even on a Mac.

```yaml
code: {framework_from: this_repository, python_paths: ["src"]}
python_packages:
  requirements_file: requirements/missions.txt
```

## `installed`: a project that uses naavik

### 1. Make the framework commit fetchable

The host installs the framework from Git, so the commit must be pushed:

```bash
cd naavik
git push origin <branch>
git rev-parse <branch>               # the full 40-character SHA
```

Always pin the full SHA: branch names and short SHAs are refused, since they
can point at different code later.

### 2. Project layout

```
my-project/
  pyproject.toml
  naavik-project.yaml      # what the mission files share (step 6)
  my_project/
    __init__.py
    components.py          # session_config.components_package
  configs/train.yaml
  missions/sweep.yaml
  requirements/missions.txt     # generated in step 4
```

### 3. Depend on the framework at that SHA

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "naavik[vision] @ git+https://github.com/abhishekbisht1429/naavik.git@<40-character sha>",
  # your own dependencies
]
```

Leave out `[vision]` if you do not use the built-in image datasets.

### 4. Generate the requirements

```bash
cd my-project
uv pip compile pyproject.toml \
    --python-version 3.12 \
    --python-platform linux \
    -o requirements/missions.txt
grep -i '^naavik' requirements/missions.txt
# naavik @ git+https://github.com/abhishekbisht1429/naavik.git@<the same sha>
```

The framework must appear exactly once, at the full SHA; torch,
torchvision, and the rest are pinned with it.

### 5. Install the same framework on your workstation

The CLI runs locally, from the same commit:

```bash
pip install "naavik @ git+https://github.com/abhishekbisht1429/naavik.git@<the same sha>"
```

### 6. The project file and a mission file

```yaml
# naavik-project.yaml
code:
  framework_from: installed      # the framework comes from the environment
  python_paths: ["."]            # puts my_project/ on PYTHONPATH (["src"] for a src layout)
python_packages:
  requirements_file: requirements/missions.txt
  python_version: "3.12"         # optional with interpreter_from: site
```

```yaml
# missions/sweep.yaml
format_version: 1
name: my-sweep
project_file: naavik-project.yaml
base:
  config_file: configs/train.yaml
jobs:
  cpu:
    per_experiment: {walltime: "00:15:00", cpus: 2, memory: 8G}
    recipes:
      - name: seeds
        vary: [{path: session_config.rng_seed, values: [1, 2]}]
```

In `configs/train.yaml`, set `session_config.components_package:
my_project.components`.

### 7. The host

Let naavik build the environment:

```yaml
python_environment:
  interpreter_from: site
  site_interpreter: {...}         # from show-site-interpreter
environment_build: {on: login_node}   # pip fetches from GitHub and PyPI on the login node
```

Or use `python_environment.use_existing: /path/to/env`, an environment that
already has this framework commit and its dependencies (install them there
with the same `pip install ... @<sha>`).

### A host for this machine

To run the same mission file on your workstation, without committing each
change and in the environment your IDE already uses, let `naavik init` add a
host for this machine (it asks before adding to an existing hosts file):

```bash
naavik init --host here
```

It writes:

```yaml
hosts:
  here:
    connection: local
    scheduler: {kind: builtin, gpus: all}         # any free GPU; or a count, or [indices]
    run_code_from: working_tree                   # run the working tree as it is
    python_environment: {use_existing: current}   # the interpreter running naavik
    directories: {...}                            # absolute, under ~/.local/share/naavik
```

`naavik launch missions/sweep.yaml --host here` then needs no requirements
file (the environment exists) and no commit; see
[Hosts](08-missions.md#hosts-this-machine-an-ssh-machine-or-a-cluster).

### 8. Commit, validate, launch

```bash
git add -A && git commit -m "Set up missions"
naavik validate missions/sweep.yaml
naavik launch missions/sweep.yaml
naavik status <mission-id>
```

`validate` checks that the framework is pinned exactly once and everything
else is pinned; the environment build's preflight checks that
`naavik` imports from the environment.

### A private framework repository

`pip install` on the host must then authenticate to GitHub. Either:

- an SSH URL, `git+ssh://git@github.com/...@<sha>`, with a deploy key on the
  host; or
- a wheel: build it (`python -m build` in the framework repository), copy it
  where the host can read it, and pin it with its archive hash:
  `naavik @ file:///N/.../naavik-0.5.0-py3-none-any.whl#sha256=<hash>`
  (`sha256sum` prints the hash).

## CUDA builds of torch

torch's Linux wheels on PyPI include CUDA, so no extra index is needed for
the default build. For a specific CUDA build, add PyTorch's index (find the
right `cuXXX` for your torch version on pytorch.org):

```bash
uv pip compile ... \
    --extra-index-url https://download.pytorch.org/whl/<cuXXX> \
    --index-strategy unsafe-best-match \
    --emit-index-url
```

`--emit-index-url` writes the index into the file, where it is allowed.

## Without uv: freezing a tested environment

If you have a working environment on Linux x86_64 with the host's Python
version (for example a conda environment on a training machine):

```bash
python -m pip list --format=freeze --exclude-editable > requirements/missions.txt
```

Use `pip list --format=freeze`, not `pip freeze`: for packages installed by
conda, `pip freeze` writes `name @ file:///...` paths that do not exist on
the host (and `validate` refuses). Then edit the file:

- remove conda tooling (`conda`, `conda-package-handling`, `mkl-service`,
  ...), whose versions may not exist on PyPI;
- `framework_from: this_repository`: remove `naavik`;
  `installed`: replace it with the `git+...@<sha>` line;
- a `torch==X.Y.Z+cuXXX` line needs `--extra-index-url
  https://download.pytorch.org/whl/cuXXX` at the top.

This is less reliable than `uv pip compile`: conda builds can differ from
the PyPI ones `pip install --no-deps` will fetch.

A conda environment *on the host* is a separate choice:
`python_environment.kind: conda` creates it with Python and
`python_environment.conda_packages`, then installs this file into it with
pip, so the file holds pip packages only.

## Updating

When `pyproject.toml` changes (or, with `framework_from: installed`, to move
to a new framework commit: push it, put its SHA in `pyproject.toml`, and
reinstall the CLI at that SHA), run the same `uv pip compile` command again
(add `--upgrade` to take newer versions within your ranges) and commit the
file. The next mission builds a new environment once; earlier missions keep
theirs.
