# Naavik

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Package version](https://img.shields.io/badge/version-0.5.0-blue.svg)](./pyproject.toml)
[![Python tests](https://github.com/abhishekbisht1429/naavik/actions/workflows/python-tests.yaml/badge.svg)](https://github.com/abhishekbisht1429/naavik/actions/workflows/python-tests.yaml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](./LICENSE)

Naavik is a component-based framework for training and analysing PyTorch models. It helps you launch, monitor, and compare experiments on your machine, over SSH, or on a Slurm cluster. Its command is available as `naavik` or `navik`.

Start with [your first Naavik run](docs/guide/00-naavik-first-run.md), or follow the component example below if you want to build a training workflow.

Workflow code is organized into reusable **resources**, **hooks**, and **steps**. `TrainingSession` and `AnalysisSession` share the lifecycle implemented by the abstract `Session` base, while using separate component registries and defaults. `TrainingEngine` constructs or restores the appropriate session in the parent process, serializes its state, launches one or more spawned workers, and monitors them for completion, errors, interrupts, and missed heartbeats.

> **Project status:** This project is under active development. The current API is suitable for experimentation and framework development, but review [Current behavior and limitations](docs/concepts/limitations.md) before using it for long-running or production workloads.


## Features

- Configuration-driven session construction with YAML and OmegaConf overrides
- Decorator-based registration and recursive component-package discovery
- Shared components with optional per-session-type registry overrides
- Explicit resource, hook, and step dependencies
- Opt-in rollback for partially initialized resources and session hooks
- Topological execution ordering and dependency-cycle detection
- Built-in batch, forward and loss steps configured in YAML, ordered by the `iteration_context` keys they read and write
- Session-level and iteration-level shared contexts
- Stateful component checkpointing and session restoration
- Restoration of Python, NumPy, PyTorch, and CUDA RNG state
- Spawn-based worker processes with error forwarding and heartbeat monitoring
- Single-node PyTorch DistributedDataParallel setup with coordinated stopping
- Resume and extend operations for saved sessions
- Analysis sessions driven by a trained-session checkpoint
- Mission launches, monitoring, run records, and comparisons with Naavik
- Built-in model loading, logging, checkpointing, DDP, data management, optimization, timing, TensorBoard, and infinite samplers

## Requirements

The package currently declares:

- Python 3.12 or newer
- PyTorch 2.13.0 (pinned)
- NumPy 2.4.4 or newer
- OmegaConf 2.3 or newer
- Matplotlib 3.10.9 or newer
- TensorBoard 2.20 or newer
- PyYAML
- pytest

See [`pyproject.toml`](./pyproject.toml) for the complete dependency list.

## Installation

### Install from GitHub

```bash
python -m pip install \
  "git+https://github.com/abhishekbisht1429/naavik.git@main"
```

### Development installation

```bash
git clone https://github.com/abhishekbisht1429/naavik.git
cd naavik
python -m pip install --upgrade pip
python -m pip install -e .
```

Missions need nothing extra: the launcher is part of the core install. For
the built-in image datasets, install the `vision` extra; from a clone, use
`python -m pip install -e '.[vision]'`. See
[mission setup](docs/guide/10-mission-projects.md) for environment and CUDA
considerations.

### Optional: Graphviz

The `model_diagram` built-in lays out its pictures with Graphviz when the
`dot` program is installed, and falls back to matplotlib otherwise. Graphviz
is a system program, so it is not a pip dependency:

```bash
conda install -c conda-forge graphviz   # or: sudo apt-get install graphviz
```

## Quick start

This example defines a stateful resource, a training step, and a lifecycle hook. The framework discovers the component module, creates the components from YAML, spawns a worker, and runs five iterations.

### 1. Create a project package

```text
my_project/
├── __init__.py
├── config.yaml
└── components/
    ├── __init__.py
    └── demo.py
```

The package containing decorated components must be importable by both the parent process and spawned worker processes.

### 2. Define components

Create `my_project/components/demo.py`:

```python
from naavik.components import (
    LifecycleHook,
    StatefulResource,
    Step,
    hook,
    reads,
    requires_resource,
    resource,
    step,
    writes,
)
from naavik.session import TrainingSession


@resource("counter")
class CounterResource(StatefulResource):
    def __init__(self, config: dict):
        self.value = int(config.get("start", 0))

    def setup(self, session: TrainingSession) -> None:
        pass

    def teardown(self, session: TrainingSession) -> None:
        pass

    def get_state(self) -> dict[str, int]:
        return {"value": self.value}

    def set_state(self, state: dict[str, int]) -> None:
        self.value = state["value"]


@step("increment")
@requires_resource("counter")
@writes("counter_value")
class IncrementStep(Step):
    def __init__(self, config: dict):
        self.amount = int(config.get("amount", 1))

    def run(self, session: TrainingSession) -> int:
        counter = self.get_dependency("counter")
        counter.value += self.amount
        return counter.value


@hook("progress")
@reads("counter_value")
class ProgressHook(LifecycleHook):
    def __init__(self, config: dict):
        self.call_every = int(config.get("call_every", 1))

    def pre_session(self, session: TrainingSession) -> None:
        pass

    def post_session(self, session: TrainingSession) -> None:
        pass

    def pre_iteration_callback(self, session: TrainingSession) -> None:
        pass

    def post_iteration_callback(
        self, session: TrainingSession, counter_value: int
    ) -> None:
        print(f"iteration={session.iteration}, counter={counter_value}")
```

Each configured component class receives its YAML mapping as one `config` argument.
The step and the hook share a value through the iteration's context: `@writes`
stores what `run` returns under `counter_value`, and `@reads` passes it to
`post_iteration_callback` as the `counter_value` argument (see
[Ordering by dataflow](docs/guide/02-wiring-components.md#ordering-by-dataflow)).

### 3. Create the YAML configuration

Create `my_project/config.yaml`:

```yaml
session_config:
  rng_seed: 42
  sessions_dir: ./runs
  max_iterations: 5
  components_package: my_project.components
  device: cpu

resources:
  counter:
    start: 0

steps:
  increment:
    amount: 2

hooks:
  progress:
    call_every: 1
  logger:
    log_every: 1
  checkpointer:
    checkpoint_every: 5
```

Components are listed by kind under `resources`, `steps` and `hooks`, and each
entry must name a component visible to the active session type; listing one
under the wrong kind is an error. The groups are optional: a component may
also be listed directly at the top level of the file, next to `session_config`, and the
two layouts can be mixed. Listing order is not execution order -- steps run in
the order their data flows. `role_bindings` maps a role name to a registered
implementation, and configuration belongs under that implementation name (see
[wiring](docs/guide/02-wiring-components.md)). An entry with an empty value
(`counter:`) configures a component with no settings. Required components that
inherit `Component.__init__` are activated without an entry; required
components with a custom constructor must have one.

### 4. Set up naavik for this machine (once)

From the directory containing `my_project`, make it a Git repository with a
first commit (naavik runs your working tree as a snapshot on top of it),
then let `naavik init` write the project file and a host for this machine:

```bash
git init && git add my_project && git commit -m "first commit"
naavik init
```

`naavik init` writes `naavik-project.yaml` (commit it) and a host named
`here` in `~/.config/naavik/hosts.yaml`, and prints what it chose. It never
overwrites a file; with an existing hosts file, it asks before adding the
host. See [your first run](docs/guide/00-naavik-first-run.md) for what the
two files hold.

### 5. Run it

```bash
naavik run my_project/config.yaml
```

`naavik run` checks the config with your code, runs it as a mission of one
experiment on this machine, and follows its log until it ends. The run is
recorded (`naavik runs list`), uncommitted edits included.

### Running the engine from your own program

The engine also runs inside a program of your own, without the launcher
(`python -m my_project.train --config my_project/config.yaml`): to embed a
session, to debug worker processes, or to extend a checkpoint. See
[Running the engine from Python](docs/reference/engine.md).

## Training a real model

The quick start writes its own step. For ordinary training you do not have to:
the built-in `load_batch`, `forward` and `compute` steps and the `optimizer`
resource do the work from configuration, so the only code is a dataset and a
model, registered as resources:

```python
import torch
from torch import nn

from naavik.components import ModuleResource, Resource, resource


@resource("toy_dataset")
class ToyDataset(Resource):
    def __init__(self, config=None):
        self.inputs = torch.randn(256, 8)
        self.targets = (self.inputs.sum(dim=1) > 0).long()

    def setup(self, session):
        pass

    def teardown(self, session):
        pass

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


@resource("classifier")
class Classifier(ModuleResource):
    def __init__(self, config=None):
        super().__init__(config)
        self.net = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, inputs):
        return self.net(inputs)
```

In the session's YAML, next to `session_config`:

```yaml
role_bindings: {model: classifier, dataset: toy_dataset}

resources:
  classifier:
  toy_dataset:
  ddp: {world_size: 1, backend: gloo}
  data_manager: {batch_size: 32, num_workers: 0, pin_memory: false}
  optimizer: {optimizer: {name: AdamW, kwargs: {lr: 0.001}}}

steps:
  load_batch: {fields: [inputs, targets]}         # batch -> inputs, targets
  forward: {args: [inputs], outputs: logits}      # classifier(inputs)
  compute#loss: {function: cross_entropy, args: [logits, targets], outputs: loss}
```

Each step names the `iteration_context` keys it reads and writes, and the
session orders the steps by them; `optimizer` brings in `backward` (which reads
`loss`) and the update after it. [Building an
iteration](docs/guide/03-building-an-iteration.md) walks through this run and
varies it: several loss terms, a second model, two views of one input.

## Documentation

The [documentation index](docs/README.md) helps you find a path for running
experiments or building components, along with the command and API references.
To launch a sweep or use another machine, see [running missions](docs/guide/08-missions.md).
To inspect and compare results, see [run records](docs/guide/13-run-registry.md)
and [comparing runs](docs/guide/12-comparing-runs.md).

Working on the framework itself? See
[development and testing](docs/development.md).

## License

This repository is licensed under the [Apache License 2.0](./LICENSE).

---
**Author Note**
This README was created with the help of ChatGPT/Claude. Please open an issue if you find anything missing or inconsistent.
