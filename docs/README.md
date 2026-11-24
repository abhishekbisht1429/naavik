# Documentation

[Project README](../README.md)

Naavik helps you launch training experiments, follow them across machines,
and compare their results. Its training framework provides the sessions and
components those experiments run. Start with the path that fits your work;
you can return to the reference and concepts pages whenever you need them.

## Run experiments with Naavik

1. Follow [your first Naavik run](guide/00-naavik-first-run.md) for installation,
   setup, and a local experiment.
2. Run one configuration or a sweep locally, over SSH, or on a cluster with
   [missions](guide/08-missions.md). For a single configuration, see
   [`naavik launch --session`](guide/08-missions.md#one-session-file-followed-naavik-launch---session),
   or run it here, without a mission, with
   [`naavik run`](guide/04-configuration.md#running-a-config).
3. Use the same guide to [follow a mission](guide/08-missions.md#7-keeping-track-of-your-missions),
   [resume unfinished experiments](guide/08-missions.md#10-resuming-unfinished-experiments),
   and [fetch results](guide/08-missions.md#11-getting-results-back).
4. Find and inspect recorded runs with the [run registry](guide/13-run-registry.md),
   then [compare their sessions' results](guide/12-comparing-sessions.md).

A **mission** holds experiments; each attempt of an experiment is a **run**;
a run builds or continues a **session**. These and the other terms are
defined in [Missions, experiments, sessions, and runs](concepts/missions-sessions-runs.md).

For a new project, see [setting up a project for missions](guide/10-mission-projects.md).
For IU's cluster, follow the [Big Red 200 checklist](guide/09-big-red-200.md).

## Learn it

### Build training workflows

Start with the [quick start](../README.md#quick-start), then use these guides
to build or customize a session. You can read them in order or choose the
task you need.

1. [Resources, hooks, and steps](guide/01-resources-hooks-steps.md) — the three
   kinds of component, giving one state that survives a checkpoint, and how the
   framework discovers your classes.
2. [Wiring components together](guide/02-wiring-components.md) — activating
   components from YAML, binding a role to an implementation, declaring
   dependencies and wrapping.
3. [Building an iteration](guide/03-building-an-iteration.md) — a training
   run from the built-in `load_batch`, `forward` and `compute` steps and the
   optimizer, configured in YAML; recipes for several losses, a second model
   and two views.
4. [Configuration](guide/04-configuration.md) — the session file,
   `session_config` fields, and command-line overrides.
5. [Checkpoints, resume, and extend](guide/05-checkpoints-and-resume.md) — what
   a checkpoint holds, continuing a run, and changing hyperparameters safely.
6. [Distributed training](guide/06-distributed-training.md) — DDP configuration,
   how a launch decides its topology, and coordinated stopping.
7. [Analysis sessions](guide/07-analysis-sessions.md) — driving a trained
   checkpoint through an analysis workflow.
8. [Running missions: here, over SSH, or on a cluster](guide/08-missions.md) — sweeps of
   experiments from one mission file on this machine, a machine over SSH, or a
   Slurm cluster (committed or as a snapshot of your working tree), and
   keeping track of every mission you launched.
9. [First runs on Big Red 200](guide/09-big-red-200.md) — SSH, host settings,
   and a staged checklist for bringing missions up on IU's cluster.
10. [Setting up a project for missions](guide/10-mission-projects.md) — where
    the framework comes from, generating the pinned requirements, and projects that depend on
    naavik.
11. [Self-supervised training with DINO](guide/11-self-supervised-dino.md) —
    a ViT-S/16 DINO run built part by part, checked on a small scale,
    resumed, and evaluated with a linear probe; what changes for BYOL.
12. [Comparing sessions](guide/12-comparing-sessions.md) — one report over a sweep's
    sessions, local or from a mission: the hyperparameters that differ, each metric, and
    sessions that differ only in their seed as mean ± std.
13. [Every run is recorded](guide/13-run-registry.md) — the record each
    launch leaves (config, code, environment, devices, outcome), and
    `naavik runs` to list, show, compare (`diff`), and sync them.

## Look it up

- [The hosts file](reference/hosts-file.md) — the machines you run on
  (this machine, over SSH, or a Slurm cluster): connection, scheduler,
  directories, Python environments; read by missions (`naavik launch`) and
  `naavik runs sync`.
- [Mission files and the project file](reference/mission-files.md) — every
  key of the committed files: code, packages, data sources, jobs, recipes.
- [Earlier names](reference/earlier-names.md) — old file, key, and command
  names that are still read, and what they are called now.
- [Built-in components](reference/builtin-components.md) — every component the
  framework ships with, and its configuration keys.
- [Generic steps](reference/generic-steps.md) — `load_batch`, `forward` and
  `compute`: every key, and how values go in and out of `iteration_context`.
- [Optimization](reference/optimization.md) — `optimizer`, the steps from
  `backward` to `optimizer_step`, accumulation, precision, custom gradient
  stages.
- [Importing components](reference/import-components.md) —
  `import_components`: another run's model, and what it was wired to, as this
  session's own components.
- [Fine-tuning](reference/fine-tuning.md) — `module_part` and
  `backbone_with_head`: a pretrained backbone from another run, frozen layers,
  pooling, and your own head.
- [Self-supervised training](reference/self-supervised.md) — DINO and BYOL
  from generic parts: `projection_head`, `multi_view`, `ema_teacher`,
  `ema_update`, `dino_distillation_loss`, and dataset views.
- [Transformer blocks](reference/transformer-blocks.md) — the swappable
  transformer building blocks and three composite models.
- [Image datasets](reference/datasets.md) — optional torchvision datasets
  (CIFAR-10, Flowers-102, Stanford Cars, iNaturalist, ImageNet) for the
  `dataset` role.
- [Infinite samplers](reference/samplers.md) — `InfiniteSampler` and
  `DistributedInfiniteSampler`.
- [Profiler](reference/profiler.md) — `profiler`: `torch.profiler` traces
  of chosen iterations and ranks, and what they show.
- [CLI reference](reference/cli.md) — every `naavik` command and flag, and the
  `summarize` command for comparing sessions.
- [Running the engine from Python](reference/engine.md) — the engine without
  the launcher: your own entry point, its flags (`--resume-session`,
  `--extend-session`, `--debug`, ...), and how its overrides are read.
- [API summary](reference/api.md) — public imports and type members.

## Understand it

- [Missions, experiments, sessions, and runs](concepts/missions-sessions-runs.md)
  — every naavik term, how they nest, where each lives, and which command
  shows it.
- [Architecture and process model](concepts/architecture.md) — parent and
  worker responsibilities, spawning, supervision, heartbeats.
- [Session lifecycle](concepts/session-lifecycle.md) — exact phase ordering,
  failure and rollback, the two shared contexts.
- [The component model](concepts/component-model.md) — what a component may do
  while being constructed, configuration schemas, dependency errors.
- [`ModuleResource`](concepts/module-resource.md) — composing `nn.Module`
  resources and who owns which weights.
- [Current behavior and limitations](concepts/limitations.md) — known
  constraints, grouped by area.

## Contributing

- [Development and testing](development.md) — running the suite, project layout.
