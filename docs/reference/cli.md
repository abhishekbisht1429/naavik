# CLI Reference

[← Docs](../README.md) · [Project README](../../README.md)

Use `naavik` to launch missions and inspect their results. This page lists its
commands and the options accepted by a custom Python training entry point. For
a walkthrough, start with [Running missions](../guide/08-missions.md).

## The `naavik` command

The package installs `naavik` for launching missions and reading results.
You can also use `navik` or `python -m naavik.cli`:


| Command | Does | Guide |
|---|---|---|
| `naavik validate / render / launch MISSION_FILE` | check, preview, or launch a mission | [Running missions](../guide/08-missions.md) |
| `naavik run CONFIG [--override PATH=VALUE] [--note TEXT] [--gpus N] [--walltime T] [--detach]` | one config as a mission of one experiment, followed to its end | [`naavik run`](../guide/08-missions.md#one-config-followed-naavik-run) |
| `naavik validate / render / launch MISSION_FILE --job J --recipe J/R` | only part of a mission file | [Running missions](../guide/08-missions.md#5-checking-a-mission-validate-and-render) |
| `naavik missions / status / logs / cancel / resume / fetch` | follow, stop, continue, and collect missions | [Running missions](../guide/08-missions.md#7-keeping-track-of-your-missions) |
| `naavik missions --all-hosts` | the missions on every host | [Every host at once](../guide/08-missions.md#every-host-at-once---all-hosts) |
| `naavik show-site-interpreter` | print a site interpreter's identity | [Running missions](../guide/08-missions.md#1-the-hosts-file-your-machines-on-your-machine) |
| `naavik summarize ROOT... [--runs RUN] [--mission MISSION]` | compare runs | [Comparing runs](../guide/12-comparing-runs.md) |
| `naavik runs list / show / diff / sync` | every recorded run | [Every run is recorded](../guide/13-run-registry.md) |

`naavik <command> --help` lists every option. The earlier entry points run
the same code: `python -m naavik.hpc <command>` and
`python -m naavik.reporting summarize`; so do the same paths under the
framework's earlier name, `training_framework` ([the framework's own
names](earlier-names.md#the-frameworks-own-names)). Earlier names are kept
as aliases: `submit` (`launch`), `batches` (`missions`), `probe-python`
(`show-site-interpreter`), `--target` / `--profile` (`--host`), `--targets` /
`--profiles` (`--hosts-file`), `--task` (`--experiment`), `--tasks`
(`--experiments`). Flags are spelled out in full: an abbreviation is
refused.

`validate`, `render`, `launch`, and `run` check every experiment's config with the
mission's own code ([step 9](../guide/08-missions.md#5-checking-a-mission-validate-and-render));
`NAAVIK_CODE_CHECK=0` skips it, with a warning, recorded in the
receipt. `naavik run` exits with the experiment's exit status (`130` when Ctrl-C
stopped following).

## `naavik runs`

The [run registry](../guide/13-run-registry.md): every engine launch's
record. Also `python -m naavik.runs`.

| Command | Does |
|---|---|
| `naavik runs list [--since AGE] [--state S] [--host H] [--mission TEXT] [--where KEY=VALUE] [--limit N] [--json]` | recorded runs, newest first (at most 50 by default; `--limit 0`: all) |
| `naavik runs show RUN [--json]` | one run in detail: launch, code, environment, devices, sessions, checkpoints, outcome |
| `naavik runs diff A B [--all-keys]` | what differs in code, config, environment, and devices (`--all-keys`: also `sessions_dir` and the rendezvous address and port) |
| `naavik runs sync [--host H ... \| --all-hosts] [--hosts-file FILE]` | copy runs recorded on SSH hosts; pass `--host` explicitly with current hosts files (see [sync limitation](../guide/13-run-registry.md#across-machines)) |

`RUN`, `A`, `B`: a run id or any unique beginning of one. Every command takes
`--registry PATH` (default `$NAAVIK_RUNS`, else
`~/.local/share/naavik/runs`). Exit status `2` with the reason
on standard error when a run is not found, is ambiguous, or the registry is
turned off. Environment variables: `NAAVIK_RUNS=PATH|off`,
`NAAVIK_RUNS_SNAPSHOT=0`.

## Run summaries

`summarize` is a separate program, not a flag of your training entry point:
it reads finished (or running) runs and compares them. See
[Comparing runs](../guide/12-comparing-runs.md) for what the report holds.

```bash
naavik summarize ROOT [ROOT ...] [options]
# or: python -m naavik.reporting summarize ROOT [ROOT ...] [options]
```

| Argument | Default | Meaning |
|---|---|---|
| `ROOT` | | A directory to look for runs (`session_*` directories) in: a `sessions_dir`, a directory above several, one session directory, a mission directory, or `hpc-results/<mission-id>`. A ROOT, `--runs` or `--mission` is needed |
| `--runs RUN` | | The session directories of a [recorded run](../guide/13-run-registry.md) (an id or a unique beginning; repeatable) |
| `--mission MISSION` | | A mission (an id or its receipt): read on this machine when it ran here, else from its fetched copy; never downloaded (repeatable) |
| `--fetched DIR` | `hpc-results` | Where `naavik fetch` put missions from other hosts |
| `--metric TAG[:min\|:max]` | every tag, last only | A scalar tag to report; with a direction its best value too (repeatable) |
| `--tags GLOB` | | Also report every tag matching `GLOB`, last value only (repeatable) |
| `--last-n N` | | Also report the mean of each tag's last `N` points |
| `--sort TAG` | run id | Order groups and runs by the best value of this `--metric` (it needs a direction) |
| `--include GLOB` / `--exclude GLOB` | | Choose runs by id (repeatable) |
| `--seed-key KEY` | `session_config.rng_seed` | A seed: a config path, or `component:path` inside that component's entry wherever it is listed; replaces the default (repeatable) |
| `--no-seed-groups` | | Report runs only, without grouping over seeds |
| `--ignore-key PATH` | | A config path not to compare, added to the defaults `session_config.sessions_dir`, `ddp.master_addr`, `ddp.master_port`, `resources.ddp.master_addr`, `resources.ddp.master_port` (repeatable) |
| `--no-default-ignores` | | Compare the default ignored keys too |
| `--format markdown\|csv\|json` | `markdown` | Output format |
| `--table groups\|runs` | `groups` (`runs` with `--no-seed-groups`) | With `--format csv`: which table |
| `--output PATH`, `-o PATH` | standard output | Where to write the report; never inside a run directory |
| `--strict` | | Exit with status 1 when the report lists problems |

Exit status: `0` the report was written; `1` it was written, has problems,
and `--strict` was given; `2` no report could be made (no runs, a tag no run
writes, an unknown seed key, conflicting options), with the reason on
standard error.

## Your Python training entry point

The following options apply when you launch your own entry point, such as `python -m my_project.train`. See [Configuration](../guide/04-configuration.md) for session settings and [Checkpoints, resume, and extend](../guide/05-checkpoints-and-resume.md) for continuing a session.

## Operations

The three operations, `--config`, `--resume-session`, and
`--extend-session`, are mutually exclusive and one is required. New-session
type selection and constructor-specific arguments belong to each
`sessions[]` entry.

| Argument | Operation | Notes |
|---|---|---|
| `--config PATH` | Start new sessions from a YAML file | Its session entries may [include other files](../guide/04-configuration.md#sharing-configuration-between-files); overrides are addressed with a `sessions[i].` prefix and applied after the includes; see [Configuration](../guide/04-configuration.md#omegaconf-overrides) |
| `--resume-session PATH` | Continue a checkpoint unchanged | Accepts only the three launch-topology overrides; see [Resume](../guide/05-checkpoints-and-resume.md#resume) |
| `--extend-session PATH` | Continue a checkpoint with changed, extension-safe settings | Overrides are session-relative, without a `sessions[0]` prefix; see [Extend](../guide/05-checkpoints-and-resume.md#extend) |

`--override KEY=VALUE [KEY=VALUE ...]` applies to any of the three: the
keys each operation accepts, and how they are addressed, are in the notes
above.

The launch-topology keys `ddp.world_size`, `ddp.master_addr` and
`ddp.master_port` are accepted by every operation, because they describe the
machine rather than the session — see
[The launch decides the topology](../guide/06-distributed-training.md#the-launch-decides-the-topology).

## Debugging

Pass `--debug` for debugger-managed worker processes. Workers are spawned
normally, and the parent waits for them using plain process joins without
heartbeat, failure, timeout, or termination monitoring:

```bash
python -m my_project.train --config my_project/config.yaml --debug
```

## Process-monitoring options

```bash
python -m my_project.train \
  --config my_project/config.yaml \
  --heartbeat-timeout 60 \
  --stop-sync-grace-period 0.01 \
  --stop-sync-poll-interval 0.005 \
  --process_timeout_on_join 30
```

| Argument | Default | Meaning |
|---|---:|---|
| `--heartbeat-timeout` | `30.0` | Maximum seconds a live worker may go without a stage change or `session.send_heartbeat()` call |
| `--stop-sync-grace-period` | `0.01` | Seconds to poll a DDP stop vote still pending when it is read before sleeping |
| `--stop-sync-poll-interval` | `0.005` | Sleep duration between later polls of a pending DDP stop vote |
| `--process_timeout_on_join` | `30.0` | Graceful-shutdown period before surviving workers are terminated |

The stop-sync grace period must be a finite non-negative value. The poll
interval must be finite and greater than zero. These are process-runtime
settings and are not written into session configuration or checkpoints.

---

**See also:** [Architecture and process model](../concepts/architecture.md)
explains what these timeouts are measuring.
