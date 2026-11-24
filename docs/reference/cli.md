# CLI Reference

[← Docs](../README.md) · [Project README](../../README.md)

Use `naavik` to run experiments and inspect their results. This page lists its
commands; the engine's own options, for a program of your own, are on
[Running the engine from Python](engine.md). For a walkthrough, start with
[your first run](../guide/00-naavik-first-run.md), then
[Running missions](../guide/08-missions.md).

## The `naavik` command

The package installs `naavik` for launching missions and reading results.
You can also use `navik` or `python -m naavik.cli`:


| Command | Does | Guide |
|---|---|---|
| `naavik init [--host NAME] [--hosts-file PATH] [--gpus all\|none\|N\|0,1] [--python-paths P] [--dry-run] [--yes]` | write the project file and a host for this machine, for `naavik launch` (never overwrites; asks before adding to an existing hosts file) | [Your first run](../guide/00-naavik-first-run.md#launch-it-as-a-mission) |
| `naavik validate / render / launch MISSION_FILE` | check, preview, or launch a mission | [Running missions](../guide/08-missions.md) |
| `naavik run CONFIG [--override PATH=VALUE] [--note TEXT] [--debug]` | run the session of CONFIG here, in this process: one run, no mission | [Running a config](../guide/04-configuration.md#running-a-config) |
| `naavik run --resume-session SESSION_DIR` / `--resume-session-from-checkpoint CHECKPOINT` | continue a session from its latest checkpoint (said in the log), or from the checkpoint given | [Resume](../guide/05-checkpoints-and-resume.md#resume) |
| `naavik launch --session CONFIG [--override PATH=VALUE] [--note TEXT] [--gpus N] [--walltime T] [--detach]` | CONFIG as a mission of one experiment on a host, followed to its end | [`naavik launch --session`](../guide/08-missions.md#one-session-file-followed-naavik-launch---session) |
| `naavik validate / render / launch MISSION_FILE --job J --recipe J/R` | only part of a mission file | [Running missions](../guide/08-missions.md#5-checking-a-mission-validate-and-render) |
| `naavik missions / status / logs / cancel / resume / fetch` | follow, stop, continue, and collect missions | [Running missions](../guide/08-missions.md#7-keeping-track-of-your-missions) |
| `naavik missions --all-hosts` | the missions on every host | [Every host at once](../guide/08-missions.md#every-host-at-once---all-hosts) |
| `naavik show-site-interpreter` | print a site interpreter's identity | [Running missions](../guide/08-missions.md#1-the-hosts-file-your-machines-on-your-machine) |
| `naavik summarize ROOT... [--runs RUN] [--mission MISSION]` | compare sessions | [Comparing sessions](../guide/12-comparing-sessions.md) |
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

`validate`, `render`, and `launch` check every experiment's config with the
mission's own code ([step 9](../guide/08-missions.md#5-checking-a-mission-validate-and-render));
`NAAVIK_CODE_CHECK=0` skips it, with a warning, recorded in the
receipt. `naavik launch --session` exits with the experiment's exit status (`130`
when Ctrl-C stopped following). `naavik run` takes no project file and no host:
it puts the current directory first on the import path (as `python -m`
does), prints the run id and the session directory, and refuses the mission
options (`--gpus`, `--walltime`, `--detach`, `--host`, ...), naming `naavik
launch --session`.

## `naavik runs`

The [run registry](../guide/13-run-registry.md): every engine launch's
record. Also `python -m naavik.runs`.

| Command | Does |
|---|---|
| `naavik runs list [--since AGE] [--state S] [--host H] [--mission TEXT] [--where KEY=VALUE] [--limit N] [--json]` | recorded runs, newest first (at most 50 by default; `--limit 0`: all) |
| `naavik runs show RUN [--json]` | one run in detail: launch, code, environment, devices, sessions, checkpoints, outcome |
| `naavik runs diff A B [--all-keys]` | what differs in code, config, environment, and devices (`--all-keys`: also `sessions_dir` and the rendezvous address and port) |
| `naavik runs sync [--host H ... \| --all-hosts] [--hosts-file FILE]` | copy runs recorded on SSH hosts (default: the hosts file's `default_host`); see [Across machines](../guide/13-run-registry.md#across-machines) |

`RUN`, `A`, `B`: a run id or any unique beginning of one. Every command takes
`--registry PATH` (default `$NAAVIK_RUNS`, else
`~/.local/share/naavik/runs`). Exit status `2` with the reason
on standard error when a run is not found, is ambiguous, or the registry is
turned off. Environment variables: `NAAVIK_RUNS=PATH|off`,
`NAAVIK_RUNS_SNAPSHOT=0`.

## Session summaries

`summarize` is a separate program, not a flag of your training entry point:
it reads finished (or running) sessions and compares them, one row per
session directory (a resumed session is one session, of several runs). See
[Comparing sessions](../guide/12-comparing-sessions.md) for what the report holds.

```bash
naavik summarize ROOT [ROOT ...] [options]
# or: python -m naavik.reporting summarize ROOT [ROOT ...] [options]
```

| Argument | Default | Meaning |
|---|---|---|
| `ROOT` | | A directory to look for sessions (`session_*` directories) in: a `sessions_dir`, a directory above several, one session directory, a mission directory, or `hpc-results/<mission-id>`. A ROOT, `--runs` or `--mission` is needed |
| `--runs RUN` | | The session directories of a [recorded run](../guide/13-run-registry.md) (a run id or a unique beginning; repeatable) |
| `--mission MISSION` | | A mission (an id or its receipt): read on this machine when it ran here, else from its fetched copy; never downloaded (repeatable) |
| `--fetched DIR` | `hpc-results` | Where `naavik fetch` put missions from other hosts |
| `--metric TAG[:min\|:max]` | every tag, last only | A scalar tag to report; with a direction its best value too (repeatable) |
| `--tags GLOB` | | Also report every tag matching `GLOB`, last value only (repeatable) |
| `--last-n N` | | Also report the mean of each tag's last `N` points |
| `--sort TAG` | session id | Order groups and sessions by the best value of this `--metric` (it needs a direction) |
| `--include GLOB` / `--exclude GLOB` | | Choose sessions by id (repeatable) |
| `--seed-key KEY` | `session_config.rng_seed` | A seed: a config path, or `component:path` inside that component's entry wherever it is listed; replaces the default (repeatable) |
| `--no-seed-groups` | | Report sessions only, without grouping over seeds |
| `--ignore-key PATH` | | A config path not to compare, added to the defaults `session_config.sessions_dir`, `ddp.master_addr`, `ddp.master_port`, `resources.ddp.master_addr`, `resources.ddp.master_port` (repeatable) |
| `--no-default-ignores` | | Compare the default ignored keys too |
| `--format markdown\|csv\|json` | `markdown` | Output format |
| `--table groups\|sessions` | `groups` (`sessions` with `--no-seed-groups`) | With `--format csv`: which table |
| `--output PATH`, `-o PATH` | standard output | Where to write the report; never inside a session directory |
| `--strict` | | Exit with status 1 when the report lists problems |

Exit status: `0` the report was written; `1` it was written, has problems,
and `--strict` was given; `2` no report could be made (no sessions, a tag no session
writes, an unknown seed key, conflicting options), with the reason on
standard error.

## Running the engine yourself

The engine's own options (`--config`, `--resume-session`, `--extend-session`,
`--override`, `--debug`, and the process-monitoring timeouts), for a program
of your own such as `python -m my_project.train`, are on
[Running the engine from Python](engine.md).

---

**See also:** [Architecture and process model](../concepts/architecture.md)
explains what these timeouts are measuring.
