# Running the engine from Python

[← Docs](../README.md)

`naavik run CONFIG` runs a session file's session with the engine, here,
and `naavik launch` runs missions; both build the engine for you. This page
is for running the engine yourself, from a program of your own. Use it to:

- embed a session in a larger program or a notebook;
- extend a session (`--extend-session`): naavik has no `extend` yet.

Resuming (`--resume-session`) and `--debug` are also on `naavik run`, which
needs no entry point of your own.

Such a launch is a run like any other, recorded in the
[run registry](../guide/13-run-registry.md); it is not a mission: no
receipt, no `naavik status`.

## The entry point

```python
# my_project/train.py
from naavik.engine import Configurator
from naavik.engine import TrainingEngine


def main() -> None:
    configurator = Configurator()

    with TrainingEngine(configurator) as engine:
        engine.start_session()


if __name__ == "__main__":
    main()
```

```bash
python -m my_project.train --config my_project/config.yaml
```

The `if __name__ == "__main__"` guard is required for safe process spawning.
The parent process creates the session and worker configuration; the worker
process reconstructs the session and executes the configured components.
`Configurator` reads the options below from the command line; see the
[API summary](api.md) for its methods and `TrainingEngine`.

## Operations

The operations are mutually exclusive and one is required. New-session
type selection and constructor-specific arguments belong in the session file
(`session_type`, `session_kwargs`).

| Argument | Operation | Notes |
|---|---|---|
| `--config PATH` | Start a new session from a YAML file that is the session | It may [include other files](../guide/04-configuration.md#sharing-configuration-between-files); overrides are relative to the session and applied after the includes; see [Overrides](#overrides) |
| `--resume-session SESSION_DIR` | Continue a session, unchanged, from its latest checkpoint | The latest complete checkpoint (highest iteration) under the session directory; the log says which. Accepts only the three launch-topology overrides; see [Resume](../guide/05-checkpoints-and-resume.md#resume) |
| `--resume-session-from-checkpoint CHECKPOINT` | Continue a session, unchanged, from this checkpoint | As above, from the checkpoint given |
| `--extend-session SESSION_DIR` | Continue a session from its latest checkpoint with changed, extension-safe settings | Overrides are relative to the session, as with `--config`; see [Extend](../guide/05-checkpoints-and-resume.md#extend) |
| `--extend-session-from-checkpoint CHECKPOINT` | The same, from this checkpoint | Also takes the deprecated `CHECKPOINT NEW_MAX_ITERATIONS` |

A checkpoint given to `--resume-session` or `--extend-session` is refused,
naming the `-from-checkpoint` flag: those take a session directory. A
checkpoint outside its session directory (an explicit
`checkpointer.checkpoints_dir`) is not found by the session flags; give it
with the `-from-checkpoint` ones.

`--override KEY=VALUE [KEY=VALUE ...]` applies to any of them: the
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

## Overrides

`--override KEY=VALUE [KEY=VALUE ...]` takes OmegaConf dot-list items,
relative to the session: `logger.log_every=25`,
`steps.forward.args=[x]`. A file still in the earlier `sessions: [...]`
form also takes `sessions[0].logger.log_every=25`; on a session file that
form is refused with the path to write.

Values are read by OmegaConf, not as YAML as `naavik run --override` reads
them: `3e-4` is the number 0.0003 here but the string `"3e-4"` for
`naavik run`. Write `3.0e-4`, a number in both.

A mission passes some of these options to every experiment through
`base.engine_options` (`--heartbeat-timeout`, `--stop-sync-grace-period`,
`--stop-sync-poll-interval`, `--process_timeout_on_join`; see
[mission files](mission-files.md#base)).

---

**See also:** [Architecture and process model](../concepts/architecture.md)
explains what these timeouts are measuring;
[Checkpoints, resume, and extend](../guide/05-checkpoints-and-resume.md) what
resume and extend restore and allow.
