from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any
import warnings

import torch


TRAINING_SESSION_TYPE = "training"
ANALYSIS_SESSION_TYPE = "analysis"

#: The ways a configuration can say how long a session runs. Exactly one is
#: given; the session resolves it to `SessionConfig.max_iterations`.
RUN_LENGTH_FIELDS = ("max_iterations", "epochs")

_REQUIRED_SESSION_CONFIG_FIELDS = frozenset({
    "rng_seed",
    "sessions_dir",
    "components_package",
})
_OPTIONAL_SESSION_CONFIG_DEFAULTS = {
    "device": "cpu",
    "show_execution_graph": True,
}
_SESSION_CONFIG_FIELDS = (
    _REQUIRED_SESSION_CONFIG_FIELDS
    | _OPTIONAL_SESSION_CONFIG_DEFAULTS.keys()
    | set(RUN_LENGTH_FIELDS)
)


@dataclass(frozen=True)
class SessionConfig:
    rng_seed: int
    session_dir: str
    #: How many iterations the session runs, whichever way its length was
    #: configured. Everything that needs the length of the run reads this.
    max_iterations: int
    #: The epochs `max_iterations` was resolved from, when the configuration
    #: gave the length in epochs; None when it gave `max_iterations`.
    epochs: int | None = None


class SessionPhase(Enum):
    NEW = auto()
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    FINISHED = auto()
    INTERRUPTED = auto()


def normalize_session_type(session_type: str) -> str:
    if not isinstance(session_type, str):
        raise TypeError("session_type must be a string")
    normalized = session_type.strip()
    if not normalized:
        raise ValueError("session_type must not be empty")
    return normalized


def normalize_session_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("'session_config' must be a mapping")

    missing = sorted(_REQUIRED_SESSION_CONFIG_FIELDS - config.keys())
    if missing:
        raise ValueError(
            "Missing required session_config fields: " + ", ".join(missing)
        )
    unknown = sorted(config.keys() - _SESSION_CONFIG_FIELDS)
    if unknown:
        raise ValueError(
            "Unknown session_config fields: " + ", ".join(unknown)
        )

    normalized = dict(config)
    for name, default in _OPTIONAL_SESSION_CONFIG_DEFAULTS.items():
        normalized.setdefault(name, default)

    rng_seed = normalized["rng_seed"]
    if not isinstance(rng_seed, int) or isinstance(rng_seed, bool):
        raise TypeError("session_config.rng_seed must be an integer")
    field = run_length_field(normalized)
    length = normalized[field]
    if (
            not isinstance(length, int)
            or isinstance(length, bool)
            or length < 0
    ):
        raise ValueError(
            f"session_config.{field} must be a non-negative integer"
        )
    for name in ("sessions_dir", "components_package", "device"):
        value = normalized[name]
        if not isinstance(value, str) or not value:
            raise ValueError(f"session_config.{name} must be a non-empty string")
    if not isinstance(normalized["show_execution_graph"], bool):
        raise TypeError("session_config.show_execution_graph must be a boolean")

    return normalized


#: The device types `session_config.device` may name.
SUPPORTED_DEVICE_TYPES = ("cpu", "cuda")


def device_spelling_problem(spec: str) -> str | None:
    """Why `spec` is not a device a session can name on any machine, or None.

    Depends on nothing about this machine, so a config can be checked before
    it reaches the machine that runs it; `device_problem` adds what does.
    """
    try:
        device = torch.device(spec)
    except (RuntimeError, ValueError):
        return (
            f"session_config.device {spec!r} is not a device; give one of "
            "cpu, cuda or cuda:N"
        )
    if device.type not in SUPPORTED_DEVICE_TYPES:
        return (
            f"session_config.device {spec!r} is a {device.type} device; "
            f"sessions run on {' or '.join(SUPPORTED_DEVICE_TYPES)}"
        )
    return None


def device_problem(spec: str) -> str | None:
    """Why `spec` cannot be this session's device here, or None when it can.

    `spec` is a `session_config.device` string: `cpu`, `cuda`, or `cuda:N`,
    where N counts the devices `CUDA_VISIBLE_DEVICES` leaves visible.
    """
    spelling_problem = device_spelling_problem(spec)
    if spelling_problem is not None:
        return spelling_problem
    device = torch.device(spec)
    if device.type != "cuda":
        return None
    if not torch.cuda.is_available():
        return (
            f"session_config.device is {spec!r}, but CUDA is not available "
            "to this process"
        )
    count = torch.cuda.device_count()
    if device.index is not None and device.index >= count:
        return (
            f"session_config.device is {spec!r}, but only {count} CUDA "
            "device(s) are visible to this process. N in cuda:N counts the "
            "devices CUDA_VISIBLE_DEVICES leaves visible, from 0."
        )
    return None


def resolve_device(spec: str, *, strict: bool) -> torch.device:
    """The device `session_config.device` asks for.

    `strict` -- a session built from a configuration -- raises when it
    cannot be used here. Otherwise -- a session restored from a checkpoint,
    whose stored device may name a GPU this machine lacks -- a device that is
    not a usable CUDA one falls back to CPU with a warning, as checkpoints
    have always loaded; an out-of-range `cuda:N` is kept, as it always was.
    """
    problem = device_problem(spec)
    if problem is None:
        return torch.device(spec)
    if strict:
        raise ValueError(problem)
    try:
        device = torch.device(spec)
    except (RuntimeError, ValueError):
        device = None
    if (
            device is not None
            and device.type == "cuda"
            and torch.cuda.is_available()
    ):
        # Only the ordinal is out of range; a worker pins it and says so.
        return device
    warnings.warn(f"{problem}; running on cpu instead", stacklevel=3)
    return torch.device("cpu")


def run_length_field(settings: Mapping[str, Any]) -> str:
    """The one field of `RUN_LENGTH_FIELDS` that `settings` gives."""
    given = [field for field in RUN_LENGTH_FIELDS if field in settings]
    if len(given) != 1:
        raise ValueError(
            "session_config must give exactly one of "
            + ", ".join(RUN_LENGTH_FIELDS)
            + (f"; it gives {', '.join(given)}" if given else "")
        )
    return given[0]


def replace_run_length(
        settings: Mapping[str, Any],
        field: str,
        value: Any,
) -> dict[str, Any]:
    """`settings` with the run length given as `field`, and only as it."""
    if field not in RUN_LENGTH_FIELDS:
        raise ValueError(
            f"{field!r} is not one of {', '.join(RUN_LENGTH_FIELDS)}"
        )
    replaced = {
        name: setting for name, setting in settings.items()
        if name not in RUN_LENGTH_FIELDS
    }
    replaced[field] = value
    return replaced
