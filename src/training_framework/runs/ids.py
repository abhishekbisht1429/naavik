"""Run ids: 26 characters, sorting by the time a run started.

48 bits of milliseconds since the epoch, then 80 random bits, in Crockford's
base 32 (no I, L, O, U), like a ULID: unique without coordination between
machines, and a plain sort lists runs oldest first.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
LENGTH = 26
_TIME_CHARS = 10


def _encode(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        value, digit = divmod(value, 32)
        chars.append(ALPHABET[digit])
    return "".join(reversed(chars))


def new_run_id(now_ms: int | None = None, random_bits: int | None = None) -> str:
    """A new id; `now_ms` and `random_bits` only for tests."""
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    if not 0 <= now_ms < 2 ** 48:
        raise ValueError(f"time {now_ms} ms does not fit a run id")
    random_bits = int.from_bytes(os.urandom(10), "big") if random_bits is None else random_bits
    return _encode(now_ms, _TIME_CHARS) + _encode(random_bits, LENGTH - _TIME_CHARS)


def is_run_id(value: object) -> bool:
    # 48 bits of time in 10 characters (50 bits): the first is 0-3.
    return (isinstance(value, str) and len(value) == LENGTH
            and all(char in ALPHABET for char in value) and value[0] in "0123")


def run_id_time(run_id: str) -> datetime:
    """When the run with this id started."""
    if not is_run_id(run_id):
        raise ValueError(f"{run_id!r} is not a run id")
    value = 0
    for char in run_id[:_TIME_CHARS]:
        value = value * 32 + ALPHABET.index(char)
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
