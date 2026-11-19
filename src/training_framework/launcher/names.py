"""Names and hashes shared by every part of a plan (building block B2)."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_SAFE_NAME = re.compile(r"\A[A-Za-z0-9_-]+\Z")


def check_name(value: Any, what: str) -> str | None:
    """The problem with `value` as a name, or None. Names become directory
    names, batch IDs, and job node IDs."""
    if not isinstance(value, str) or not _SAFE_NAME.match(value):
        return (
            f"{what} must be a non-empty string of letters, digits, '_' or "
            f"'-'; got {value!r}"
        )
    return None


#: A mission's id, as `compile_plan` makes it: `<name>-<UTC time>-<commit[:12]>`.
_MISSION_ID = re.compile(r"\A[A-Za-z0-9_-]+-\d{8}T\d{6}Z-[0-9a-f]{12}\Z")


def check_mission_id(value: Any, what: str) -> str | None:
    """The problem with `value` as a mission id, or None."""
    if not isinstance(value, str) or not _MISSION_ID.match(value):
        return (
            f"{what} must be a mission id (<name>-<YYYYMMDDTHHMMSSZ>-<12 hex digits>, as "
            f"`naavik missions` lists it); got {value!r}"
        )
    return None


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def hash_value(value: Any) -> str:
    """SHA-256 of `value` as canonical JSON."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
