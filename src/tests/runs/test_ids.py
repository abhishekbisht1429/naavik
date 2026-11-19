"""Run ids: unique, and sorting by when the run started."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from naavik.runs.ids import LENGTH, is_run_id, new_run_id, run_id_time


def test_an_id_is_26_crockford_characters_that_say_when_it_started():
    run_id = new_run_id(now_ms=1_763_000_000_123, random_bits=0)
    assert len(run_id) == LENGTH == 26 and is_run_id(run_id)
    assert not set(run_id) & set("ILOU")
    assert run_id_time(run_id) == datetime.fromtimestamp(1_763_000_000.123, tz=timezone.utc)


def test_ids_sort_by_start_time_whatever_their_random_part():
    earlier = new_run_id(now_ms=1_000, random_bits=2 ** 80 - 1)
    later = new_run_id(now_ms=1_001, random_bits=0)
    assert earlier < later


def test_ids_made_in_the_same_millisecond_differ():
    assert len({new_run_id(now_ms=5) for _ in range(1000)}) == 1000


@pytest.mark.parametrize("value", [
    "", "x", "01JCXK7Q3F", "01JCXK7Q3F" * 3, "81JCXK7Q3FAAAAAAAAAAAAAAAA",
    "01JCXK7Q3FAAAAAAAAAAAAAAAI", 26, None,
])
def test_what_is_not_a_run_id(value):
    assert not is_run_id(value)


def test_a_time_that_does_not_fit_is_refused():
    with pytest.raises(ValueError):
        new_run_id(now_ms=2 ** 48)
    with pytest.raises(ValueError):
        run_id_time("not-an-id")
