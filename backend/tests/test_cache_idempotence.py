"""
Property-Based Test for Cache Idempotence (Property 6)

**Validates: Requirements 5.2, 5.3**

Property 6: Cache Idempotence
For any video_id that has been successfully processed once, a second call to
/generate with the same video_id SHALL return an identical timestamps array
and the pipeline SHALL NOT be re-executed.

This test verifies the property directly against the cache module:
- get/set/get cycle returns identical results
- A simulated pipeline counter increments only once per unique video_id
"""

from __future__ import annotations

import sys
import os

# Allow imports from the backend directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import importlib
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

import cache
from models import Timestamp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_timestamps(n: int, base_title: str = "Section") -> list[Timestamp]:
    """Create a deterministic list of n Timestamp objects."""
    return [
        Timestamp(time=f"{i}:00", title=f"{base_title} {i}")
        for i in range(n)
    ]


def _reset_cache() -> None:
    """Clear the module-level cache dict between test runs."""
    cache._cache.clear()


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid video_id: non-empty, stripped strings (YouTube IDs are typically
# alphanumeric + hyphens/underscores, but the cache accepts any non-empty str)
video_id_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="-_",
    ),
    min_size=1,
    max_size=32,
).filter(lambda s: s.strip() != "")

# Timestamp list strategy: 1–20 timestamps with realistic time/title values
timestamp_list_strategy = st.lists(
    st.builds(
        Timestamp,
        time=st.from_regex(r"\d{1,2}:\d{2}", fullmatch=True),
        title=st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
            min_size=1,
            max_size=60,
        ),
    ),
    min_size=1,
    max_size=20,
)


# ---------------------------------------------------------------------------
# Property 6a: Cache get/set/get returns identical result
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(video_id=video_id_strategy, timestamps=timestamp_list_strategy)
def test_cache_get_after_set_returns_identical_timestamps(
    video_id: str, timestamps: list[Timestamp]
) -> None:
    """
    **Validates: Requirements 5.2, 5.3**

    For any video_id and any list of timestamps:
    - After cache.set(video_id, timestamps), cache.get(video_id) returns
      a list equal to the stored list.
    - A second call to cache.get(video_id) returns the same result again
      (idempotent reads).
    """
    _reset_cache()

    # Initially the cache should be empty for this video_id
    assert cache.get(video_id) is None

    # Store timestamps
    cache.set(video_id, timestamps)

    # First retrieval
    first_result = cache.get(video_id)
    assert first_result is not None, "cache.get should return stored timestamps"
    assert first_result == timestamps, (
        f"First retrieval mismatch for video_id={video_id!r}"
    )

    # Second retrieval — must be identical (idempotent)
    second_result = cache.get(video_id)
    assert second_result == first_result, (
        f"Second retrieval differs from first for video_id={video_id!r}"
    )


# ---------------------------------------------------------------------------
# Property 6b: Pipeline runs only once per video_id
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(video_id=video_id_strategy, timestamps=timestamp_list_strategy)
def test_pipeline_runs_only_once_per_video_id(
    video_id: str, timestamps: list[Timestamp]
) -> None:
    """
    **Validates: Requirements 5.2, 5.3**

    Simulates the /generate endpoint cache-check logic:
    - A pipeline callable is invoked only when the cache misses.
    - On a second request for the same video_id the pipeline is NOT called again.
    - Both calls return identical timestamp lists.
    """
    _reset_cache()

    pipeline_call_count = 0

    def fake_pipeline(vid: str) -> list[Timestamp]:
        nonlocal pipeline_call_count
        pipeline_call_count += 1
        return timestamps

    def simulate_generate(vid: str) -> list[Timestamp]:
        """Mirrors the cache-check logic in the /generate endpoint."""
        cached = cache.get(vid)
        if cached is not None:
            return cached
        result = fake_pipeline(vid)
        cache.set(vid, result)
        return result

    # First call — cache miss, pipeline should run
    first_response = simulate_generate(video_id)
    assert pipeline_call_count == 1, (
        f"Pipeline should run exactly once on first call, got {pipeline_call_count}"
    )

    # Second call — cache hit, pipeline must NOT run again
    second_response = simulate_generate(video_id)
    assert pipeline_call_count == 1, (
        f"Pipeline should NOT run on second call (cache hit), "
        f"but ran {pipeline_call_count} times total"
    )

    # Both responses must be identical
    assert first_response == second_response, (
        f"Responses differ between first and second call for video_id={video_id!r}"
    )


# ---------------------------------------------------------------------------
# Property 6c: Different video_ids are cached independently
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    video_id_a=video_id_strategy,
    video_id_b=video_id_strategy,
    timestamps_a=timestamp_list_strategy,
    timestamps_b=timestamp_list_strategy,
)
def test_different_video_ids_cached_independently(
    video_id_a: str,
    video_id_b: str,
    timestamps_a: list[Timestamp],
    timestamps_b: list[Timestamp],
) -> None:
    """
    **Validates: Requirements 5.2, 5.3**

    Storing timestamps for video_id_a does not affect the cache entry for
    video_id_b, and vice versa. Each video_id's pipeline runs independently.
    """
    _reset_cache()

    pipeline_calls: dict[str, int] = {}

    def fake_pipeline(vid: str, ts: list[Timestamp]) -> list[Timestamp]:
        pipeline_calls[vid] = pipeline_calls.get(vid, 0) + 1
        return ts

    def simulate_generate(vid: str, ts: list[Timestamp]) -> list[Timestamp]:
        cached = cache.get(vid)
        if cached is not None:
            return cached
        result = fake_pipeline(vid, ts)
        cache.set(vid, result)
        return result

    # Process both video IDs once
    result_a1 = simulate_generate(video_id_a, timestamps_a)
    result_b1 = simulate_generate(video_id_b, timestamps_b)

    # Process both again — should hit cache
    result_a2 = simulate_generate(video_id_a, timestamps_a)
    result_b2 = simulate_generate(video_id_b, timestamps_b)

    # Each pipeline ran at most once per unique video_id
    assert pipeline_calls.get(video_id_a, 0) == 1, (
        f"Pipeline for video_id_a={video_id_a!r} ran "
        f"{pipeline_calls.get(video_id_a, 0)} times, expected 1"
    )
    assert pipeline_calls.get(video_id_b, 0) == 1, (
        f"Pipeline for video_id_b={video_id_b!r} ran "
        f"{pipeline_calls.get(video_id_b, 0)} times, expected 1"
    )

    # Results are stable across calls
    assert result_a1 == result_a2, "video_id_a results differ between calls"
    assert result_b1 == result_b2, "video_id_b results differ between calls"
