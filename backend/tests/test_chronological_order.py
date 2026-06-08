"""
Property-Based Test for Output Chronological Order (Property 12)

**Validates: Requirements 10.1**

Property 12: Output Chronological Order
For any successful /generate response, the `timestamps` array SHALL be sorted
in strictly non-decreasing order of their time values (converted to seconds).

This test verifies the property directly against `generate_titles()`:
- For any list of boundary windows (with any start_times), the returned
  Timestamp objects are sorted in non-decreasing order of their time values
  converted to seconds.
"""

from __future__ import annotations

import asyncio
import json
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Allow imports from the backend directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from pipeline.preprocess import TextWindow
from pipeline.titles import (
    generate_titles,
    _time_str_to_seconds,
    _unique_seconds_to_time_str,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_seconds(time_str: str) -> int:
    """Parse 'M:SS' or 'H:MM:SS' to integer seconds (mirrors titles.py helper)."""
    return _time_str_to_seconds(time_str)


def _make_mock_groq_response(titles: list[str]) -> MagicMock:
    """Build a mock Groq API response that returns the given titles as JSON."""
    mock_message = MagicMock()
    mock_message.content = json.dumps(titles)

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    return mock_response


def _make_mock_groq_client(titles: list[str]) -> MagicMock:
    """Build a mock AsyncGroq client whose chat.completions.create returns titles."""
    mock_response = _make_mock_groq_response(titles)

    mock_completions = MagicMock()
    mock_completions.create = AsyncMock(return_value=mock_response)

    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    mock_client = MagicMock()
    mock_client.chat = mock_chat

    return mock_client


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Non-negative start_time values (seconds) — any realistic video duration
_start_time_strategy = st.floats(
    min_value=0.0,
    max_value=86400.0,  # up to 24 hours
    allow_nan=False,
    allow_infinity=False,
)

# A list of start_times (1–20 windows) — deliberately NOT pre-sorted so we
# test that generate_titles() sorts regardless of input order
_start_times_strategy = st.lists(
    _start_time_strategy,
    min_size=1,
    max_size=20,
)

# Simple title text (non-empty)
_title_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
    min_size=1,
    max_size=60,
)


# ---------------------------------------------------------------------------
# Property 12: Output Chronological Order
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(start_times=_start_times_strategy)
def test_generate_titles_returns_timestamps_in_chronological_order(
    start_times: list[float],
) -> None:
    """
    **Validates: Requirements 10.1**

    For any list of boundary windows with arbitrary start_times,
    generate_titles() SHALL return Timestamp objects sorted in non-decreasing
    order of their time values (converted to seconds).
    """
    # Build TextWindow objects with the generated start_times
    windows = [
        TextWindow(text=f"Transcript excerpt {i}.", start_time=t)
        for i, t in enumerate(start_times)
    ]

    # Generate one title per window (content doesn't matter for ordering)
    titles = [f"Chapter {i}" for i in range(len(windows))]
    mock_client = _make_mock_groq_client(titles)

    with patch("pipeline.titles.AsyncGroq", return_value=mock_client), \
         patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        result = _run(generate_titles(windows))

    # The result must have the same number of timestamps as input windows
    assert len(result) == len(windows), (
        f"Expected {len(windows)} timestamps, got {len(result)}"
    )

    # Convert each timestamp's time string to seconds for comparison
    result_seconds = [_to_seconds(ts.time) for ts in result]

    # Verify non-decreasing order
    for i in range(len(result_seconds) - 1):
        assert result_seconds[i] <= result_seconds[i + 1], (
            f"Timestamps are not in non-decreasing order at index {i}: "
            f"{result[i].time} ({result_seconds[i]}s) > "
            f"{result[i + 1].time} ({result_seconds[i + 1]}s)\n"
            f"Full result: {[(ts.time, ts.title) for ts in result]}"
        )


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(start_times=_start_times_strategy)
def test_generate_titles_sorted_order_matches_sorted_start_times(
    start_times: list[float],
) -> None:
    """
    **Validates: Requirements 10.1**

    The time values in the returned timestamps, when converted to seconds,
    SHALL match unique display labels derived from sorted start_times
    (truncated to int, with +1s bumps when labels would collide).
    """
    windows = [
        TextWindow(text=f"Excerpt {i}.", start_time=t)
        for i, t in enumerate(start_times)
    ]

    titles = [f"Title {i}" for i in range(len(windows))]
    mock_client = _make_mock_groq_client(titles)

    with patch("pipeline.titles.AsyncGroq", return_value=mock_client), \
         patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        result = _run(generate_titles(windows))

    used: set[str] = set()
    expected_sorted_seconds = [
        _to_seconds(_unique_seconds_to_time_str(t, used))
        for t in sorted(start_times)
    ]
    actual_seconds = [_to_seconds(ts.time) for ts in result]

    assert actual_seconds == expected_sorted_seconds, (
        f"Returned time values (in seconds) do not match sorted input start_times.\n"
        f"Input start_times (truncated): {[int(t) for t in start_times]}\n"
        f"Expected sorted:               {expected_sorted_seconds}\n"
        f"Actual:                        {actual_seconds}"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_windows_returns_empty_list() -> None:
    """generate_titles([]) must return [] without calling the Groq API."""
    with patch("pipeline.titles.AsyncGroq") as mock_groq_cls, \
         patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        result = _run(generate_titles([]))

    assert result == [], f"Expected [], got {result}"
    mock_groq_cls.assert_not_called()


def test_single_window_returns_single_timestamp_in_order() -> None:
    """A single window must return exactly one timestamp."""
    window = TextWindow(text="Only excerpt.", start_time=42.0)
    mock_client = _make_mock_groq_client(["Only Chapter"])

    with patch("pipeline.titles.AsyncGroq", return_value=mock_client), \
         patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        result = _run(generate_titles([window]))

    assert len(result) == 1
    assert _to_seconds(result[0].time) == 42


def test_already_sorted_windows_remain_sorted() -> None:
    """Windows already in ascending order must produce sorted timestamps."""
    windows = [
        TextWindow(text=f"Excerpt {i}.", start_time=float(i * 60))
        for i in range(5)
    ]
    titles = [f"Chapter {i}" for i in range(5)]
    mock_client = _make_mock_groq_client(titles)

    with patch("pipeline.titles.AsyncGroq", return_value=mock_client), \
         patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        result = _run(generate_titles(windows))

    seconds = [_to_seconds(ts.time) for ts in result]
    assert seconds == sorted(seconds), f"Expected sorted, got {seconds}"


def test_reverse_sorted_windows_produce_sorted_timestamps() -> None:
    """Windows in descending order must still produce ascending timestamps."""
    windows = [
        TextWindow(text=f"Excerpt {i}.", start_time=float((4 - i) * 60))
        for i in range(5)
    ]
    titles = [f"Chapter {i}" for i in range(5)]
    mock_client = _make_mock_groq_client(titles)

    with patch("pipeline.titles.AsyncGroq", return_value=mock_client), \
         patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        result = _run(generate_titles(windows))

    seconds = [_to_seconds(ts.time) for ts in result]
    assert seconds == sorted(seconds), (
        f"Reverse-sorted input must produce ascending output, got {seconds}"
    )


def test_duplicate_start_times_produce_non_decreasing_order() -> None:
    """Windows with identical start_times must produce non-decreasing timestamps."""
    windows = [
        TextWindow(text=f"Excerpt {i}.", start_time=120.0)
        for i in range(4)
    ]
    titles = [f"Chapter {i}" for i in range(4)]
    mock_client = _make_mock_groq_client(titles)

    with patch("pipeline.titles.AsyncGroq", return_value=mock_client), \
         patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        result = _run(generate_titles(windows))

    seconds = [_to_seconds(ts.time) for ts in result]
    for i in range(len(seconds) - 1):
        assert seconds[i] <= seconds[i + 1], (
            f"Duplicate start_times must produce non-decreasing order, got {seconds}"
        )
