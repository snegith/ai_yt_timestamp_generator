"""
Property-Based Test for Single Batched Groq Call (Property 11)

**Validates: Requirements 9.1**

Property 11: Single Batched Groq Call
For any list of N ≥ 1 boundary windows, `generate_titles(boundary_windows)`
SHALL make exactly one call to the Groq API and SHALL return exactly N
`Timestamp` objects.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Allow imports from the backend directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from models import Timestamp
from pipeline.preprocess import TextWindow
from pipeline.titles import generate_titles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_window(index: int, start_time: float = 0.0) -> TextWindow:
    """Create a TextWindow with deterministic text content."""
    return TextWindow(
        text=f"This is the transcript excerpt for window number {index}.",
        start_time=start_time,
    )


def _make_groq_response(titles: list[str]) -> MagicMock:
    """
    Build a mock Groq chat completion response whose first choice contains
    a JSON-encoded list of title strings.
    """
    message = MagicMock()
    message.content = json.dumps(titles)

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# N: number of boundary windows (at least 1)
window_count_strategy = st.integers(min_value=1, max_value=30)

# Strategy for a single title string (short, printable text)
title_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
    min_size=1,
    max_size=60,
)


# ---------------------------------------------------------------------------
# Property 11a: Exactly 1 Groq API call is made for any N ≥ 1 windows
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(n=window_count_strategy)
def test_exactly_one_groq_api_call(n: int) -> None:
    """
    **Validates: Requirements 9.1**

    For any N ≥ 1 boundary windows, generate_titles() makes exactly 1 call
    to the Groq API (chat.completions.create), regardless of N.
    """
    windows = [_make_window(i, float(i * 60)) for i in range(n)]
    titles = [f"Title {i}" for i in range(n)]
    mock_response = _make_groq_response(titles)

    mock_create = AsyncMock(return_value=mock_response)
    mock_completions = MagicMock()
    mock_completions.create = mock_create
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client = MagicMock()
    mock_client.chat = mock_chat

    with patch("pipeline.titles.AsyncGroq", return_value=mock_client), \
         patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        _run(generate_titles(windows))

    call_count = mock_create.call_count
    assert call_count == 1, (
        f"Expected exactly 1 Groq API call for {n} windows, "
        f"but got {call_count} calls"
    )


# ---------------------------------------------------------------------------
# Property 11b: Exactly N Timestamp objects are returned for N windows
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(n=window_count_strategy)
def test_returns_exactly_n_timestamps(n: int) -> None:
    """
    **Validates: Requirements 9.1**

    For any N ≥ 1 boundary windows, generate_titles() returns exactly N
    Timestamp objects.
    """
    windows = [_make_window(i, float(i * 60)) for i in range(n)]
    titles = [f"Title {i}" for i in range(n)]
    mock_response = _make_groq_response(titles)

    mock_create = AsyncMock(return_value=mock_response)
    mock_completions = MagicMock()
    mock_completions.create = mock_create
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client = MagicMock()
    mock_client.chat = mock_chat

    with patch("pipeline.titles.AsyncGroq", return_value=mock_client), \
         patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        result = _run(generate_titles(windows))

    assert len(result) == n, (
        f"Expected exactly {n} Timestamp objects for {n} windows, "
        f"but got {len(result)}"
    )
    assert all(isinstance(ts, Timestamp) for ts in result), (
        "All returned objects must be Timestamp instances"
    )


# ---------------------------------------------------------------------------
# Property 11 combined: 1 API call AND N Timestamps for any N ≥ 1
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(n=window_count_strategy)
def test_single_call_and_n_timestamps_combined(n: int) -> None:
    """
    **Validates: Requirements 9.1**

    Combined Property 11: for any N ≥ 1 boundary windows,
    generate_titles() SHALL:
      (a) make exactly 1 Groq API call, and
      (b) return exactly N Timestamp objects.
    """
    windows = [_make_window(i, float(i * 60)) for i in range(n)]
    titles = [f"Chapter {i}" for i in range(n)]
    mock_response = _make_groq_response(titles)

    mock_create = AsyncMock(return_value=mock_response)
    mock_completions = MagicMock()
    mock_completions.create = mock_create
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client = MagicMock()
    mock_client.chat = mock_chat

    with patch("pipeline.titles.AsyncGroq", return_value=mock_client), \
         patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        result = _run(generate_titles(windows))

    # (a) exactly 1 API call
    assert mock_create.call_count == 1, (
        f"Expected exactly 1 Groq API call for {n} windows, "
        f"but got {mock_create.call_count} calls"
    )

    # (b) exactly N Timestamp objects
    assert len(result) == n, (
        f"Expected exactly {n} Timestamp objects for {n} windows, "
        f"but got {len(result)}"
    )
    assert all(isinstance(ts, Timestamp) for ts in result), (
        "All returned objects must be Timestamp instances"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_windows_returns_empty_no_api_call() -> None:
    """generate_titles([]) must return [] without making any Groq API call."""
    mock_create = AsyncMock()
    mock_completions = MagicMock()
    mock_completions.create = mock_create
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client = MagicMock()
    mock_client.chat = mock_chat

    with patch("pipeline.titles.AsyncGroq", return_value=mock_client), \
         patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        result = _run(generate_titles([]))

    assert result == [], f"Expected [] for empty input, got {result}"
    assert mock_create.call_count == 0, (
        f"Expected 0 API calls for empty input, got {mock_create.call_count}"
    )


def test_single_window_one_call_one_timestamp() -> None:
    """For exactly 1 window, exactly 1 API call and 1 Timestamp are produced."""
    windows = [_make_window(0, 0.0)]
    titles = ["Introduction"]
    mock_response = _make_groq_response(titles)

    mock_create = AsyncMock(return_value=mock_response)
    mock_completions = MagicMock()
    mock_completions.create = mock_create
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client = MagicMock()
    mock_client.chat = mock_chat

    with patch("pipeline.titles.AsyncGroq", return_value=mock_client), \
         patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        result = _run(generate_titles(windows))

    assert mock_create.call_count == 1, (
        f"Expected 1 API call for 1 window, got {mock_create.call_count}"
    )
    assert len(result) == 1, f"Expected 1 Timestamp, got {len(result)}"
    assert isinstance(result[0], Timestamp)


def test_all_windows_sent_in_single_prompt() -> None:
    """
    The single API call must include all N window texts in its prompt,
    not split across multiple calls.
    """
    n = 5
    windows = [_make_window(i, float(i * 30)) for i in range(n)]
    titles = [f"Title {i}" for i in range(n)]
    mock_response = _make_groq_response(titles)

    mock_create = AsyncMock(return_value=mock_response)
    mock_completions = MagicMock()
    mock_completions.create = mock_create
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client = MagicMock()
    mock_client.chat = mock_chat

    with patch("pipeline.titles.AsyncGroq", return_value=mock_client), \
         patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        result = _run(generate_titles(windows))

    # Only 1 call was made
    assert mock_create.call_count == 1

    # The user message in that single call must reference all N excerpts
    call_kwargs = mock_create.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs["messages"]
    # Find the user message
    user_messages = [m for m in messages if m.get("role") == "user"]
    assert len(user_messages) == 1, "Expected exactly one user message in the prompt"
    user_content = user_messages[0]["content"]

    for i in range(1, n + 1):
        assert f"Excerpt {i}" in user_content, (
            f"Expected 'Excerpt {i}' to appear in the batched prompt, "
            f"but it was missing"
        )
