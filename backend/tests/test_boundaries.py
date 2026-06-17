"""
Property-Based Test for Adaptive Topic Boundary Detection (Property 10)

**Validates: Requirements 8.3, 8.4, 10.3**

Property 10: Adaptive Topic Boundary Detection and First-Window Invariant
For a sequence of windows long enough to analyse, `detect_boundaries(windows)`
includes window at index i > 0 iff the cosine similarity between
embed[i-1] and embed[i] falls below the per-video adaptive cutoff
(mean - _STD_FACTOR * std of the consecutive similarities). It always includes
windows[0] and returns a non-empty list for non-empty input.
"""

from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Allow imports from the backend directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from pipeline.preprocess import TextWindow
from pipeline.boundaries import (
    cosine_similarity,
    detect_boundaries,
    _STD_FACTOR,
    _MIN_WINDOWS_FOR_ANALYSIS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_window(text: str = "some text", start_time: float = 0.0) -> TextWindow:
    return TextWindow(text=text, start_time=start_time)


def _make_mock_model(embeddings: np.ndarray) -> MagicMock:
    """Return a mock SentenceTransformer whose encode() returns `embeddings`."""
    mock = MagicMock()
    mock.encode.return_value = embeddings
    return mock


def _build_embeddings_for_similarities(similarities: list[float]) -> np.ndarray:
    """
    Build an embedding matrix such that cosine_similarity(embed[i], embed[i+1])
    equals similarities[i] for each i.

    Each window gets its own perpendicular axis so that controlling one pair's
    similarity does not disturb the others. Every embedding is a unit vector.
    """
    n = len(similarities) + 1  # total number of windows
    actual_dim = max(4, 1 + 2 * len(similarities))
    embeddings = np.zeros((n, actual_dim), dtype=float)

    embeddings[0, 0] = 1.0

    for i, s in enumerate(similarities):
        s_clamped = float(np.clip(s, -1.0, 1.0))
        perp_dim = 1 + 2 * i + 1
        sin_theta = np.sqrt(max(0.0, 1.0 - s_clamped ** 2))
        embeddings[i + 1] = s_clamped * embeddings[i]
        embeddings[i + 1, perp_dim] += sin_theta

        norm = np.linalg.norm(embeddings[i + 1])
        if norm > 0:
            embeddings[i + 1] /= norm

    return embeddings


def _adaptive_cutoff(similarities: list[float]) -> float:
    """Mirror the cutoff computed inside detect_boundaries()."""
    sims = np.asarray(similarities, dtype=float)
    return float(sims.mean()) - _STD_FACTOR * float(sims.std())


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_similarity_value = st.floats(
    min_value=-1.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)

# At least (_MIN_WINDOWS_FOR_ANALYSIS - 1) pairs so the analysed branch runs.
_similarity_list_strategy = st.lists(
    _similarity_value,
    min_size=_MIN_WINDOWS_FOR_ANALYSIS - 1,
    max_size=20,
)

_window_count_strategy = st.integers(min_value=1, max_value=21)


# ---------------------------------------------------------------------------
# Property 10a: windows[0] is always included
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(n=_window_count_strategy)
def test_first_window_always_included(n: int) -> None:
    """
    **Validates: Requirements 8.4**

    For any non-empty list of windows, detect_boundaries() SHALL always include
    windows[0] as the first element of the result.
    """
    windows = [_make_window(f"window {i}", float(i * 10)) for i in range(n)]
    embeddings = np.eye(n, max(n, 1), dtype=float)

    mock_model = _make_mock_model(embeddings)

    with patch("pipeline.boundaries._get_model", return_value=mock_model):
        result = detect_boundaries(windows)

    assert len(result) >= 1, "Result must be non-empty for non-empty input"
    assert result[0] is windows[0], (
        f"windows[0] must always be the first element of the result, "
        f"but got {result[0]!r} instead of {windows[0]!r}"
    )


# ---------------------------------------------------------------------------
# Property 10b: result is non-empty for non-empty input
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(n=_window_count_strategy)
def test_result_non_empty_for_non_empty_input(n: int) -> None:
    """
    **Validates: Requirements 10.3**

    For any non-empty list of windows, detect_boundaries() SHALL return a
    non-empty list (windows[0] is always included).
    """
    windows = [_make_window(f"window {i}", float(i * 10)) for i in range(n)]
    # Identical unit vectors → all similarities equal → std == 0 → only windows[0].
    embeddings = np.ones((n, 4), dtype=float)
    embeddings /= np.linalg.norm(embeddings[0])

    mock_model = _make_mock_model(embeddings)

    with patch("pipeline.boundaries._get_model", return_value=mock_model):
        result = detect_boundaries(windows)

    assert len(result) >= 1, (
        f"detect_boundaries() must return a non-empty list for {n} input windows, "
        f"but returned an empty list"
    )


# ---------------------------------------------------------------------------
# Property 10c: window i>0 included iff similarity < adaptive cutoff
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(similarities=_similarity_list_strategy)
def test_boundary_threshold_iff_similarity_below_adaptive_cutoff(
    similarities: list[float],
) -> None:
    """
    **Validates: Requirements 8.3**

    For a sequence long enough to analyse, detect_boundaries() SHALL include
    window[i] (i > 0) iff cosine_similarity(embed[i-1], embed[i]) is below the
    per-video adaptive cutoff (mean - _STD_FACTOR * std).
    """
    n = len(similarities) + 1
    windows = [_make_window(f"window {i}", float(i * 10)) for i in range(n)]

    embeddings = _build_embeddings_for_similarities(similarities)

    # Recompute the actual similarities from the constructed embeddings so the
    # cutoff in the test exactly matches what detect_boundaries() computes.
    actual_sims = [
        cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(similarities))
    ]
    cutoff = _adaptive_cutoff(actual_sims)

    mock_model = _make_mock_model(embeddings)

    with patch("pipeline.boundaries._get_model", return_value=mock_model):
        result = detect_boundaries(windows)

    result_ids = {id(w) for w in result}

    assert id(windows[0]) in result_ids, "windows[0] must always be in the result"

    for i, sim in enumerate(actual_sims):
        window = windows[i + 1]
        should_include = sim < cutoff
        if should_include:
            assert id(window) in result_ids, (
                f"Window {i + 1} should be included "
                f"(similarity={sim:.6f} < cutoff={cutoff:.6f}), but was absent"
            )
        else:
            assert id(window) not in result_ids, (
                f"Window {i + 1} should NOT be included "
                f"(similarity={sim:.6f} >= cutoff={cutoff:.6f}), but was present"
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_windows_returns_empty() -> None:
    """detect_boundaries([]) must return []."""
    with patch("pipeline.boundaries._get_model"):
        result = detect_boundaries([])
    assert result == []


def test_single_window_returns_that_window() -> None:
    """A single window must always be returned (it is windows[0])."""
    window = _make_window("only window", 0.0)
    with patch("pipeline.boundaries._get_model") as mock_get_model:
        result = detect_boundaries([window])
    assert result == [window], f"Expected [{window}], got {result}"
    # Too few windows to analyse → the model is never invoked.
    mock_get_model.assert_not_called()


def test_few_windows_all_returned_without_embedding() -> None:
    """Fewer than _MIN_WINDOWS_FOR_ANALYSIS windows are all kept as chapters."""
    n = _MIN_WINDOWS_FOR_ANALYSIS - 1
    windows = [_make_window(f"w{i}", float(i)) for i in range(n)]

    with patch("pipeline.boundaries._get_model") as mock_get_model:
        result = detect_boundaries(windows)

    assert result == windows, f"Expected all {n} windows, got {len(result)}"
    mock_get_model.assert_not_called()


def test_uniform_similarity_returns_only_first_window() -> None:
    """When all consecutive similarities are identical, std == 0 → only windows[0]."""
    n = 6
    windows = [_make_window(f"w{i}", float(i)) for i in range(n)]
    embeddings = np.ones((n, 4), dtype=float)
    embeddings /= np.linalg.norm(embeddings[0])

    mock_model = _make_mock_model(embeddings)

    with patch("pipeline.boundaries._get_model", return_value=mock_model):
        result = detect_boundaries(windows)

    assert result == [windows[0]], (
        f"Uniform similarity should yield only windows[0], got {len(result)} windows"
    )


def test_clear_topic_dip_is_flagged() -> None:
    """A single sharp similarity dip is detected as a boundary."""
    # High similarity everywhere except one clear dip at pair index 2.
    similarities = [0.8, 0.8, 0.1, 0.8, 0.8]
    n = len(similarities) + 1
    windows = [_make_window(f"w{i}", float(i * 60)) for i in range(n)]
    embeddings = _build_embeddings_for_similarities(similarities)

    mock_model = _make_mock_model(embeddings)

    with patch("pipeline.boundaries._get_model", return_value=mock_model):
        result = detect_boundaries(windows)

    result_ids = {id(w) for w in result}
    # windows[3] follows the dip (pair index 2 == similarity between w2 and w3).
    assert id(windows[3]) in result_ids, "The window after a clear dip must be a boundary"
    assert id(windows[0]) in result_ids, "windows[0] must always be included"
