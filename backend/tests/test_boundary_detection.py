"""
Property-Based Test for Boundary Detection Threshold and First-Window Invariant (Property 10)

**Validates: Requirements 8.3, 8.4, 10.3**

Property 10: Topic Boundary Detection Threshold and First-Window Invariant
`detect_boundaries(windows)` includes window at index i > 0 iff
cosine_similarity(embed[i-1], embed[i]) < 0.35, always includes windows[0],
and result is non-empty for non-empty input.
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
from pipeline.boundaries import cosine_similarity, detect_boundaries, _SIMILARITY_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_window(text: str = "some text", start_time: float = 0.0) -> TextWindow:
    return TextWindow(text=text, start_time=start_time)


def _unit_vector(dim: int, index: int) -> np.ndarray:
    """Return a unit vector with a 1 at position `index` and 0s elsewhere."""
    v = np.zeros(dim, dtype=float)
    v[index] = 1.0
    return v


def _make_mock_model(embeddings: np.ndarray) -> MagicMock:
    """Return a mock SentenceTransformer whose encode() returns `embeddings`."""
    mock = MagicMock()
    mock.encode.return_value = embeddings
    return mock


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# A similarity value strictly below the threshold (boundary should be included)
below_threshold = st.floats(
    min_value=-1.0,
    max_value=_SIMILARITY_THRESHOLD - 1e-9,
    allow_nan=False,
    allow_infinity=False,
)

# A similarity value at or above the threshold (boundary should NOT be included)
above_or_equal_threshold = st.floats(
    min_value=_SIMILARITY_THRESHOLD,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)

# Strategy for a list of similarity values (one per consecutive pair)
# Each value is either below or above the threshold.
similarity_list_strategy = st.lists(
    st.one_of(below_threshold, above_or_equal_threshold),
    min_size=1,   # at least 1 pair → at least 2 windows
    max_size=20,
)

# Strategy for window count (1 to 21 windows)
window_count_strategy = st.integers(min_value=1, max_value=21)


def _build_embeddings_for_similarities(similarities: list[float], dim: int = 4) -> np.ndarray:
    """
    Build an embedding matrix such that cosine_similarity(embed[i], embed[i+1])
    equals similarities[i] for each i.

    Strategy: embed[0] = e_0 (unit vector along axis 0).
    For each subsequent embed[i+1], construct a vector in the plane spanned by
    e_0 and e_1 such that its cosine similarity with embed[i] equals similarities[i].

    Since we want exact control, we use a simpler approach:
    - embed[i] = e_i (orthogonal unit vectors) for i < dim
    - cosine_similarity(e_i, e_{i+1}) = 0.0 for all i

    For controlled similarity, we use 2D vectors:
    embed[0] = [1, 0, 0, ...]
    embed[i+1] = [cos(theta), sin(theta), 0, ...] where cos(theta) = similarities[i]
    But this only works for the first pair.

    Better approach: use pairs of vectors where each pair is independent.
    For pair (i, i+1):
      embed[i]   = [cos_i, sin_i, 0, ..., 0]  in a dedicated 2D subspace
      embed[i+1] = [1,     0,     0, ..., 0]  in the same subspace

    Actually the simplest correct approach:
    We build embeddings so that consecutive pairs have the desired similarity.
    Use 2D subspaces: pair i uses dimensions (2i, 2i+1).

    embed[0] = e_{0}  (unit vector in dim 0)
    For each i in range(len(similarities)):
        s = similarities[i]
        s = clamp(s, -1, 1)
        # embed[i+1] must have cosine_similarity with embed[i] == s
        # embed[i] is a unit vector in some direction
        # embed[i+1] = s * embed[i] + sqrt(1-s^2) * perp
        # where perp is a unit vector orthogonal to embed[i]
    """
    n = len(similarities) + 1  # number of windows
    # Use enough dimensions: each window gets its own 2D subspace
    actual_dim = max(dim, 2 * n)
    embeddings = np.zeros((n, actual_dim), dtype=float)

    # embed[0] = unit vector in dimension 0
    embeddings[0, 0] = 1.0

    for i, s in enumerate(similarities):
        # Clamp to valid cosine range
        s_clamped = float(np.clip(s, -1.0, 1.0))
        # embed[i] is already set; build embed[i+1] such that
        # cosine_similarity(embed[i], embed[i+1]) == s_clamped
        # embed[i+1] = s_clamped * embed[i] + sqrt(1 - s_clamped^2) * perp
        # perp = unit vector in a fresh dimension (2*(i+1)+1)
        perp_dim = 2 * (i + 1) + 1
        sin_theta = np.sqrt(max(0.0, 1.0 - s_clamped ** 2))
        embeddings[i + 1] = s_clamped * embeddings[i]
        embeddings[i + 1, perp_dim] += sin_theta

        # Normalize embed[i+1] to be a unit vector
        norm = np.linalg.norm(embeddings[i + 1])
        if norm > 0:
            embeddings[i + 1] /= norm

    return embeddings


# ---------------------------------------------------------------------------
# Property 10a: windows[0] is always included
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(n=window_count_strategy)
def test_first_window_always_included(n: int) -> None:
    """
    **Validates: Requirements 8.4**

    For any non-empty list of windows, detect_boundaries() always includes
    windows[0] as the first element of the result.
    """
    windows = [_make_window(f"window {i}", float(i * 10)) for i in range(n)]
    # Use orthogonal unit vectors → all similarities = 0.0 < 0.35 → all included
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
@given(n=window_count_strategy)
def test_result_non_empty_for_non_empty_input(n: int) -> None:
    """
    **Validates: Requirements 10.3**

    For any non-empty list of windows, detect_boundaries() returns a
    non-empty list.
    """
    windows = [_make_window(f"window {i}", float(i * 10)) for i in range(n)]
    # High similarity (0.9) → only windows[0] included, but result is still non-empty
    embeddings = np.ones((n, 4), dtype=float)
    # Normalize each row
    embeddings /= np.linalg.norm(embeddings[0])

    mock_model = _make_mock_model(embeddings)

    with patch("pipeline.boundaries._get_model", return_value=mock_model):
        result = detect_boundaries(windows)

    assert len(result) >= 1, (
        f"detect_boundaries() must return a non-empty list for {n} input windows, "
        f"but returned an empty list"
    )


# ---------------------------------------------------------------------------
# Property 10c: window at index i>0 included iff similarity < 0.35
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(similarities=similarity_list_strategy)
def test_boundary_threshold_iff_similarity_below_threshold(
    similarities: list[float],
) -> None:
    """
    **Validates: Requirements 8.3**

    For any sequence of windows, detect_boundaries() includes window[i] (i > 0)
    if and only if cosine_similarity(embed[i-1], embed[i]) < 0.35.

    We construct embeddings with exact controlled cosine similarities between
    consecutive pairs, then verify the inclusion/exclusion matches the threshold.
    """
    n = len(similarities) + 1  # number of windows
    windows = [_make_window(f"window {i}", float(i * 10)) for i in range(n)]

    # Build embeddings with the desired pairwise similarities
    embeddings = _build_embeddings_for_similarities(similarities)

    # Verify our embedding construction is correct (sanity check)
    for i, expected_sim in enumerate(similarities):
        actual_sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        # Allow small floating-point tolerance
        assert abs(actual_sim - float(np.clip(expected_sim, -1.0, 1.0))) < 1e-6, (
            f"Embedding construction error at pair {i}: "
            f"expected similarity {expected_sim:.6f}, got {actual_sim:.6f}"
        )

    mock_model = _make_mock_model(embeddings)

    with patch("pipeline.boundaries._get_model", return_value=mock_model):
        result = detect_boundaries(windows)

    result_set = set(id(w) for w in result)

    # windows[0] must always be included
    assert id(windows[0]) in result_set, "windows[0] must always be in the result"

    # For each i > 0, check inclusion iff similarity < threshold
    for i, sim in enumerate(similarities):
        actual_sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        window = windows[i + 1]
        should_include = actual_sim < _SIMILARITY_THRESHOLD

        if should_include:
            assert id(window) in result_set, (
                f"Window {i + 1} should be included (similarity={actual_sim:.6f} < "
                f"{_SIMILARITY_THRESHOLD}), but was not found in result"
            )
        else:
            assert id(window) not in result_set, (
                f"Window {i + 1} should NOT be included (similarity={actual_sim:.6f} >= "
                f"{_SIMILARITY_THRESHOLD}), but was found in result"
            )


# ---------------------------------------------------------------------------
# Property 10 combined: all three sub-properties together
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(similarities=similarity_list_strategy)
def test_boundary_detection_combined_property(similarities: list[float]) -> None:
    """
    **Validates: Requirements 8.3, 8.4, 10.3**

    Combined Property 10: for any sequence of windows with controlled embeddings,
    detect_boundaries() SHALL:
      (a) always include windows[0],
      (b) return a non-empty result for non-empty input,
      (c) include window[i] (i > 0) iff cosine_similarity(embed[i-1], embed[i]) < 0.35.
    """
    n = len(similarities) + 1
    windows = [_make_window(f"window {i}", float(i * 10)) for i in range(n)]
    embeddings = _build_embeddings_for_similarities(similarities)

    mock_model = _make_mock_model(embeddings)

    with patch("pipeline.boundaries._get_model", return_value=mock_model):
        result = detect_boundaries(windows)

    # (a) windows[0] always included
    assert result[0] is windows[0], "windows[0] must be the first element of the result"

    # (b) non-empty result
    assert len(result) >= 1, "Result must be non-empty for non-empty input"

    result_set = set(id(w) for w in result)

    # (c) threshold iff condition
    for i, sim in enumerate(similarities):
        actual_sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        window = windows[i + 1]
        should_include = actual_sim < _SIMILARITY_THRESHOLD

        if should_include:
            assert id(window) in result_set, (
                f"Window {i + 1} should be included (sim={actual_sim:.6f} < "
                f"{_SIMILARITY_THRESHOLD}), but was absent"
            )
        else:
            assert id(window) not in result_set, (
                f"Window {i + 1} should NOT be included (sim={actual_sim:.6f} >= "
                f"{_SIMILARITY_THRESHOLD}), but was present"
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
    embeddings = np.array([[1.0, 0.0, 0.0, 0.0]])

    mock_model = _make_mock_model(embeddings)

    with patch("pipeline.boundaries._get_model", return_value=mock_model):
        result = detect_boundaries([window])

    assert result == [window], f"Expected [{window}], got {result}"


def test_all_high_similarity_only_first_window_returned() -> None:
    """When all consecutive similarities are >= 0.35, only windows[0] is returned."""
    n = 5
    windows = [_make_window(f"w{i}", float(i)) for i in range(n)]
    # All identical embeddings → similarity = 1.0 >= 0.35
    embeddings = np.ones((n, 4), dtype=float)
    embeddings /= np.linalg.norm(embeddings[0])

    mock_model = _make_mock_model(embeddings)

    with patch("pipeline.boundaries._get_model", return_value=mock_model):
        result = detect_boundaries(windows)

    assert result == [windows[0]], (
        f"With all similarities >= 0.35, only windows[0] should be returned, "
        f"but got {len(result)} windows"
    )


def test_all_low_similarity_all_windows_returned() -> None:
    """When all consecutive similarities are < 0.35, all windows are returned."""
    n = 5
    windows = [_make_window(f"w{i}", float(i)) for i in range(n)]
    # Orthogonal unit vectors → similarity = 0.0 < 0.35
    embeddings = np.eye(n, dtype=float)

    mock_model = _make_mock_model(embeddings)

    with patch("pipeline.boundaries._get_model", return_value=mock_model):
        result = detect_boundaries(windows)

    assert result == windows, (
        f"With all similarities < 0.35, all {n} windows should be returned, "
        f"but got {len(result)}"
    )


def test_threshold_boundary_exact_value_not_included() -> None:
    """A window with similarity exactly equal to 0.35 should NOT be included."""
    # Two windows: similarity between them = exactly 0.35
    windows = [_make_window("w0", 0.0), _make_window("w1", 10.0)]
    # Construct embeddings with cosine similarity = 0.35
    theta = np.arccos(0.35)
    embeddings = np.array([
        [1.0, 0.0],
        [np.cos(theta), np.sin(theta)],
    ])

    mock_model = _make_mock_model(embeddings)

    with patch("pipeline.boundaries._get_model", return_value=mock_model):
        result = detect_boundaries(windows)

    # Verify the actual similarity is 0.35
    actual_sim = cosine_similarity(embeddings[0], embeddings[1])
    assert abs(actual_sim - 0.35) < 1e-9, f"Expected similarity 0.35, got {actual_sim}"

    # windows[1] should NOT be included (similarity == 0.35, not strictly < 0.35)
    assert result == [windows[0]], (
        f"Window with similarity exactly 0.35 should NOT be included, "
        f"but result was {result}"
    )


def test_threshold_just_below_is_included() -> None:
    """A window with similarity just below 0.35 should be included."""
    windows = [_make_window("w0", 0.0), _make_window("w1", 10.0)]
    # Construct embeddings with cosine similarity = 0.35 - epsilon
    target_sim = 0.35 - 1e-7
    theta = np.arccos(target_sim)
    embeddings = np.array([
        [1.0, 0.0],
        [np.cos(theta), np.sin(theta)],
    ])

    mock_model = _make_mock_model(embeddings)

    with patch("pipeline.boundaries._get_model", return_value=mock_model):
        result = detect_boundaries(windows)

    actual_sim = cosine_similarity(embeddings[0], embeddings[1])
    assert actual_sim < _SIMILARITY_THRESHOLD, (
        f"Expected similarity < 0.35, got {actual_sim}"
    )

    assert result == windows, (
        f"Window with similarity just below 0.35 should be included, "
        f"but result was {result}"
    )
