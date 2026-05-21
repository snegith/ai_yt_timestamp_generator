"""
Property-Based Test for Cosine Similarity Symmetry and Bounds (Property 9)

**Validates: Requirements 8.2**

Property 9: Cosine Similarity Symmetry and Bounds
For any two non-zero vectors, `cosine_similarity(a, b) == cosine_similarity(b, a)`
and the result is always in the closed interval [-1.0, 1.0].
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pipeline.boundaries import cosine_similarity


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# A non-zero 1-D float vector of length 1–128.
# We generate finite floats and then filter out the all-zero case.
_vector_strategy = st.integers(min_value=1, max_value=128).flatmap(
    lambda n: arrays(
        dtype=np.float64,
        shape=(n,),
        elements=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
).filter(lambda v: np.linalg.norm(v) > 0.0)


# A pair of non-zero vectors with the same length.
_vector_pair_strategy = st.integers(min_value=1, max_value=128).flatmap(
    lambda n: st.tuples(
        arrays(
            dtype=np.float64,
            shape=(n,),
            elements=st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
            ),
        ),
        arrays(
            dtype=np.float64,
            shape=(n,),
            elements=st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
            ),
        ),
    ).filter(
        lambda pair: np.linalg.norm(pair[0]) > 0.0 and np.linalg.norm(pair[1]) > 0.0
    )
)


# ---------------------------------------------------------------------------
# Property 9a: Symmetry — cosine_similarity(a, b) == cosine_similarity(b, a)
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(pair=_vector_pair_strategy)
def test_cosine_similarity_is_symmetric(pair: tuple[np.ndarray, np.ndarray]) -> None:
    """
    **Validates: Requirements 8.2**

    For any two non-zero vectors a and b,
    cosine_similarity(a, b) SHALL equal cosine_similarity(b, a).
    """
    a, b = pair
    result_ab = cosine_similarity(a, b)
    result_ba = cosine_similarity(b, a)

    assert result_ab == result_ba, (
        f"Symmetry violated: cosine_similarity(a, b)={result_ab} != "
        f"cosine_similarity(b, a)={result_ba}\n"
        f"  a = {a}\n"
        f"  b = {b}"
    )


# ---------------------------------------------------------------------------
# Property 9b: Bounds — result is always in [-1.0, 1.0]
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(pair=_vector_pair_strategy)
def test_cosine_similarity_is_within_bounds(
    pair: tuple[np.ndarray, np.ndarray],
) -> None:
    """
    **Validates: Requirements 8.2**

    For any two non-zero vectors, cosine_similarity(a, b) SHALL be in
    the closed interval [-1.0, 1.0].
    """
    a, b = pair
    result = cosine_similarity(a, b)

    assert -1.0 <= result <= 1.0, (
        f"Result {result} is outside [-1.0, 1.0].\n"
        f"  a = {a}\n"
        f"  b = {b}"
    )


# ---------------------------------------------------------------------------
# Property 9c: Combined — symmetry AND bounds for the same inputs
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(pair=_vector_pair_strategy)
def test_cosine_similarity_symmetry_and_bounds_combined(
    pair: tuple[np.ndarray, np.ndarray],
) -> None:
    """
    **Validates: Requirements 8.2**

    Combined property: for any two non-zero vectors a and b,
      (a) cosine_similarity(a, b) == cosine_similarity(b, a)  [symmetry], and
      (b) the result is in [-1.0, 1.0]                        [bounds].

    This is the canonical statement of Property 9.
    """
    a, b = pair
    result_ab = cosine_similarity(a, b)
    result_ba = cosine_similarity(b, a)

    # --- symmetry ---
    assert result_ab == result_ba, (
        f"Symmetry violated: cosine_similarity(a, b)={result_ab} != "
        f"cosine_similarity(b, a)={result_ba}"
    )

    # --- bounds ---
    assert -1.0 <= result_ab <= 1.0, (
        f"Result {result_ab} is outside [-1.0, 1.0]."
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical_vectors_returns_one() -> None:
    """
    **Validates: Requirements 8.2**

    The cosine similarity of a vector with itself SHALL be 1.0.
    """
    v = np.array([1.0, 2.0, 3.0])
    result = cosine_similarity(v, v)
    assert abs(result - 1.0) < 1e-9, f"Expected 1.0, got {result}"


def test_cosine_similarity_opposite_vectors_returns_minus_one() -> None:
    """
    **Validates: Requirements 8.2**

    The cosine similarity of a vector and its negation SHALL be -1.0.
    """
    v = np.array([1.0, 2.0, 3.0])
    result = cosine_similarity(v, -v)
    assert abs(result - (-1.0)) < 1e-9, f"Expected -1.0, got {result}"


def test_cosine_similarity_orthogonal_vectors_returns_zero() -> None:
    """
    **Validates: Requirements 8.2**

    The cosine similarity of two orthogonal vectors SHALL be 0.0.
    """
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    result = cosine_similarity(a, b)
    assert abs(result) < 1e-9, f"Expected 0.0, got {result}"


def test_cosine_similarity_zero_vector_returns_zero() -> None:
    """
    **Validates: Requirements 8.2**

    When either vector is the zero vector, cosine_similarity SHALL return 0.0
    (guarded division-by-zero case).
    """
    zero = np.array([0.0, 0.0, 0.0])
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(zero, v) == 0.0
    assert cosine_similarity(v, zero) == 0.0
    assert cosine_similarity(zero, zero) == 0.0
