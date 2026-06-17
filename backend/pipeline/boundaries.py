"""pipeline/boundaries.py — Embedding, cosine similarity, topic boundary detection."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.preprocess import TextWindow

# ---------------------------------------------------------------------------
# Model — loaded once at module import time to avoid repeated disk I/O.
# ---------------------------------------------------------------------------

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer instance, loading it on first call."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D numpy arrays.

    Returns a value in the closed interval [-1.0, 1.0].
    If either vector has zero norm the function returns 0.0 to avoid
    division by zero.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    raw = float(np.dot(a, b) / (norm_a * norm_b))
    # Clamp to [-1.0, 1.0] to guard against floating-point rounding errors
    # (e.g. identical vectors can yield 1.0000000000000002).
    return max(-1.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Boundary detection
# ---------------------------------------------------------------------------

# How many standard deviations below the mean a consecutive-window similarity
# must fall to count as a topic boundary. A purely *absolute* threshold (the
# old 0.35 value) does not work in practice: consecutive ~200-word windows from
# the same video share enough vocabulary that all-MiniLM-L6-v2 almost always
# scores them well above 0.35, so virtually nothing was ever flagged as a
# boundary and most videos collapsed to 1-2 timestamps. An *adaptive* cutoff
# computed from each video's own similarity distribution flags the relative
# dips (the real topic changes) regardless of the absolute baseline.
_STD_FACTOR = 0.5

# Videos with very few windows are too short to analyse meaningfully — treat
# every window as its own chapter rather than collapsing to a single one.
_MIN_WINDOWS_FOR_ANALYSIS = 4


def detect_boundaries(windows: list[TextWindow]) -> list[TextWindow]:
    """Detect topic boundaries in a sequence of text windows.

    Algorithm
    ---------
    1. Encode every window's text with ``all-MiniLM-L6-v2``.
    2. For each consecutive pair ``(i-1, i)``, compute cosine similarity.
    3. Compute an *adaptive* cutoff from the per-video similarity distribution:
       ``cutoff = mean(similarities) - _STD_FACTOR * std(similarities)``.
    4. Mark window ``i`` as a boundary when its similarity to the previous
       window falls below that cutoff (i.e. a relative dip = topic change).
    5. Always include ``windows[0]`` regardless of similarity.

    Using an adaptive cutoff (instead of a fixed absolute threshold) means the
    detector flags the genuinely low-similarity transitions for *this* video,
    which produces a sensible number of chapters across videos with very
    different baseline similarity levels.

    Parameters
    ----------
    windows:
        Ordered list of :class:`~pipeline.preprocess.TextWindow` objects
        produced by :func:`~pipeline.preprocess.preprocess`.

    Returns
    -------
    list[TextWindow]
        Non-empty list of boundary windows (for non-empty input).
        ``windows[0]`` is always the first element.
    """
    if not windows:
        return []

    # Too few windows to compute a meaningful distribution — keep them all.
    if len(windows) < _MIN_WINDOWS_FOR_ANALYSIS:
        return list(windows)

    model = _get_model()

    # Encode all window texts in a single batched call for efficiency.
    texts = [w.text for w in windows]
    embeddings: np.ndarray = model.encode(texts, convert_to_numpy=True)

    # Cosine similarity between each consecutive pair (length == len(windows) - 1).
    similarities = np.array(
        [
            cosine_similarity(embeddings[i - 1], embeddings[i])
            for i in range(1, len(windows))
        ]
    )

    cutoff = float(similarities.mean()) - _STD_FACTOR * float(similarities.std())

    boundary_windows: list[TextWindow] = [windows[0]]

    for i in range(1, len(windows)):
        if similarities[i - 1] < cutoff:
            boundary_windows.append(windows[i])

    return boundary_windows
