# cache.py — In-memory cache keyed by video_id
# Success entries use LRU eviction; failures expire after a TTL.

from __future__ import annotations

import time
from collections import OrderedDict

from models import Timestamp

_cache: OrderedDict[str, list[Timestamp]] = OrderedDict()
_failure_cache: dict[str, tuple[str, float]] = {}  # video_id -> (error_message, expires_at)

# Seconds to suppress retries after a pipeline failure (limits retry storms).
FAILURE_TTL_SECONDS = 60

# Maximum successful cache entries before LRU eviction.
MAX_CACHE_SIZE = 500


def _evict_expired_failures() -> None:
    now = time.monotonic()
    expired = [vid for vid, (_, exp) in _failure_cache.items() if now >= exp]
    for vid in expired:
        _failure_cache.pop(vid, None)


def get(video_id: str) -> list[Timestamp] | None:
    """Return the cached timestamp list for *video_id*, or None if not present."""
    entry = _cache.get(video_id)
    if entry is not None:
        _cache.move_to_end(video_id)
    return entry


def set(video_id: str, timestamps: list[Timestamp]) -> None:
    """Store *timestamps* in the cache under *video_id*."""
    _cache[video_id] = timestamps
    _cache.move_to_end(video_id)
    _failure_cache.pop(video_id, None)

    evicted: list[str] = []
    while len(_cache) > MAX_CACHE_SIZE:
        evicted.append(_cache.popitem(last=False)[0])
    if evicted:
        _evict_expired_failures()


def clear_failure(video_id: str) -> None:
    """Remove a cached failure for *video_id* so the pipeline can run again."""
    _failure_cache.pop(video_id, None)


def get_failure(video_id: str) -> str | None:
    """Return a cached failure message if still within TTL, else None."""
    _evict_expired_failures()
    entry = _failure_cache.get(video_id)
    if entry is None:
        return None
    message, expires_at = entry
    if time.monotonic() >= expires_at:
        _failure_cache.pop(video_id, None)
        return None
    return message


def set_failure(
    video_id: str,
    message: str,
    ttl_seconds: float = FAILURE_TTL_SECONDS,
) -> None:
    """Cache a pipeline failure so repeated requests fail fast for *ttl_seconds*."""
    _evict_expired_failures()
    _failure_cache[video_id] = (message, time.monotonic() + ttl_seconds)
