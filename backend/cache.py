# cache.py — In-memory cache keyed by video_id
# Simple dict — no TTL, lives for the lifetime of the process

from __future__ import annotations

from models import Timestamp

_cache: dict[str, list[Timestamp]] = {}


def get(video_id: str) -> list[Timestamp] | None:
    """Return the cached timestamp list for *video_id*, or None if not present."""
    return _cache.get(video_id)


def set(video_id: str, timestamps: list[Timestamp]) -> None:
    """Store *timestamps* in the cache under *video_id*."""
    _cache[video_id] = timestamps
