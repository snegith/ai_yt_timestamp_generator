"""Tests for per-video pipeline lock lifecycle and pruning."""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("GROQ_API_KEY", "test-key")

import cache
from main import _MAX_PIPELINE_LOCKS, _pipeline_locks, _prune_pipeline_locks, app


@pytest.fixture(autouse=True)
def reset_state():
    cache._cache.clear()
    cache._failure_cache.clear()
    _pipeline_locks.clear()
    yield
    cache._cache.clear()
    cache._failure_cache.clear()
    _pipeline_locks.clear()


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_pipeline_lock_released_after_failure(client: TestClient) -> None:
    with patch("main._run_pipeline", new=AsyncMock(side_effect=RuntimeError("boom"))):
        response = client.post("/generate", json={"video_id": "sameVideo1X"})

    assert response.status_code == 502
    assert "sameVideo1X" not in _pipeline_locks


def test_prune_pipeline_locks_removes_unlocked_entries() -> None:
    for i in range(_MAX_PIPELINE_LOCKS + 10):
        _pipeline_locks[f"vid{i:04d}"] = asyncio.Lock()

    _prune_pipeline_locks()

    assert len(_pipeline_locks) == 0
