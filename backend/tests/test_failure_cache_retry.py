"""Tests for failure-cache bypass when the client sends force_retry."""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("GROQ_API_KEY", "test-key")

import cache
from main import app, _pipeline_locks
from models import Timestamp


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


def test_failure_cache_blocks_retry_by_default(client: TestClient) -> None:
    cache.set_failure("sameVideo1X", "previous pipeline error")

    with patch("main._run_pipeline", new=AsyncMock()) as mock_pipeline:
        response = client.post("/generate", json={"video_id": "sameVideo1X"})

    assert response.status_code == 502
    assert response.json()["detail"] == "previous pipeline error"
    mock_pipeline.assert_not_called()


def test_force_retry_bypasses_failure_cache(client: TestClient) -> None:
    cache.set_failure("sameVideo1X", "previous pipeline error")
    timestamps = [Timestamp(time="0:00", title="Intro")]

    with patch("main._run_pipeline", new=AsyncMock(return_value=timestamps)) as mock_pipeline:
        response = client.post(
            "/generate",
            json={"video_id": "sameVideo1X", "force_retry": True},
        )

    assert response.status_code == 200
    assert response.json()["timestamps"][0]["title"] == "Intro"
    mock_pipeline.assert_called_once_with("sameVideo1X")
    assert cache.get_failure("sameVideo1X") is None


def test_force_retry_still_returns_success_cache(client: TestClient) -> None:
    timestamps = [Timestamp(time="0:00", title="Cached")]
    cache.set("sameVideo1X", timestamps)

    with patch("main._run_pipeline", new=AsyncMock()) as mock_pipeline:
        response = client.post(
            "/generate",
            json={"video_id": "sameVideo1X", "force_retry": True},
        )

    assert response.status_code == 200
    assert response.json()["timestamps"][0]["title"] == "Cached"
    mock_pipeline.assert_not_called()
