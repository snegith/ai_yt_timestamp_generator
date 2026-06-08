"""Concurrent /generate requests for the same video_id share one pipeline run."""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import patch

import pytest

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


def test_concurrent_requests_run_pipeline_once():
    timestamps = [Timestamp(time="0:00", title="Intro")]

    pipeline_calls = 0

    async def slow_pipeline(video_id: str):
        nonlocal pipeline_calls
        pipeline_calls += 1
        await asyncio.sleep(0.05)
        return timestamps

    async def run_requests():
        with patch("main._run_pipeline", side_effect=slow_pipeline):
            from httpx import ASGITransport, AsyncClient

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                return await asyncio.gather(
                    client.post("/generate", json={"video_id": "sameVideo1X"}),
                    client.post("/generate", json={"video_id": "sameVideo1X"}),
                )

    results = asyncio.run(run_requests())

    assert pipeline_calls == 1
    for response in results:
        assert response.status_code == 200
        assert response.json()["timestamps"][0]["title"] == "Intro"
