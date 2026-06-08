# conftest.py — ensures the backend root is on sys.path for all pytest runs
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))


def run_async(coro):
    """Run a coroutine in a fresh event loop (safe after asyncio.run in other tests)."""
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _restore_event_loop_after_test():
    """Legacy tests use get_event_loop(); asyncio.run() in one test closes it for others."""
    yield
    asyncio.set_event_loop(asyncio.new_event_loop())
