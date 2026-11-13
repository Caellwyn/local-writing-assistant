"""Shared test setup: no test talks to a real Ollama about how to run a model."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import runtime  # noqa: E402


@pytest.fixture(autouse=True)
def _run_models_as_named(request, monkeypatch):
    """Requests go to the named model with Ollama's own placement (runtime.py has its own tests)."""
    if request.node.get_closest_marker("real_runtime"):
        return
    monkeypatch.setattr(runtime, "resolve", lambda tag, num_ctx, force_ok=True: (tag, {}))
    monkeypatch.setattr(runtime, "confirm", lambda run_tag, options: True)


def pytest_configure(config):
    config.addinivalue_line("markers", "real_runtime: use the real runtime.resolve/confirm")
