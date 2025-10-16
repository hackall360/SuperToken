"""Pytest configuration for SuperToken."""

try:  # pragma: no cover - best effort import to keep real torch available during tests
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - optional dependency scenario
    pass
