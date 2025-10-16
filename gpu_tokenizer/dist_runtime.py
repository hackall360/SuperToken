"""Distributed runtime configuration and helpers for GPU tokenization workers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Iterable, List, Optional

# Configure a module level logger. Downstream applications can adjust the
# configuration to taste but this ensures we at least emit something when the
# module is used in isolation.
logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:  # pragma: no cover - configuration guard
    logging.basicConfig(level=logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class DeviceRequest:
    """Describe a type of accelerator device that should be provisioned."""

    kind: str
    count: int
    capability: Optional[str] = None

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        if self.count <= 0:
            raise ValueError("DeviceRequest.count must be positive")
        if not self.kind:
            raise ValueError("DeviceRequest.kind must be non-empty")


@dataclass(frozen=True)
class ChunkTarget:
    """Declare how many samples each worker should accumulate per chunk."""

    tokens: int
    sequences: int

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        if self.tokens <= 0:
            raise ValueError("ChunkTarget.tokens must be positive")
        if self.sequences <= 0:
            raise ValueError("ChunkTarget.sequences must be positive")


@dataclass(frozen=True)
class LaunchConfig:
    """Aggregate the full configuration for launching distributed workers."""

    devices: List[DeviceRequest] = field(default_factory=list)
    chunk_target: Optional[ChunkTarget] = None
    rebalance_cadence: Optional[timedelta] = None
    environment: dict[str, str] = field(default_factory=dict)

    def iter_env(self) -> Iterable[tuple[str, str]]:
        """Iterate over environment variables with defaults applied."""

        for key, value in self.environment.items():
            yield key, value


def get_env_flag(name: str, default: bool = False) -> bool:
    """Return a boolean from an environment variable."""

    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    return raw in {"1", "true", "yes", "on"}


def get_env_int(name: str, default: int) -> int:
    """Return an integer from an environment variable, with a default."""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Environment variable {name!r} must be an integer") from exc


def launch_workers(config: LaunchConfig) -> None:
    """Spawn worker processes as described by *config*.

    Future iterations of the distributed runtime will implement orchestration
    logic here. For now this placeholder makes the API explicit while clearly
    signalling that the functionality is not yet implemented.
    """

    raise NotImplementedError("Distributed worker launch is not implemented yet")


def worker_main() -> None:
    """Entry point for a tokenizer worker process."""

    raise NotImplementedError("Distributed worker execution is not implemented yet")
