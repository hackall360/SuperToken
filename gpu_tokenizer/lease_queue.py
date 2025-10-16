"""Lease management primitives for coordinating distributed workers.

This module defines the public surface area for managing work leases. Later
phases will provide concrete implementations; for now we document the expected
behaviour and raise ``NotImplementedError`` from the stub methods so callers
understand that the functionality is pending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LeaseState:
    """Placeholder structure describing the state associated with a lease."""

    lease_id: str
    payload: Dict[str, Any]
    owner: Optional[str] = None


class LeaseNotary:
    """Coordinate the lifecycle of work leases across distributed workers.

    Invariants (to be upheld by the future implementation):

    * At most one active lease is granted for a given ``lease_id`` at a time.
    * ``heartbeat`` calls extend the validity of the most recently granted lease
      for a worker; missing heartbeats results in the lease being eligible for
      ``requeue``.
    * ``serialize_state`` produces a deterministic snapshot of all outstanding
      leases suitable for persisting and later restoration.
    """

    def __init__(self) -> None:
        raise NotImplementedError

    def grant_lease(self, *, worker_id: str) -> LeaseState:
        """Return the next available lease for ``worker_id``."""

        raise NotImplementedError

    def requeue(self, lease: LeaseState) -> None:
        """Make ``lease`` available for future ``grant_lease`` calls."""

        raise NotImplementedError

    def heartbeat(self, *, lease: LeaseState, worker_id: str) -> None:
        """Refresh the lease owned by ``worker_id`` to avoid reassignment."""

        raise NotImplementedError

    def serialize_state(self) -> Dict[str, Any]:
        """Capture the durable state necessary to restore the queue."""

        raise NotImplementedError
