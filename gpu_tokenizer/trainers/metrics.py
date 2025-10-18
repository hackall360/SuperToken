"""Shared trainer metrics utilities."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Mapping


@dataclass
class TrainerMetricsEWMA:
    """Maintain exponential moving averages for trainer throughput metrics."""

    alpha: float = 0.2
    window_size: int = 16
    enabled: bool = False
    overlap_enabled: bool = True
    _tokens_per_s: float | None = None
    _lease_per_s: float | None = None
    _stage_windows: dict[str, deque[float]] = field(default_factory=dict)
    rank: int | None = None
    _copy_window: deque[float] = field(init=False, repr=False)
    _compute_window: deque[float] = field(init=False, repr=False)
    _iteration_window: deque[float] = field(init=False, repr=False)
    _reduction_window: deque[float] = field(init=False, repr=False)
    _reduction_share: float | None = None
    _reduction_share_latest: float | None = None
    _rank_stats: dict[int, dict[str, float]] = field(init=False, repr=False)

    _COPY_STAGES = frozenset({"h2d", "d2h"})
    _COMPUTE_STAGES = frozenset({"kernel", "reduction"})
    _REDUCTION_GROW_THRESHOLD = 0.15
    _REDUCTION_SHRINK_THRESHOLD = 0.08

    def __post_init__(self) -> None:
        # Clamp configuration to sensible defaults while tolerating bad inputs.
        try:
            alpha = float(self.alpha)
        except (TypeError, ValueError):
            alpha = 0.2
        if alpha <= 0.0:
            alpha = 0.01
        if alpha > 1.0:
            alpha = 1.0
        self.alpha = alpha
        try:
            window = int(self.window_size)
        except (TypeError, ValueError):
            window = 16
        if window <= 0:
            window = 1
        self.window_size = window
        self.enabled = bool(self.enabled)
        self.overlap_enabled = bool(self.overlap_enabled)
        self._copy_window = deque(maxlen=self.window_size)
        self._compute_window = deque(maxlen=self.window_size)
        self._iteration_window = deque(maxlen=self.window_size)
        self._reduction_window = deque(maxlen=self.window_size)
        self._reduction_share = None
        self._reduction_share_latest = None
        self._rank_stats = {}

    def reset(self) -> None:
        """Clear accumulated metrics."""

        self._tokens_per_s = None
        self._lease_per_s = None
        self._stage_windows.clear()
        self._copy_window = deque(maxlen=self.window_size)
        self._compute_window = deque(maxlen=self.window_size)
        self._iteration_window = deque(maxlen=self.window_size)
        self._reduction_window = deque(maxlen=self.window_size)
        self._reduction_share = None
        self._reduction_share_latest = None
        self._rank_stats = {}

    @property
    def tokens_per_s(self) -> float | None:
        return self._tokens_per_s

    @property
    def lease_per_s(self) -> float | None:
        return self._lease_per_s

    @property
    def reduction_overhead(self) -> float | None:
        """Return the EWMA of reduction overhead, if available."""

        return self._reduction_share

    def record_stage(self, stage: str, duration_s: float) -> str:
        if not self.enabled or duration_s < 0:
            return "other"
        window = self._stage_windows.setdefault(stage, deque(maxlen=self.window_size))
        window.append(float(duration_s))
        stage_key = stage.lower()
        kind = "other"
        if stage_key in self._COPY_STAGES:
            self._copy_window.append(float(duration_s))
            kind = "copy"
        elif stage_key in self._COMPUTE_STAGES:
            self._compute_window.append(float(duration_s))
            kind = "compute"
        return kind

    def set_rank(self, rank: int | None) -> None:
        """Declare the distributed rank associated with this metrics tracker."""

        if rank is None:
            self.rank = None
        else:
            self.rank = int(rank)

    def _update_rank_entry(
        self,
        rank: int,
        *,
        tokens_per_s: float | None,
        lease_per_s: float | None,
        samples: float = 0.0,
        extra: Mapping[str, float] | None = None,
    ) -> None:
        entry = self._rank_stats.setdefault(
            int(rank),
            {"tokens_per_s": None, "lease_per_s": None, "samples": 0.0},
        )
        if tokens_per_s is not None and math.isfinite(tokens_per_s):
            entry["tokens_per_s"] = float(tokens_per_s)
        if lease_per_s is not None and math.isfinite(lease_per_s):
            entry["lease_per_s"] = float(lease_per_s)
        if samples > 0.0 and math.isfinite(samples):
            entry["samples"] = float(entry.get("samples", 0.0)) + float(samples)
        if extra:
            for key, value in extra.items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric):
                    continue
                entry[key] = numeric

    def update_rank_snapshot(self, snapshot: Mapping[str, object]) -> None:
        """Merge a per-rank snapshot gathered from a peer."""

        try:
            rank = int(snapshot.get("rank"))  # type: ignore[arg-type]
        except Exception:
            return
        tokens_obj = snapshot.get("tokens_per_s")
        leases_obj = snapshot.get("lease_per_s")
        samples_obj = snapshot.get("samples", 0.0)
        idle_obj = snapshot.get("idle_ms") or snapshot.get("idle_ewma_ms")
        width_obj = snapshot.get("lease_width")
        max_active_obj = snapshot.get("max_active")
        try:
            tokens_val = float(tokens_obj) if tokens_obj is not None else None
        except (TypeError, ValueError):
            tokens_val = None
        try:
            leases_val = float(leases_obj) if leases_obj is not None else None
        except (TypeError, ValueError):
            leases_val = None
        try:
            samples_val = float(samples_obj) if samples_obj is not None else 0.0
        except (TypeError, ValueError):
            samples_val = 0.0
        extras: dict[str, float] = {}
        try:
            if idle_obj is not None:
                extras["idle_ms"] = float(idle_obj)
        except (TypeError, ValueError):
            pass
        try:
            if width_obj is not None:
                extras["lease_width"] = float(width_obj)
        except (TypeError, ValueError):
            pass
        try:
            if max_active_obj is not None:
                extras["max_active"] = float(max_active_obj)
        except (TypeError, ValueError):
            pass
        self._update_rank_entry(
            rank,
            tokens_per_s=tokens_val,
            lease_per_s=leases_val,
            samples=samples_val,
            extra=extras if extras else None,
        )

    def record_tokens(
        self,
        tokens: int,
        duration_s: float,
        *,
        leases: int | None = None,
        rank: int | None = None,
    ) -> None:
        if not self.enabled or duration_s <= 0:
            return
        rate = float(tokens) / float(duration_s) if tokens > 0 else 0.0
        if self._tokens_per_s is None:
            self._tokens_per_s = rate
        else:
            self._tokens_per_s = (self.alpha * rate) + ((1.0 - self.alpha) * self._tokens_per_s)
        if leases is not None:
            lease_rate = float(leases) / float(duration_s) if leases > 0 else 0.0
            if self._lease_per_s is None:
                self._lease_per_s = lease_rate
            else:
                self._lease_per_s = (
                    self.alpha * lease_rate
                    + (1.0 - self.alpha) * self._lease_per_s
                )
        rank_id = self.rank if self.rank is not None else rank
        if rank_id is not None:
            self._update_rank_entry(
                int(rank_id),
                tokens_per_s=self._tokens_per_s,
                lease_per_s=self._lease_per_s,
                samples=1.0,
            )

    def record_feedback(
        self,
        *,
        rank: int | None = None,
        idle_ms: float | None = None,
        lease_width: float | None = None,
        max_active: float | None = None,
    ) -> None:
        """Track supplemental lease telemetry for ``rank``."""

        extras: dict[str, float] = {}
        if idle_ms is not None:
            try:
                value = float(idle_ms)
            except (TypeError, ValueError):
                value = None
            if value is not None and math.isfinite(value):
                extras["idle_ms"] = value
        if lease_width is not None:
            try:
                width_val = float(lease_width)
            except (TypeError, ValueError):
                width_val = None
            if width_val is not None and math.isfinite(width_val):
                extras["lease_width"] = width_val
        if max_active is not None:
            try:
                max_val = float(max_active)
            except (TypeError, ValueError):
                max_val = None
            if max_val is not None and math.isfinite(max_val):
                extras["max_active"] = max_val
        if not extras:
            return
        target_rank = rank if rank is not None else self.rank
        if target_rank is None:
            return
        self._update_rank_entry(
            int(target_rank),
            tokens_per_s=None,
            lease_per_s=None,
            samples=0.0,
            extra=extras,
        )

    def record_iteration(self, total_duration_s: float, reduction_s: float) -> None:
        """Track the wall clock time spent on reductions versus the iteration total."""

        if not self.enabled:
            return

        total = float(total_duration_s)
        reduction = float(reduction_s)
        if total <= 0.0:
            return

        if reduction < 0.0:
            reduction = 0.0
        if reduction > total:
            reduction = total

        share = reduction / total if total > 0.0 else 0.0
        self._iteration_window.append(total)
        self._reduction_window.append(reduction)
        self._reduction_share_latest = share

        if self._reduction_share is None:
            self._reduction_share = share
        else:
            self._reduction_share = (self.alpha * share) + (
                (1.0 - self.alpha) * self._reduction_share
            )

    def recommend_reduction_cadence(
        self,
        current: int,
        *,
        min_cadence: int,
        max_cadence: int,
        share_override: float | None = None,
    ) -> int:
        """Return an updated cadence based on observed reduction overhead."""

        if not self.enabled:
            return current

        share = self._reduction_share if share_override is None else float(share_override)
        if share is None:
            return max(min_cadence, min(current, max_cadence))

        clamped = max(min_cadence, min(current, max_cadence))

        if len(self._iteration_window) == 0:
            return clamped

        if share > self._REDUCTION_GROW_THRESHOLD and clamped < max_cadence:
            return min(max_cadence, clamped + 1)

        if share < self._REDUCTION_SHRINK_THRESHOLD and clamped > min_cadence:
            return max(min_cadence, clamped - 1)

        return clamped

    def summaries(self) -> dict[str, object]:
        stage_summary: dict[str, dict[str, object]] = {}
        for name, window in self._stage_windows.items():
            samples = list(window)
            count = len(samples)
            avg = sum(samples) / count if count else 0.0
            stage_summary[name] = {
                "samples": count,
                "avg_s": avg,
                "latest_s": samples[-1] if samples else 0.0,
                "window": samples,
            }

        def _window_stats(window: deque[float]) -> dict[str, object]:
            values = list(window)
            count = len(values)
            avg = sum(values) / count if count else 0.0
            latest = values[-1] if values else 0.0
            return {
                "samples": count,
                "avg_s": avg,
                "latest_s": latest,
                "window": values,
            }

        copy_stats = _window_stats(self._copy_window)
        compute_stats = _window_stats(self._compute_window)
        reduction_total = list(self._iteration_window)
        reduction_values = list(self._reduction_window)
        reduction_samples = len(reduction_total)
        reduction_avg_total = sum(reduction_total) / reduction_samples if reduction_samples else 0.0
        reduction_avg = sum(reduction_values) / reduction_samples if reduction_samples else 0.0

        return {
            "enabled": self.enabled,
            "overlap_enabled": self.overlap_enabled,
            "tokens_per_s": self._tokens_per_s,
            "lease_per_s": self._lease_per_s,
            "stages": stage_summary,
            "copy": copy_stats,
            "compute": compute_stats,
            "reduction": {
                "samples": reduction_samples,
                "avg_total_s": reduction_avg_total,
                "avg_reduction_s": reduction_avg,
                "latest_total_s": reduction_total[-1] if reduction_total else 0.0,
                "latest_reduction_s": reduction_values[-1] if reduction_values else 0.0,
                "share_latest": self._reduction_share_latest,
                "share_ewma": self._reduction_share,
            },
        }

    def snapshot(self) -> dict[str, object]:
        """Return an instantaneous view of the throughput metrics."""

        rank_stats = {rank: dict(stats) for rank, stats in self._rank_stats.items()}
        local_samples = 0.0
        if self.rank is not None:
            entry = rank_stats.get(int(self.rank))
            if entry is not None:
                try:
                    local_samples = float(entry.get("samples", 0.0))
                except (TypeError, ValueError):
                    local_samples = 0.0
        snapshot = {
            "rank": self.rank,
            "tokens_per_s": self._tokens_per_s,
            "lease_per_s": self._lease_per_s,
            "samples": local_samples,
            "per_rank": rank_stats,
        }
        return snapshot


__all__ = ["TrainerMetricsEWMA"]
