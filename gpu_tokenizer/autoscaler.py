"""Adaptive autoscaler for GPU BPE training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - fallback when psutil missing
    psutil = None

import torch


@dataclass
class ScaleState:
    batch_size: int
    cpu_workers: int
    h2d_mb: int


class AutoScaler:
    """Simple autoscaler targeting configurable resource utilization."""

    def __init__(
        self,
        target_util: float = 0.80,
        min_bs: int = 256,
        max_bs: int = 8192,
        min_workers: int = 2,
        max_workers: Optional[int] = None,
        init_h2d_mb: int = 512,
        device: Optional[str] = None,
    ) -> None:
        self.tu = float(max(0.1, min(target_util, 0.95)))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_bs, self.max_bs = min_bs, max_bs
        self.min_workers = min_workers
        self.max_workers = max_workers or max(min_workers, os.cpu_count() or 4)
        self.state: Optional[ScaleState] = None
        self._h2d_mb = init_h2d_mb

    def _gpu_caps(self) -> tuple[int, int]:
        if self.device == "cpu" or not torch.cuda.is_available():
            return 0, 0
        free, total = torch.cuda.mem_get_info()
        return int(free), int(total)

    def _cpu_caps(self) -> tuple[int, int, int, float]:
        cores = os.cpu_count() or 4
        mem_total = 0
        mem_free = 0
        if psutil is not None:
            vm = psutil.virtual_memory()
            mem_total = int(vm.total)
            mem_free = int(vm.available)
            cpu_util = psutil.cpu_percent(interval=0.05)
        else:
            cpu_util = 0.0
        return cores, mem_free, mem_total, cpu_util

    def suggest(self, token_bytes_per_example: int = 2048) -> ScaleState:
        free, total = self._gpu_caps()
        cores, mem_free, _mem_total, _cpu_util = self._cpu_caps()
        gpu_budget = int(total * self.tu)
        gpu_free_budget = max(0, min(free, gpu_budget))
        bytes_per_ex = int(token_bytes_per_example * 1.2) or 1024
        bs_gpu = max(self.min_bs, min(self.max_bs, max(1, gpu_free_budget // bytes_per_ex)))
        workers = max(self.min_workers, int(cores * self.tu))
        workers = min(workers, self.max_workers)
        h2d_mb = max(256, min(8192, int((mem_free * self.tu) / (1024 * 1024 * 8))))
        if self.state is None:
            self._h2d_mb = h2d_mb or self._h2d_mb
        else:
            h2d_mb = self._h2d_mb
        self.state = ScaleState(batch_size=bs_gpu, cpu_workers=workers, h2d_mb=h2d_mb)
        return self.state

    def feedback(self, step_time_s: float | None = None, oom: bool = False) -> None:
        if self.state is None:
            return
        if oom:
            self.state = ScaleState(
                batch_size=max(self.min_bs, self.state.batch_size // 2),
                cpu_workers=max(self.min_workers, self.state.cpu_workers - 1),
                h2d_mb=self.state.h2d_mb,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return
        free, total = self._gpu_caps()
        if total == 0:
            return
        headroom = free / (total + 1e-9)
        target_head = 1.0 - self.tu
        if headroom > target_head + 0.10:
            self.state = ScaleState(
                batch_size=min(self.max_bs, int(self.state.batch_size * 1.25)),
                cpu_workers=min(self.max_workers, self.state.cpu_workers + 1),
                h2d_mb=min(8192, int(self.state.h2d_mb * 1.2)),
            )
        elif headroom < target_head - 0.05:
            self.state = ScaleState(
                batch_size=max(self.min_bs, int(self.state.batch_size * 0.9)),
                cpu_workers=max(self.min_workers, self.state.cpu_workers - 1),
                h2d_mb=max(256, int(self.state.h2d_mb * 0.9)),
            )
        if step_time_s is not None:
            self._h2d_mb = self.state.h2d_mb


__all__ = ["AutoScaler", "ScaleState"]
