"""Adaptive autoscaler for GPU BPE training."""

from __future__ import annotations

import json
import logging
import math
import os
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, Optional

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - fallback when psutil missing
    psutil = None

import torch


logger = logging.getLogger(__name__)


@dataclass
class ScaleState:
    batch_size: int
    cpu_workers: int
    h2d_mb: int
    cpu_fallback_rate: float = 0.0


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
        window_size: int = 10,
        device: Optional[str] = None,
    ) -> None:
        self.tu = float(max(0.1, min(target_util, 0.95)))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_bs, self.max_bs = min_bs, max_bs
        self.min_workers = min_workers
        self.max_workers = max_workers or max(min_workers, os.cpu_count() or 4)
        self.state: Optional[ScaleState] = None
        self._h2d_mb = init_h2d_mb
        self._window_size = max(3, window_size)
        self._step_times: Deque[float] = deque(maxlen=self._window_size)
        self._vram_fracs: Deque[float] = deque(maxlen=self._window_size)

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
        fallback_rate = self.state.cpu_fallback_rate if self.state is not None else 0.0
        self.state = ScaleState(
            batch_size=bs_gpu,
            cpu_workers=workers,
            h2d_mb=h2d_mb,
            cpu_fallback_rate=fallback_rate,
        )
        return self.state

    def feedback(
        self,
        step_time_s: float | None = None,
        oom: bool = False,
        cpu_fallback_rate: float | None = None,
    ) -> None:
        if self.state is None:
            return
        fallback_rate = self.state.cpu_fallback_rate
        if cpu_fallback_rate is not None:
            fallback_rate = max(0.0, min(1.0, float(cpu_fallback_rate)))
        if oom:
            prev = self.state
            self.state = ScaleState(
                batch_size=max(self.min_bs, self.state.batch_size // 2),
                cpu_workers=max(self.min_workers, self.state.cpu_workers - 1),
                h2d_mb=self.state.h2d_mb,
                cpu_fallback_rate=fallback_rate,
            )
            self._log_adjustment(prev, self.state, event="oom")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return
        free, total = self._gpu_caps()
        if total == 0:
            return
        used = max(0, total - free)
        self._record_metrics(step_time_s, used, total)
        if len(self._vram_fracs) < max(3, self._window_size // 2):
            return
        mean_vram, var_vram = self._stats(self._vram_fracs)
        mean_step, _ = self._stats(self._step_times)
        prev_state = self.state
        new_state = self._tuned_state(mean_vram, var_vram, mean_step, fallback_rate)
        if new_state != prev_state:
            self.state = new_state
            self._h2d_mb = self.state.h2d_mb
            self._log_adjustment(prev_state, new_state, mean_vram, var_vram, mean_step)
        else:
            self.state = ScaleState(
                batch_size=self.state.batch_size,
                cpu_workers=self.state.cpu_workers,
                h2d_mb=self.state.h2d_mb,
                cpu_fallback_rate=fallback_rate,
            )

    def _record_metrics(self, step_time_s: float | None, used_bytes: int, total_bytes: int) -> None:
        if step_time_s is not None and math.isfinite(step_time_s):
            self._step_times.append(float(step_time_s))
        if total_bytes > 0:
            frac = max(0.0, min(1.0, used_bytes / float(total_bytes)))
            self._vram_fracs.append(frac)

    def _stats(self, values: Deque[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        mean = sum(values) / len(values)
        var = 0.0
        if len(values) > 1:
            var = sum((v - mean) ** 2 for v in values) / len(values)
        return mean, var

    def _scale_from_gap(self, gap: float, variance: float) -> float:
        if gap <= 0:
            return 0.0
        base = min(0.25, max(0.05, gap * 0.6))
        if variance > 0.01:
            base *= 0.3
        elif variance > 0.005:
            base *= 0.5
        return base

    def _tuned_state(
        self,
        mean_vram: float,
        var_vram: float,
        mean_step: float,
        fallback_rate: float,
    ) -> ScaleState:
        lower_bound = max(0.75, self.tu - 0.05)
        upper_bound = min(0.95, self.tu + 0.10)
        new_bs = self.state.batch_size
        new_workers = self.state.cpu_workers
        new_h2d = self.state.h2d_mb
        if mean_vram < lower_bound:
            gap = lower_bound - mean_vram
            scale = self._scale_from_gap(gap, var_vram)
            if scale > 0:
                new_bs = min(self.max_bs, max(self.min_bs, int(self.state.batch_size * (1 + scale))))
                new_workers = min(
                    self.max_workers,
                    max(self.min_workers, int(round(self.state.cpu_workers * (1 + scale * 0.5)))),
                )
                h2d_scale = max(0.02, scale * 0.5)
                new_h2d = min(8192, max(256, int(self.state.h2d_mb * (1 + h2d_scale))))
        elif mean_vram > upper_bound:
            gap = mean_vram - upper_bound
            scale = self._scale_from_gap(gap, var_vram)
            if scale > 0:
                new_bs = max(self.min_bs, int(self.state.batch_size * (1 - scale)))
                new_workers = max(self.min_workers, int(self.state.cpu_workers * (1 - scale * 0.5)))
                h2d_scale = max(0.02, scale * 0.5)
                new_h2d = max(256, int(self.state.h2d_mb * (1 - h2d_scale)))
        if fallback_rate > 0.25:
            new_bs = max(self.min_bs, int(new_bs * 0.9))
            new_workers = min(self.max_workers, max(new_workers, self.state.cpu_workers + 1))
        elif fallback_rate < 0.05 and new_workers < self.max_workers:
            new_workers = min(self.max_workers, max(new_workers, self.state.cpu_workers))
        if mean_step > 0 and len(self._step_times) >= self._window_size:
            # Light smoothing: if step times are trending up, avoid aggressive growth
            recent = list(self._step_times)[-3:]
            if len(recent) >= 3 and recent[-1] > recent[0] * 1.1 and new_bs > self.state.batch_size:
                new_bs = max(self.state.batch_size, int((self.state.batch_size + new_bs) / 2))
        return ScaleState(
            batch_size=new_bs,
            cpu_workers=new_workers,
            h2d_mb=new_h2d,
            cpu_fallback_rate=fallback_rate,
        )

    def snapshot_metrics(self) -> dict[str, object]:
        state_dict = asdict(self.state) if self.state is not None else None
        mean_step, var_step = self._stats(self._step_times)
        mean_vram, var_vram = self._stats(self._vram_fracs)
        return {
            "device": self.device,
            "target_util": self.tu,
            "window_size": self._window_size,
            "state": state_dict,
            "step_times": list(self._step_times),
            "step_stats": {"mean": mean_step, "variance": var_step},
            "vram_utilization": list(self._vram_fracs),
            "vram_stats": {"mean": mean_vram, "variance": var_vram},
        }

    def _log_adjustment(
        self,
        prev: ScaleState,
        new: ScaleState,
        mean_vram: float | None = None,
        var_vram: float | None = None,
        mean_step: float | None = None,
        event: str = "feedback",
    ) -> None:
        payload = {
            "event": event,
            "batch_size": new.batch_size,
            "prev_batch_size": prev.batch_size,
            "h2d_mb": new.h2d_mb,
            "prev_h2d_mb": prev.h2d_mb,
            "cpu_workers": new.cpu_workers,
            "prev_cpu_workers": prev.cpu_workers,
            "cpu_fallback_rate": new.cpu_fallback_rate,
        }
        if mean_vram is not None:
            payload["mean_vram_util"] = round(mean_vram, 4)
        if var_vram is not None:
            payload["var_vram_util"] = round(var_vram, 6)
        if mean_step is not None and mean_step > 0:
            payload["mean_step_time_s"] = round(mean_step, 4)
        logger.info("autoscale.adjust %s", json.dumps(payload, sort_keys=True))


__all__ = ["AutoScaler", "ScaleState"]
