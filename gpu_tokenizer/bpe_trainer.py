"""GPU-accelerated BPE trainer with optional autoscaling."""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional, Tuple

import torch

from .autoscaler import AutoScaler
from .utils import apply_merge_once, count_pairs


class GPUBPETrainer:
    """Train byte-pair encodings on the GPU."""

    def __init__(
        self,
        base_vocab: int = 256,
        merges: int = 50_000,
        device: str | None = None,
        autoscaler: Optional[AutoScaler] = None,
    ) -> None:
        self.base_vocab = base_vocab
        self.target_merges = merges
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = base_vocab
        self.merges: List[Tuple[int, int]] = []
        self.autoscaler = autoscaler or AutoScaler()

    def fit(
        self, batches: List[Tuple[torch.Tensor, torch.Tensor]], log_every: int = 100
    ) -> dict[str, object]:
        step = 0
        while step < self.target_merges:
            state = self.autoscaler.suggest(token_bytes_per_example=int(8 * 1024))
            _ = state  # currently unused but keeps autoscaler state fresh
            t0 = time.time()
            global_pairs = None
            global_counts = None
            for (x_cpu, v_cpu) in batches:
                try:
                    x = x_cpu.to(self.device, non_blocking=True)
                except RuntimeError as exc:
                    if "CUDA out of memory" in str(exc):
                        self.autoscaler.feedback(oom=True)
                        continue
                    raise
                v = v_cpu.to(self.device, non_blocking=True)
                pairs, counts = count_pairs(x, v)
                if pairs.numel() == 0:
                    continue
                if global_pairs is None:
                    global_pairs = pairs
                    global_counts = counts
                else:
                    global_pairs = torch.cat([global_pairs, pairs], dim=0)
                    global_counts = torch.cat([global_counts, counts], dim=0)
            if global_pairs is None or global_pairs.numel() == 0:
                print("No pairs left to merge.")
                break
            uniq, inv = torch.unique(global_pairs, dim=0, return_inverse=True)
            counts = torch.zeros(uniq.size(0), dtype=torch.long, device=self.device)
            counts.scatter_add_(0, inv, global_counts)
            best_idx = torch.argmax(counts)
            a_id, b_id = uniq[best_idx].tolist()
            best_count = int(counts[best_idx].item())
            if best_count <= 1:
                print("Stopping: no frequent pairs.")
                break
            new_id = self.vocab_size
            if step % log_every == 0:
                print(f"merge {step:6d}: ({a_id},{b_id}) -> {new_id}  count={best_count}")
            self.merges.append((a_id, b_id))
            self.vocab_size += 1
            new_batches: list[Tuple[torch.Tensor, torch.Tensor]] = []
            oom_seen = False
            for (x_cpu, v_cpu) in batches:
                try:
                    x = x_cpu.to(self.device, non_blocking=True)
                    v = v_cpu.to(self.device, non_blocking=True)
                    x2, v2 = apply_merge_once(x, v, a_id, b_id, new_id)
                    x2_cpu = x2.to("cpu", non_blocking=True, copy=True).pin_memory()
                    v2_cpu = v2.to("cpu", non_blocking=True, copy=True).pin_memory()
                    new_batches.append((x2_cpu, v2_cpu))
                except RuntimeError as exc:
                    if "CUDA out of memory" in str(exc):
                        oom_seen = True
                        torch.cuda.empty_cache()
                        new_batches.append((x_cpu, v_cpu))
                    else:
                        raise
            batches = new_batches
            step += 1
            self.autoscaler.feedback(step_time_s=time.time() - t0, oom=oom_seen)
        return {
            "base_vocab": self.base_vocab,
            "vocab_size": self.vocab_size,
            "merges": self.merges,
        }

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        meta = {
            "base_vocab": self.base_vocab,
            "vocab_size": self.vocab_size,
            "merges": self.merges,
        }
        with open(os.path.join(out_dir, "bpe_merges.json"), "w") as f:
            json.dump(meta, f)
        print(f"Saved merges → {out_dir}/bpe_merges.json")


__all__ = ["GPUBPETrainer"]
