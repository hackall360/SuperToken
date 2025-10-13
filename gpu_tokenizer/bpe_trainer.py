"""GPU-accelerated BPE trainer with optional autoscaling."""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional, Tuple

import torch

from .autoscaler import AutoScaler
from .utils import apply_merge_once, count_pairs

def _aggregate_pair_keys(
    keys: torch.Tensor, counts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate duplicate 64‑bit pair keys by summing their counts.

    Given a 1‑D tensor of packed pair keys and a 1‑D tensor of counts of the
    same length, this helper will sort the keys, perform a run–length encoding
    on the sorted keys, and sum the corresponding counts for each unique key.
    It returns a tuple of unique keys and their aggregated counts.  Both
    tensors are on the same device as the inputs.

    The packed key format is ``(a_id << 32) | b_id``, so values should be
    64‑bit integers.  See the training loop for how keys are constructed.
    """
    # Ensure inputs are 1‑D and on the same device
    if keys.numel() == 0:
        return keys, counts
    device = keys.device
    # Sort the keys; keep corresponding counts aligned
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    sorted_counts = counts[order]
    # Identify boundaries where the key changes (i.e. new unique starts)
    diff = torch.ones_like(sorted_keys, dtype=torch.bool)
    diff[1:] = sorted_keys[1:] != sorted_keys[:-1]
    # Positions of the first occurrence of each unique key
    pos = torch.nonzero(diff, as_tuple=False).flatten()
    # Compute prefix sums of counts with a leading zero for convenience
    prefix = torch.cat(
        [torch.zeros((1,), dtype=sorted_counts.dtype, device=device), torch.cumsum(sorted_counts, dim=0)]
    )
    # Starts are after the previous unique position; ends are at the current unique position
    starts = torch.cat([
        torch.zeros((1,), dtype=pos.dtype, device=device), pos[:-1] + 1
    ])
    ends = pos
    aggregated_counts = prefix[ends + 1] - prefix[starts]
    aggregated_keys = sorted_keys[pos]
    return aggregated_keys, aggregated_counts


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
        # Outer loop over the number of merges
        while step < self.target_merges:
            # Query autoscaler to refresh internal state.  The suggested batch
            # size is unused here but influences later feedback.
            _ = self.autoscaler.suggest(token_bytes_per_example=int(8 * 1024))
            t0 = time.time()
            # We accumulate packed pair keys and their local counts across batches.
            global_keys: Optional[torch.Tensor] = None
            global_counts: Optional[torch.Tensor] = None
            for (x_cpu, v_cpu) in batches:
                # Transfer batch to device with non‑blocking copies; handle OOM
                try:
                    x = x_cpu.to(self.device, non_blocking=True)
                except RuntimeError as exc:
                    if "CUDA out of memory" in str(exc):
                        # Notify autoscaler and skip this batch
                        self.autoscaler.feedback(oom=True)
                        continue
                    raise
                v = v_cpu.to(self.device, non_blocking=True)
                # Count unique adjacent pairs per batch using existing util
                pairs, counts = count_pairs(x, v)
                # Skip if no valid pairs in this batch
                if pairs.numel() == 0:
                    continue
                # Pack the pair ids into 64‑bit keys for more efficient aggregation
                # Format: key = (a_id << 32) | b_id, using 64‑bit integer type
                a_ids = pairs[:, 0].to(torch.long)
                b_ids = pairs[:, 1].to(torch.long)
                keys = (a_ids << 32) | b_ids
                # Accumulate keys and counts for this iteration
                if global_keys is None:
                    global_keys = keys
                    global_counts = counts.to(torch.long)
                else:
                    global_keys = torch.cat([global_keys, keys], dim=0)
                    global_counts = torch.cat([global_counts, counts.to(torch.long)], dim=0)
            # If no pairs were found, terminate early
            if global_keys is None or global_keys.numel() == 0:
                print("No pairs left to merge.")
                break
            # Aggregate duplicate keys by summing their counts
            agg_keys, agg_counts = _aggregate_pair_keys(global_keys, global_counts)
            # Select the most frequent pair
            best_idx = torch.argmax(agg_counts)
            best_key = agg_keys[best_idx]
            a_id = int((best_key >> 32).item())
            b_id = int((best_key & ((1 << 32) - 1)).item())
            best_count = int(agg_counts[best_idx].item())
            # Stop if no pair appears more than once
            if best_count <= 1:
                print("Stopping: no frequent pairs.")
                break
            new_id = self.vocab_size
            # Log progress periodically
            if step % log_every == 0:
                print(f"merge {step:6d}: ({a_id},{b_id}) -> {new_id}  count={best_count}")
            # Record the merge and update vocab size
            self.merges.append((a_id, b_id))
            self.vocab_size += 1
            # Apply the merge to each batch and build the next round of batches
            new_batches: list[Tuple[torch.Tensor, torch.Tensor]] = []
            oom_seen = False
            for (x_cpu, v_cpu) in batches:
                try:
                    # Move batch to device
                    x = x_cpu.to(self.device, non_blocking=True)
                    v = v_cpu.to(self.device, non_blocking=True)
                    # Apply merge once on GPU
                    x2, v2 = apply_merge_once(x, v, a_id, b_id, new_id)
                    # Transfer back to CPU (pinned) to free GPU memory
                    x2_cpu = x2.to("cpu", non_blocking=True, copy=True).pin_memory()
                    v2_cpu = v2.to("cpu", non_blocking=True, copy=True).pin_memory()
                    new_batches.append((x2_cpu, v2_cpu))
                except RuntimeError as exc:
                    if "CUDA out of memory" in str(exc):
                        # On OOM, free GPU memory and keep the unmodified batch
                        oom_seen = True
                        torch.cuda.empty_cache()
                        new_batches.append((x_cpu, v_cpu))
                    else:
                        raise
            batches = new_batches
            step += 1
            # Provide feedback to the autoscaler about iteration time and OOM status
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
