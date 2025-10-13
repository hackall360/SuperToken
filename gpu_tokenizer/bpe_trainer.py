"""GPU-accelerated BPE trainer with optional autoscaling."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Iterable, List, Optional, Tuple, cast

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
    # Next boundaries are the subsequent start positions (or the end of the array)
    next_pos = torch.cat(
        [pos[1:], torch.tensor([sorted_keys.numel()], dtype=pos.dtype, device=device)]
    )
    aggregated_counts = prefix[next_pos] - prefix[pos]
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
        self,
        batches: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        log_every: int = 100,
        profile_streams: bool = False,
    ) -> dict[str, object]:
        """Train merges using a pipelined GPU workflow when possible."""

        device = torch.device(self.device)
        step = 0
        current_batches: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = batches

        def _ensure_workspace(
            workspace: dict[str, Optional[torch.Tensor]],
            x_src: torch.Tensor,
            v_src: torch.Tensor,
        ) -> None:
            if workspace["x"] is None or workspace["x"].shape != x_src.shape or workspace["x"].dtype != x_src.dtype:
                workspace["x"] = torch.empty_like(x_src, device=device)
            if workspace["v"] is None or workspace["v"].shape != v_src.shape or workspace["v"].dtype != v_src.dtype:
                workspace["v"] = torch.empty_like(v_src, device=device)

        cpu_device = torch.device("cpu")

        def _count_pairs_cpu(
            batch_iter: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
            global_keys: Optional[torch.Tensor] = None
            global_counts: Optional[torch.Tensor] = None
            consumed: list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for (x_cpu, v_cpu, lengths_cpu) in batch_iter:
                x_snapshot = x_cpu.clone()
                v_snapshot = v_cpu.clone()
                lengths_snapshot = lengths_cpu.clone()
                consumed.append((x_snapshot, v_snapshot, lengths_snapshot))
                pairs, counts = count_pairs(x_snapshot.to(cpu_device), v_snapshot.to(cpu_device))
                if pairs.numel() == 0:
                    continue
                a_ids = pairs[:, 0].to(torch.long)
                b_ids = pairs[:, 1].to(torch.long)
                keys = (a_ids << 32) | b_ids
                if global_keys is None:
                    global_keys = keys
                    global_counts = counts.to(torch.long)
                else:
                    global_keys = torch.cat([global_keys, keys], dim=0)
                    global_counts = torch.cat([global_counts, counts.to(torch.long)], dim=0)
            return global_keys, global_counts, consumed

        def _apply_merge_cpu(
            batch_iter: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            a_id: int,
            b_id: int,
            new_id: int,
        ) -> tuple[list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], bool]:
            new_batches: list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            oom_seen = False
            for (x_cpu, v_cpu, lengths_cpu) in batch_iter:
                try:
                    x = x_cpu.to(cpu_device)
                    v = v_cpu.to(cpu_device)
                    x2, v2 = apply_merge_once(x, v, a_id, b_id, new_id)
                    x2_cpu = x2.to("cpu", copy=True)
                    v2_cpu = v2.to("cpu", copy=True)
                    lengths2_cpu = v2_cpu.sum(dim=-1)
                    new_batches.append((x2_cpu, v2_cpu, lengths2_cpu))
                except RuntimeError as exc:
                    if "CUDA out of memory" in str(exc):
                        oom_seen = True
                        torch.cuda.empty_cache()
                        new_batches.append((x_cpu, v_cpu, lengths_cpu))
                    else:
                        raise
            return new_batches, oom_seen

        use_cuda = device.type == "cuda" and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(device)
            copy_stream = torch.cuda.Stream(device=device)
            compute_stream = torch.cuda.Stream(device=device)
            workspaces = [
                {"x": None, "v": None},
                {"x": None, "v": None},
            ]
            copy_events = [torch.cuda.Event(blocking=False) for _ in range(2)]
            compute_events = [torch.cuda.Event(blocking=False) for _ in range(2)]

            def _count_pairs_gpu(
                batch_iter: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
            ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
                global_keys: Optional[torch.Tensor] = None
                global_counts: Optional[torch.Tensor] = None
                consumed: list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
                buffer_idx = 0
                compute_recorded = [False, False]
                pending_copy = False
                pending_compute = False
                pair_results: list[tuple[torch.Tensor, torch.Tensor]] = []
                for (x_cpu, v_cpu, lengths_cpu) in batch_iter:
                    x_snapshot = x_cpu.clone().pin_memory()
                    v_snapshot = v_cpu.clone().pin_memory()
                    lengths_snapshot = lengths_cpu.clone()
                    consumed.append((x_snapshot, v_snapshot, lengths_snapshot))
                    workspace = workspaces[buffer_idx]
                    _ensure_workspace(workspace, x_snapshot, v_snapshot)
                    with torch.cuda.stream(copy_stream):
                        if compute_recorded[buffer_idx]:
                            copy_stream.wait_event(compute_events[buffer_idx])
                        workspace["x"].copy_(x_snapshot, non_blocking=True)
                        workspace["v"].copy_(v_snapshot, non_blocking=True)
                    copy_events[buffer_idx].record(copy_stream)
                    pending_copy = True
                    try:
                        with torch.cuda.stream(compute_stream):
                            compute_stream.wait_event(copy_events[buffer_idx])
                            pairs, counts = count_pairs(workspace["x"], workspace["v"])
                        compute_events[buffer_idx].record(compute_stream)
                        compute_recorded[buffer_idx] = True
                        pending_compute = True
                        pair_results.append((pairs, counts))
                    except RuntimeError as exc:
                        compute_recorded[buffer_idx] = False
                        if "CUDA out of memory" in str(exc):
                            self.autoscaler.feedback(oom=True)
                            torch.cuda.empty_cache()
                            buffer_idx = 1 - buffer_idx
                            continue
                        raise
                    buffer_idx = 1 - buffer_idx
                torch.cuda.current_stream(device).wait_stream(compute_stream)
                torch.cuda.current_stream(device).wait_stream(copy_stream)
                if profile_streams and (pending_copy or pending_compute):
                    torch.cuda.synchronize(device)
                    assert copy_stream.query(), "copy stream did not drain"
                    assert compute_stream.query(), "compute stream did not drain"
                for pairs, counts in pair_results:
                    if pairs.numel() == 0:
                        continue
                    a_ids = pairs[:, 0].to(torch.long)
                    b_ids = pairs[:, 1].to(torch.long)
                    keys = (a_ids << 32) | b_ids
                    if global_keys is None:
                        global_keys = keys
                        global_counts = counts.to(torch.long)
                    else:
                        global_keys = torch.cat([global_keys, keys], dim=0)
                        global_counts = torch.cat([global_counts, counts.to(torch.long)], dim=0)
                return global_keys, global_counts, consumed

            def _apply_merge_gpu(
                batch_iter: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                a_id: int,
                b_id: int,
                new_id: int,
            ) -> tuple[list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], bool]:
                new_batches: list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
                batch_results: list[tuple[str, Any]] = []
                oom_seen = False
                buffer_idx = 0
                compute_recorded = [False, False]
                pending_copy = False
                pending_compute = False
                for (x_cpu, v_cpu, lengths_cpu) in batch_iter:
                    workspace = workspaces[buffer_idx]
                    _ensure_workspace(workspace, x_cpu, v_cpu)
                    with torch.cuda.stream(copy_stream):
                        if compute_recorded[buffer_idx]:
                            copy_stream.wait_event(compute_events[buffer_idx])
                        workspace["x"].copy_(x_cpu, non_blocking=True)
                        workspace["v"].copy_(v_cpu, non_blocking=True)
                    copy_events[buffer_idx].record(copy_stream)
                    pending_copy = True
                    try:
                        with torch.cuda.stream(compute_stream):
                            compute_stream.wait_event(copy_events[buffer_idx])
                            x_dev, v_dev = apply_merge_once(
                                workspace["x"], workspace["v"], a_id, b_id, new_id
                            )
                        compute_events[buffer_idx].record(compute_stream)
                        compute_recorded[buffer_idx] = True
                        pending_compute = True
                    except RuntimeError as exc:
                        compute_recorded[buffer_idx] = False
                        if "CUDA out of memory" in str(exc):
                            oom_seen = True
                            torch.cuda.empty_cache()
                            batch_results.append(("oom", (x_cpu, v_cpu, lengths_cpu)))
                            buffer_idx = 1 - buffer_idx
                            continue
                        raise
                    finish_event = torch.cuda.Event(blocking=False)
                    with torch.cuda.stream(copy_stream):
                        copy_stream.wait_event(compute_events[buffer_idx])
                        x_cpu_out = x_dev.to("cpu", non_blocking=True, copy=True).pin_memory()
                        v_cpu_out = v_dev.to("cpu", non_blocking=True, copy=True).pin_memory()
                        finish_event.record(copy_stream)
                    batch_results.append(("ready", (x_cpu_out, v_cpu_out, finish_event)))
                    buffer_idx = 1 - buffer_idx
                torch.cuda.current_stream(device).wait_stream(copy_stream)
                torch.cuda.current_stream(device).wait_stream(compute_stream)
                if profile_streams and (pending_copy or pending_compute):
                    torch.cuda.synchronize(device)
                    assert copy_stream.query(), "copy stream did not drain"
                    assert compute_stream.query(), "compute stream did not drain"
                for tag, payload in batch_results:
                    if tag == "oom":
                        x_keep, v_keep, lengths_keep = cast(
                            Tuple[torch.Tensor, torch.Tensor, torch.Tensor], payload
                        )
                        new_batches.append((x_keep, v_keep, lengths_keep))
                    else:
                        x_cpu_out, v_cpu_out, finish_event = cast(
                            Tuple[torch.Tensor, torch.Tensor, torch.cuda.Event], payload
                        )
                        finish_event.synchronize()
                        lengths_cpu_out = v_cpu_out.sum(dim=-1)
                        new_batches.append((x_cpu_out, v_cpu_out, lengths_cpu_out))
                return new_batches, oom_seen

        while step < self.target_merges:
            _ = self.autoscaler.suggest(token_bytes_per_example=int(8 * 1024))
            t0 = time.time()
            if use_cuda:
                global_keys, global_counts, consumed_batches = _count_pairs_gpu(current_batches)
            else:
                global_keys, global_counts, consumed_batches = _count_pairs_cpu(current_batches)
            if global_keys is None or global_keys.numel() == 0:
                print("No pairs left to merge.")
                break
            current_batches = consumed_batches
            agg_keys, agg_counts = _aggregate_pair_keys(global_keys, global_counts)
            best_idx = torch.argmax(agg_counts)
            best_key = agg_keys[best_idx]
            a_id = int((best_key >> 32).item())
            b_id = int((best_key & ((1 << 32) - 1)).item())
            best_count = int(agg_counts[best_idx].item())
            if best_count <= 1:
                print("Stopping: no frequent pairs.")
                break
            new_id = self.vocab_size
            if step % log_every == 0:
                print(f"merge {step:6d}: ({a_id},{b_id}) -> {new_id}  count={best_count}")
            self.merges.append((a_id, b_id))
            self.vocab_size += 1
            if use_cuda:
                current_batches, oom_seen = _apply_merge_gpu(current_batches, a_id, b_id, new_id)
            else:
                current_batches, oom_seen = _apply_merge_cpu(current_batches, a_id, b_id, new_id)
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
