# gpu_tokenizer/
# ├─ __init__.py
# ├─ bpe_trainer.py          # GPU-accelerated BPE merges (PyTorch CUDA), CPU packer
# ├─ unigram_trainer.py      # GPU-forward/backward EM scaffold (batched); WIP but runnable on small vocab
# ├─ cpu_packer.py           # Fast file → integer-ids packer (streaming), light CPU
# ├─ datasets.py             # Iterable dataset with padding + masks, pinned-memory host→GPU
# ├─ utils.py                # Helpers: prefix sums, compaction, pair counting
# └─ cli_train_bpe.py        # Simple CLI for BPE training on consumer GPUs

# =============================== __init__.py ===============================
from .bpe_trainer import GPUBPETrainer
from .unigram_trainer import GPUUnigramTrainer

__all__ = ["GPUBPETrainer", "GPUUnigramTrainer"]

# =============================== utils.py ==================================
from __future__ import annotations
import torch

def device_of(x: torch.Tensor) -> torch.device:
    return x.device

@torch.jit.script
def prefix_sum_int(mask: torch.Tensor) -> torch.Tensor:
    """Inclusive prefix sum over last dim (int64). mask: [B, L] (0/1)."""
    return torch.cumsum(mask, dim=-1)

@torch.jit.script
def compact_by_mask(vals: torch.Tensor, keep: torch.Tensor, max_len: int) -> torch.Tensor:
    """Compact vals along last dim using keep (0/1). Returns [B, max_len].
    Any slots beyond new length are filled with -1.
    vals, keep: [B, L]
    max_len: int
    """
    B, L = vals.shape
    idx = prefix_sum_int(keep) - 1  # [B, L], positions for kept elements
    # Flatten batch, scatter into output buffer
    out = vals.new_full((B, max_len), -1)
    b_ids = torch.arange(B, device=vals.device).unsqueeze(1).expand(B, L)
    take = keep.bool()
    if take.any():
        out[b_ids[take], idx[take]] = vals[take]
    return out

@torch.jit.script
def apply_merge_once(seqs: torch.Tensor, valid: torch.Tensor, a_id: int, b_id: int, new_id: int):
    """Apply a single BPE merge (a_id,b_id)->new_id on a padded batch of seqs.
    seqs: [B, L] int64 (token ids >=0), padded with -1 where invalid
    valid: [B, L] 0/1 mask (1 where token is valid)
    Returns new_seqs [B, L], new_valid [B, L]
    """
    B, L = seqs.shape
    # Adjacent pairs
    lhs = seqs[:, :-1]
    rhs = seqs[:, 1:]
    v_l = valid[:, :-1]
    v_r = valid[:, 1:]
    pair_match = (lhs == a_id) & (rhs == b_id) & (v_l.bool()) & (v_r.bool())  # [B, L-1]

    # Tokens to keep: all valid tokens except the RHS of matched pairs
    keep = valid.clone()
    keep[:, 1:] = keep[:, 1:] & (~pair_match)

    # Replace the LHS of matched pairs with new_id
    seqs = seqs.clone()
    seqs[:, :-1] = torch.where(pair_match, torch.as_tensor(new_id, device=seqs.device, dtype=seqs.dtype), lhs)
    seqs[:, -1] = seqs[:, -1]  # keep last column

    # Now compact based on keep
    new_lens = keep.sum(dim=-1)
    max_new = int(new_lens.max().item())
    new_seqs = compact_by_mask(seqs, keep, max_new)
    new_valid = torch.zeros_like(seqs)
    new_valid[:, :max_new] = 0
    # Build valid mask for compacted output (1 for slots we filled)
    b_ids = torch.arange(B, device=seqs.device).unsqueeze(1)
    rng = torch.arange(max_new, device=seqs.device).unsqueeze(0)
    # A slot is valid iff its value != -1
    new_valid = (new_seqs != -1).long()
    return new_seqs, new_valid

@torch.jit.script
def count_pairs(seqs: torch.Tensor, valid: torch.Tensor):
    """Return unique pairs and counts for all adjacent valid tokens in batch.
    seqs: [B, L] int64, valid: [B, L] 0/1
    Returns: pairs [N,2], counts [N]
    """
    lhs = seqs[:, :-1]
    rhs = seqs[:, 1:]
    mask = valid[:, :-1].bool() & valid[:, 1:].bool()
    if not mask.any():
        return seqs.new_empty((0,2)), seqs.new_empty((0,), dtype=torch.long)
    pairs = torch.stack([lhs[mask], rhs[mask]], dim=1)  # [M,2]
    # unique over rows
    uniq, inv = torch.unique(pairs, dim=0, return_inverse=True)
    counts = torch.bincount(inv, minlength=uniq.size(0))
    return uniq, counts

# =============================== cpu_packer.py ===============================
from __future__ import annotations
import os, mmap
from typing import List, Iterable

# Light CPU: bytes→ids. Default: byte-level vocab 0..255; -1 for pad.
class BytePacker:
    def __init__(self, bos: int | None = None, eos: int | None = None):
        self.bos = bos
        self.eos = eos

    def encode_file(self, path: str) -> list[int]:
        with open(path, 'rb') as f:
            data = f.read()
        out = []
        if self.bos is not None:
            out.append(self.bos)
        out.extend(data)
        if self.eos is not None:
            out.append(self.eos)
        return out

# =============================== datasets.py ================================
from __future__ import annotations
import glob, random
from typing import List, Iterator, Tuple
import torch

class PackedBatcher:
    """Streams packed integer sequences, pads per-batch, pinned host memory for fast H2D.
    """
    def __init__(self, sequences: List[list[int]], batch_size: int = 1024, seed: int = 1337):
        self.sequences = sequences
        self.bs = batch_size
        random.Random(seed).shuffle(self.sequences)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for i in range(0, len(self.sequences), self.bs):
            chunk = self.sequences[i:i+self.bs]
            max_len = max(len(x) for x in chunk)
            x = torch.full((len(chunk), max_len), -1, dtype=torch.long, pin_memory=True)
            v = torch.zeros((len(chunk), max_len), dtype=torch.long, pin_memory=True)
            for r, seq in enumerate(chunk):
                L = len(seq)
                x[r, :L] = torch.tensor(seq, dtype=torch.long)
                v[r, :L] = 1
            yield x, v

# =============================== bpe_trainer.py ==============================
from __future__ import annotations
import os, json, math
from typing import List, Tuple
import torch
from .utils import count_pairs, apply_merge_once

class GPUBPETrainer:
    """GPU-accelerated BPE trainer (consumer GPU friendly).

    Core loop:
      1) H2D batch of padded sequences (int64 ids, -1 padded).
      2) Count all adjacent pairs on GPU via torch.unique+bincount.
      3) Pick most frequent pair globally.
      4) Apply merge to all batches on GPU via vectorized compaction.
      5) Repeat.

    Notes:
      • This is a single-process, single-GPU MVP. Multi-GPU can shard batches.
      • Initial vocab is 0..base_vocab-1 (e.g., 256 for bytes). New ids append sequentially.
    """
    def __init__(self, base_vocab: int = 256, merges: int = 50000, device: str | None = None):
        self.base_vocab = base_vocab
        self.target_merges = merges
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = base_vocab
        self.merges: List[Tuple[int,int]] = []

    def fit(self, batches: List[Tuple[torch.Tensor, torch.Tensor]], log_every: int = 100):
        """batches: list of (seqs, valid) on CPU pinned memory. We'll move to device per-iter.
        """
        # Move first batch to get device dtype right
        step = 0
        while step < self.target_merges:
            # 1) Global pair counts over all batches (accumulate on GPU)
            global_pairs = None
            global_counts = None
            for (x_cpu, v_cpu) in batches:
                x = x_cpu.to(self.device, non_blocking=True)
                v = v_cpu.to(self.device, non_blocking=True)
                pairs, counts = count_pairs(x, v)
                if pairs.numel() == 0:
                    continue
                if global_pairs is None:
                    global_pairs = pairs
                    global_counts = counts
                else:
                    # concat then re-unique to accumulate
                    global_pairs = torch.cat([global_pairs, pairs], dim=0)
                    global_counts = torch.cat([global_counts, counts], dim=0)
            if global_pairs is None or global_pairs.numel() == 0:
                print("No pairs left to merge.")
                break
            # reduce duplicates
            uniq, inv = torch.unique(global_pairs, dim=0, return_inverse=True)
            counts = torch.zeros(uniq.size(0), dtype=torch.long, device=self.device)
            counts.scatter_add_(0, inv, global_counts)
            # 2) pick most frequent
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

            # 3) apply merge to all batches in-place
            new_batches = []
            for (x_cpu, v_cpu) in batches:
                x = x_cpu.to(self.device, non_blocking=True)
                v = v_cpu.to(self.device, non_blocking=True)
                x2, v2 = apply_merge_once(x, v, a_id, b_id, new_id)
                # Keep tensors on CPU pinned to reduce GPU RAM growth between iterations
                x2_cpu = x2.to("cpu", non_blocking=True, torch_dtype=None, copy=True).pin_memory()
                v2_cpu = v2.to("cpu", non_blocking=True, torch_dtype=None, copy=True).pin_memory()
                new_batches.append((x2_cpu, v2_cpu))
            batches = new_batches
            step += 1
        return {
            "base_vocab": self.base_vocab,
            "vocab_size": self.vocab_size,
            "merges": self.merges,
        }

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        meta = {
            "base_vocab": self.base_vocab,
            "vocab_size": self.vocab_size,
            "merges": self.merges,
        }
        with open(os.path.join(out_dir, "bpe_merges.json"), "w") as f:
            json.dump(meta, f)
        print(f"Saved merges → {out_dir}/bpe_merges.json")

# ============================== unigram_trainer.py ===========================
from __future__ import annotations
import torch, math
from typing import List, Dict, Tuple

class GPUUnigramTrainer:
    """Minimal Unigram (SentencePiece-like) trainer scaffold with GPU E/M steps.

    This is a *working prototype* for small vocab/short max_subword (e.g. 8).
    It runs forward-backward per sequence in a batch using log-space DP.
    For production-scale corpora, you’d add pruning and tries.
    """
    def __init__(self, base_vocab=256, vocab_size=50000, max_subword_len=8, device=None):
        self.base_vocab = base_vocab
        self.target_vocab = vocab_size
        self.max_len = max_subword_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize subwords as all byte unigrams only; extend later via sampled candidates
        self.id2piece: Dict[int, bytes] = {i: bytes([i]) for i in range(base_vocab)}
        self.piece2id: Dict[bytes, int] = {bytes([i]): i for i in range(base_vocab)}
        self.logp = torch.full((base_vocab,), -math.log(base_vocab), device=self.device)

    def _extend_candidates(self, sequences: torch.Tensor):
        """Heuristic candidate extension: collect frequent n-grams (2..max_len) on GPU and add top-K.
        sequences: [B, L] int64, -1 padded.
        """
        B, L = sequences.shape
        valid = (sequences >= 0)
        # collect 2-grams up to max_len (simplified; real impl would sample/limit)
        grams: List[Tuple[bytes, int]] = []
        for n in range(2, self.max_len+1):
            # Rolling windows via slice stack
            if L < n:
                break
            spans_valid = torch.ones((B, L-n+1), dtype=torch.bool, device=sequences.device)
            for k in range(n):
                spans_valid &= valid[:, k:L-n+1+k]
            if not spans_valid.any():
                continue
            # Represent n-grams as tuples of ints and hash on CPU (proto)
            # (In practice use GPU hashing; for MVP we bring small sample back)
            idx = torch.nonzero(spans_valid, as_tuple=False)
            if idx.numel() == 0:
                continue
            # sample up to 1e6 spans to limit host traffic
            take = min(idx.size(0), 1000000)
            idx = idx[torch.randperm(idx.size(0), device=idx.device)[:take]]
            rows = idx[:,0]
            cols = idx[:,1]
            ng = torch.stack([sequences[rows, cols+k] for k in range(n)], dim=1).to("cpu")
            # Count on CPU for MVP
            from collections import Counter
            c = Counter(map(tuple, ng.tolist()))
            for key, cnt in c.items():
                grams.append((bytes(key), cnt))
        # Add top candidates until hitting target (very rough MVP)
        grams.sort(key=lambda x: -x[1])
        for piece, _ in grams:
            if piece not in self.piece2id:
                new_id = len(self.id2piece)
                self.id2piece[new_id] = piece
                self.piece2id[piece] = new_id
                if len(self.id2piece) >= self.target_vocab:
                    break
        # initialize logp uniform over current pieces
        V = len(self.id2piece)
        self.logp = torch.full((V,), -math.log(V), device=self.device)

    def _forward_backward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log-prob of seq and expected counts per piece via lattice DP.
        seq: [L] int64 token ids 0..255
        Returns: logZ (scalar), expected_counts [V]
        """
        device = self.device
        ids = torch.arange(len(self.id2piece), device=device)
        # Build matches: for each position, which pieces can start here
        L = seq.size(0)
        # For MVP, limit to pieces length<=max_len and that fully fit
        starts = [[] for _ in range(L)]
        for pid, piece in self.id2piece.items():
            plen = len(piece)
            if plen==0 or plen>self.max_len or plen>L:
                continue
            # naive scan; optimize with trie later
            for i in range(0, L-plen+1):
                if torch.all(seq[i:i+plen].cpu() == torch.tensor(list(piece), dtype=torch.long)):
                    starts[i].append(pid)
        # Forward
        logZ = torch.full((L+1,), -1e9, device=device)
        logZ[0] = 0.0
        trans = []  # list of (i, pid, j)
        for i in range(L):
            zi = logZ[i]
            if zi < -1e8:
                continue
            for pid in starts[i]:
                plen = len(self.id2piece[pid])
                j = i + plen
                logZ[j] = torch.logaddexp(logZ[j], zi + self.logp[pid])
                trans.append((i, pid, j))
        # Backward expectations
        if logZ[L] < -1e8:
            # no coverage; fallback to bytes-only segmentation
            exp = torch.zeros((len(self.id2piece),), device=device)
            for i in range(L):
                pid = int(seq[i].item())
                exp[pid] += 1
            return logZ[L], exp
        # compute posterior for each transition
        back = torch.full((L+1,), -1e9, device=device)
        back[L] = 0.0
        # reverse edges
        for i, pid, j in reversed(trans):
            back[i] = torch.logaddexp(back[i], back[j] + self.logp[pid])
        exp = torch.zeros((len(self.id2piece),), device=device)
        logZL = logZ[L]
        for i, pid, j in trans:
            # posterior of using piece pid at i→j is proportional to fwd(i)+logp(pid)+bwd(j) - logZ
            post = (logZ[i] + self.logp[pid] + back[j] - logZL).exp()
            exp[pid] += post
        return logZL, exp

    def fit_epoch(self, batches: List[torch.Tensor]):
        # Ensure we have candidates beyond bytes
        if len(self.id2piece) < self.target_vocab:
            # roughly extend with frequent n-grams from a sample batch
            self._extend_candidates(batches[0])
        # E-step: accumulate expected counts
        V = len(self.id2piece)
        exp_counts = torch.zeros((V,), device=self.device)
        total_logZ = 0.0
        for x in batches:
            x = x.to(self.device)
            valid = x >= 0
            for row in range(x.size(0)):
                seq = x[row][valid[row]].to(self.device)
                _, exp = self._forward_backward(seq)
                exp_counts[:exp.numel()] += exp  # guard
        # M-step: normalize to probabilities
        smoothed = exp_counts + 1e-6
        logp = (smoothed / smoothed.sum()).clamp_min(1e-12).log()
        self.logp = logp
        return {"vocab": len(self.id2piece)}

# =============================== cli_train_bpe.py ============================
from __future__ import annotations
import argparse, glob
from typing import List
import torch
from .cpu_packer import BytePacker
from .datasets import PackedBatcher
from .bpe_trainer import GPUBPETrainer

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", help="Globs of input files (any bytes)")
    ap.add_argument("--merges", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=2048)
    args = ap.parse_args()

    # CPU pack: bytes→ids (fast path)
    packer = BytePacker()
    paths: List[str] = []
    for g in args.data:
        paths.extend(glob.glob(g, recursive=True))
    seqs = [packer.encode_file(p) for p in paths]

    # Batch + pad on CPU (pinned)
    batcher = list(PackedBatcher(seqs, batch_size=args.bs))

    # Train BPE merges on GPU
    trainer = GPUBPETrainer(base_vocab=256, merges=args.merges)
    meta = trainer.fit(batcher, log_every=100)
    trainer.save("./bpe_out")
    print(meta)


# =============================== autoscaler.py ==============================
from __future__ import annotations
import time, os
from dataclasses import dataclass
from typing import Optional

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

import torch

@dataclass
class ScaleState:
    batch_size: int
    cpu_workers: int
    h2d_mb: int

class AutoScaler:
    """Adaptive autoscaler with ~80% utilization ceilings for CPU+GPU.

    • Probes CPU cores, system RAM, GPU VRAM, and (optionally) live utilization.
    • Suggests: batch size, # dataloader workers, and H2D staging size (MB).
    • Monitors OOMs/slow steps and backs off; if headroom grows, scales up.
    """
    def __init__(self,
                 target_util: float = 0.80,
                 min_bs: int = 256,
                 max_bs: int = 8192,
                 min_workers: int = 2,
                 max_workers: Optional[int] = None,
                 init_h2d_mb: int = 512,
                 device: Optional[str] = None):
        self.tu = float(max(0.1, min(target_util, 0.95)))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_bs, self.max_bs = min_bs, max_bs
        self.min_workers = min_workers
        self.max_workers = max_workers or max(min_workers, os.cpu_count() or 4)
        self.state: Optional[ScaleState] = None

    def _gpu_caps(self):
        if self.device == "cpu" or not torch.cuda.is_available():
            return 0, 0
        free, total = torch.cuda.mem_get_info()
        return int(free), int(total)

    def _cpu_caps(self):
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
        cores, mem_free, mem_total, cpu_util = self._cpu_caps()
        gpu_budget = int(total * self.tu)
        gpu_free_budget = max(0, min(free, gpu_budget))
        bytes_per_ex = int(token_bytes_per_example * 1.2) or 1024
        bs_gpu = max(self.min_bs, min(self.max_bs, max(1, gpu_free_budget // bytes_per_ex)))
        workers = max(self.min_workers, int((cores * self.tu)))
        workers = min(workers, self.max_workers)
        h2d_mb = max(256, min(4096, int((mem_free * self.tu) / (1024*1024*8))))
        self.state = ScaleState(batch_size=bs_gpu, cpu_workers=workers, h2d_mb=h2d_mb)
        return self.state

    def feedback(self, step_time_s: float | None = None, oom: bool = False):
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

# =============================== bpe_trainer.py (autoscale-ready) ==============================
from __future__ import annotations
import os, json, math, time
from typing import List, Tuple, Optional
import torch
from .utils import count_pairs, apply_merge_once
from .autoscaler import AutoScaler

class GPUBPETrainer:
    def __init__(self, base_vocab: int = 256, merges: int = 50000, device: str | None = None,
                 autoscaler: Optional[AutoScaler] = None):
        self.base_vocab = base_vocab
        self.target_merges = merges
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = base_vocab
        self.merges: List[Tuple[int,int]] = []
        self.autoscaler = autoscaler or AutoScaler()

    def fit(self, batches: List[Tuple[torch.Tensor, torch.Tensor]], log_every: int = 100):
        step = 0
        while step < self.target_merges:
            state = self.autoscaler.suggest(token_bytes_per_example=int(8 * 1024))
            t0 = time.time()
            # 1) count pairs globally
            global_pairs = None
            global_counts = None
            for (x_cpu, v_cpu) in batches:
                try:
                    x = x_cpu.to(self.device, non_blocking=True)
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        self.autoscaler.feedback(oom=True)
                        continue
                    else:
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
            # 3) apply merge to all batches
            new_batches = []
            oom_seen = False
            for (x_cpu, v_cpu) in batches:
                try:
                    x = x_cpu.to(self.device, non_blocking=True)
                    v = v_cpu.to(self.device, non_blocking=True)
                    x2, v2 = apply_merge_once(x, v, a_id, b_id, new_id)
                    x2_cpu = x2.to("cpu", non_blocking=True, copy=True).pin_memory()
                    v2_cpu = v2.to("cpu", non_blocking=True, copy=True).pin_memory()
                    new_batches.append((x2_cpu, v2_cpu))
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        oom_seen = True
                        torch.cuda.empty_cache()
                        # fallback: keep original batch for now
                        new_batches.append((x_cpu, v_cpu))
                    else:
                        raise
            batches = new_batches
            step += 1
            self.autoscaler.feedback(step_time_s=time.time()-t0, oom=oom_seen)
        return {
            "base_vocab": self.base_vocab,
            "vocab_size": self.vocab_size,
            "merges": self.merges,
        }

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        meta = {
            "base_vocab": self.base_vocab,
            "vocab_size": self.vocab_size,
            "merges": self.merges,
        }
        with open(os.path.join(out_dir, "bpe_merges.json"), "w") as f:
            json.dump(meta, f)
        print(f"Saved merges → {out_dir}/bpe_merges.json")

# =============================== cli_train_bpe.py (autoscale-ready) ============================
from __future__ import annotations
import argparse, glob
from typing import List
import torch
from .cpu_packer import BytePacker
from .datasets import PackedBatcher
from .bpe_trainer import GPUBPETrainer
from .autoscaler import AutoScaler

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", help="Globs of input files (any bytes)")
    ap.add_argument("--merges", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=2048, help="Max initial batch size; autoscaler may pick smaller")
    args = ap.parse_args()

    packer = BytePacker()
    paths: List[str] = []
    for g in args.data:
        paths.extend(glob.glob(g, recursive=True))
    seqs = [packer.encode_file(p) for p in paths]

    scaler = AutoScaler(target_util=0.80)
    init = scaler.suggest(token_bytes_per_example=8*1024)
    bs = min(args.bs, init.batch_size)
    batcher = list(PackedBatcher(seqs, batch_size=bs))

    trainer = GPUBPETrainer(base_vocab=256, merges=args.merges, autoscaler=scaler)
    meta = trainer.fit(batcher, log_every=100)
    trainer.save("./bpe_out")
    print(meta)
