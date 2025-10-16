# SuperToken Multi-GPU Integration Plan
*High-performance distributed tokenizer training for heterogeneous NVIDIA GPUs*

---

## Overview

This plan defines the architecture, phases, and step-by-step implementation process for integrating **heterogeneous multi-GPU distributed training** into the SuperToken project.  
The goal is to enable *extremely fast, low-latency communication* between **any NVIDIA compute-compatible, CUDA-capable GPUs**, maximizing throughput even when the GPUs have mismatched capabilities (e.g., RTX 4070 SUPER + RTX 2060).

The system will use:
- **Throughput-proportional sharding**
- **Lease-based work queue (host notary)**
- **Work stealing**
- **EWMA adaptive rebalancing**
- **Asynchronous I/O and compute overlap**

Control stays on the host, data stays on GPUs. No VRAM-atomics or heavy inter-GPU chatter.

---

# Phase 0 – Instrumentation and Preparation

### Goals
- Establish integration points in the SuperToken repo.
- Add lightweight performance metrics to measure throughput.

### Tasks
1. Create new files:
   - `gpu_tokenizer/dist_runtime.py`
   - `gpu_tokenizer/lease_queue.py`
2. Update existing files:
   - `gpu_tokenizer/bpe_trainer.py` → add `TrainerMetricsEWMA`
   - `gpu_tokenizer/utils.py` → add `can_peer()` helper
   - `gpu_tokenizer/io.py` → add chunker factory
   - `gpu_tokenizer/cli_train_bpe.py` → add CLI flags (`--dist`, `--gpus`, etc.)
3. Add timing metrics for:
   - H2D copy, kernels, D2H copy, reduction
4. Ensure CLI shows per-stage timings per iteration.

### Acceptance Criteria
- Single GPU performance unchanged (±1% regression max).
- CLI prints timing metrics cleanly.

---

# Phase 1 – Multi-GPU Launcher (NCCL + Spawn Workers)

### Goals
- Launch one process per GPU.
- Each process (rank) runs independently with its own autoscaler.

### Implementation
- Add distributed setup in `dist_runtime.py` using `torch.multiprocessing.spawn`.
- Initialize `torch.distributed.init_process_group(backend="nccl")`.
- Each rank binds to its CUDA device.
- Add signal handling for clean shutdown.

### Acceptance Criteria
- Multi-GPU launch completes on two GPUs without deadlock.
- Logs show both ranks active.

---

# Phase 2 – Lease-Based Global Work Queue (Host Notary)

### Goals
- Replace static corpus split with a dynamic **ticket dispenser**.
- Minimize synchronization and eliminate collisions.

### Implementation Steps
1. Add `LeaseNotary` class in `lease_queue.py`:
   - Rank 0 owns `next_idx` and `N`.
   - `grant_lease(K)` → returns `(start, end)`.
   - Broadcast small lease messages to workers.
2. Each worker processes `[start:end)` independently.
3. Add requeue mechanism for failed/slow ranks.
4. Integrate `make_chunker()` in `io.py` for creating ~100ms compute chunks.

### Acceptance Criteria
- 0 duplicate work across ranks.
- Fast GPU keeps working; slow GPU contributes without stalling.

---

# Phase 3 – Throughput-Proportional Sharding + Work Stealing

### Goals
- Distribute work according to GPU speed (tokens/sec).
- Implement automatic rebalancing.

### Implementation Steps
1. Calibration round on startup (each GPU measures TPS).
2. Compute weights: `w_i = tps_i / sum(tps_all)`.
3. Determine lease size per GPU: `K_i` ~ proportional to throughput.
4. Use “pull-based” work stealing — faster GPUs request next leases sooner.

### Acceptance Criteria
- Wall clock training speed ≈ sum of GPU throughputs × 0.88.
- No idle time >50 ms on faster GPU.

---

# Phase 4 – Asynchronous I/O and Compute Overlap

### Goals
- Maximize throughput by overlapping copy, compute, and reduce.

### Implementation Steps
1. Two CUDA streams per rank (`stream_io`, `stream_compute`).
2. Double-buffered pinned host memory for I/O.
3. Add `can_peer(src, dst)` in `utils.py`.
4. Enable peer-to-peer transfers (`cudaMemcpyPeerAsync`) if supported.

### Acceptance Criteria
- H2D + D2H time overlaps with compute.
- Disabling overlap reduces throughput significantly.

---

# Phase 5 – Aggregation and Adaptive Cadence

### Goals
- Optimize reduction performance and prevent sync stalls.

### Implementation Steps
1. Locally compact pair keys/counts before reduce.
2. Aggregate every R=8–16 chunks via `dist.all_reduce`.
3. Adaptive cadence: increase R if reduction >15% of total time.

### Acceptance Criteria
- Reduction ≤15% of total step time.
- Results match single-GPU output exactly.

---

# Phase 6 – EWMA Adaptive Rebalance

### Goals
- Rebalance dynamically based on observed throughput.

### Implementation Steps
1. Track tokens/sec via EWMA (`alpha=0.2`).
2. Rebalance every `rebalance_secs` (10s default).
3. Update weights softly: `w_i = 0.5*w_old + 0.5*w_new`.
4. Adjust lease size and inflight limits per rank.

### Acceptance Criteria
- Rebalancing stabilizes within 30 seconds after thermal throttling or load shifts.

---

# Phase 7 – Fault Tolerance and Recovery

### Goals
- Prevent lost work when a GPU fails or slows down.

### Implementation Steps
1. Heartbeat every 2 seconds to rank 0.
2. Rank 0 requeues unacknowledged leases after timeout.
3. Ensure chunk processing is idempotent.

### Acceptance Criteria
- Killing one rank doesn’t cause job failure.
- All chunks are processed once, no duplicates.

---

# Phase 8 – CLI Integration

### Goals
- Simplify distributed execution for users.

### Implementation Steps
- Extend CLI with new args:
  ```bash
  python -m gpu_tokenizer.cli_train_bpe --dist --gpus 0,1 --lease-size 16 --rebalance-secs 10 --target-chunk-ms 100
  ```
- Show per-rank table:
  - GPU ID, TPS, lease size, inflight leases, stage times, ETA.

### Acceptance Criteria
- Works with or without `--dist` flag.
- User-friendly logs.

---

# Phase 9 – Testing and Benchmarking

### Goals
- Validate correctness, scaling, and fault tolerance.

### Implementation Steps
1. Unit tests:
   - Lease uniqueness.
   - Requeue and heartbeat recovery.
   - EWMA logic.
2. Integration tests:
   - Multi-GPU vs single-GPU vocab parity.
3. Benchmarks:
   - Single GPU baseline.
   - Mixed GPUs (4070S + 2060) → ≥88% ideal sum throughput.

### Acceptance Criteria
- All tests pass.
- Benchmark results documented in `docs/performance.md`.

---

# Code Modules and Responsibilities

| Module | Responsibility |
|--------|----------------|
| `dist_runtime.py` | Distributed process launcher, calibration, rebalancing loop |
| `lease_queue.py` | Centralized work lease management (rank 0) |
| `io.py` | Corpus chunker and data pipeline |
| `utils.py` | Peer access checks, distributed utilities |
| `bpe_trainer.py` | GPU kernels, metrics, trainer step integration |
| `cli_train_bpe.py` | CLI flags and runtime configuration |
| `autoscaler.py` | Per-rank dynamic batch adjustment |

---

# Key Design Principles

1. **Control on host, data on GPU.**
2. **Peer access only for bulk data, never for control flow.**
3. **Chunk granularity = 50–150 ms per task on fast GPU.**
4. **Leases amortize comms overhead (8–64 chunks).**
5. **Soft rebalancing avoids oscillations.**
6. **Asymmetric batch sizes per GPU.**
7. **All reduce operations are compact and infrequent.**

---

# Success Metrics

| Metric | Target |
|--------|--------|
| Multi-GPU Speedup | ≥88% of ideal sum of GPU throughputs |
| Fast GPU Idle Time | <2% over steady state |
| Reduction Overhead | ≤15% of step time |
| Rebalance Convergence | <30 seconds |
| Correctness | Identical vocab to single-GPU run |

---

# Deliverables

- [ ] `dist_runtime.py` – distributed runtime system
- [ ] `lease_queue.py` – lease-based work queue
- [ ] `make_chunker()` in `io.py`
- [ ] CLI flags and UX integration
- [ ] Peer access utilities
- [ ] Tests (`tests/test_dist_runtime.py`, `tests/test_lease_queue.py`)
- [ ] `docs/performance.md` benchmark results

---

# Notes and Best Practices

- Avoid global `torch.cuda.synchronize()` in hot loops.
- Use `torch.distributed.barrier()` only for startup/shutdown.
- Validate peer access via `torch.cuda.device_can_access_peer()`.
- Monitor GPU utilization via `nvidia-smi dmon` during tuning.
- For debugging, enable `TORCH_DISTRIBUTED_DEBUG=DETAIL`.

---

# Summary

Once implemented, this system will make SuperToken one of the **fastest GPU-based tokenizer trainers** available, even on mismatched consumer hardware. It scales linearly across devices, self-balances under variable loads, and maintains correctness while pushing hardware to its limits.
