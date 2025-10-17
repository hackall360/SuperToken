# Hybrid Tokenizer — Full Project Plan

This plan is a complete, hand‑off‑ready blueprint for building a **hybrid tokenizer** system that supports **CPU and GPU** training, **BPE + Unigram** algorithms, an optional **hybrid schedule**, **compact AST / code-aware mode**, **multi‑GPU** scaling with lease scheduling, **autoscaling**, **checkpointing**, **benchmarks**, and **embedding model integration**. It is written so a coding agent can implement it end‑to‑end on consumer hardware while preserving quality and reproducibility.

---

## 0) Scope at a Glance

**In scope**
- Trainers: BPE, Unigram, and Hybrid (BPE→Unigram cycles) with identical outputs on CPU/GPU (bit‑for‑bit determinism where practical).
- Backends: CPU (NumPy/PyTorch CPU), GPU (PyTorch CUDA + custom CUDA/Triton kernels).
- Multi-GPU: NCCL + lease-based work scheduling; heterogeneous GPU support.
- Streaming data ingestion with compression (none/zstd/lz4), memory‑mapping, backpressure.
- Autoscaler maintaining ~80% device utilization target.
- Checkpointing & resume (trainer state + autoscaler + dataset cursors).
- Configurable privacy hardening (tie‑break randomization seed, merge-rule hashing redaction for export).
- Code-aware mode with **compact AST**, **symbol tables**, and optional **meta‑token compression**.
- Embedding workflow: export for downstream models; optional co‑training & pruning.
- CLI-first UX; Python API parity.
- Comprehensive tests and benchmarks.

**Out of scope (for now)**
- Training an LLM; this project focuses on tokenization + embeddings export.
- Non‑CUDA GPUs (e.g., ROCm) – future stretch goal.
- On‑device inference tokenization (runtime tokenization for user prompts) – we export artifacts compatible with common runtimes instead.

---

## 1) Success Criteria (KPIs)

1. **Throughput (training):** ≥ 8k tokens/s on a single RTX 4070‑class card with a 50k merge run (representative corpora), scaling to ≥ 85% efficiency using 2 heterogeneous GPUs.
2. **Parity:** CPU vs GPU trainers produce **identical vocab artifacts** within deterministic mode (fixed seeds, fixed order) for the same config.
3. **Stability:** No OOM with default autoscaler at 80% target on 8–16 GB VRAM devices; graceful backoff logged.
4. **Resume:** Interrupt + resume yields the same final vocab as uninterrupted run.
5. **Code mode:** Compact‑AST tokenization reduces code token counts by ≥ 25% vs baseline subword tokenization on a mixed code corpus while preserving lossless round‑trip (via symbol table + sidecar dictionary).
6. **Bench Suite:** Benchmark harness outputs JSON telemetry and plots; CI asserts schema stability and scaling thresholds.
7. **Docs & CLI:** New users can train, benchmark, export, and evaluate with **three copy‑paste commands** from README/CLI docs.

---

## 2) Architecture Overview

### 2.1 Core Components
- **Datasets & Streaming**: iterable corpora, compression adapters, memory‑mapping, prefetch workers, bounded queues, backpressure.
- **Packers**: contiguous sequence builders (device‑aware), BOS/EOS insertion, padding rules.
- **Autoscaler**: utilization‑aware batch chooser (PID‑style or EWMA), CPU and GPU modes.
- **Trainer Interface**: common lifecycle (`train()`, `save()`, `state_dict()`), telemetry hooks.
- **Kernels**: CUDA/Triton kernels for pair‑counts, RLE/histograms, EM updates; CPU fast‑paths.
- **Distributed Runtime**: NCCL initialization + **lease queue** for heterogeneous GPUs.
- **Checkpointing**: trainer state, autoscaler state, dataset cursors, RNG seeds.
- **Exporters**: standard artifacts (merges.txt, vocab.json, model.json, unigram.prob), hybrid manifest, code‑mode sidecars (symbol table, meta‑token dict).
- **CLI**: `train-bpe`, `train-unigram`, `train-hybrid`, `benchmark`, `export`, `evaluate`, `resume`.

### 2.2 Data Flow
```
[Shard discovery] -> [Streaming I/O] -> [Packing] -> [Autoscaled Batches]
      -> [Trainer Kernels] -> [Telemetry + Checkpointing] -> [Export Artifacts]
```

### 2.3 Devices & Scaling
- **CPU mode:** vectorized NumPy/PyTorch ops; thread pools; identical logic as GPU.
- **GPU mode (single):** kernels for counting, scoring, EM; device‑resident tensors; minimize host↔device hops.
- **GPU mode (multi):** per‑rank trainers + **lease queue**; NCCL all‑reduce for histograms; rank 0 notary for leases; peer‑to‑peer where available.

---

## 3) Algorithms

### 3.1 BPE (Byte Pair Encoding)
- **Input:** byte sequences (UTF‑8), BOS/EOS optional.
- **Loop (M merges):**
  1. Count adjacent token pairs (packed‑key trick → sort + run‑length encode).
  2. Select highest‑frequency pair (stable/deterministic tie‑break; optional randomization for privacy).
  3. Apply merge to tokenization; update tokens.
  4. Emit telemetry (pairs/s, peak VRAM, batch latency).
- **GPU path:** fused kernels for pair extraction → sort → RLE; device histogram; minimal host sync.
- **CPU path:** vectorized pair extraction; radix sort where available; packed 64‑bit keys.
- **Stop conditions:** reached `--merges`, or frequency under threshold.
- **Artifacts:** `merges.txt`, `vocab.json`, `model.json` (SP‑compatible where possible).

### 3.2 Unigram (Kudo & Richardson style)
- **Init vocab:** bytes + seed substrings (up to `--token-bytes`), or compose from short BPE warm‑up.
- **EM training (E‑step, M‑step) for `--epochs`:**
  - E‑step: compute tokenization probabilities, sample segmentations (with subword regularization); accumulate expected counts.
  - M‑step: re‑estimate token probabilities; **prune** those with minimal contribution; renormalize.
- **GPU:** parallel path scoring & soft count accumulation.
- **CPU:** vectorized probability & DP segmentation with bounded candidate windows.
- **Artifacts:** `unigram.vocab`, `unigram.prob`, SP‑compatible exports.

### 3.3 Hybrid Schedule
- **Warm‑start**: run BPE for `k` merges to capture frequent pairs quickly.
- **Refine:** run Unigram EM+pruning for `e` epochs.
- **Iterate (optional):** alternation cycles (small BPE burst → Unigram prune) until target vocab or convergence.
- **Determinism:** same seeds & merge order yield identical vocab across devices.

### 3.4 Diffusion/Entropy Augmentation (optional, experimental)
- For each shard: generate N lightly perturbed variants (character‑level dropout/substitution; whitespace jitter; Unicode normalization variants).
- Compute candidate merges’ entropy reduction across variants; prefer merges with consistent information gain.
- Gate behind `--augmentation none|entropy|diffusion` and `--aug-strength`, with clean fallback to classic BPE/Unigram.

### 3.5 Morphology Pre‑Segmentation (optional)
- Rule‑based normalizer for agglutinative languages (e.g., suffix class mapping, case tags).
- Hook: `--morphology <lang>` enables the pass before packing.
- Guarantee **off** by default to avoid bias; maintain injective mapping (lossless reconstruction).

### 3.6 Code-Aware Mode: Compact AST + Symbol Tables
- **Frontends:** Python (AST), JS/TS (ESTree), others via plugins.
- **Linearization:** produce a compact sequence of **construct tokens** (e.g., `DEF`, `IF`, `FOR`, `CALL`, `RET`) with typed edges.
- **Symbols:** replace identifiers with `SYM<n>` and write a per‑file `symbols.json` sidecar.
- **Meta-token compression:** discover repeated multi‑token patterns and map them to `META<n>` with a `meta_dict.json` (lossless reconstruction).
- **Fallback:** if AST parse fails, default to subword tokenization for that file with a flag in shard metadata.
- **Goal:** reduce token counts and bleed less semantics via names while enabling exact round‑trip.

---

## 4) Autoscaler

- **Inputs:** tokens/s, kernel time, host wait time, VRAM watermark (GPU) or CPU load.
- **Policy:** EWMA + proportional band targeting (e.g., 75–85% util). Cooldown to prevent oscillations.
- **Actions:** adjust batch size; adjust number of in‑flight leases (multi‑GPU); signal backpressure to I/O.
- **Interfaces:** `suggest_batch_size(metrics)`, `update(metrics)`, `state_dict()`. Separate CPU/GPU heuristics under a shared facade.
- **Safety:** max/min batch clamps; OOM catcher that steps back and marks safe range.

---

## 5) Distributed Runtime (Multi‑GPU)

- **Init:** NCCL/Torch Distributed; rank assignment; seed setting.
- **Lease Queue:** rank 0 notary doles out *leases* (chunks) to ranks; faster GPUs consume more leases (work‑stealing).
- **Synchronization:** periodic all‑reduce on histograms / sufficient statistics; strict shape agreement.
- **Heterogeneous weights:** per‑device target share (optional) to predict expected scaling; record actual vs expected.
- **Failure Handling:** if a rank drops, shrink world and continue if safe; otherwise checkpoint and exit with a clear message.

---

## 6) Streaming I/O & Packing

- **Inputs:** `--data` globs; codecs `--compression none|zstd|lz4`.
- **Workers:** configurable `--io-workers` prefetch threads/processes into a bounded queue.
- **Memory mapping:** try mmap first; fall back to buffered reads.
- **Packing:** produce contiguous byte tensors with BOS/EOS flags; device‑aware allocation; padding + mask if needed.
- **Backpressure:** autoscaler signals consumer pace; queue throttles prefetch to avoid VRAM spikes.

---

## 7) Checkpointing & Reproducibility

- **State:** trainer internals (merges, probabilities, vocab tables), autoscaler, dataset cursors, RNG seeds, CLI config.
- **Cadence:** `--checkpoint-every N` batches or `--time-minutes`.
- **Resume:** auto‑discover latest checkpoint; verify config compatibility (warn on deltas with a diff printout).
- **Determinism:** fixed seed → identical merge order / prunes; option `--deterministic` to pin kernels where supported.

---

## 8) Privacy & Safety

- **BPE leak mitigation:** `--privacy tie-randomize|hash-merges|none`. Hash exported merges or randomize tie‑breaks.
- **PII hygiene hooks:** optional redactors before packing (off by default).
- **Telemetry hygiene:** avoid logging raw text; log counts/IDs only.

---

## 9) CLI Design

### 9.1 Commands
- `train-bpe` — Train BPE tokenizer.
- `train-unigram` — Train Unigram tokenizer.
- `train-hybrid` — Hybrid scheduler (BPE warm‑start → Unigram refine).
- `benchmark` — Synthetic + real corpora sweep; scaling and parity checks.
- `export` — Convert internal artifacts to SentencePiece/HF-compatible files.
- `evaluate` — Report token count compression, OOV stats, morphology purity, code‑mode compression ratio.
- `resume` — Shortcut to resume a previous training run.

### 9.2 Shared Flags (examples)
```
--data "data/**/*.txt" --compression zstd --io-workers 4 --prefetch-batches 8
--device cpu|cuda|cuda:0,1 --target-util 0.80 --checkpoint-dir ./ckpts --checkpoint-every 2000
--bos 1 --eos 2 --token-bytes 8192 --log-every 50 --out-dir ./artifacts/run1
```

### 9.3 Algorithm‑specific
- BPE: `--merges 50000 --min-pair-freq 2 --deterministic`
- Unigram: `--vocab-size 50000 --epochs 3 --subword-reg strength --seed 123`
- Hybrid: `--bpe-warm-merges 5000 --unigram-epochs 2 --cycles 3`

### 9.4 Code‑mode & Morphology
- Code mode: `--code-mode on --code-lang py,ts --meta-compress on --meta-max-len 16`
- Morphology: `--morphology tr --morph-case-token on --morph-affix-class on`

---

## 10) Exports & Formats

- **BPE**: `merges.txt`, `vocab.json`, `model.json` (+ README notes).
- **Unigram**: `unigram.vocab`, `unigram.prob` (SP format), `model.json`.
- **Hybrid**: `hybrid_manifest.json` (records the BPE→Unigram schedule, seeds, and hashes of stage outputs).
- **Code mode**: `symbols.json` (per‑file symbol table), `meta_dict.json` (meta‑token dictionary), `code_manifest.json` (round‑trip metadata).

---

## 11) Metrics & Benchmarking

- **Throughput**: tokens/s, per‑stage timings, occupancy (GPU) / CPU load.
- **Memory**: VRAM/host peak, fragmentation counters.
- **Compression**: tokens per byte, average tokens per document; code‑mode reduction %.
- **Quality (proxy)**: morphology purity (if enabled), OOV rate, token/type count distribution.
- **Scaling**: single vs multi‑GPU efficiency; heterogeneous speedup vs expected.
- **Stability**: OOM recoveries, autoscaler oscillation score.

**Artifacts**
- `bench_*.json` snapshots (config + telemetry).
- `trend_table.md`, `trend_plot.png` for longitudinal runs.

---

## 12) Testing Strategy & Acceptance Criteria

### 12.1 Unit Tests
- Pair counting: GPU vs CPU parity on synthetic corpora (random + adversarial).
- Merge application: deterministic order; tie‑break logic; end‑to‑end merges = expected.
- Unigram EM: known toy corpora produce expected prunes/probabilities.
- Code mode: round‑trip reconstruction equals source (AST → compact → expand).
- Exporters: files parse in SentencePiece / HF; schema validation.
- Autoscaler: monotonic convergence to target band under synthetic load ramps.
- Checkpoint/Resume: interrupted runs produce same final vocab as uninterrupted.
- Lease Queue: no deadlocks; fairness under heterogeneous latencies.

### 12.2 Integration Tests
- CLI smoke: `train-bpe`, `train-unigram`, `train-hybrid` complete on tiny corpora.
- Multi‑GPU: `torchrun` 2×GPUs synthetic job reaches ≥ 85% expected scaling.
- Streaming: zstd and lz4 decode; mmap vs buffered fallback correctness.

### 12.3 Performance Tests
- Single‑GPU throughput ≥ target; VRAM within cap; autoscaler steady.
- CPU throughput baseline recorded; identical artifacts vs GPU (deterministic).

**Acceptance**: All unit + integration tests pass; performance thresholds hit; artifacts load in downstream toolchains; docs & CLI examples usable copy‑paste.

---

## 13) Implementation Phases

### Phase A — Infrastructure & Parity
1. **Trainer Interface + CLI skeleton**
   - Tasks: define abstract `BaseTrainer`, wire `main.py` commands, add telemetry hooks.
   - Acceptance: CLI prints config + progress; no‑op trainer runs.
2. **CPU Fast‑Path**
   - Tasks: implement packed‑key pair counting, RLE histogram; Unigram CPU EM; deterministic merges/prunes.
   - Acceptance: pass unit tests; parity against “golden” fixtures.
3. **GPU Kernels (single GPU)**
   - Tasks: CUDA/Triton for pair extraction → sort → RLE; EM updates; device‑resident packers.
   - Acceptance: ≥ 3× speedup vs CPU baseline on 4070‑class; identical artifacts in deterministic mode.
4. **Streaming I/O + Autoscaler**
   - Tasks: prefetch workers; bounded queue; autoscaler (CPU+GPU policies); backpressure signals.
   - Acceptance: stable utilization around target; graceful OOM backoff; metrics logged.
5. **Checkpoint/Resume**
   - Tasks: serialize trainer + autoscaler + dataset cursor + RNG seeds; loader with config diff.
   - Acceptance: resume matches uninterrupted outputs byte‑for‑byte.

### Phase B — Multi‑GPU & Hybrid
6. **Distributed Runtime + Lease Queue**
   - Tasks: rank 0 notary; lease messages; all‑reduce histograms; fault handling.
   - Acceptance: ≥ 85% efficiency with 2 GPUs; fairness under heterogeneity.
7. **Hybrid Scheduler**
   - Tasks: BPE warm‑start → Unigram refine; optional cycles; config + manifest export.
   - Acceptance: artifacts stable; hybrid improves compression or quality proxies vs single algorithm.

### Phase C — Extensions & Tooling
8. **Code‑Aware Mode (Compact AST + Symbols + Meta‑tokens)**
   - Tasks: parsers for Python + JS/TS; linearizer; symbol tables; meta‑token discovery; round‑trip expander.
   - Acceptance: ≥ 25% token reduction on code corpora; exact reconstruction; fallback path verified.
9. **Morphology Pre‑Segmentation**
   - Tasks: plug‑in architecture; TR demo (case token, affix classes); injective mapping.
   - Acceptance: togglable, off by default; documented; round‑trip tested.
10. **Embedding Workflow + Pruning**
   - Tasks: export vocab for embedding training; optional co‑training scaffolding; pruning unused tokens; de‑dup near‑duplicates.
   - Acceptance: exports load into simple embedding trainer; pruning demonstrably reduces params without accuracy drop on a small eval.

### Phase D — Privacy, Docs, Bench, CI
11. **Privacy Hardening**
   - Tasks: tie‑break randomization seed; merge hashing; export redaction modes.
   - Acceptance: modes switchable and documented; unit tests for determinism vs randomization.
12. **Benchmarks & Trend Reports**
   - Tasks: synthetic + real dataset scenarios; trend plot pipeline; scaling JSON schema + CI checks.
   - Acceptance: artifacts generated; CI passes with threshold assertions.
13. **Documentation & Examples**
   - Tasks: README quickstart; CLI guide; API reference; Cookbook (morphology, code‑mode, hybrid).
   - Acceptance: three copy‑paste examples succeed on fresh envs.

---

## 14) Risks & Mitigations

- **VRAM pressure / OOM**: autoscaler backoff; chunked histograms; fused kernels; stream‑aware allocators.
- **Parity drift CPU↔GPU**: single source of truth for merges/prunes; determinism test suite; seed pinning.
- **Distributed flakiness**: lease timeouts; heartbeats; resume on failure; robust NCCL init with env validation.
- **AST fragility**: fallback to subword on parse errors; per‑language plugins; golden round‑trip tests.
- **Privacy leakage via merges**: tie‑randomization + hashing export modes; documented trade‑offs.

---

## 15) Deliverables & Repo Layout

**Deliverables**
- Tokenizer artifacts for BPE/Unigram/Hybrid; code‑mode sidecars; hybrid manifest.
- Benchmarks JSON + plots; trend table.
- Full docs (Architecture, CLI, API, Cookbook).
- Test suite + CI.

**Suggested Tree**
```
gpu_tokenizer/
  trainers/ (bpe.py, unigram.py, hybrid.py, base.py)
  kernels/ (cuda_kernels.cu, triton_kernels.py)
  io/ (adapters.py, memmap.py, workers.py)
  packers/ (cpu.py, cuda.py)
  dist/ (lease_queue.py, runtime.py)
  code_mode/ (py_frontend.py, ts_frontend.py, linearizer.py, symbols.py, meta_compress.py)
  morphology/ (tr_rules.py, common.py, plugins/)
  export/ (sp.py, hf.py, manifest.py)
  autoscale/ (controller.py, metrics.py)
  utils/ (logging.py, determinism.py, privacy.py)
benchmarks/ (runner.py, configs/, samples/, reports/)
docs/ (README, cli.md, api.md, architecture.md, cookbook/)
tests/ (unit/, integration/, performance/)
main.py (CLI entrypoint)
```

---

## 16) Operating Targets (Consumer Hardware)

- **GPU**: RTX 3060–4090 class; VRAM 8–24 GB.
- **CPU**: 6–16 cores; 16–64 GB RAM.
- **OS**: Linux (primary), Windows WSL2 (best effort), macOS CPU‑only fallback.
- **Dependencies**: Python ≥ 3.10, PyTorch (CUDA), Triton (optional), NumPy, zstandard, lz4, pandas/matplotlib for benchmarks.

---

## 17) Quickstart Recipes

**Train BPE (GPU)**
```
python main.py train-bpe \
  --data "data/**/*.txt" --merges 50000 \
  --token-bytes 8192 --target-util 0.80 \
  --checkpoint-dir ./artifacts/bpe_ckpts \
  --out-dir ./artifacts/bpe
```

**Train Unigram (CPU)**
```
python main.py train-unigram \
  --device cpu --vocab-size 50000 --epochs 3 \
  --data "data/**/*.txt" \
  --out-dir ./artifacts/unigram
```

**Hybrid on two GPUs**
```
torchrun --standalone --nproc_per_node 2 \
  python main.py train-hybrid \
  --data "data/**/*.txt" \
  --bpe-warm-merges 5000 --unigram-epochs 2 --cycles 3 \
  --checkpoint-dir ./artifacts/hybrid_ckpts \
  --out-dir ./artifacts/hybrid
```

**Benchmark suite**
```
python main.py benchmark \
  --data "data/**/*.txt" \
  --synthetic-docs 2000 --synthetic-min-len 16 --synthetic-max-len 128 \
  --output-dir ./artifacts/benchmarks
```

---

## 18) Definition of Done

- All phases A–D completed.
- KPIs in §1 met or exceeded.
- Docs and examples tested by a fresh user on CPU‑only and single‑GPU configs.
- Artifacts consumed by at least one external embedding trainer & tokenizer loader.
- CI runs unit/integration/perf smoke tests on every PR with gates on scaling and schema.

---

## 19) Appendix: Configuration Keys (Reference)

**Global**
- `device`, `target_util`, `deterministic`, `seed`, `bos`, `eos`, `token_bytes`, `checkpoint_dir`, `checkpoint_every`, `log_every`

**BPE**
- `merges`, `min_pair_freq`, `privacy_mode (none|tie-randomize|hash-merges)`

**Unigram**
- `vocab_size`, `epochs`, `subword_reg_strength`, `min_prob`, `prune_floor`

**Hybrid**
- `bpe_warm_merges`, `unigram_epochs`, `cycles`

**I/O**
- `data_globs`, `compression`, `io_workers`, `prefetch_batches`

**Distributed**
- `dist` (bool), `gpus` (list), `lease_size`, `rebalance_secs`, `target_chunk_ms`

**Code‑Mode**
- `code_mode`, `code_langs`, `meta_compress`, `meta_max_len`

**Morphology**
- `morphology_lang`, `morph_case_token`, `morph_affix_class`

---

_End of plan._
