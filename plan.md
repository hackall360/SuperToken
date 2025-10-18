# Hybrid Tokenizer — Complete Implementation Plan (v2)

This plan supersedes earlier drafts. It is a hand‑off‑ready blueprint for a **hybrid tokenizer** that supports **CPU and GPU** training, **BPE + Unigram** algorithms, a **hybrid schedule**, **compact AST / code‑aware mode**, **multi‑GPU** scaling with a lease scheduler, **autoscaling**, **checkpointing/resume**, **privacy hardening**, **benchmarks**, and an **evaluate** command. It includes the gap‑closure items identified in the latest repo audit and specifies acceptance tests so a coding agent can implement everything end‑to‑end on consumer hardware.

---

## 0) Scope at a Glance

**In scope**
- Trainers: BPE, Unigram, and Hybrid (BPE→Unigram cycles) with **CPU** and **GPU** backends producing identical artifacts in deterministic mode.
- Multi‑GPU: NCCL + **lease‑based** work scheduling; heterogeneous GPU support.
- Streaming data ingestion with compression (none/zstd/lz4), memory‑mapping, prefetch, backpressure.
- Autoscaler (CPU/GPU aware) maintaining ~80% target utilization.
- Checkpointing & **resume** for all trainers.
- Privacy features: tie‑break randomization, hash‑merges, logfile hygiene.
- Code‑aware mode: **compact AST**, **symbol tables**, **meta‑token compression** with exact round‑trip.
- Morphology pre‑segmentation plug‑ins (opt‑in).
- Embedding workflow: artifact export; optional near‑duplicate dedupe and (opt‑in) co‑training.
- CLI-first UX; Python API parity; CI+benchmarks.

**Out of scope (for now)**
- ROCm path; non‑CUDA accelerators.
- Full LLM training; this project focuses on tokenization and embedding artifacts.

---

## 1) Success Criteria (KPIs)

1. **Throughput (single‑GPU):** ≥ 8k tokens/s on an RTX 4070‑class GPU for a 50k merge BPE run (representative corpus).
2. **Scaling (2 GPUs):** ≥ 85% efficiency on heterogeneous pair (e.g., 4070 + 3060) with the lease queue runtime.
3. **Parity (CPU↔GPU):** Deterministic mode yields byte‑for‑byte identical artifacts for the same config/seed.
4. **Stability:** Default autoscaler avoids OOM on 8–16 GB VRAM devices and backs off cleanly.
5. **Resume:** Interrupt + resume produces the same final artifacts as uninterrupted runs (BPE/Unigram/Hybrid).
6. **Code‑mode:** ≥ 25% token reduction vs baseline subword on a mixed code corpus while preserving exact round‑trip.
7. **Evaluate CLI:** Generates a JSON report with compression, OOV, morphology purity (if enabled), and code‑mode reduction; reproducible on CI.
8. **Docs:** Three copy‑paste paths (BPE, Unigram, Hybrid) work on fresh envs (CPU‑only and single‑GPU).

---

## 2) Architecture Overview

### 2.1 Components
- **Datasets & Streaming:** iterable corpora, compression adapters, mmap, prefetch workers, bounded queues, backpressure hooks.
- **Packers:** device‑aware contiguous sequence builders; BOS/EOS; padding masks (if needed).
- **Autoscaler:** EWMA/band controller with CPU vs GPU heuristics.
- **Trainers:** common lifecycle (`train()`, `save()`, `state_dict()/load_state_dict()`), telemetry hooks.
- **Kernels:** CUDA/Triton for pair extraction→sort→RLE/histograms and Unigram EM updates; CPU fast‑paths.
- **Distributed Runtime:** NCCL init + **lease queue** with rank‑0 notary; all‑reduce of histograms/sufficient stats.
- **Checkpointing:** trainer + autoscaler + dataset cursor + RNG seeds + CLI config.
- **Exports:** HF/SP‑compatible vocab/merges/prob files; hybrid manifest; code‑mode sidecars.
- **CLI:** `train-bpe`, `train-unigram`, `train-hybrid`, `benchmark`, `export`, **`evaluate`**, `resume-*`.

### 2.2 Data Flow
```
[Shard discovery] -> [Streaming I/O] -> [Packing] -> [Autoscaled Batches]
      -> [Trainer Kernels] -> [Telemetry + Checkpointing] -> [Exports]
```

### 2.3 Devices & Scaling
- **CPU mode:** vectorized NumPy/PyTorch ops; thread pools; identical semantics to GPU.
- **GPU (single):** device‑resident tensors; fused kernels; minimal host↔device hops.
- **GPU (multi):** lease queue keeps fast devices busy; periodic all‑reduce; P2P when available.

---

## 3) Algorithms

### 3.1 BPE
- **Loop:** count adjacent pairs → select highest frequency (stable tie‑break; optional randomized ties) → apply merge → repeat to `--merges` or floor.
- **GPU:** packed 64‑bit pair keys, radix sort, RLE histogram; on‑device merge application where practical.
- **CPU:** identical logic with vectorized extraction and memory‑efficient histograms.
- **Artifacts:** `merges.txt`, `vocab.json`, `model.json` (SP‑compatible when possible).

### 3.2 Unigram
- **Init:** bytes + seed substrings (up to `--token-bytes`) or short BPE warm‑up.
- **EM:** E‑step DP/semi‑Viterbi with subword regularization → M‑step probability re‑estimation → prune floor.
- **GPU:** parallel path scoring and expected‑count accumulation; device‑side pruning candidate selection.
- **CPU:** DP with bounded window; vectorized probability updates.
- **Artifacts:** `unigram.vocab`, `unigram.prob`, SP exports.

### 3.3 Hybrid Schedule
- **Warm‑start:** BPE for `k` merges to capture frequent pairs.
- **Refine:** Unigram EM/prune for `e` epochs.
- **Cycle (optional):** small BPE bursts + Unigram pruning until target vocab or convergence.
- **Manifest:** `hybrid_manifest.json` records schedule, seeds, and hashes of stage outputs.

### 3.4 Experimental Augmentation (opt‑in)
- **Modes:** `--augmentation none|entropy|diffusion` with `--aug-strength`.
- **Entropy:** create N light variants of each sample; prefer merges reducing entropy across variants.
- **Diffusion (alias):** N stochastic character/space jitters; evaluate robustness of piece discovery.
- **Off by default;** must not alter outputs when disabled.

### 3.5 Morphology Pre‑Segmentation (opt‑in)
- **Plugins:** language‑specific case/affix tags; injective mapping for exact round‑trip.
- **Flags:** `--morphology <lang>`, `--morph-case-token`, `--morph-affix-class`.

### 3.6 Code‑Aware Mode (Compact AST)
- **Frontends:** Python (AST), JS/TS (ESTree); plugin interface for others.
- **Linearization:** structural tokens (`DEF`, `IF`, `CALL`, …) + typed edges; identifiers → `SYM<n>` stored in `symbols.json`.
- **Meta‑token compression:** frequent multi‑token patterns → `META<n>`; dictionary `meta_dict.json`.
- **Fallback:** parse failures drop to subword with a flag in shard metadata.
- **Guarantee:** exact round‑trip reconstruction.

---

## 4) Autoscaler

- **Inputs:** tokens/s, kernel time, host wait time, VRAM watermark (GPU) / CPU load.
- **Policy:** target band (e.g., 75–85%); cooldown to avoid oscillation; min/max clamps.
- **Actions:** batch size adjustments; in‑flight leases (multi‑GPU); backpressure signals to I/O.
- **State:** `state_dict()`/`load_state_dict()` persisted in checkpoints.

---

## 5) Distributed Runtime (Multi‑GPU)

- **Init:** `torch.distributed` NCCL; ranks, seeds, env validation.
- **Lease Queue:** rank‑0 notary serves leases; faster devices fetch more work (work‑stealing).
- **Sync:** periodic all‑reduce of histograms/sufficient stats with strict shape agreement.
- **Heterogeneous weights:** optional device weights to compute expected scaling for reports.
- **Failure:** rank drop → shrink if safe; otherwise checkpoint & fail with actionable message.

---

## 6) Streaming I/O & Packing

- **Compression:** `--compression none|zstd|lz4`.
- **Workers:** `--io-workers` prefetchers into bounded queue.
- **mmap:** try mmap; fallback buffered reads.
- **Packing:** device‑aware contiguous tensors; BOS/EOS; mask/pad rules.
- **Backpressure:** autoscaler feedback to throttle prefetch on VRAM spikes.

---

## 7) Checkpointing & Resume

- **State:** trainer internals (merges/probabilities), autoscaler, dataset cursors, RNG seeds, CLI config.
- **Cadence:** `--checkpoint-every N` batches or `--time-minutes`.
- **Resume:** auto‑load latest snapshot; config diff printed with warnings for safe deltas.
- **Determinism:** `--deterministic` pins seeds and stable merges/prunes where supported.

---

## 8) Privacy & Safety

- **BPE leakage:** `--privacy none|tie-randomize|hash-merges` to mitigate mixture inference from merge lists.
- **Log hygiene:** avoid raw text in logs; only counts/IDs; redactable telemetry.
- **PII hooks:** optional redactors (off by default).

---

## 9) CLI Design

### 9.1 Commands
- `train-bpe` — Train BPE.
- `train-unigram` — Train Unigram.
- `train-hybrid` — Hybrid schedule.
- `benchmark` — Synthetic/real sweep + scaling.
- `export` — HF/SP exports + embedding tables.
- **`evaluate`** — Report compression/OOV/morphology/ code‑mode reduction.
- **`resume-bpe`** — Resume a BPE run from checkpoints.
- `resume` flags — Add `--resume-from`/`--checkpoint-*` to Unigram and Hybrid.

### 9.2 Shared Flags (examples)
```
--data "data/**/*.txt" --compression zstd --io-workers 4 --prefetch-batches 8
--device cpu|cuda|cuda:0,1 --target-util 0.80 --deterministic --seed 123
--checkpoint-dir ./ckpts --checkpoint-every 2000
--bos 1 --eos 2 --token-bytes 8192 --log-every 50 --out-dir ./artifacts/run1
```

### 9.3 Algorithm‑specific
- **BPE:** `--merges 50000 --min-pair-freq 2 --privacy tie-randomize`
- **Unigram:** `--vocab-size 50000 --epochs 3 --subword-reg-strength 0.1`
- **Hybrid:** `--bpe-warm-merges 5000 --unigram-epochs 2 --cycles 3`

### 9.4 Code‑mode & Morphology
- **Code‑mode:** `--code-mode on --code-lang py,ts --meta-compress on --meta-max-len 16`
- **Morphology:** `--morphology tr --morph-case-token on --morph-affix-class on`

### 9.5 Evaluate (new)
```
python main.py evaluate \
  --artifacts ./artifacts/bpe \
  --sample "eval/**/*.txt" \
  --report ./artifacts/eval_report.json
```
**Report fields:** tokens_per_byte, oov_rate, avg_tokens_per_doc, length_histogram, code_mode_reduction_pct, morphology_purity (if enabled), notes.

---

## 10) Exports & Formats

- **BPE:** `merges.txt`, `vocab.json`, `model.json` (+ README notes).
- **Unigram:** `unigram.vocab`, `unigram.prob`, `model.json`.
- **Hybrid:** `hybrid_manifest.json` (schedule + hashes + seeds).
- **Code‑mode:** `symbols.json`, `meta_dict.json`, `code_manifest.json` (round‑trip metadata).
- **Embedding:** `embeddings.npy`/`.pt`, `token_to_id.json`, `pruning.json` (if dedupe/pruning applied).

---

## 11) Metrics & Benchmarking

- **Throughput:** tokens/s, stage timings, occupancy (GPU) / load (CPU).
- **Memory:** VRAM/host peak; allocator stats.
- **Compression:** tokens/byte, tokens/doc; code‑mode reduction %.
- **Quality proxies:** OOV rate; morphology purity; token/type distribution.
- **Scaling:** efficiency vs expected (heterogeneous weights supported).
- **Stability:** OOM recoveries; autoscaler oscillation score.

**Artifacts:** `bench_*.json`, `trend_table.md`, `trend_plot.png`.

---

## 12) Testing Strategy & Acceptance Criteria

### 12.1 Unit
- Pair counting parity (GPU vs CPU); adversarial corpora; deterministic ties.
- Merge application; stable order; `--privacy tie-randomize` tested for non‑determinism where expected.
- Unigram EM correctness on toy corpora.
- Code‑mode exact round‑trip; meta‑token encode/decode.
- Exports parse in SP/HF; schema validation.
- Autoscaler convergence under synthetic ramps.
- Checkpoint/resume: same final artifacts as uninterrupted runs.
- Evaluate metrics: fixed small sample → fixed JSON with golden snapshot.

### 12.2 Integration
- CLI smoke (`train-bpe`, `train-unigram`, `train-hybrid`) on tiny corpora (CPU & GPU).
- Multi‑GPU: 2× GPUs synthetic job reaches ≥ 85% efficiency.
- Streaming: zstd/lz4 decode; mmap vs buffered fallback.

### 12.3 Performance
- Single‑GPU ≥ target; autoscaler stable; no OOM with defaults.
- CPU run completes and matches GPU artifacts in deterministic mode.

**Acceptance:** All tests pass; performance thresholds hit; docs examples work fresh; evaluate report produced.

---

## 13) Implementation Phases (with gap‑closure tasks)

### Phase A — Core & Parity
1. **Trainer Interface + CLI skeleton** → _done or confirm_
2. **CPU Fast‑path for BPE/Unigram** → _done or confirm_
3. **GPU Kernels (single‑GPU)** → _done or confirm_
4. **Streaming I/O + Autoscaler** → _done or confirm_
5. **Checkpoint/Resume base** → _done or confirm_

### Phase B — Distributed & Hybrid
6. **Distributed runtime + Lease queue** → _done or confirm_
7. **Hybrid scheduler + manifest** → _done or confirm_

### Phase C — Extensions
8. **Code‑mode (AST + symbols + meta)** → _done or confirm_
9. **Morphology plug‑ins** → _done or confirm_
10. **Embedding export + optional dedupe** → _implement dedupe flag `--dedupe-similarity τ` (merge near‑duplicate tokens; record in `pruning.json`)._

### Phase D — Gap Closure (NEW)
11. **`evaluate` CLI**  
    - Add subparser + `gpu_tokenizer/evaluate.py`.  
    - Computes compression/OOV/morphology/code‑mode metrics; writes JSON.  
    - Tests: `tests/test_cli_evaluate.py` (golden snapshot).
12. **Resume wiring for Unigram & Hybrid**  
    - Add `--resume-from`, `--checkpoint-dir`, `--checkpoint-every` to `train-unigram`; wire into epoch loop.  
    - Add `--resume-from` to `train-hybrid`.  
    - Tests: CLI resume smoke + equality.
13. **`resume-bpe` parser**  
    - Wire `_cmd_resume_bpe` into `build_parser()` and mirror key flags.  
    - Docs updated.
14. **Experimental augmentation toggles**  
    - Add `--augmentation/--aug-strength`; implement “entropy” mode minimal pass.  
    - Tests: correctness off; effect measurable when on.
15. **Docs sync**  
    - Update `docs/cli.md` (`evaluate`, `resume-*`, new flags).  
    - README quick example for `evaluate`.  
    - Cookbook recipes for code‑mode and morphology.

---

## 14) Risks & Mitigations

- **VRAM pressure / OOM** → autoscaler backoff; chunked histograms; fused kernels.
- **Parity drift CPU↔GPU** → shared merge/prune code; determinism suite; seed pinning.
- **Distributed flakes** → lease timeouts, heartbeats, safe shrink; robust env checks.
- **AST fragility** → parser fallback; per‑language plugins; golden round‑trip tests.
- **Privacy leakage** → hashing + tie randomization modes; documented trade‑offs.

---

## 15) Deliverables & Repo Layout

**Deliverables**
- BPE/Unigram/Hybrid artifacts; code‑mode sidecars; hybrid manifest.
- Evaluate JSON; bench JSON; plots; trend table.
- Full docs (Architecture, CLI, API, Cookbook).
- Test suite; CI with scaling/ schema gates.

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
  export/ (sp.py, hf.py, manifest.py, artifacts.py)
  autoscale/ (controller.py, metrics.py)
  evaluate.py    <-- NEW
  utils/ (logging.py, determinism.py, privacy.py)
benchmarks/ (runner.py, configs/, samples/, reports/)
docs/ (README, cli.md, api.md, architecture.md, cookbook/)
tests/ (unit/, integration/, performance/, test_cli_evaluate.py)  <-- NEW
main.py (CLI entrypoint)
```

---

## 16) Operating Targets (Consumer Hardware)

- **GPU:** RTX 3060–4090 (8–24 GB).
- **CPU:** 6–16 cores; 16–64 GB RAM.
- **OS:** Linux primary; Windows (WSL2) best‑effort; macOS CPU‑only.
- **Deps:** Python ≥ 3.10, PyTorch (CUDA), Triton (opt), NumPy, zstandard, lz4, pandas/matplotlib (bench).

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
  --checkpoint-dir ./artifacts/unigram_ckpts --checkpoint-every 1 \
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

**Evaluate artifacts**
```
python main.py evaluate \
  --artifacts ./artifacts/bpe \
  --sample "eval/**/*.txt" \
  --report ./artifacts/eval_report.json
```

---

## 18) Definition of Done

- KPIs in §1 met.
- All gap‑closure tasks in §13 Phase D merged.
- Unit/integration/perf tests and CI gates pass.
- Docs & CLI examples verified on CPU‑only and single‑GPU machines.
- Evaluate JSON produced and schematized in CI artifacts.

---

## 19) Configuration Keys (Reference)

**Global**
- `device`, `target_util`, `deterministic`, `seed`, `bos`, `eos`, `token_bytes`,
  `checkpoint_dir`, `checkpoint_every`, `resume_from`, `log_every`, `out_dir`

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

**Augmentation (experimental)**
- `augmentation`, `aug_strength`

---

_End of plan (v2)._
