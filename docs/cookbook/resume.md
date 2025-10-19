# Resuming Interrupted Training

Tokenizer runs can stretch for hours, especially when you sweep over large corpora or iterate through multiple hybrid cycles. SuperToken's CLI exposes a consistent checkpointing contract so you can pause work, move it between machines, or schedule maintenance windows without losing progress.

## Shared Flags

Every training subcommand—`train-bpe`, `train-unigram`, and `train-hybrid`—accepts the same control knobs:

- `--checkpoint-dir`: Directory where checkpoints are written. The trainers create it if it does not exist.
- `--checkpoint-every`: Frequency of intermediate checkpoints. For BPE it counts merge steps, for unigram it counts epochs, and for hybrid it counts complete BPE→unigram cycles. Setting the value to `0` keeps only the final checkpoint.
- `--resume-from`: Path to an existing checkpoint directory created by `--checkpoint-dir`.
- `--time-minutes`: Optional wall-clock budget. When the timer expires the trainers save a checkpoint (when `--checkpoint-dir` is set) and exit cleanly.

The same flags work across fresh runs and resumes—you can tweak unrelated arguments such as `--out-dir` or `--privacy` when restarting.

## BPE

```bash
python main.py train-bpe \
  --data "data/**/*.txt" \
  --merges 50000 \
  --checkpoint-dir ./runs/bpe_ckpt \
  --checkpoint-every 200 \
  --time-minutes 120
```

If the job stops you can continue from the last merge by pointing `--resume-from` at the same directory:

```bash
python main.py train-bpe \
  --data "data/**/*.txt" \
  --merges 50000 \
  --resume-from ./runs/bpe_ckpt
```

Behind the scenes the CLI reloads autoscaler state, restores dataset cursors, and continues with the remaining merges.

## Unigram

Unigram training replays pre-packed batches, which makes resumption straightforward. The CLI keeps track of completed epochs inside the checkpoint payload:

```bash
python main.py train-unigram \
  --data "data/**/*.txt" \
  --epochs 6 \
  --checkpoint-dir ./runs/unigram_ckpt \
  --checkpoint-every 2
```

Interrupting the run after two epochs leaves a checkpoint containing the learned vocabulary, log-probabilities, and epoch history. Restart the job with:

```bash
python main.py train-unigram \
  --data "data/**/*.txt" \
  --epochs 6 \
  --resume-from ./runs/unigram_ckpt
```

Only the remaining epochs run. Supplying `--time-minutes` is helpful when you want to cap nightly maintenance windows without guessing how many epochs will fit.

## Hybrid

Hybrid runs alternate between BPE warm-ups and unigram refinement. The checkpoint stores merge history, per-cycle telemetry, and the most recent BPE/unigram snapshots, so resuming a multi-cycle schedule is as simple as:

```bash
python main.py train-hybrid \
  --data "data/**/*.txt" \
  --merges 25000 \
  --cycles 4 \
  --checkpoint-dir ./runs/hybrid_ckpt \
  --checkpoint-every 1
```

To resume from the last completed cycle, reuse the directory with `--resume-from`. You can combine it with `--time-minutes` to pause after a fixed wall-clock window—handy when alternating workloads on shared hardware.

## Tips

- When resuming, you can omit `--checkpoint-dir`; the CLI falls back to the path supplied via `--resume-from`.
- Periodic checkpoints can be noisy on local disks. Consider setting `--checkpoint-every` to a larger value and rely on the automatic checkpoint triggered by `--time-minutes` if you need a graceful stop.
- The emitted `state.json` files include progress metadata (epochs or cycles completed). They are designed to be human-inspectable, making it easy to audit the exact point where training paused.
