# Code-aware BPE training in three commands

This recipe exercises the AST-aware pipeline, meta-token compression, and privacy redaction flags without leaving your shell. Copy and paste the commands in order.

1. Create a tiny polyglot manifest with inline JSON entries:
   ```bash
   cat <<'JSON' > repo.jsonl
   {"language": "python", "filename": "metrics.py", "source": "def scale(xs, k):\n    return [x * k for x in xs]\n"}
   {"language": "typescript", "filename": "vector.ts", "source": "export const dot = (a, b) => a.x * b.x + a.y * b.y;"}
   JSON
   ```
2. Train a BPE model with code-mode enabled, capturing the run summary:
   ```bash
   python main.py train-bpe \
     --data repo.jsonl \
     --merges 128 \
     --token-bytes 2048 \
     --code-mode \
     --code-langs python typescript \
     --meta-compress \
     --privacy hash-merges \
     --out-dir artifacts/code-bpe \
     | tee artifacts/code-bpe/run.log
   ```
3. Inspect the `code_mode` telemetry block to confirm AST coverage and compression wins:
   ```bash
   python - <<'PY'
   import json
   from pathlib import Path

   log = Path('artifacts/code-bpe/run.log').read_text(encoding='utf-8')
   payload = json.loads(log.strip().splitlines()[-1])
   summary = payload.get('code_mode', {})
   print(json.dumps(summary, indent=2, sort_keys=True))
   PY
   ```

The printed JSON lists AST vs. fallback samples, the average packed length, and whether meta-token compression and privacy hashing were active.
