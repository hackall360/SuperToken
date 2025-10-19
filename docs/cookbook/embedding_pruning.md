# Embedding pruning and export in three commands

Use the embedding export CLI to generate a pruned vocabulary, embedding matrix, and manifest suitable for lightweight models.

1. Create a toy vocabulary and frequency file for demonstration:
   ```bash
   python - <<'PY'
   import json
   from pathlib import Path

   Path('artifacts/emb-demo').mkdir(parents=True, exist_ok=True)
   vocab = {"<pad>": 0, "<unk>": 1, "hello": 2, "istanbul": 3, "token": 4}
   stats = {
       "tokens": {
           "<pad>": {"count": 0},
           "<unk>": {"count": 0},
           "hello": {"count": 42},
           "istanbul": {"count": 8},
           "token": {"count": 1}
       }
   }
   Path('artifacts/emb-demo/vocab.json').write_text(json.dumps(vocab), encoding='utf-8')
   Path('artifacts/emb-demo/stats.json').write_text(json.dumps(stats), encoding='utf-8')
   PY
   ```
2. Export embeddings while pruning tokens seen fewer than five times:
   ```bash
   python main.py export-embeddings \
     --vocab artifacts/emb-demo/vocab.json \
     --token-stats artifacts/emb-demo/stats.json \
     --dedupe-similarity 0.97 \
     --min-frequency 5 \
     --keep-token <pad> \
     --keep-token <unk> \
     --embedding-dim 32 \
     --embedding-seed 7 \
     --out-dir artifacts/emb-demo/export
   ```
3. Summarise the manifest and pruning report to verify which tokens survived:
   ```bash
   python - <<'PY'
   import json
   from pathlib import Path

   manifest = json.loads(Path('artifacts/emb-demo/export/manifest.json').read_text(encoding='utf-8'))
   pruning = json.loads(Path('artifacts/emb-demo/export/pruning.json').read_text(encoding='utf-8'))
   print(json.dumps({
       'exported_tokens': manifest['exported_token_count'],
       'preserved_tokens': manifest['preserved_tokens'],
       'pruned': pruning
   }, indent=2, sort_keys=True))
   PY
   ```

The dedupe threshold runs before pruning, so similar tokens collapse into a canonical entry before the frequency filter executes. Merged tokens show up in `pruning.json` with an `action` of `deduped`, while low-frequency removals retain the historical schema. You can feed the resulting `vocab.json` and `embeddings.json` into downstream trainers knowing which tokens were removed and why.
