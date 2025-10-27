# Hybrid schedule with morphology and privacy in three commands

Hybrid training warm-starts a BPE vocabulary before refining it with unigram updates. The following commands generate a toy corpus, run the schedule, and surface the exported manifest. Consult the [threat model](../../README.md#threat-model) before choosing a privacy mode—`tie-randomize` offers the strongest protection against merge reconstruction, while lighter guards keep reproducibility intact for trusted environments.

1. Seed a small bilingual corpus with simple sentences:
   ```bash
   cat <<'TXT' > corpus.txt
   merhaba dünya
   günaydın istanbul
   hello world
   good morning istanbul
   TXT
   ```
2. Launch the hybrid trainer with morphology segmentation and tie-randomised privacy (suitable when artifacts leave your trust boundary):
   ```bash
   python main.py train-hybrid \
     --data corpus.txt \
     --merges 64 \
     --cycles 2 \
     --unigram-epochs 1 \
     --morphology-lang tr \
     --morphology-case-markers \
     --privacy tie-randomize \
     --out-dir artifacts/hybrid-demo
   ```
3. Review the hybrid manifest to confirm phase summaries and privacy metadata:
   ```bash
   python - <<'PY'
   import json
   from pathlib import Path

   manifest = Path('artifacts/hybrid-demo/hybrid_manifest.json').read_text(encoding='utf-8')
   payload = json.loads(manifest)
   print(json.dumps({
       'cycles': len(payload.get('cycles', [])),
       'privacy': payload.get('privacy'),
       'morphology': payload.get('morphology')
   }, indent=2, sort_keys=True))
   PY
   ```

The output highlights how many cycles ran, which morphology options were active, and which privacy guard protected the exported merges.
