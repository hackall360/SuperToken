# Benchmarks

The benchmarking CLI (`python main.py benchmark`) orchestrates synthetic and
real-corpus training runs and emits JSON summaries under the directory provided
via `--output-dir`.

## Attaching evaluation reports

Benchmark outputs can now embed tokenizer evaluation metrics. After producing an
evaluation JSON (for example with `python main.py evaluate`), pass the report to
`main.py benchmark` using the new `--evaluation-report` flag:

```bash
python main.py evaluate \
  --data path/to/corpus.txt \
  --artifacts path/to/exported/tokenizer \
  --deterministic \
  --output artifacts/eval/report.json

python main.py benchmark \
  --synthetic-docs 8 \
  --output-dir artifacts/benchmarks \
  --evaluation-report artifacts/eval/report.json
```

The resulting `benchmark_*.json` document will include the evaluation payload
under an `"evaluation"` key, allowing downstream automation to surface
compression, OOV, and morphology statistics alongside throughput metrics.

## Skipping evaluation for local runs

The evaluation CLI honours the `SUPERTOKEN_SKIP_EVALUATION` environment variable.
Set it to any truthy value (`1`, `true`, `on`, …) to skip expensive evaluation
work when running quick local smoke tests:

```bash
export SUPERTOKEN_SKIP_EVALUATION=1
python main.py evaluate --data path/to/corpus.txt --artifacts path/to/tokenizer
```

CI pipelines can enforce evaluation by omitting the environment variable or by
supplying the `--force-evaluation` flag, which overrides the skip toggle.
