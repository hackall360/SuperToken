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
  --output artifacts/eval/report.json \
  --summary-format table

python main.py benchmark \
  --synthetic-docs 8 \
  --output-dir artifacts/benchmarks \
  --evaluation-report artifacts/eval/report.json
```

The resulting `benchmark_*.json` document will include the evaluation payload
under an `"evaluation"` key, allowing downstream automation to surface
compression, OOV, and morphology statistics alongside throughput metrics.

## Benchmarking reference tokenizers

Pass one or more `--baseline-corpus` values to benchmark pre-trained
SentencePiece and Hugging Face tokenizers against built-in corpora:

```bash
python main.py benchmark \
  --baseline-corpus wikitext-103 \
  --baseline-corpus the-stack-sm \
  --sentencepiece-model path/to/model.model \
  --huggingface-tokenizer tests/data/models/bpe/tokenizer.json \
  --synthetic-docs 0 \
  --output-dir artifacts/benchmarks
```

The CLI recognises small excerpts from Wikitext-103 and The Stack (Python) out
of the box. The resulting benchmark JSON embeds a `"baseline_tokenizers"`
section with per-tokenizer throughput (`tokens_per_s`), compression efficiency
(`bytes_per_token`), and average loss metrics. The summary table printed to
stdout mirrors these figures so regression dashboards can surface deltas
alongside the GPU trainer throughput statistics.

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

SentencePiece exports can be evaluated directly by replacing `--artifacts` with `--vocab path/to/unigram.vocab --model-type unigram`. The command will still emit the JSON report under `reports/evaluate.json` when `--output` is omitted.
