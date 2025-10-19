# Evaluation workflows

The `evaluate` subcommand measures how well a trained tokenizer represents a held-out corpus. This recipe shows how to collect artifacts, run the command, and interpret the resulting JSON report.

## 1. Gather artifacts

Make sure the directory passed to `--artifacts` contains at least a `vocab.json`. Supplying `merges.json` (or `merges.txt`) and `tokenizer.json` enriches the report with merge statistics and provenance hints.

```bash
ls artifacts/bpe
# => merges.json  tokenizer.json  vocab.json
```

## 2. Run the evaluator

Invoke the CLI with the same preprocessing flags used during training. Enabling `--deterministic` keeps golden reports stable for regression tests.

```bash
python main.py evaluate \
  --data "datasets/eval/*.txt" \
  --artifacts artifacts/bpe \
  --morphology-lang tr \
  --code-mode \
  --code-langs python typescript \
  --meta-compress \
  --deterministic \
  --output reports/eval.json
```

## 3. Read the report

The JSON file is designed to be machine- and human-friendly. The table below summarises the top-level keys:

| Section | Purpose |
| --- | --- |
| `artifacts` | Canonical paths and merge counts used during compression. |
| `corpus` | Document totals plus per-document byte/token averages. |
| `compression` | Ratios such as `tokens_per_byte` after applying merge rules. |
| `oov` | Counts and identifiers for out-of-vocabulary tokens. |
| `morphology` | Segmentation statistics plus the resolved CLI configuration. |
| `code_mode` | Whether AST ingestion was active, language coverage, and meta-token stats. |

## 4. Automate regression gates

Because deterministic runs always emit the same JSON for identical inputs, you can check reports into version control and compare them in CI. For example, the repository’s `tests/test_cli_evaluate.py` asserts that a tiny corpus keeps its `oov.rate` at `0.05`. Teams can extend the pattern to enforce target compression ratios or morphology coverage before shipping new vocabularies.
