# Evaluation workflows

The `evaluate` subcommand measures how well a trained tokenizer represents a held-out corpus. This recipe shows how to collect artifacts, run the command, and interpret the resulting JSON report.

## 1. Gather artifacts

The evaluator needs a vocabulary mapping and optionally the merge history and tokenizer manifest produced during exports. Point `--artifacts` at a directory containing the files below or override individual paths with `--vocab`, `--merges`, and `--tokenizer`.

```bash
ls artifacts/bpe
# => merges.json  tokenizer.json  vocab.json
```

At minimum `vocab.json` must be present; merge metadata enriches compression statistics and `tokenizer.json` records extra provenance in the report.

## 2. Run the evaluator

Invoke the CLI with the same preprocessing flags used during training so morphology and code-mode settings match the corpus. Enabling `--deterministic` keeps golden reports stable for regression tests.

```bash
python main.py evaluate \
  --data tests/data/evaluate_corpus/plain.txt \
  --artifacts tests/data/models/bpe \
  --morphology-lang tr \
  --deterministic \
  --output reports/evaluate.json
```

If you evaluate structured code manifests, forward the same `--code-mode`, `--code-langs`, and `--meta-compress` flags that were active during training. Use `--meta-max-length` to cap the length of discovered meta-tokens when AST compression is enabled.

## 3. Read the report

The JSON file is designed to be machine- and human-friendly. The table below summarises the top-level keys:

| Section | Purpose |
| --- | --- |
| `artifacts` | Canonical paths, vocabulary size, and the number of merge rules applied. |
| `corpus` | Document totals plus per-document byte and token averages. |
| `compression` | Ratios such as `tokens_per_byte` after applying merge rules. |
| `oov` | Counts, rates, and identifiers for out-of-vocabulary tokens. |
| `morphology` | Segmentation statistics plus the resolved CLI configuration. |
| `code_mode` | Whether AST ingestion was active, language coverage, and meta-token stats. |

Example entries from the repository’s golden regression report:

```json
{
  "compression": {
    "tokens_per_byte": 0.7142857142857143
  },
  "oov": {
    "instances": 1,
    "rate": 0.05
  }
}
```

## 4. Automate regression gates

Deterministic runs always emit the same JSON for identical inputs, so you can check reports into version control and compare them in CI. The repository’s `tests/test_evaluate_report.py` asserts that helper utilities keep merge application, morphology summaries, and the final metrics stable. Extend the pattern to enforce target compression ratios, OOV budgets, or morphology coverage before shipping new vocabularies.
