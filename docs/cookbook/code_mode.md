# Code mode token count comparison

This recipe demonstrates how to evaluate the impact of code-mode preprocessing and meta-token compression on a polyglot repository. The workflow runs a tiny `train-bpe` session twice—first with plain code-mode linearisation and then with `--meta-compress` enabled—and compares the reported token counts.

## 1. Prepare a polyglot manifest

Code mode accepts newline-delimited JSON (`.jsonl`) where each entry provides a `language`, `source`, and optional `filename`. Create a manifest containing a mix of Python and TypeScript files:

```json
{"language": "python", "filename": "vector.py", "source": "def scale(v, factor):\n    return [x * factor for x in v]\n"}
{"language": "typescript", "filename": "point.ts", "source": "export function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }"}
```

Save the file as `repo.jsonl` inside your working directory. You can generate larger manifests by sampling from your repository, formatting each file as a JSON object.

## 2. Collect baseline code-mode metrics

Run `train-bpe` with a small merge budget so the job finishes quickly. The command prints a `code_mode` block that includes the number of samples, AST vs. fallback counts, and the average packed sequence length:

```bash
python main.py train-bpe \
  --data repo.jsonl \
  --merges 64 \
  --min-batch 2 \
  --max-batch 2 \
  --token-bytes 2048 \
  --code-mode \
  --code-langs python typescript
```

Example tail output:

```json
{
  "code_mode": {
    "samples": 2,
    "ast_samples": 2,
    "fallback_samples": 0,
    "average_sequence_length": 742.5,
    "meta_compress": false
  },
  "telemetry": {
    "autoscaler": {
      "window": [...]
    }
  }
}
```

Record the `average_sequence_length`—it represents the per-sample token count for plain AST linearisation.

## 3. Enable meta-token compression

Repeat the run with `--meta-compress` to collapse repeated identifier and structural patterns. The CLI updates the same `code_mode` section with the compressed sequence statistics:

```bash
python main.py train-bpe \
  --data repo.jsonl \
  --merges 64 \
  --min-batch 2 \
  --max-batch 2 \
  --token-bytes 2048 \
  --code-mode \
  --code-langs python typescript \
  --meta-compress
```

Sample output highlighting the improvement:

```json
{
  "code_mode": {
    "samples": 2,
    "ast_samples": 2,
    "fallback_samples": 0,
    "average_sequence_length": 511.0,
    "meta_compress": true,
    "meta_token_count": 4
  }
}
```

The reduced `average_sequence_length` (742.5 → 511.0 in this example) confirms that meta-token compression shaved roughly 31 % off the packed AST representation. The `meta_token_count` field reports how many reusable patterns were discovered across the corpus.

## 4. Investigate fallbacks

If the manifest contains files that fail AST parsing, the CLI marks them as `fallback_samples`. Each affected entry falls back to byte-level encoding but retains metadata so you can surface the filename and reason. Addressing syntax errors or unsupported languages increases the share of AST tokens and maximises the benefit of meta compression.

## 5. Apply to larger runs

Once the small-scale test looks promising, scale up merges and batch sizes for real training. The code-mode summary continues to reflect aggregate token statistics, making it easy to track improvements across repositories or meta-token parameters. Combine the summaries with your training logs to build dashboards that quantify the payoff of AST-aware preprocessing across language families.
