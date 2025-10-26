# Test data fixtures

This directory contains synthetic corpora and tokenizer artifacts used by the
unit test suite. All files were created specifically for the project and are
safe to redistribute under the repository licence.

## `evaluate_corpus/`

* `plain.txt` – Turkish-influenced base shard that keeps morphology coverage
  stable for compression and OOV regression tests.
* `emoji.txt` – Short sentences containing multi-codepoint emoji (including
  ZWJ sequences) to exercise byte-level normalization.
* `rtl.txt` – Mixed Arabic/Hebrew text that includes explicit directionality
  markers so RTL handling remains covered.
* `cjk.txt` – Simplified Chinese and Japanese lines to ensure wide glyphs
  tokenize deterministically.
* `indic.txt` – Devanagari and Bengali phrases covering Indic scripts.
* `code.jsonl` – Four structured entries: a Python AST sample, an invalid
  Python snippet that forces byte fallback, a TypeScript example, and a
  `toml` manifest to ensure unsupported languages exercise the fallback path.

## `models/`

### `models/bpe/`

Hand-authored Byte-Pair Encoding artifacts that align with `plain.txt`:

* `vocab.json` lists the individual byte tokens and merged units for the sample
  corpus (for example `meta`, `kedi`, and `programci`). The byte `y` is omitted
  on purpose so OOV handling can be asserted deterministically.
* `merges.json` sequences the corresponding merge operations. The new token IDs
  start at 256 to mimic a byte-level base vocabulary.
* `tokenizer.json` is a minimal placeholder since the tests only check that the
  manifest exists.

### `models/unigram/`

SentencePiece artifacts trained locally against `plain.txt` and
`code.jsonl` (via `sentencepiece==0.2.1` with `vocab_size=64`). Only the
textual `unigram.vocab` file is checked into version control; the binary
`unigram.model` generated alongside it is omitted so the repository stays free
of binary blobs. Tests that need a SentencePiece model generate one on the fly
during execution.

### `models/hybrid/`

A combined bundle that reuses the BPE merges and SentencePiece model while
adding hybrid-specific metadata:

* `unigram.prob` converts the SentencePiece scores to probabilities so loader
  tests can validate the format without shipping the binary model.
* `hybrid_manifest.json` records merge pairs, unigram weights, and a privacy
  summary consistent with a non-redacted export. The file was generated with a
  short Python script that serialises the SentencePiece metadata.

## `expected/`

* `evaluate_report.json` – Golden JSON produced by calling
  `gpu_tokenizer.evaluate()` with deterministic settings against all
  `evaluate_corpus/*.txt` shards and the BPE artifacts above. The report
  includes morphology, compression, and OOV metrics and mirrors the CLI test
  output.
