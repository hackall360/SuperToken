# Morphology-aware training with reconstruction checks

This recipe walks through enabling the Turkish morphology plugin, training a tiny `train-bpe` run, and validating that the
segmentation preserves round-trip fidelity. Morphology passes are **opt-in** and safe by default—SuperToken leaves byte streams
untouched unless you request a plugin explicitly.

## 1. Inspect available plugins

Discover which plugins ship with your install by querying the registry:

```python
from gpu_tokenizer import morphology
print(morphology.available_plugins())
```

The output includes `"tr"`, `"ja"`, and `"ko"` for the bundled Turkish, Japanese, and Korean segmenters. Additional plugins can
be added at import time via `morphology.register_plugin(name, cls)`.

## 2. Prepare a sample corpus

Create a UTF-8 text file containing a few Turkish sentences. The plugin recognises common agglutinative suffixes and optionally
annotates case markers and productive affixes.

```bash
echo "Ankara'daki müzeleri gezmeyi çok seviyorum." > turkish.txt
echo "Kitaplarını öğrencilerine hızla dağıttı." >> turkish.txt
```

## 3. Run a morphology-enabled training dry run

Use `--morphology-lang tr` to activate the plugin. Additional flags toggle opt-in annotations that the Turkish implementation
understands. The Japanese (`ja`) and Korean (`ko`) plugins expose the same CLI surface but focus on script-aware segmentation
with no extra flags:

```bash
python main.py train-bpe \
  --data turkish.txt \
  --merges 128 \
  --token-bytes 2048 \
  --min-batch 2 \
  --max-batch 4 \
  --dry-run \
  --morphology-lang tr \
  --morphology-case-markers \
  --morphology-affix-tags
```

The CLI logs a `morphology` block confirming the plugin, language code, and enabled annotations. Remove the `--dry-run` flag to
perform full training once you are comfortable with the configuration.

## 4. Verify reconstruction fidelity

Before integrating morphology into large-scale training, validate that segmentation and recomposition round-trip without loss.
All bundled plugins ship with unit tests that assert this property. The following snippet uses the same plugin instance as the
CLI and checks that the original bytes match the recomposed output:

```python
from gpu_tokenizer.morphology import create_plugin
from gpu_tokenizer.cpu_packer import BytePacker

sample = "Ankara'daki müzeleri gezmeyi çok seviyorum.".encode("utf-8")
plugin = create_plugin("tr", case_markers=True, affix_tags=True)
segments = list(plugin.presegment(sample))
assert plugin.recompose(segments) == sample

packer = BytePacker(morphology=plugin)
ids = list(packer.encode_view(sample))
print(f"Token count with morphology: {len(ids)}")
```

Comparing `len(ids)` against a baseline `BytePacker(morphology=None)` measurement highlights how morphology influences packed
sequence lengths.

## 5. Roll out safely

Morphology is useful when you expect systematic affix patterns, but it also changes downstream token statistics. Keep the flags
unset for existing pipelines to preserve historical baselines, and enable them selectively for experiments. Track the
`morphology` section in CLI config logs to audit which runs received preprocessing.
