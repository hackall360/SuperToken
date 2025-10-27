"""Helpers for running tokenizer training benchmarks."""

from __future__ import annotations

import json
import random
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Sequence

import torch

from gpu_tokenizer import (
    AutoScaler,
    BytePacker,
    GPUBPETrainer,
    GPUUnigramTrainer,
    PackedBatcher,
)
from gpu_tokenizer.io import MemoryMappedShard

try:  # pragma: no cover - optional dependency
    import sentencepiece as _sentencepiece
except ImportError:  # pragma: no cover - dependency guard
    _sentencepiece = None

try:  # pragma: no cover - optional dependency
    from tokenizers import Tokenizer as _HFTokenizer
except ImportError:  # pragma: no cover - dependency guard
    _HFTokenizer = None

from .schema import validate_benchmark_output

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from tokenizers import Tokenizer as HFTokenizer

    from gpu_tokenizer.morphology import MorphologyPlugin

@dataclass
class CorpusSummary:
    sequences: int
    tokens: int
    max_length: int
    sources: list[dict[str, object]]


@dataclass
class BPERunSpec:
    """Configuration describing a single BPE benchmark invocation."""

    name: str
    batch_size: int
    device: str | None = None
    devices: list[str] | None = None
    overlap: bool = True
    scaling_reference: str | None = None
    device_weights: list[float] | None = None
    target_efficiency: float = 0.88

    def resolve_device(self) -> str | None:
        if self.device:
            return self.device
        if self.devices:
            return self.devices[0]
        return None

    def normalized_weights(self) -> list[float]:
        if self.device_weights:
            return [float(w) for w in self.device_weights]
        count = len(self.devices or [])
        if count <= 0:
            return []
        return [1.0] * count


@dataclass(frozen=True)
class BaselineCorpus:
    """Description of a small baseline corpus used for tokenizer benchmarks."""

    name: str
    description: str
    documents: tuple[str, ...]


_BASELINE_CORPORA: dict[str, BaselineCorpus] = {
    "wikitext-103": BaselineCorpus(
        name="wikitext-103",
        description="Wikitext-103 validation excerpt",
        documents=(
            "= Valkyrie Profile =\nValkyrie Profile is a role-playing video game developed by tri-Ace and published by Enix.",
            "The story weaves Norse mythology with original characters, following the valkyrie Lenneth as she recruits the souls of fallen warriors to fight in Ragnarok.",
        ),
    ),
    "the-stack-sm": BaselineCorpus(
        name="the-stack-sm",
        description="Python snippet inspired by The Stack sample dataset",
        documents=(
            "def fibonacci(n: int) -> list[int]:\n    sequence = [0, 1]\n    while len(sequence) < n:\n        sequence.append(sequence[-1] + sequence[-2])\n    return sequence[:n]\n",
            "class Example:\n    def __init__(self, payload: str) -> None:\n        self.payload = payload\n\n    def render(self) -> str:\n        return f\"Example({self.payload})\"\n",
        ),
    ),
}


_BASELINE_ALIASES: dict[str, str] = {
    "wikitext": "wikitext-103",
    "wikitext103": "wikitext-103",
    "the-stack": "the-stack-sm",
    "stack": "the-stack-sm",
    "the-stack-python": "the-stack-sm",
}


def available_baseline_corpora() -> list[str]:
    """Return the sorted list of builtin baseline corpus identifiers."""

    return sorted(_BASELINE_CORPORA.keys())


def resolve_baseline_corpora(names: Sequence[str]) -> list[BaselineCorpus]:
    """Resolve canonical :class:`BaselineCorpus` entries for *names*."""

    resolved: list[BaselineCorpus] = []
    seen: set[str] = set()
    for raw_name in names:
        if not raw_name:
            continue
        key = raw_name.lower()
        canonical = _BASELINE_ALIASES.get(key, key)
        corpus = _BASELINE_CORPORA.get(canonical)
        if corpus is None:
            raise KeyError(canonical)
        if corpus.name in seen:
            continue
        seen.add(corpus.name)
        resolved.append(corpus)
    return resolved


def _load_sentencepiece_processor(model_path: Path | str | None) -> Any:
    if model_path is None:
        return None
    if _sentencepiece is None:
        raise RuntimeError(
            "sentencepiece must be installed to benchmark SentencePiece tokenizers"
        )
    processor = _sentencepiece.SentencePieceProcessor()
    if not processor.Load(str(model_path)):
        raise RuntimeError(f"Failed to load SentencePiece model from {model_path}")
    return processor


def _load_huggingface_tokenizer(tokenizer_path: Path | str | None) -> Any:
    if tokenizer_path is None:
        return None
    if _HFTokenizer is None:
        raise RuntimeError(
            "The `tokenizers` package is required to benchmark Hugging Face tokenizers"
        )
    return _HFTokenizer.from_file(str(tokenizer_path))


def _benchmark_sentencepiece(
    processor: Any,
    documents: Sequence[str],
    *,
    total_bytes: int,
    model_path: str | None,
) -> dict[str, object]:
    total_tokens = 0
    total_loss = 0.0
    wall_start = time.perf_counter()
    for text in documents:
        pieces = processor.encode(text, out_type=int)
        total_tokens += len(pieces)
        if pieces:
            total_loss -= sum(float(processor.get_score(piece)) for piece in pieces)
    wall = time.perf_counter() - wall_start
    tokens_per_s: float | None = None
    if wall > 0 and total_tokens > 0:
        tokens_per_s = total_tokens / wall
    bytes_per_token: float | None = None
    if total_tokens > 0 and total_bytes > 0:
        bytes_per_token = total_bytes / total_tokens
    loss_per_token: float | None = None
    if total_tokens > 0:
        loss_per_token = total_loss / total_tokens
    return {
        "model_path": model_path,
        "wall_time_s": wall,
        "tokens": total_tokens,
        "tokens_per_s": tokens_per_s,
        "bytes_per_token": bytes_per_token,
        "loss_per_token": loss_per_token,
    }


def _benchmark_huggingface(
    tokenizer: Any,
    documents: Sequence[str],
    *,
    total_bytes: int,
    tokenizer_path: str | None,
) -> dict[str, object]:
    total_tokens = 0
    wall_start = time.perf_counter()
    for text in documents:
        encoding = tokenizer.encode(text)
        total_tokens += len(getattr(encoding, "ids", []) or [])
    wall = time.perf_counter() - wall_start
    tokens_per_s: float | None = None
    if wall > 0 and total_tokens > 0:
        tokens_per_s = total_tokens / wall
    bytes_per_token: float | None = None
    if total_tokens > 0 and total_bytes > 0:
        bytes_per_token = total_bytes / total_tokens
    return {
        "tokenizer_path": tokenizer_path,
        "wall_time_s": wall,
        "tokens": total_tokens,
        "tokens_per_s": tokens_per_s,
        "bytes_per_token": bytes_per_token,
        "loss_per_token": None,
    }


def run_reference_tokenizers(
    corpora: Sequence[BaselineCorpus],
    *,
    sentencepiece_model: Path | str | None = None,
    huggingface_tokenizer: Path | str | None = None,
) -> list[dict[str, object]]:
    """Benchmark SentencePiece and Hugging Face tokenizers on *corpora*."""

    processor = _load_sentencepiece_processor(sentencepiece_model)
    hf_tokenizer = _load_huggingface_tokenizer(huggingface_tokenizer)
    sp_model_str = str(sentencepiece_model) if sentencepiece_model else None
    hf_tokenizer_str = str(huggingface_tokenizer) if huggingface_tokenizer else None
    results: list[dict[str, object]] = []
    for corpus in corpora:
        documents = [doc for doc in corpus.documents if doc]
        total_bytes = sum(len(doc.encode("utf-8")) for doc in documents)
        tokenizer_stats: dict[str, dict[str, object]] = {}
        if processor is not None:
            tokenizer_stats["sentencepiece"] = _benchmark_sentencepiece(
                processor,
                documents,
                total_bytes=total_bytes,
                model_path=sp_model_str,
            )
        if hf_tokenizer is not None:
            tokenizer_stats["huggingface"] = _benchmark_huggingface(
                hf_tokenizer,
                documents,
                total_bytes=total_bytes,
                tokenizer_path=hf_tokenizer_str,
            )
        results.append(
            {
                "name": corpus.name,
                "description": corpus.description,
                "documents": len(documents),
                "total_bytes": total_bytes,
                "tokenizers": tokenizer_stats,
            }
        )
    return results


def generate_streaming_compression_runs(
    *,
    batch_size: int,
    device: str,
    target_efficiency: float = 0.9,
    baseline_name: str = "streaming_baseline",
    overlap_name: str = "streaming_overlap",
) -> list[BPERunSpec]:
    """Create run specifications highlighting streaming compression modes.

    The generated scenarios capture a baseline single-device run where host to
    device transfers occur sequentially and a second run that enables overlapped
    transfers. The latter references the baseline so downstream scaling
    reporting can quantify the win from overlapping compression/decompression
    work with GPU execution.

    Parameters
    ----------
    batch_size:
        Number of sequences per packed batch for both scenarios.
    device:
        CUDA device string to target (for example ``"cuda:0"``).
    target_efficiency:
        Minimum acceptable efficiency relative to the baseline when overlap is
        enabled. The default ``0.9`` requires at least a 90% throughput match.
    baseline_name, overlap_name:
        Identifiers used in the emitted :class:`BPERunSpec` objects. They are
        exposed as keyword-only arguments so callers can align them with
        existing reporting conventions.

    Returns
    -------
    list[BPERunSpec]
        Two BPE run specifications: a sequential-transfer baseline and an
        overlapped streaming configuration that references the baseline for
        scaling analysis.
    """

    baseline = BPERunSpec(
        name=baseline_name,
        batch_size=batch_size,
        device=device,
        overlap=False,
    )
    streaming = BPERunSpec(
        name=overlap_name,
        batch_size=batch_size,
        device=device,
        overlap=True,
        scaling_reference=baseline.name,
        target_efficiency=target_efficiency,
    )
    return [baseline, streaming]


def generate_multi_gpu_runs(
    *,
    batch_size: int,
    baseline_device: str,
    data_parallel_devices: Sequence[str],
    target_efficiency: float = 0.88,
    baseline_name: str = "single_gpu",
    multi_name: str = "multi_gpu",
) -> list[BPERunSpec]:
    """Build run specifications for evaluating multi-GPU throughput scaling.

    A single-GPU baseline is paired with a multi-GPU configuration that lists
    all participating devices. The ``scaling_reference`` field is wired up so
    :func:`run_bpe_suite` can compute efficiency relative to the baseline using
    :class:`BPERunSpec.normalized_weights`.

    Parameters
    ----------
    batch_size:
        Batch size shared by all generated runs.
    baseline_device:
        CUDA device used for the baseline measurement.
    data_parallel_devices:
        Iterable of CUDA devices to use for the multi-GPU run.
    target_efficiency:
        Minimum acceptable efficiency relative to the baseline. Defaults to the
        standard 88%% efficiency threshold used in our scaling reports.
    baseline_name, multi_name:
        Identifiers used in the generated :class:`BPERunSpec` objects.

    Returns
    -------
    list[BPERunSpec]
        Two run specifications: the baseline and a multi-GPU configuration
        referencing that baseline.
    """

    devices = [str(device) for device in data_parallel_devices]
    if not devices:
        raise ValueError("data_parallel_devices must contain at least one device")
    baseline = BPERunSpec(
        name=baseline_name,
        batch_size=batch_size,
        device=baseline_device,
        overlap=True,
    )
    multi = BPERunSpec(
        name=multi_name,
        batch_size=batch_size,
        devices=devices,
        overlap=True,
        scaling_reference=baseline.name,
        device_weights=[1.0] * len(devices),
        target_efficiency=target_efficiency,
    )
    return [baseline, multi]


def generate_hybrid_runs(
    *,
    batch_size: int,
    fast_device: str,
    helper_devices: Sequence[str],
    helper_weight: float = 0.75,
    target_efficiency: float = 0.85,
    baseline_name: str = "hybrid_baseline",
    hybrid_name: str = "hybrid_pipeline",
) -> list[BPERunSpec]:
    """Generate run specifications blending single- and multi-GPU execution.

    Hybrid scenarios model setups where a primary GPU performs most of the
    compute while one or more helper devices focus on auxiliary tasks (for
    example, host staging or decompression). The generated configuration assigns
    custom ``device_weights`` so scaling comparisons factor in that asymmetric
    contribution.

    Parameters
    ----------
    batch_size:
        Batch size to reuse for both runs.
    fast_device:
        Device capturing the single-GPU baseline and leading the hybrid run.
    helper_devices:
        Sequence of helper GPU identifiers that augment the baseline device.
    helper_weight:
        Relative contribution of each helper GPU when projecting expected
        throughput. ``0.75`` means each helper is expected to deliver 75%% of the
        throughput of the fast device.
    target_efficiency:
        Required efficiency relative to the expected aggregate throughput.
    baseline_name, hybrid_name:
        Identifiers used for the generated :class:`BPERunSpec` objects.

    Returns
    -------
    list[BPERunSpec]
        Baseline and hybrid run specifications with scaling metadata.
    """

    helpers = [str(device) for device in helper_devices]
    if not helpers:
        raise ValueError("helper_devices must contain at least one device")
    baseline = BPERunSpec(
        name=baseline_name,
        batch_size=batch_size,
        device=fast_device,
        overlap=True,
    )
    weights = [1.0] + [helper_weight for _ in helpers]
    hybrid = BPERunSpec(
        name=hybrid_name,
        batch_size=batch_size,
        devices=[fast_device, *helpers],
        overlap=True,
        scaling_reference=baseline.name,
        device_weights=weights,
        target_efficiency=target_efficiency,
    )
    return [baseline, hybrid]


def _ensure_trainers_available() -> None:
    """Validate that GPU-backed trainer classes were successfully imported.

    Returns
    -------
    None
        The function exists purely for its side effect of raising when the
        trainers cannot be used.

    Side Effects
    ------------
    None.

    Raises
    ------
    RuntimeError
        Raised when either :class:`GPUBPETrainer` or :class:`GPUUnigramTrainer`
        resolved to ``None`` because PyTorch lacks CUDA support.
    """

    if GPUBPETrainer is None or GPUUnigramTrainer is None:  # type: ignore[truthy-function]
        raise RuntimeError(
            "Both GPUBPETrainer and GPUUnigramTrainer require torch with GPU support."
        )


def synthesize_corpus(
    *,
    documents: int,
    min_length: int,
    max_length: int,
    vocab_size: int,
    seed: int,
) -> list[list[int]]:
    """Create a synthetic integer-token corpus for benchmarking.

    The helper draws uniformly random token IDs in the ``[0, vocab_size)`` range
    for ``documents`` pseudo-documents. Each document length is sampled from the
    inclusive range between ``min_length`` and ``max_length``. The sampling is
    deterministic for a given ``seed`` so that synthetic corpora can be shared
    across benchmark runs.

    Parameters
    ----------
    documents:
        Number of documents (sequences) to generate. ``0`` or negative values
        short-circuit to an empty corpus.
    min_length:
        Inclusive lower bound on the number of tokens per document. Values less
        than ``1`` are treated as ``1`` to ensure non-empty sequences when
        documents are requested.
    max_length:
        Inclusive upper bound on the number of tokens per document. When
        ``max_length`` is smaller than ``min_length`` the effective bounds are
        adjusted so both become ``min_length``.
    vocab_size:
        Size of the synthetic vocabulary. Tokens are sampled from
        ``range(vocab_size)``.
    seed:
        Seed forwarded to :class:`random.Random` for deterministic sampling.

    Returns
    -------
    list[list[int]]
        A list containing one integer sequence per generated document. Each
        inner list represents a 1D token vector whose length matches the sampled
        document length. The list is empty if ``documents`` is ``0`` or negative.

    Side Effects
    ------------
    None.

    Notes
    -----
    The routine performs purely CPU-side random number generation; GPU hardware
    is not consulted.

    Raises
    ------
    ValueError
        Never raised directly; caller should ensure ``vocab_size`` is positive
        to avoid an implicit ``ValueError`` from ``randrange``.
    """
    if documents <= 0:
        return []
    rng = random.Random(seed)
    corpus: list[list[int]] = []
    for _ in range(documents):
        length = rng.randint(max(1, min_length), max(min_length, max_length))
        seq = [rng.randrange(vocab_size) for _ in range(length)]
        corpus.append(seq)
    return corpus


def _iter_shard_sequences(
    shard: MemoryMappedShard,
    packer: BytePacker,
) -> Iterator[list[int]]:
    """Yield decoded token sequences from a memory-mapped shard.

    Parameters
    ----------
    shard:
        Open :class:`gpu_tokenizer.io.MemoryMappedShard` providing access to the
        byte-packed corpus contents.
    packer:
        :class:`gpu_tokenizer.BytePacker` configured with the desired BOS/EOS
        settings.

    Returns
    -------
    Iterator[list[int]]
        Iterator producing one list of integer token IDs per stored sequence.
        Each yielded list corresponds to a packed shard record decoded into a
        1D CPU tensor before conversion to Python integers.

    Notes
    -----
    The helper streams data directly from the memory map without requiring GPU
    availability. Callers are responsible for keeping the ``shard`` context
    alive for as long as iteration continues.

    Side Effects
    ------------
    None. The caller is responsible for managing the lifetime of ``shard``.
    """
    encoded = packer.encode_shard(shard)
    seq = list(encoded)
    if seq:
        yield seq


def load_real_corpus(
    paths: Sequence[Path],
    *,
    bos: int | None,
    eos: int | None,
    limit: int | None,
    morphology: "MorphologyPlugin" | None = None,
) -> list[list[int]]:
    """Load token sequences from byte-packed shards on disk.

    Each path is memory mapped via :class:`gpu_tokenizer.io.MemoryMappedShard`
    and decoded with :class:`gpu_tokenizer.BytePacker`. Optional beginning of
    sequence (``bos``) and end of sequence (``eos``) IDs can be provided to
    prepend/append to every decoded sequence. When ``limit`` is provided the
    loader stops as soon as ``limit`` sequences have been read.

    Parameters
    ----------
    paths:
        Sequence of shard paths. An empty sequence returns an empty corpus
        without touching the filesystem.
    bos, eos:
        Optional integer IDs inserted at the beginning/end of each sequence.
        Use ``None`` to disable the corresponding token. Defaults are implicit
        ``None`` because the function only accepts keyword arguments.
    limit:
        Optional maximum number of sequences to load. ``None`` means no limit.

    Returns
    -------
    list[list[int]]
        Token sequences decoded from the provided shards. Each sequence is a 1D
        list of integer token IDs, and the outer list preserves the order of
        ``paths``.

    Side Effects
    ------------
    Opens each provided shard path and keeps it memory mapped for the duration
    of the load. The function closes all file handles before returning thanks to
    :class:`contextlib.ExitStack`.

    Notes
    -----
    Decoding takes place entirely on the CPU; GPU availability is irrelevant at
    this stage of the pipeline.

    Raises
    ------
    FileNotFoundError
        If any shard path is missing.
    OSError
        If the underlying memory-mapped file cannot be opened.
    """
    if not paths:
        return []
    packer = BytePacker(bos=bos, eos=eos, morphology=morphology)
    sequences: list[list[int]] = []
    # ExitStack ensures shards are closed in LIFO order even if iteration
    # exits early, so we acquire it before touching any filesystem resources.
    with ExitStack() as stack:
        for path in paths:
            shard = stack.enter_context(MemoryMappedShard(path))
            for seq in _iter_shard_sequences(shard, packer):
                sequences.append(seq)
                if limit is not None and len(sequences) >= limit:
                    return sequences
    return sequences


def summarize_corpus(
    sequences: Sequence[Sequence[int]],
    *,
    sources: list[dict[str, object]] | None = None,
) -> CorpusSummary:
    """Compute high-level statistics for a tokenized corpus.

    Parameters
    ----------
    sequences:
        Iterable of token sequences to summarize.
    sources:
        Optional metadata describing how the corpus was created. When ``None``
        an empty list is stored in the summary. The default is ``None`` to avoid
        sharing mutable state between calls.

    Returns
    -------
    CorpusSummary
        Dataclass containing the number of sequences, total token count, the
        maximum sequence length, and the preserved ``sources`` metadata. The
        ``max_length`` field reflects the maximum 1D sequence length in tokens.

    Side Effects
    ------------
    None.

    Notes
    -----
    The summary is computed with standard Python iteration and arithmetic; GPU
    devices are not involved.

    Raises
    ------
    None.
    """
    total_tokens = sum(len(seq) for seq in sequences)
    max_len = max((len(seq) for seq in sequences), default=0)
    return CorpusSummary(
        sequences=len(sequences),
        tokens=total_tokens,
        max_length=max_len,
        sources=list(sources or []),
    )


def _build_bpe_batches(
    sequences: Sequence[Sequence[int]],
    *,
    batch_size: int,
    seed: int,
) -> PackedBatcher:
    """Create a :class:`PackedBatcher` suited for BPE training.

    The packed batcher preserves padding metadata and yields packed tensors and
    sequence descriptors when iterated.

    Parameters
    ----------
    sequences:
        Token sequences to batch.
    batch_size:
        Number of sequences per packed batch.
    seed:
        Seed used to shuffle sequences before packing for reproducibility.

    Returns
    -------
    PackedBatcher
        The ready-to-iterate batcher. Iterating yields ``(tokens, valid,
        lengths)`` tuples where ``tokens`` is a 2D tensor shaped ``[batch,
        max_tokens]`` representing packed sequences on the CPU, ``valid`` masks
        padding positions, and ``lengths`` contains per-sequence lengths.

    Side Effects
    ------------
    None, aside from any lazy work performed when the batcher is iterated.

    Notes
    -----
    Packing operates on CPU tensors only and therefore does not require GPU
    availability.

    Raises
    ------
    None directly; construction errors from :class:`PackedBatcher` propagate.
    """
    return PackedBatcher(sequences, batch_size=batch_size, seed=seed)


def _build_unigram_batches(
    sequences: Sequence[Sequence[int]],
    *,
    batch_size: int,
    seed: int,
) -> list[torch.Tensor]:
    """Pack sequences and materialize tensors for unigram training.

    Parameters
    ----------
    sequences:
        Token sequences to batch.
    batch_size:
        Number of sequences per packed batch.
    seed:
        Seed used to shuffle sequences before packing.

    Returns
    -------
    list[torch.Tensor]
        List of cloned 2D token tensors shaped ``[batch, max_tokens]`` with
        sequences laid out row-wise. Cloning detaches the tensors from the
        underlying memory map so that the unigram trainer can freely modify
        their contents in-place.

    Side Effects
    ------------
    Creates cloned tensors, increasing memory usage relative to the lazy BPE
    iteration.

    Notes
    -----
    All tensors are materialised on the CPU; GPU availability is not required
    until the batches are fed to a trainer.

    Raises
    ------
    None directly; errors from :class:`PackedBatcher` or tensor cloning
    propagate.
    """
    packed = PackedBatcher(sequences, batch_size=batch_size, seed=seed)
    # Clone the packed tensors so the unigram trainer can mutate batches without
    # touching the shared memory map backing the iterator.
    return [tokens.clone() for tokens, _valid, _lengths in packed]


def run_bpe_benchmark(
    sequences: Sequence[Sequence[int]],
    *,
    base_vocab: int,
    merges: int,
    batch_size: int,
    device: str | None,
    seed: int,
    log_every: int,
    overlap: bool = True,
    devices: Sequence[str] | None = None,
) -> dict[str, object]:
    """Benchmark :class:`GPUBPETrainer` on the provided sequences.

    The function configures an :class:`AutoScaler` to keep the batch size fixed
    for reproducible benchmarking, constructs packed batches, trains the BPE
    model and returns profiling metadata.

    Parameters
    ----------
    sequences:
        Token sequences to train on.
    base_vocab:
        Initial vocabulary size passed to the trainer.
    merges:
        Number of merge operations to perform.
    batch_size:
        Number of sequences per packed batch.
    device:
        Optional CUDA device (e.g. ``"cuda:0"``). ``None`` lets the trainer
        decide.
    seed:
        Shuffle seed provided to :func:`_build_bpe_batches`.
    log_every:
        Step interval for logging internal trainer metrics.

    Returns
    -------
    dict[str, object]
        Dictionary capturing the configuration, wall-clock time in seconds, and
        the metadata returned by :meth:`GPUBPETrainer.fit`. The batches are
        produced deterministically given ``seed`` to keep benchmark runs
        reproducible.

    Side Effects
    ------------
    Performs GPU work and logs through the trainer as configured.

    Notes
    -----
    Requires a CUDA-capable PyTorch installation; otherwise
    :func:`_ensure_trainers_available` raises :class:`RuntimeError` before any
    training begins.

    Raises
    ------
    RuntimeError
        If GPU-enabled trainers are unavailable (see
        :func:`_ensure_trainers_available`).

    Examples
    --------
    Train with already packed sequences:

    >>> sequences = [[1, 2, 3], [4, 5]]
    >>> result = run_bpe_benchmark(
    ...     sequences,
    ...     base_vocab=256,
    ...     merges=1000,
    ...     batch_size=2,
    ...     device="cuda:0",
    ...     seed=42,
    ...     log_every=50,
    ... )
    >>> result["config"]["merges"]
    1000
    """
    _ensure_trainers_available()
    # Keep ``min_bs`` and ``max_bs`` identical so the autoscaler never alters
    # batch sizes during benchmarking, preserving deterministic timings.
    autoscaler = AutoScaler(min_bs=batch_size, max_bs=batch_size, device=device)
    batches = _build_bpe_batches(sequences, batch_size=batch_size, seed=seed)
    kwargs: dict[str, object] = {
        "base_vocab": base_vocab,
        "merges": merges,
        "device": device,
        "autoscaler": autoscaler,
    }
    if devices:
        kwargs["devices"] = list(devices)
    trainer = GPUBPETrainer(**kwargs)
    total_tokens = sum(len(seq) for seq in sequences)
    # Capture the high-resolution wall-clock before invoking GPU work so the
    # elapsed timing includes the entire training call.
    wall_start = time.perf_counter()
    meta = trainer.fit(batches, log_every=log_every, overlap_transfers=overlap)
    wall_time = time.perf_counter() - wall_start
    tokens_per_s: float | None = None
    if wall_time > 0 and total_tokens > 0:
        tokens_per_s = total_tokens / wall_time
    autoscaler_window: list[dict[str, object]] = []
    telemetry = meta.get("telemetry") if isinstance(meta, dict) else None
    if isinstance(telemetry, dict):
        autoscaler_meta = telemetry.get("autoscaler")
        if isinstance(autoscaler_meta, dict):
            window = autoscaler_meta.get("window")
            if isinstance(window, list):
                autoscaler_window = window
    return {
        "config": {
            "base_vocab": base_vocab,
            "merges": merges,
            "batch_size": batch_size,
            "device": device,
            "log_every": log_every,
            "devices": list(devices) if devices else None,
            "overlap": overlap,
        },
        "wall_time_s": wall_time,
        "result": meta,
        "overlap_enabled": overlap,
        "tokens_processed": total_tokens,
        "tokens_per_s": tokens_per_s,
        "autoscaler_window": autoscaler_window,
    }


def load_bpe_run_config(path: Path) -> list[BPERunSpec]:
    """Load a JSON configuration describing multiple BPE benchmark runs."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    runs_raw = payload.get("runs", [])
    specs: list[BPERunSpec] = []
    for entry in runs_raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "")
        if not name:
            continue
        batch_size = int(entry.get("batch_size", 0))
        if batch_size <= 0:
            continue
        devices = entry.get("devices")
        normalized_devices: list[str] | None
        if isinstance(devices, list):
            normalized_devices = [str(dev) for dev in devices if isinstance(dev, str)]
        else:
            normalized_devices = None
        spec = BPERunSpec(
            name=name,
            batch_size=batch_size,
            device=str(entry.get("device")) if entry.get("device") else None,
            devices=normalized_devices,
            overlap=bool(entry.get("overlap", True)),
            scaling_reference=(
                str(entry.get("scaling_reference"))
                if entry.get("scaling_reference")
                else None
            ),
            device_weights=[
                float(val)
                for val in entry.get("device_weights", [])
                if isinstance(val, (int, float))
            ]
            or None,
            target_efficiency=float(entry.get("target_efficiency", 0.88)),
        )
        specs.append(spec)
    return specs


def run_bpe_suite(
    sequences: Sequence[Sequence[int]],
    *,
    base_vocab: int,
    merges: int,
    seed: int,
    log_every: int,
    run_configs: Sequence[BPERunSpec],
) -> dict[str, object]:
    """Execute a suite of BPE benchmarks derived from run specifications."""

    runs: list[dict[str, object]] = []
    throughputs: dict[str, float] = {}
    for spec in run_configs:
        benchmark = run_bpe_benchmark(
            sequences,
            base_vocab=base_vocab,
            merges=merges,
            batch_size=spec.batch_size,
            device=spec.resolve_device(),
            seed=seed,
            log_every=log_every,
            overlap=spec.overlap,
            devices=spec.devices,
        )
        run_record = {
            **benchmark,
            "name": spec.name,
        }
        throughputs[spec.name] = float(benchmark.get("tokens_per_s") or 0.0)
        runs.append(run_record)

    for spec, run_record in zip(run_configs, runs):
        scaling_info: dict[str, object] | None = None
        if spec.scaling_reference:
            baseline = throughputs.get(spec.scaling_reference, 0.0)
            weights = spec.normalized_weights()
            expected = baseline * sum(weights) if baseline > 0 and weights else 0.0
            observed = float(run_record.get("tokens_per_s") or 0.0)
            efficiency = observed / expected if expected > 0 else None
            meets_target = (
                efficiency is not None and efficiency >= spec.target_efficiency
            )
            scaling_info = {
                "reference": spec.scaling_reference,
                "device_weights": weights,
                "expected_tokens_per_s": expected if expected > 0 else None,
                "efficiency": efficiency,
                "target_efficiency": spec.target_efficiency,
                "meets_target": meets_target if efficiency is not None else None,
            }
        run_record["scaling"] = scaling_info
    return {"runs": runs}


def run_unigram_benchmark(
    sequences: Sequence[Sequence[int]],
    *,
    base_vocab: int,
    vocab_size: int,
    max_subword_len: int,
    batch_size: int,
    epochs: int,
    device: str | None,
    seed: int,
) -> dict[str, object]:
    """Benchmark :class:`GPUUnigramTrainer` across multiple epochs.

    Parameters
    ----------
    sequences:
        Token sequences to train on.
    base_vocab:
        Initial vocabulary size supplied to the trainer.
    vocab_size:
        Target vocabulary size.
    max_subword_len:
        Maximum length of candidate subwords.
    batch_size:
        Number of sequences per packed batch.
    epochs:
        Number of training epochs to run.
    device:
        Optional CUDA device string. ``None`` lets the trainer decide.
    seed:
        Shuffle seed used for batching.

    Returns
    -------
    dict[str, object]
        Dictionary containing the configuration, wall-clock time in seconds, and
        per-epoch metrics returned from :meth:`GPUUnigramTrainer.fit_epoch`. The
        deterministic batching seeded by ``seed`` ensures consistent epoch
        ordering across runs.

    Side Effects
    ------------
    Performs GPU work and mutates the internal state of the unigram trainer. The
    cloned batch tensors are also retained until Python reclaims them.

    Notes
    -----
    Requires a CUDA-capable PyTorch installation; otherwise
    :func:`_ensure_trainers_available` raises :class:`RuntimeError` before any
    training begins.

    Raises
    ------
    RuntimeError
        If GPU-enabled trainers are unavailable.

    Examples
    --------
    >>> run_unigram_benchmark(
    ...     sequences=[[1, 2, 3], [4, 5]],
    ...     base_vocab=256,
    ...     vocab_size=512,
    ...     max_subword_len=16,
    ...     batch_size=2,
    ...     epochs=3,
    ...     device=None,
    ...     seed=123,
    ... )
    {"config": {...}, "wall_time_s": ..., "epochs": [...]}
    """
    _ensure_trainers_available()
    trainer = GPUUnigramTrainer(
        base_vocab=base_vocab,
        vocab_size=vocab_size,
        max_subword_len=max_subword_len,
        device=device,
        seed=seed,
    )
    batches = _build_unigram_batches(sequences, batch_size=batch_size, seed=seed)
    # Start measuring wall-clock time immediately before the epoch loop so the
    # reported duration covers every iteration and synchronization point.
    wall_start = time.perf_counter()
    epoch_metrics: list[dict[str, object]] = []
    for epoch in range(epochs):
        stats = trainer.fit_epoch(batches)
        stats["epoch"] = epoch + 1
        epoch_metrics.append(stats)
    wall_time = time.perf_counter() - wall_start
    return {
        "config": {
            "base_vocab": base_vocab,
            "vocab_size": vocab_size,
            "max_subword_len": max_subword_len,
            "batch_size": batch_size,
            "epochs": epochs,
            "device": device,
        },
        "wall_time_s": wall_time,
        "epochs": epoch_metrics,
    }


def format_summary_table(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> str:
    """Create a simple ASCII table summarizing benchmark results."""

    materialized_rows = list(rows)
    matrix = [headers, *materialized_rows]
    widths = [max(len(str(cell)) for cell in column) for column in zip(*matrix)]

    def _fmt(row: Sequence[str]) -> str:
        return " | ".join(str(cell).ljust(width) for cell, width in zip(row, widths))

    line = "-+-".join("-" * width for width in widths)
    parts = [_fmt(headers), line]
    parts.extend(_fmt(row) for row in materialized_rows)
    return "\n".join(parts)


def emit_benchmark_summary(
    corpus: CorpusSummary,
    bpe_result: dict[str, object],
    unigram_result: dict[str, object],
    bpe_suite: dict[str, object] | None = None,
    baseline_tokenizers: Sequence[Mapping[str, object]] | None = None,
) -> str:
    """Format a human-readable summary of benchmark runs.

    Parameters
    ----------
    corpus:
        Summary statistics for the dataset used in training.
    bpe_result:
        Benchmark results dictionary produced by :func:`run_bpe_benchmark`.
    unigram_result:
        Benchmark results dictionary produced by :func:`run_unigram_benchmark`.

    baseline_tokenizers:
        Optional iterable describing reference tokenizer throughput metrics.

    Returns
    -------
    str
        Multi-line textual summary including a table of wall-clock times and
        tokens-per-second metrics.

    Side Effects
    ------------
    None.

    Raises
    ------
    KeyError
        If the expected keys are missing from ``bpe_result`` or
        ``unigram_result``.
    """
    rows: list[list[str]] = []
    bpe_tokens = bpe_result.get("tokens_per_s")
    if bpe_tokens is None and bpe_result.get("wall_time_s", 0.0) > 0:
        bpe_tokens = corpus.tokens / bpe_result["wall_time_s"]
    rows.append(
        [
            "GPUBPETrainer",
            f"{bpe_result['wall_time_s']:.2f}",
            f"{bpe_tokens:.2f}" if bpe_tokens else "n/a",
            str(bpe_result["result"].get("vocab_size", "")),
        ]
    )
    unigram_tokens = None
    if unigram_result["wall_time_s"] > 0:
        unigram_tokens = corpus.tokens / unigram_result["wall_time_s"]
    rows.append(
        [
            "GPUUnigramTrainer",
            f"{unigram_result['wall_time_s']:.2f}",
            f"{unigram_tokens:.2f}" if unigram_tokens else "n/a",
            str(unigram_result["epochs"][-1].get("vocab", ""))
            if unigram_result["epochs"]
            else "",
        ]
    )
    headers = ["Trainer", "Wall time (s)", "Tokens/s", "Final vocab"]
    summary_lines = [
        f"Corpus → {corpus.sequences} sequences, {corpus.tokens} tokens (max len {corpus.max_length})",
        format_summary_table(rows, headers),
    ]
    if bpe_suite and isinstance(bpe_suite.get("runs"), list):
        suite_rows: list[list[str]] = []
        for run in bpe_suite.get("runs", []):
            if not isinstance(run, dict):
                continue
            wall = float(run.get("wall_time_s", 0.0))
            tokens_per_s = run.get("tokens_per_s")
            if tokens_per_s is None and wall > 0:
                tokens_per_s = corpus.tokens / wall
            scaling = run.get("scaling") if isinstance(run.get("scaling"), dict) else None
            efficiency = scaling.get("efficiency") if scaling else None
            meets_target = scaling.get("meets_target") if scaling else None
            scaling_display = "n/a"
            if efficiency is not None:
                scaling_display = f"{efficiency * 100:.1f}%"
                if meets_target is True:
                    scaling_display += " ✅"
                elif meets_target is False:
                    scaling_display += " ⚠️"
            suite_rows.append(
                [
                    str(run.get("name", "")),
                    f"{wall:.2f}",
                    f"{tokens_per_s:.2f}" if tokens_per_s else "n/a",
                    scaling_display,
                ]
            )
        if suite_rows:
            summary_lines.append("")
            summary_lines.append(
                format_summary_table(
                    suite_rows,
                    ["Run", "Wall (s)", "Tokens/s", "Scaling"],
                )
            )
    if baseline_tokenizers:
        baseline_rows: list[list[str]] = []
        for record in baseline_tokenizers:
            if not isinstance(record, Mapping):
                continue
            corpus_name = str(record.get("name", ""))
            tokenizers_meta = record.get("tokenizers")
            if not isinstance(tokenizers_meta, Mapping):
                continue
            for tokenizer_name, stats in tokenizers_meta.items():
                if not isinstance(stats, Mapping):
                    continue
                tokens_per_s = stats.get("tokens_per_s")
                bytes_per_token = stats.get("bytes_per_token")
                loss_per_token = stats.get("loss_per_token")
                baseline_rows.append(
                    [
                        corpus_name,
                        str(tokenizer_name),
                        f"{float(tokens_per_s):.2f}" if tokens_per_s else "n/a",
                        f"{float(bytes_per_token):.3f}" if bytes_per_token else "n/a",
                        f"{float(loss_per_token):.4f}" if loss_per_token else "n/a",
                    ]
                )
        if baseline_rows:
            summary_lines.append("")
            summary_lines.append(
                format_summary_table(
                    baseline_rows,
                    ["Corpus", "Tokenizer", "Tokens/s", "Bytes/token", "Loss/token"],
                )
            )
    return "\n".join(summary_lines)


def serialize_run(
    output_dir: Path,
    *,
    corpus: CorpusSummary,
    config: dict[str, object],
    bpe: dict[str, object],
    unigram: dict[str, object],
    bpe_runs: dict[str, object] | None = None,
    evaluation: Mapping[str, object] | None = None,
    baseline_tokenizers: Sequence[Mapping[str, object]] | None = None,
) -> Path:
    """Persist benchmark inputs and outputs to JSON for later analysis.

    Parameters
    ----------
    output_dir:
        Directory that will contain the serialized run. Created if missing.
    corpus:
        Summary data describing the training corpus.
    config:
        Top-level configuration (e.g., command-line arguments).
    bpe:
        Result dictionary from :func:`run_bpe_benchmark`.
    unigram:
        Result dictionary from :func:`run_unigram_benchmark`.

    evaluation:
        Optional evaluation report payload that will be embedded under the
        ``"evaluation"`` key of the serialized JSON when provided.
    baseline_tokenizers:
        Optional iterable of reference tokenizer metrics captured for baseline
        corpora.

    Returns
    -------
    Path
        File path of the written JSON file. The name encodes the UTC timestamp
        of serialization.

    Side Effects
    ------------
    Creates ``output_dir`` as needed and writes a JSON document to disk.

    Raises
    ------
    OSError
        If the directory cannot be created or the file cannot be written.
    TypeError
        If the payload contains unsupported objects for JSON serialization and
        ``_json_default`` cannot handle them.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp": timestamp,
        "config": config,
        "corpus": {
            "sequences": corpus.sequences,
            "tokens": corpus.tokens,
            "max_length": corpus.max_length,
            "sources": corpus.sources,
        },
        "bpe": bpe,
        "unigram": unigram,
    }
    if bpe_runs is not None:
        payload["bpe_runs"] = bpe_runs.get("runs") if isinstance(bpe_runs, dict) else bpe_runs
    if evaluation is not None:
        payload["evaluation"] = evaluation
    if baseline_tokenizers:
        payload["baseline_tokenizers"] = list(baseline_tokenizers)
    validate_benchmark_output(payload)
    path = output_dir / f"benchmark_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, torch.device):  # type: ignore[attr-defined]
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
