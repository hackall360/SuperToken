"""Benchmark utilities for SuperToken."""

from .benchmark_runner import (
    BPERunSpec,
    CorpusSummary,
    emit_benchmark_summary,
    generate_hybrid_runs,
    generate_multi_gpu_runs,
    generate_streaming_compression_runs,
    load_bpe_run_config,
    load_real_corpus,
    run_bpe_benchmark,
    run_bpe_suite,
    run_unigram_benchmark,
    serialize_run,
    summarize_corpus,
    synthesize_corpus,
)
from .schema import BENCHMARK_OUTPUT_SCHEMA, SchemaValidationError, validate_benchmark_output

__all__ = [
    "BPERunSpec",
    "BENCHMARK_OUTPUT_SCHEMA",
    "CorpusSummary",
    "emit_benchmark_summary",
    "generate_hybrid_runs",
    "generate_multi_gpu_runs",
    "generate_streaming_compression_runs",
    "load_bpe_run_config",
    "load_real_corpus",
    "run_bpe_benchmark",
    "run_bpe_suite",
    "run_unigram_benchmark",
    "serialize_run",
    "SchemaValidationError",
    "validate_benchmark_output",
    "summarize_corpus",
    "synthesize_corpus",
]
