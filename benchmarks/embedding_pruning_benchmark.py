"""Benchmark lightweight embedding training before and after pruning.

The benchmark is intentionally small and CPU-only so it can run in CI
environments without GPU access or third-party numerical libraries. It trains a
toy embedding classifier on a synthetic task twice: once using the full
vocabulary and once after pruning infrequent tokens. The resulting metrics are
emitted as JSON for storage alongside existing tokenizer telemetry.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence


@dataclass
class DatasetSplit:
    """Container holding token indices and labels for a dataset split."""

    tokens: list[int]
    labels: list[int]

    @property
    def size(self) -> int:
        return len(self.tokens)


@dataclass
class SyntheticDataset:
    """Synthetic dataset and frequency metadata used by the benchmark."""

    train: DatasetSplit
    eval: DatasetSplit
    vocab_size: int
    class_count: int
    counts: list[int]

    @property
    def total_samples(self) -> int:
        return self.train.size + self.eval.size


@dataclass
class EpochTelemetry:
    """Loss/accuracy snapshot collected at the end of an epoch."""

    epoch: int
    loss: float
    accuracy: float


@dataclass
class TrainingSummary:
    """Aggregate training metrics for a single benchmark run."""

    wall_time_s: float
    throughput_samples_per_s: float
    final_loss: float
    final_accuracy: float
    evaluation_loss: float
    evaluation_accuracy: float
    epochs: list[EpochTelemetry]

    def to_dict(self) -> dict[str, object]:
        return {
            "wall_time_s": self.wall_time_s,
            "throughput_samples_per_s": self.throughput_samples_per_s,
            "final_loss": self.final_loss,
            "final_accuracy": self.final_accuracy,
            "evaluation": {
                "loss": self.evaluation_loss,
                "accuracy": self.evaluation_accuracy,
            },
            "telemetry": {
                "epochs": [asdict(epoch) for epoch in self.epochs],
            },
        }


@dataclass
class PruningResult:
    """Details describing how the vocabulary was pruned."""

    threshold: float
    kept: dict[int, int]
    pruned: list[dict[str, float]]

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "kept_tokens": len(self.kept),
            "pruned_tokens": self.pruned,
        }


@dataclass
class EmbeddingBenchmarkConfig:
    """Configuration for running the embedding pruning benchmark."""

    vocab_size: int = 64
    embedding_dim: int = 16
    class_count: int = 4
    train_samples: int = 4096
    eval_samples: int = 1024
    epochs: int = 8
    batch_size: int = 128
    learning_rate: float = 0.25
    prune_frequency_threshold: float = 0.01
    seed: int = 2025


def _softmax_row(logits: list[float]) -> list[float]:
    anchor = max(logits)
    exps = [math.exp(value - anchor) for value in logits]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(logits)] * len(logits)
    return [value / total for value in exps]


def _cross_entropy(probs: list[list[float]], labels: list[int]) -> float:
    total = 0.0
    for row, label in zip(probs, labels):
        prob = max(min(row[label], 1.0), 1e-12)
        total -= math.log(prob)
    return total / max(1, len(labels))


def _accuracy(predictions: list[int], labels: list[int]) -> float:
    if not labels:
        return 0.0
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    return correct / len(labels)


class EmbeddingClassifier:
    """Minimal embedding + linear classifier implemented with Python lists."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        class_count: int,
        rng: random.Random,
    ) -> None:
        scale = 1.0 / math.sqrt(max(1, embedding_dim))
        self.embeddings: list[list[float]] = [
            [rng.uniform(-scale, scale) for _ in range(embedding_dim)]
            for _ in range(vocab_size)
        ]
        self.weights: list[list[float]] = [
            [rng.uniform(-scale, scale) for _ in range(embedding_dim)]
            for _ in range(class_count)
        ]
        self.bias: list[float] = [0.0 for _ in range(class_count)]

    @property
    def class_count(self) -> int:
        return len(self.weights)

    @property
    def embedding_dim(self) -> int:
        return len(self.embeddings[0]) if self.embeddings else 0

    def forward(self, tokens: list[int]) -> list[list[float]]:
        logits_batch: list[list[float]] = []
        for token in tokens:
            embedding = self.embeddings[token]
            logits: list[float] = []
            for class_idx, weight_vector in enumerate(self.weights):
                activation = sum(value * weight for value, weight in zip(embedding, weight_vector))
                activation += self.bias[class_idx]
                logits.append(activation)
            logits_batch.append(logits)
        return logits_batch

    def predict(self, tokens: list[int]) -> list[int]:
        logits_batch = self.forward(tokens)
        predictions: list[int] = []
        for logits in logits_batch:
            best = max(range(len(logits)), key=lambda idx: logits[idx])
            predictions.append(best)
        return predictions


def evaluate_classifier(model: EmbeddingClassifier, dataset: DatasetSplit) -> tuple[float, float]:
    if dataset.size == 0:
        return 0.0, 0.0
    logits = model.forward(dataset.tokens)
    probs = [_softmax_row(row) for row in logits]
    loss = _cross_entropy(probs, dataset.labels)
    predictions = [max(range(len(row)), key=row.__getitem__) for row in logits]
    return loss, _accuracy(predictions, dataset.labels)


def train_classifier(
    model: EmbeddingClassifier,
    dataset: DatasetSplit,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[TrainingSummary, EmbeddingClassifier]:
    if dataset.size == 0:
        summary = TrainingSummary(
            wall_time_s=0.0,
            throughput_samples_per_s=0.0,
            final_loss=0.0,
            final_accuracy=0.0,
            evaluation_loss=0.0,
            evaluation_accuracy=0.0,
            epochs=[],
        )
        return summary, model

    shuffle_rng = random.Random(0)
    indices = list(range(dataset.size))
    epoch_history: list[EpochTelemetry] = []
    start = time.perf_counter()

    for epoch in range(epochs):
        shuffle_rng.shuffle(indices)
        total_loss = 0.0
        total_correct = 0
        seen = 0

        for batch_start in range(0, dataset.size, batch_size):
            batch_indices = indices[batch_start : batch_start + batch_size]
            batch_tokens = [dataset.tokens[idx] for idx in batch_indices]
            batch_labels = [dataset.labels[idx] for idx in batch_indices]
            if not batch_tokens:
                continue

            logits = model.forward(batch_tokens)
            probs = [_softmax_row(row) for row in logits]

            loss = _cross_entropy(probs, batch_labels)
            total_loss += loss * len(batch_labels)

            grad_logits: list[list[float]] = []
            for row, label in zip(probs, batch_labels):
                grads: list[float] = []
                for class_idx in range(model.class_count):
                    target = 1.0 if class_idx == label else 0.0
                    grads.append((row[class_idx] - target) / len(batch_labels))
                grad_logits.append(grads)

            embedded = [model.embeddings[token][:] for token in batch_tokens]
            grad_weights = [[0.0 for _ in range(model.embedding_dim)] for _ in range(model.class_count)]
            grad_bias = [0.0 for _ in range(model.class_count)]
            grad_embeddings = [[0.0 for _ in range(model.embedding_dim)] for _ in batch_tokens]

            for sample_idx, grads in enumerate(grad_logits):
                emb = embedded[sample_idx]
                for class_idx, grad_value in enumerate(grads):
                    if grad_value == 0.0:
                        continue
                    weight_grad = grad_weights[class_idx]
                    for dim in range(model.embedding_dim):
                        weight_grad[dim] += grad_value * emb[dim]
                    grad_bias[class_idx] += grad_value
                    grad_emb = grad_embeddings[sample_idx]
                    weights = model.weights[class_idx]
                    for dim in range(model.embedding_dim):
                        grad_emb[dim] += grad_value * weights[dim]

            for class_idx in range(model.class_count):
                for dim in range(model.embedding_dim):
                    model.weights[class_idx][dim] -= learning_rate * grad_weights[class_idx][dim]
                model.bias[class_idx] -= learning_rate * grad_bias[class_idx]

            token_gradients: dict[int, list[float]] = {}
            for sample_idx, token in enumerate(batch_tokens):
                grads = grad_embeddings[sample_idx]
                bucket = token_gradients.setdefault(token, [0.0 for _ in range(model.embedding_dim)])
                for dim in range(model.embedding_dim):
                    bucket[dim] += grads[dim]
            for token, grads in token_gradients.items():
                vector = model.embeddings[token]
                for dim in range(model.embedding_dim):
                    vector[dim] -= learning_rate * grads[dim]

            predictions = [max(range(len(row)), key=row.__getitem__) for row in probs]
            total_correct += sum(1 for pred, label in zip(predictions, batch_labels) if pred == label)
            seen += len(batch_labels)

        avg_loss = total_loss / max(1, seen)
        avg_acc = total_correct / max(1, seen)
        epoch_history.append(EpochTelemetry(epoch=epoch + 1, loss=avg_loss, accuracy=avg_acc))

    wall = time.perf_counter() - start
    throughput = dataset.size / wall if wall > 0 else 0.0
    final_loss, final_accuracy = evaluate_classifier(model, dataset)

    summary = TrainingSummary(
        wall_time_s=wall,
        throughput_samples_per_s=throughput,
        final_loss=final_loss,
        final_accuracy=final_accuracy,
        evaluation_loss=final_loss,
        evaluation_accuracy=final_accuracy,
        epochs=epoch_history,
    )
    return summary, model


def generate_synthetic_dataset(config: EmbeddingBenchmarkConfig) -> SyntheticDataset:
    rng = random.Random(config.seed)
    weights = [rng.random() + 0.1 for _ in range(config.vocab_size)]
    vocab = list(range(config.vocab_size))
    train_tokens = rng.choices(vocab, weights=weights, k=config.train_samples)
    eval_tokens = rng.choices(vocab, weights=weights, k=config.eval_samples)

    counts = [0 for _ in range(config.vocab_size)]
    for token in train_tokens:
        counts[token] += 1

    labels_train = [token % config.class_count for token in train_tokens]
    labels_eval = [token % config.class_count for token in eval_tokens]

    return SyntheticDataset(
        train=DatasetSplit(tokens=train_tokens, labels=labels_train),
        eval=DatasetSplit(tokens=eval_tokens, labels=labels_eval),
        vocab_size=config.vocab_size,
        class_count=config.class_count,
        counts=counts,
    )


def prune_vocabulary(
    counts: Sequence[int],
    *,
    threshold: float,
) -> PruningResult:
    total = float(sum(counts))
    kept: dict[int, int] = {}
    pruned: list[dict[str, float]] = []
    for token_id, count in enumerate(counts):
        freq = (count / total) if total > 0 else 0.0
        if freq >= threshold:
            kept[token_id] = len(kept)
        else:
            pruned.append({"token": token_id, "count": float(count), "frequency": freq})
    if not kept:
        best_token = max(range(len(counts)), key=counts.__getitem__) if counts else 0
        kept[best_token] = 0
        pruned = [entry for entry in pruned if int(entry["token"]) != best_token]
    return PruningResult(threshold=threshold, kept=kept, pruned=pruned)


def _reindex_split(split: DatasetSplit, mapping: dict[int, int]) -> DatasetSplit:
    filtered_tokens: list[int] = []
    filtered_labels: list[int] = []
    for token, label in zip(split.tokens, split.labels):
        new_index = mapping.get(token)
        if new_index is None:
            continue
        filtered_tokens.append(new_index)
        filtered_labels.append(label)
    return DatasetSplit(tokens=filtered_tokens, labels=filtered_labels)


@dataclass
class BenchmarkArtifacts:
    payload: dict[str, object]
    output_path: Path | None


def run_benchmark(
    config: EmbeddingBenchmarkConfig,
    *,
    output_dir: Path | None = None,
) -> BenchmarkArtifacts:
    dataset = generate_synthetic_dataset(config)

    rng = random.Random(config.seed)
    baseline_model = EmbeddingClassifier(
        dataset.vocab_size,
        config.embedding_dim,
        dataset.class_count,
        rng,
    )
    baseline_summary, baseline_model = train_classifier(
        baseline_model,
        dataset.train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
    )
    baseline_eval_loss, baseline_eval_accuracy = evaluate_classifier(baseline_model, dataset.eval)

    baseline_payload = baseline_summary.to_dict()
    baseline_payload["evaluation"] = {
        "loss": baseline_eval_loss,
        "accuracy": baseline_eval_accuracy,
    }

    pruning = prune_vocabulary(dataset.counts, threshold=config.prune_frequency_threshold)
    pruned_train = _reindex_split(dataset.train, pruning.kept)
    pruned_eval = _reindex_split(dataset.eval, pruning.kept)

    rng_pruned = random.Random(config.seed)
    pruned_model = EmbeddingClassifier(
        len(pruning.kept),
        config.embedding_dim,
        dataset.class_count,
        rng_pruned,
    )
    pruned_summary, pruned_model = train_classifier(
        pruned_model,
        pruned_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
    )
    pruned_eval_loss, pruned_eval_accuracy = evaluate_classifier(pruned_model, pruned_eval)

    pruned_payload = pruned_summary.to_dict()
    pruned_payload["evaluation"] = {
        "loss": pruned_eval_loss,
        "accuracy": pruned_eval_accuracy,
    }
    pruned_payload["dataset"] = {
        "train_samples": pruned_train.size,
        "eval_samples": pruned_eval.size,
        "vocab_size": len(pruning.kept),
    }

    payload: dict[str, object] = {
        "timestamp": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "config": asdict(config),
        "dataset": {
            "train_samples": dataset.train.size,
            "eval_samples": dataset.eval.size,
            "vocab_size": dataset.vocab_size,
            "class_count": dataset.class_count,
        },
        "baseline": baseline_payload,
        "pruned": pruned_payload,
        "pruning": pruning.to_dict(),
    }

    output_path: Path | None = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = payload["timestamp"]
        output_path = output_dir / f"embedding_benchmark_{stamp}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return BenchmarkArtifacts(payload=payload, output_path=output_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to store benchmark JSON payloads.")
    parser.add_argument("--vocab-size", type=int, default=64, help="Synthetic vocabulary size.")
    parser.add_argument("--embedding-dim", type=int, default=16, help="Embedding dimensionality for the toy model.")
    parser.add_argument("--class-count", type=int, default=4, help="Number of target classes in the synthetic task.")
    parser.add_argument("--train-samples", type=int, default=4096, help="Number of synthetic training samples.")
    parser.add_argument("--eval-samples", type=int, default=1024, help="Number of synthetic evaluation samples.")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs to run for each scenario.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size used during training.")
    parser.add_argument("--learning-rate", type=float, default=0.25, help="Learning rate for gradient descent updates.")
    parser.add_argument(
        "--prune-threshold",
        type=float,
        default=0.01,
        help="Minimum relative frequency required to keep a token.",
    )
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for dataset and weight initialisation.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = EmbeddingBenchmarkConfig(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        class_count=args.class_count,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        prune_frequency_threshold=args.prune_threshold,
        seed=args.seed,
    )
    artifacts = run_benchmark(config, output_dir=args.output_dir)
    summary = artifacts.payload
    baseline = summary["baseline"]["evaluation"]["accuracy"]
    pruned = summary["pruned"]["evaluation"]["accuracy"]
    throughput = summary["baseline"]["throughput_samples_per_s"]
    print(
        "Embedding benchmark completed. Baseline accuracy:"
        f" {baseline:.3f}, pruned accuracy: {pruned:.3f}, baseline throughput: {throughput:.1f} samples/s",
        flush=True,
    )
    if artifacts.output_path:
        print(f"Results written to {artifacts.output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

