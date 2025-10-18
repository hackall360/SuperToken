"""Hybrid trainer orchestrating alternating BPE and unigram phases."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import torch

from ..bpe_trainer import GPUBPETrainer
from ..unigram_trainer import GPUUnigramTrainer
from .base import BaseTrainer, CheckpointPayload


def _clone_bpe_batches(
    batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return deep copies of ``batches`` suitable for re-use across phases."""

    cloned: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for tokens, valid, lengths in batches:
        cloned.append((tokens.clone(), valid.clone(), lengths.clone()))
    return cloned


def _prepare_unigram_batches(
    batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> list[torch.Tensor]:
    """Convert BPE batches into the dense tensors expected by the unigram trainer."""

    prepared: list[torch.Tensor] = []
    for tokens, valid, _lengths in batches:
        dense = tokens.clone().to(dtype=torch.int32)
        mask = valid.to(dtype=torch.bool)
        dense[~mask] = -1
        prepared.append(dense)
    return prepared


def _build_piece_tables(
    base_vocab: int, merges: Sequence[tuple[int, int]]
) -> tuple[dict[int, bytes], dict[bytes, int], int]:
    """Construct ``id2piece``/``piece2id`` mappings implied by ``merges``."""

    id2piece: dict[int, bytes] = {idx: bytes([idx]) for idx in range(base_vocab)}
    piece2id: dict[bytes, int] = {bytes([idx]): idx for idx in range(base_vocab)}
    max_len = 1
    for offset, (left_id, right_id) in enumerate(merges):
        new_id = base_vocab + offset
        try:
            left = id2piece[left_id]
            right = id2piece[right_id]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(
                f"Merge pair {(left_id, right_id)} references undefined token"
            ) from exc
        piece = left + right
        id2piece[new_id] = piece
        piece2id[piece] = new_id
        if len(piece) > max_len:
            max_len = len(piece)
    return id2piece, piece2id, max_len


def _extract_tokens_per_second(payload: Mapping[str, Any] | None) -> float | None:
    """Best-effort retrieval of throughput metrics from a summary payload."""

    if not isinstance(payload, Mapping):
        return None
    direct = payload.get("tokens_per_s")
    try:
        if direct is not None:
            value = float(direct)
            if math.isfinite(value):
                return value
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        pass
    metrics_section = payload.get("metrics")
    if isinstance(metrics_section, Mapping):
        throughput = metrics_section.get("throughput")
        if isinstance(throughput, Mapping):
            token_rate = throughput.get("tokens_per_s")
            try:
                if token_rate is not None:
                    value = float(token_rate)
                    if math.isfinite(value):
                        return value
            except (TypeError, ValueError):  # pragma: no cover - defensive guard
                return None
    iteration_section = payload.get("iteration_metrics")
    if isinstance(iteration_section, Mapping):
        ewma = iteration_section.get("ewma")
        if isinstance(ewma, Mapping):
            throughput = ewma.get("throughput")
            if isinstance(throughput, Mapping):
                token_rate = throughput.get("tokens_per_s")
                try:
                    if token_rate is not None:
                        value = float(token_rate)
                        if math.isfinite(value):
                            return value
                except (TypeError, ValueError):  # pragma: no cover - defensive guard
                    return None
    return None


def _sanitize_for_json(data: Any) -> Any:
    """Convert trainer metadata into a JSON-friendly representation."""

    if isinstance(data, torch.Tensor):
        return data.detach().cpu().tolist()
    if isinstance(data, (str, int, float, type(None))):
        if isinstance(data, float) and not math.isfinite(data):
            return None
        return data
    if isinstance(data, bytes):
        return data.hex()
    if isinstance(data, Mapping):
        return {str(k): _sanitize_for_json(v) for k, v in data.items()}
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        return [_sanitize_for_json(item) for item in data]
    return data


class HybridTrainer(BaseTrainer):
    """Coordinate alternating BPE and unigram training phases."""

    def __init__(
        self,
        *,
        base_vocab: int = 256,
        merges: int = 50_000,
        cycles: int = 1,
        unigram_epochs: int = 1,
        max_unigram_len: int = 8,
        warm_start_merges: Sequence[tuple[int, int]] | None = None,
        bpe_trainer_factory: Callable[..., GPUBPETrainer] = GPUBPETrainer,
        unigram_trainer_factory: Callable[..., GPUUnigramTrainer] = GPUUnigramTrainer,
        bpe_init_kwargs: Mapping[str, Any] | None = None,
        unigram_init_kwargs: Mapping[str, Any] | None = None,
        bpe_fit_kwargs: Mapping[str, Any] | None = None,
        unigram_fit_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.base_vocab = int(base_vocab)
        self.bpe_merges = int(merges)
        self.cycles = max(int(cycles), 1)
        self.unigram_epochs = max(int(unigram_epochs), 1)
        self.max_unigram_len = max(int(max_unigram_len), 1)
        self._initial_warm_start = (
            [tuple(map(int, pair)) for pair in warm_start_merges]
            if warm_start_merges is not None
            else None
        )
        self._bpe_factory = bpe_trainer_factory
        self._unigram_factory = unigram_trainer_factory
        self._bpe_init_kwargs = dict(bpe_init_kwargs or {})
        self._unigram_init_kwargs = dict(unigram_init_kwargs or {})
        self._bpe_fit_kwargs = dict(bpe_fit_kwargs or {})
        self._unigram_fit_kwargs = dict(unigram_fit_kwargs or {})
        self._phase_history: list[dict[str, Any]] = []
        self._final_merges: list[tuple[int, int]] = []
        self._final_logp: torch.Tensor | None = None
        self._final_id2piece: dict[int, bytes] = {}
        self._last_bpe_state: Mapping[str, Any] | None = None
        self._last_unigram_state: Mapping[str, Any] | None = None
        self._completed_cycles: int = 0

    # ------------------------------------------------------------------
    # Lifecycle helpers
    def fit(
        self,
        batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        *,
        cycles: int | None = None,
        unigram_epochs: int | None = None,
        warm_start_merges: Sequence[tuple[int, int]] | None = None,
        checkpoint_dir: str | os.PathLike[str] | None = None,
        bpe_fit_kwargs: Mapping[str, Any] | None = None,
        unigram_fit_kwargs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run alternating BPE and unigram phases over ``batches``."""

        all_batches = list(batches)
        if not all_batches:
            raise ValueError("HybridTrainer.fit requires at least one batch")

        cloned_batches = _clone_bpe_batches(all_batches)
        unigram_batches = _prepare_unigram_batches(all_batches)

        total_cycles = max(int(cycles or self.cycles), 1)
        epochs_per_cycle = max(int(unigram_epochs or self.unigram_epochs), 1)
        warm_plan = (
            [tuple(map(int, pair)) for pair in warm_start_merges]
            if warm_start_merges is not None
            else (self._initial_warm_start.copy() if self._initial_warm_start else None)
        )

        bpe_phase_kwargs = dict(self._bpe_fit_kwargs)
        if bpe_fit_kwargs:
            bpe_phase_kwargs.update(bpe_fit_kwargs)
        unigram_phase_kwargs = dict(self._unigram_fit_kwargs)
        if unigram_fit_kwargs:
            unigram_phase_kwargs.update(unigram_fit_kwargs)

        checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if checkpoint_root is not None:
            checkpoint_root.mkdir(parents=True, exist_ok=True)

        self._phase_history.clear()
        self._final_merges = []
        self._final_logp = None
        self._final_id2piece = {}
        self._completed_cycles = 0

        current_warm_start = warm_plan

        for cycle_idx in range(total_cycles):
            bpe_trainer = self._bpe_factory(
                base_vocab=self.base_vocab,
                merges=self.bpe_merges,
                warm_start_merges=current_warm_start,
                **self._bpe_init_kwargs,
            )
            phase_batches = _clone_bpe_batches(cloned_batches)
            fit_kwargs = dict(bpe_phase_kwargs)
            if current_warm_start is not None:
                fit_kwargs.setdefault("warm_start_merges", current_warm_start)
            bpe_result = bpe_trainer.fit(phase_batches, **fit_kwargs)
            merges = list(tuple(map(int, pair)) for pair in bpe_trainer.merges)

            self._last_bpe_state = bpe_trainer.state_dict(include_batches=False)

            bpe_checkpoint_dir: str | None = None
            if checkpoint_root is not None:
                bpe_checkpoint_dir = os.fspath(
                    checkpoint_root / f"cycle_{cycle_idx + 1:02d}_bpe"
                )
                bpe_trainer.save_checkpoint(bpe_checkpoint_dir)

            id2piece, piece2id, max_piece_len = _build_piece_tables(
                self.base_vocab, merges
            )

            unigram_kwargs = dict(self._unigram_init_kwargs)
            unigram_kwargs.setdefault("base_vocab", self.base_vocab)
            unigram_kwargs.setdefault("vocab_size", self.base_vocab + len(merges))
            unigram_kwargs.setdefault(
                "max_subword_len", max(self.max_unigram_len, max_piece_len)
            )
            unigram_trainer = self._unigram_factory(**unigram_kwargs)

            initial_state = {
                "base_vocab": self.base_vocab,
                "target_vocab": self.base_vocab + len(merges),
                "max_len": max(self.max_unigram_len, max_piece_len),
                "device": unigram_trainer.device,
                "id2piece": id2piece,
                "piece2id": piece2id,
                "logp": torch.full(
                    (len(id2piece),),
                    -math.log(max(len(id2piece), 1)),
                    device=unigram_trainer.device,
                    dtype=torch.float32,
                ),
            }
            unigram_trainer.load_state_dict(initial_state)

            unigram_result = unigram_trainer.fit(
                unigram_batches, epochs=epochs_per_cycle, **unigram_phase_kwargs
            )

            self._last_unigram_state = unigram_trainer.state_dict()
            self._final_logp = unigram_trainer.logp.detach().clone().cpu()
            self._final_id2piece = dict(unigram_trainer.id2piece)

            unigram_checkpoint_dir: str | None = None
            if checkpoint_root is not None:
                unigram_checkpoint_dir = os.fspath(
                    checkpoint_root / f"cycle_{cycle_idx + 1:02d}_unigram"
                )
                unigram_trainer.save_checkpoint(unigram_checkpoint_dir)

            bpe_tokens_per_s = _extract_tokens_per_second(bpe_result.get("telemetry"))
            unigram_tokens_per_s = _extract_tokens_per_second(
                unigram_result.get("telemetry")
            )

            phase_record = {
                "cycle": cycle_idx + 1,
                "bpe": {
                    "vocab_size": int(bpe_result.get("vocab_size", self.base_vocab)),
                    "merge_count": len(merges),
                    "tokens_per_s": bpe_tokens_per_s,
                    "checkpoint": bpe_checkpoint_dir,
                },
                "unigram": {
                    "vocab": int(unigram_result.get("vocab", 0)),
                    "epochs": epochs_per_cycle,
                    "tokens_per_s": unigram_tokens_per_s,
                    "checkpoint": unigram_checkpoint_dir,
                },
            }
            self._phase_history.append(phase_record)

            self._final_merges = merges
            current_warm_start = merges
            self._completed_cycles = cycle_idx + 1

        summary = {
            "cycles": self._completed_cycles,
            "merges": [list(map(int, pair)) for pair in self._final_merges],
            "vocab_size": self.base_vocab + len(self._final_merges),
            "unigram_logp": (
                self._final_logp.detach().cpu().tolist()
                if isinstance(self._final_logp, torch.Tensor)
                else []
            ),
            "phase_history": _sanitize_for_json(self._phase_history),
        }
        return summary

    # ------------------------------------------------------------------
    # Checkpointing helpers
    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Capture the hybrid trainer state for checkpointing."""

        trainer_section: dict[str, Any] = {
            "base_vocab": self.base_vocab,
            "merges": [list(map(int, pair)) for pair in self._final_merges],
            "phase_history": _sanitize_for_json(self._phase_history),
            "cycles": self._completed_cycles,
            "unigram_epochs": self.unigram_epochs,
            "max_unigram_len": self.max_unigram_len,
        }
        payload = CheckpointPayload(
            version=CheckpointPayload.CURRENT_VERSION,
            trainer=trainer_section,
        )
        tensors: dict[str, Any] = {}
        if isinstance(self._final_logp, torch.Tensor):
            tensors["logp"] = self._final_logp.detach().clone()
        state = {
            "payload": payload.to_dict(),
            "tensors": tensors,
            "bpe_state": self._last_bpe_state,
            "unigram_state": self._last_unigram_state,
        }
        return state

    def load_state_dict(
        self, state_dict: Mapping[str, Any], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Restore a previously captured hybrid trainer state."""

        payload_mapping = state_dict.get("payload") if isinstance(state_dict, Mapping) else None
        payload = CheckpointPayload.from_mapping(payload_mapping)
        trainer_meta = payload.trainer
        self.base_vocab = int(trainer_meta.get("base_vocab", self.base_vocab))
        merges_raw = trainer_meta.get("merges", [])
        self._final_merges = [tuple(map(int, pair)) for pair in merges_raw]
        self._phase_history = _sanitize_for_json(trainer_meta.get("phase_history", []))
        self._completed_cycles = int(trainer_meta.get("cycles", 0))
        self.unigram_epochs = int(trainer_meta.get("unigram_epochs", self.unigram_epochs))
        self.max_unigram_len = int(trainer_meta.get("max_unigram_len", self.max_unigram_len))

        tensors = state_dict.get("tensors") if isinstance(state_dict, Mapping) else None
        if isinstance(tensors, Mapping):
            logp = tensors.get("logp")
            if isinstance(logp, torch.Tensor):
                self._final_logp = logp.detach().clone()
        self._last_bpe_state = state_dict.get("bpe_state")
        self._last_unigram_state = state_dict.get("unigram_state")
        return {
            "merges": [list(map(int, pair)) for pair in self._final_merges],
            "phase_history": self._phase_history,
        }

    def save_checkpoint(
        self, path: str | os.PathLike[str], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Persist the hybrid trainer state to ``path``."""

        state = self.state_dict()
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_path = output_dir / "state.json"
        tensor_path = output_dir / "tensors.pt"
        payload = state.get("payload", {})
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        tensors = state.get("tensors", {})
        aux_state = {
            "tensors": tensors,
            "bpe_state": state.get("bpe_state"),
            "unigram_state": state.get("unigram_state"),
        }
        torch.save(aux_state, tensor_path)
        return state

    def load_checkpoint(
        self, path: str | os.PathLike[str], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Load a previously saved hybrid trainer checkpoint."""

        meta_path = Path(path) / "state.json"
        tensor_path = Path(path) / "tensors.pt"
        if not meta_path.exists():
            raise FileNotFoundError(f"Checkpoint metadata missing at {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        tensors: MutableMapping[str, Any] = {}
        if tensor_path.exists():
            aux_state = torch.load(tensor_path, map_location="cpu")
            if isinstance(aux_state, Mapping):
                tensors.update(aux_state.get("tensors", {}))
                self._last_bpe_state = aux_state.get("bpe_state")
                self._last_unigram_state = aux_state.get("unigram_state")
        state = {
            "payload": payload,
            "tensors": tensors,
            "bpe_state": self._last_bpe_state,
            "unigram_state": self._last_unigram_state,
        }
        self.load_state_dict(state)
        return state

    # ------------------------------------------------------------------
    # Artifact helpers
    def save_artifacts(
        self, output_dir: str | os.PathLike[str], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Write the hybrid manifest describing the latest training run."""

        manifest_dir = Path(output_dir)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "hybrid_manifest.json"
        manifest = {
            "base_vocab": self.base_vocab,
            "vocab_size": self.base_vocab + len(self._final_merges),
            "cycles": self._completed_cycles,
            "merges": [list(map(int, pair)) for pair in self._final_merges],
            "unigram_logp": (
                self._final_logp.detach().cpu().tolist()
                if isinstance(self._final_logp, torch.Tensor)
                else []
            ),
            "unigram_pieces": {
                str(idx): piece.hex() for idx, piece in sorted(self._final_id2piece.items())
            },
            "phase_history": _sanitize_for_json(self._phase_history),
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        return {"manifest": os.fspath(manifest_path)}

    # ------------------------------------------------------------------
    # Metrics plumbing
    def metrics(self) -> Mapping[str, Any]:
        """Hybrid trainer exposes no live metrics trackers."""

        return self._metrics_mapping()


__all__ = ["HybridTrainer"]
