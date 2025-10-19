"""Hybrid trainer orchestrating alternating BPE and unigram phases."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import torch

from ..bpe_trainer import GPUBPETrainer
from ..unigram_trainer import GPUUnigramTrainer
from ..utils import hash_merge_pair
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


def _bytes_to_unicode() -> dict[int, str]:
    """Mirror the byte→unicode mapping used by Hugging Face BPE tokenizers."""

    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _build_bpe_tokenizer_artifacts(
    base_vocab: int, merges: Sequence[tuple[int, int]]
) -> tuple[dict[str, int], list[str], dict[str, object]]:
    """Construct vocab/merges/config triples compatible with Hugging Face."""

    byte_encoder = _bytes_to_unicode()
    token_strings: list[str] = []
    added_tokens: list[dict[str, object]] = []

    for token_id in range(base_vocab):
        if 0 <= token_id < 256:
            token_strings.append(byte_encoder[token_id])
        else:
            content = f"<|{token_id}|>"
            token_strings.append(content)
            added_tokens.append(
                {
                    "id": token_id,
                    "content": content,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                }
            )

    for idx, (left_id, right_id) in enumerate(merges):
        try:
            left = token_strings[left_id]
            right = token_strings[right_id]
        except IndexError as exc:  # pragma: no cover - defensive guard
            raise ValueError(
                f"Invalid merge pair {(left_id, right_id)} at position {idx}"
            ) from exc
        token_strings.append(left + right)

    expected_vocab = base_vocab + len(merges)
    vocab = {token: idx for idx, token in enumerate(token_strings)}
    if len(vocab) != expected_vocab:
        raise ValueError(
            "Mismatch between expected vocab size and constructed vocabulary length"
        )

    merge_strings = [
        f"{token_strings[left]} {token_strings[right]}" for left, right in merges
    ]

    tokenizer_config: dict[str, object] = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "post_processor": None,
        "decoder": {"type": "ByteLevel"},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merge_strings,
        },
    }

    return vocab, merge_strings, tokenizer_config


def _write_bpe_tokenizer_files(
    output_dir: Path,
    vocab: Mapping[str, int],
    merges: Sequence[str],
    tokenizer_config: Mapping[str, object],
) -> dict[str, str]:
    """Persist Hugging Face compatible tokenizer artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"
    tokenizer_path = output_dir / "tokenizer.json"

    with open(vocab_path, "w", encoding="utf-8") as handle:
        json.dump(dict(vocab), handle, ensure_ascii=False)

    with open(merges_path, "w", encoding="utf-8") as handle:
        handle.write("#version: 0.2\n")
        for merge in merges:
            handle.write(f"{merge}\n")

    with open(tokenizer_path, "w", encoding="utf-8") as handle:
        json.dump(dict(tokenizer_config), handle, ensure_ascii=False)

    return {
        "vocab": os.fspath(vocab_path),
        "merges": os.fspath(merges_path),
        "tokenizer": os.fspath(tokenizer_path),
    }


def _encode_unigram_piece(piece: bytes) -> str:
    text = piece.decode("latin-1")
    return text.replace(" ", "▁")


def _write_unigram_probabilities(
    path: Path, logp: torch.Tensor, id2piece: Mapping[int, bytes]
) -> Path:
    """Write a text file containing sentencepiece-compatible probabilities."""

    if not id2piece:
        raise ValueError("id2piece mapping is empty; cannot export unigram probabilities")

    vocab_size = max(id2piece) + 1
    if logp.numel() < vocab_size:
        raise ValueError("log probability tensor does not cover the full vocabulary")

    probs = torch.softmax(logp.detach().cpu(), dim=0)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# piece\tprobability\n")
        for idx in range(vocab_size):
            piece = id2piece.get(idx)
            if piece is None:
                raise ValueError(f"Missing unigram piece for id {idx}")
            encoded = _encode_unigram_piece(piece)
            prob = float(probs[idx].item())
            handle.write(f"{encoded}\t{prob:.12f}\n")
    return path


def _write_sentencepiece_model(
    output_dir: Path,
    id2piece: Mapping[int, bytes],
    logp: torch.Tensor,
    max_piece_len: int,
) -> Path | None:
    """Emit a SentencePiece model if the dependency is available."""

    try:
        from sentencepiece import sentencepiece_model_pb2 as sp_pb2  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None

    vocab_size = max(id2piece) + 1 if id2piece else logp.numel()
    if logp.numel() < vocab_size:
        raise ValueError("log probability tensor does not cover the full vocabulary")

    model = sp_pb2.ModelProto()
    model.model_type = sp_pb2.ModelProto.UNIGRAM
    trainer_spec = model.trainer_spec
    trainer_spec.model_type = sp_pb2.TrainerSpec.UNIGRAM
    trainer_spec.vocab_size = vocab_size
    trainer_spec.character_coverage = 1.0
    trainer_spec.byte_fallback = True
    trainer_spec.max_sentencepiece_length = max_piece_len
    trainer_spec.add_dummy_prefix = False
    trainer_spec.remove_extra_whitespaces = False
    trainer_spec.split_by_whitespace = False
    trainer_spec.pad_id = -1
    trainer_spec.unk_id = -1
    trainer_spec.bos_id = -1
    trainer_spec.eos_id = -1
    trainer_spec.shuffle_input_sentence = False
    trainer_spec.seed_sentencepiece_size = 0
    trainer_spec.input_sentence_size = 0

    normalizer_spec = model.normalizer_spec
    normalizer_spec.name = "identity"
    normalizer_spec.precompiled_charsmap = b""

    for idx in range(vocab_size):
        piece_bytes = id2piece.get(idx)
        if piece_bytes is None:
            raise ValueError(f"Missing unigram piece for id {idx}")
        piece = model.pieces.add()
        piece.piece = _encode_unigram_piece(piece_bytes)
        piece.score = float(logp[idx].item())
        piece.type = sp_pb2.ModelProto.SentencePiece.NORMAL

    model_path = output_dir / "unigram.model"
    output_dir.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        handle.write(model.SerializeToString())
    return model_path


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
        privacy_mode: bool = False,
        randomize_ties: bool | None = None,
        tie_seed: int | None = None,
        privacy_salt: bytes | bytearray | str | None = None,
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
        privacy_flag = bool(privacy_mode)
        if randomize_ties is None:
            randomize_flag = privacy_flag
        else:
            randomize_flag = bool(randomize_ties)
        tie_seed_value: int | None = int(tie_seed) if tie_seed is not None else None
        if isinstance(privacy_salt, (bytes, bytearray)):
            salt_bytes = bytes(privacy_salt)
        elif isinstance(privacy_salt, str):
            salt_bytes = privacy_salt.encode("utf-8")
        else:
            salt_bytes = b""
        if "privacy_mode" not in self._bpe_init_kwargs:
            self._bpe_init_kwargs["privacy_mode"] = privacy_flag
        if "randomize_ties" not in self._bpe_init_kwargs and (
            randomize_ties is not None or privacy_flag
        ):
            self._bpe_init_kwargs["randomize_ties"] = randomize_flag
        if "tie_seed" not in self._bpe_init_kwargs and (
            tie_seed_value is not None or randomize_flag
        ):
            seed_payload = int(tie_seed_value) if tie_seed_value is not None else 0
            self._bpe_init_kwargs["tie_seed"] = seed_payload
        if "privacy_salt" not in self._bpe_init_kwargs and (
            privacy_salt is not None or privacy_flag
        ):
            salt_payload: bytes | bytearray | str | None
            if privacy_salt is not None:
                salt_payload = privacy_salt
            else:
                salt_payload = salt_bytes if salt_bytes else None
            self._bpe_init_kwargs["privacy_salt"] = salt_payload
        self.privacy_mode = privacy_flag
        self.randomize_ties = randomize_flag
        self.tie_seed = tie_seed_value
        self._privacy_salt_input = privacy_salt
        self._privacy_hash_salt = salt_bytes
        self._phase_history: list[dict[str, Any]] = []
        self._final_merges: list[tuple[int, int]] = []
        self._final_logp: torch.Tensor | None = None
        self._final_id2piece: dict[int, bytes] = {}
        self._last_bpe_state: Mapping[str, Any] | None = None
        self._last_unigram_state: Mapping[str, Any] | None = None
        self._completed_cycles: int = 0
        self._stopped_early: bool = False

    # ------------------------------------------------------------------
    # Lifecycle helpers
    def _manifest_merges(self) -> list[object]:
        if not self.privacy_mode:
            return [list(map(int, pair)) for pair in self._final_merges]
        salt = self._privacy_hash_salt
        return [hash_merge_pair((int(left), int(right)), salt) for left, right in self._final_merges]

    def _privacy_summary(self) -> dict[str, object]:
        mode = "none"
        if self.randomize_ties:
            mode = "tie-randomize"
        elif self.privacy_mode:
            mode = "hash-merges"
        summary: dict[str, object] = {
            "mode": mode,
            "merges_redacted": bool(self.privacy_mode),
            "randomize_ties": bool(self.randomize_ties),
        }
        if self.tie_seed is not None or self.randomize_ties:
            summary["tie_seed"] = int(self.tie_seed) if self.tie_seed is not None else 0
        if self._privacy_hash_salt:
            summary["salt_configured"] = True
        return summary

    def fit(
        self,
        batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        *,
        cycles: int | None = None,
        unigram_epochs: int | None = None,
        warm_start_merges: Sequence[tuple[int, int]] | None = None,
        checkpoint_dir: str | os.PathLike[str] | None = None,
        checkpoint_interval: int | None = None,
        time_limit_s: float | None = None,
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

        resuming = self._completed_cycles > 0 and bool(self._phase_history)
        if not resuming:
            self._phase_history.clear()
            self._final_merges = []
            self._final_logp = None
            self._final_id2piece = {}
            self._completed_cycles = 0

        current_warm_start = warm_plan
        if resuming and self._final_merges:
            current_warm_start = [tuple(map(int, pair)) for pair in self._final_merges]

        start_cycle = self._completed_cycles if resuming else 0
        target_cycles = max(total_cycles, start_cycle)
        deadline = None
        if time_limit_s is not None and time_limit_s > 0:
            deadline = time.perf_counter() + time_limit_s
        self._stopped_early = False

        for cycle_idx in range(start_cycle, target_cycles):
            if deadline is not None and time.perf_counter() >= deadline:
                self._stopped_early = True
                break
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

            cycle_number = cycle_idx + 1
            phase_record = {
                "cycle": cycle_number,
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
            self._completed_cycles = cycle_number

            if (
                checkpoint_root is not None
                and checkpoint_interval is not None
                and checkpoint_interval > 0
                and cycle_number % checkpoint_interval == 0
            ):
                self.save_checkpoint(checkpoint_root)

            if deadline is not None and time.perf_counter() >= deadline:
                self._stopped_early = True
                break

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
            "stopped_early": bool(self._stopped_early),
        }
        if checkpoint_root is not None:
            self.save_checkpoint(checkpoint_root)
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
        trainer_section["privacy"] = self._privacy_summary()
        trainer_section["progress"] = {
            "completed_cycles": int(self._completed_cycles),
            "history": _sanitize_for_json(self._phase_history),
            "stopped_early": bool(self._stopped_early),
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
        privacy_meta = trainer_meta.get("privacy")
        if isinstance(privacy_meta, Mapping):
            mode_label = str(privacy_meta.get("mode", "")).lower()
            if mode_label == "none":
                self.privacy_mode = False
            elif mode_label in {"hash-merges", "tie-randomize"}:
                self.privacy_mode = True
            randomize_meta = privacy_meta.get("randomize_ties")
            if randomize_meta is not None:
                self.randomize_ties = bool(randomize_meta)
            if "tie_seed" in privacy_meta:
                tie_seed_raw = privacy_meta.get("tie_seed")
                try:
                    self.tie_seed = int(tie_seed_raw) if tie_seed_raw is not None else None
                except (TypeError, ValueError):
                    self.tie_seed = None

        progress_meta = trainer_meta.get("progress")
        if isinstance(progress_meta, Mapping):
            completed_meta = progress_meta.get("completed_cycles")
            try:
                self._completed_cycles = int(completed_meta)
            except (TypeError, ValueError):
                pass
            history_meta = progress_meta.get("history")
            if isinstance(history_meta, list):
                self._phase_history = _sanitize_for_json(history_meta)
            stopped_meta = progress_meta.get("stopped_early")
            if stopped_meta is not None:
                self._stopped_early = bool(stopped_meta)
        else:
            self._stopped_early = False

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

    @property
    def completed_cycles(self) -> int:
        """Return the number of completed cycles captured in the trainer state."""

        return self._completed_cycles

    @property
    def stopped_early(self) -> bool:
        """Indicate whether the most recent fit invocation halted early."""

        return self._stopped_early

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
            "merges": self._manifest_merges(),
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
        manifest["privacy"] = self._privacy_summary()
        if self.privacy_mode:
            manifest["privacy_mode"] = True
            manifest["merge_count"] = len(self._final_merges)
            manifest["randomize_ties"] = bool(self.randomize_ties)
            if self.tie_seed is not None or self.randomize_ties:
                manifest["tie_seed"] = (
                    int(self.tie_seed) if self.tie_seed is not None else 0
                )
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        return {"manifest": os.fspath(manifest_path)}

    def save(self, output_dir: str | os.PathLike[str]) -> dict[str, str]:
        """Persist combined BPE and unigram artifacts for hybrid models."""

        if not self._final_merges:
            raise RuntimeError("HybridTrainer.save requires trained BPE merges")
        if not isinstance(self._final_logp, torch.Tensor):
            raise RuntimeError("HybridTrainer.save requires unigram log probabilities")
        if not self._final_id2piece:
            raise RuntimeError("HybridTrainer.save requires populated unigram pieces")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        artifacts = self.save_artifacts(output_path)
        vocab, merges, tokenizer_config = _build_bpe_tokenizer_artifacts(
            self.base_vocab, self._final_merges
        )
        bpe_paths = _write_bpe_tokenizer_files(output_path, vocab, merges, tokenizer_config)

        logp = self._final_logp.detach().clone().cpu()
        unigram_prob_path = _write_unigram_probabilities(
            output_path / "unigram.prob", logp, self._final_id2piece
        )
        sp_path = _write_sentencepiece_model(
            output_path, self._final_id2piece, logp, self.max_unigram_len
        )

        combined: dict[str, str] = dict(artifacts)
        combined.update(bpe_paths)
        combined["unigram_prob"] = os.fspath(unigram_prob_path)
        if sp_path is not None:
            combined["unigram_model"] = os.fspath(sp_path)

        printed_paths = ", ".join(sorted(combined.values()))
        print(f"Saved hybrid artifacts → {printed_paths}")
        return combined

    # ------------------------------------------------------------------
    # Metrics plumbing
    def metrics(self) -> Mapping[str, Any]:
        """Hybrid trainer exposes no live metrics trackers."""

        return self._metrics_mapping()


__all__ = ["HybridTrainer"]
