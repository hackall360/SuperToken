"""GPU unigram tokenizer trainer prototype."""

from __future__ import annotations

import copy
import json
import math
import os
import time
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast

import torch

from . import cuda_kernels
from .trainers.base import BaseTrainer, CheckpointPayload
from .trainers.metrics import TrainerMetricsEWMA


class GPUUnigramTrainer(BaseTrainer):
    """Minimal unigram trainer using GPU-accelerated forward/backward passes."""

    def __init__(
        self,
        base_vocab: int = 256,
        vocab_size: int = 50_000,
        max_subword_len: int = 8,
        device: str | None = None,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.base_vocab = base_vocab
        self.target_vocab = vocab_size
        self.max_len = max_subword_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.id2piece: Dict[int, bytes] = {i: bytes([i]) for i in range(base_vocab)}
        self.piece2id: Dict[bytes, int] = {bytes([i]): i for i in range(base_vocab)}
        fill_value = -math.log(max(base_vocab, 1))
        if hasattr(torch, "full"):
            self.logp = torch.full(
                (base_vocab,), fill_value, device=self.device, dtype=torch.float32
            )
        else:  # pragma: no cover - exercised by minimal torch stubs in tests
            try:
                zeros = getattr(torch, "zeros", None)
                if zeros is None:
                    raise AttributeError("zeros")
                self.logp = zeros((base_vocab,), dtype=torch.float32)
                if hasattr(self.logp, "to"):
                    self.logp = self.logp.to(self.device)
                if base_vocab > 0:
                    if hasattr(self.logp, "fill_"):
                        self.logp.fill_(fill_value)
                    else:
                        self.logp += fill_value
            except Exception as exc:  # pragma: no cover - stub path
                raise RuntimeError("torch tensor constructors unavailable") from exc
        self._rng_template_state: torch.Tensor | None = None
        self._rng_template_device: torch.device | None = None
        self._base_powers: torch.Tensor | None = None
        self._trie_dirty = True
        self._vocab_trie_next: torch.Tensor | None = None
        self._vocab_trie_terminal: torch.Tensor | None = None
        self._piece_lens_tensor: torch.Tensor | None = None
        self._trie_initialized = False
        self._cpu_piece_cache: List[torch.Tensor] | None = None
        self._metrics = TrainerMetricsEWMA(enabled=False)
        self.register_metrics_tracker("throughput", self._metrics)
        self._completed_epochs: int = 0
        self._epoch_history: list[dict[str, object]] = []
        self.reset_rng(seed=seed, generator=generator)

        if self.device == "cuda" and torch.cuda.is_available():
            self._rebuild_vocab_trie()

    def _clone_generator(self, generator: torch.Generator) -> torch.Generator:
        cloned = torch.Generator(device=generator.device)
        cloned.set_state(generator.get_state())
        return cloned

    def _create_generator(
        self,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Generator:
        if generator is not None:
            return self._clone_generator(generator)
        new_gen = torch.Generator(device=torch.device("cpu"))
        if seed is not None:
            new_gen.manual_seed(seed)
        return new_gen

    @staticmethod
    def _encode_piece(piece: bytes) -> str:
        return piece.hex()

    @staticmethod
    def _decode_piece(value: object) -> bytes:
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        if isinstance(value, str):
            try:
                return bytes.fromhex(value)
            except ValueError:
                return value.encode("latin-1", errors="ignore")
        if isinstance(value, Sequence):
            try:
                return bytes(int(elem) & 0xFF for elem in value)
            except Exception:
                return bytes()
        return bytes(str(value), "latin-1", errors="ignore")

    def _serialize_rng_state(self) -> dict[str, object]:
        rng_meta: dict[str, object] = {}
        template_state = getattr(self, "_rng_template_state", None)
        if isinstance(template_state, torch.Tensor):
            rng_meta["template_state"] = template_state.detach().cpu().tolist()
        template_device = getattr(self, "_rng_template_device", None)
        if template_device is not None:
            rng_meta["template_device"] = str(template_device)
        current = getattr(self, "_rng", None)
        if isinstance(current, torch.Generator):
            try:
                state_tensor = current.get_state()
            except Exception:
                state_tensor = None
            if state_tensor is not None:
                rng_meta["current_state"] = state_tensor.detach().cpu().tolist()
                device = getattr(current, "device", torch.device("cpu"))
                rng_meta["current_device"] = str(device)
        return rng_meta

    def _restore_rng_state(self, rng_meta: Mapping[str, Any]) -> None:
        if not isinstance(rng_meta, Mapping):
            return
        template_state = rng_meta.get("template_state")
        template_device = rng_meta.get("template_device")
        if isinstance(template_state, list):
            try:
                tensor = torch.tensor(template_state, dtype=torch.uint8)
            except Exception:
                tensor = None
            if tensor is not None:
                self._rng_template_state = tensor
                try:
                    device = torch.device(str(template_device)) if template_device is not None else torch.device("cpu")
                except Exception:
                    device = torch.device("cpu")
                self._rng_template_device = device
        current_state = rng_meta.get("current_state")
        current_device = rng_meta.get("current_device", template_device)
        if isinstance(current_state, list):
            try:
                generator_device = torch.device(str(current_device)) if current_device is not None else torch.device("cpu")
            except Exception:
                generator_device = torch.device("cpu")
            try:
                generator = torch.Generator(device=generator_device)
                state_tensor = torch.tensor(current_state, dtype=torch.uint8)
                generator.set_state(state_tensor)
            except Exception:
                generator = None
            if generator is not None:
                self._rng = generator
                return
        if getattr(self, "_rng_template_state", None) is not None:
            self.reset_rng()

    def reset_rng(
        self,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        """Reset the trainer's RNG to a deterministic state.

        Providing ``seed`` or ``generator`` establishes the template state that
        subsequent calls without arguments will restore, making it easy to keep
        multi-epoch runs reproducible by invoking ``reset_rng()`` at the start of
        each epoch.
        """

        if seed is not None or generator is not None or self._rng_template_state is None:
            self._rng = self._create_generator(seed=seed, generator=generator)
            self._rng_template_state = self._rng.get_state()
            self._rng_template_device = self._rng.device
        else:
            assert self._rng_template_device is not None  # ``_rng_template_state`` guards this path.
            self._rng = torch.Generator(device=self._rng_template_device)
            self._rng.set_state(self._rng_template_state.clone())

    def state_dict(self) -> Dict[str, object]:
        """Capture the mutable trainer state for checkpointing."""

        return {
            "base_vocab": self.base_vocab,
            "target_vocab": self.target_vocab,
            "max_len": self.max_len,
            "device": self.device,
            "id2piece": dict(self.id2piece),
            "piece2id": dict(self.piece2id),
            "logp": self.logp.detach().cpu(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> dict[str, object]:
        """Restore the trainer state from ``state_dict``."""

        if isinstance(state_dict, Mapping) and "payload" in state_dict:
            payload_map = state_dict.get("payload")
            tensors_map = state_dict.get("tensors")
            if isinstance(payload_map, Mapping):
                payload = CheckpointPayload.from_mapping(payload_map)
                trainer_meta = payload.trainer
                model_section = trainer_meta.get("model")
                if not isinstance(model_section, Mapping):
                    model_section = trainer_meta
                vocab_section = trainer_meta.get("vocab")
                if not isinstance(vocab_section, Mapping):
                    vocab_section = {}
                id2piece = vocab_section.get("id2piece", {})
                piece2id = vocab_section.get("piece2id", {})
                tensor_payload = tensors_map if isinstance(tensors_map, Mapping) else {}
                progress_section = trainer_meta.get("progress")
                state_dict = {
                    "base_vocab": model_section.get("base_vocab", self.base_vocab),
                    "target_vocab": model_section.get("target_vocab", self.target_vocab),
                    "max_len": model_section.get("max_len", self.max_len),
                    "device": model_section.get("device", self.device),
                    "id2piece": id2piece,
                    "piece2id": piece2id,
                    "logp": tensor_payload.get("logp"),
                    "progress": progress_section,
                }
                self._restore_rng_state(payload.rng)
            else:
                state_dict = {}

        self.base_vocab = int(state_dict.get("base_vocab", self.base_vocab))
        self.target_vocab = int(state_dict.get("target_vocab", self.target_vocab))
        self.max_len = int(state_dict.get("max_len", self.max_len))
        self.device = str(state_dict.get("device", self.device))

        id2piece = state_dict.get("id2piece")
        piece2id = state_dict.get("piece2id")
        if isinstance(id2piece, Mapping):
            decoded: Dict[int, bytes] = {}
            for k, v in id2piece.items():
                try:
                    idx = int(k)
                except (TypeError, ValueError):
                    continue
                decoded[idx] = self._decode_piece(v)
            if decoded:
                self.id2piece = decoded
        if isinstance(piece2id, Mapping):
            decoded_rev: Dict[bytes, int] = {}
            for k, v in piece2id.items():
                try:
                    idx = int(v)
                except (TypeError, ValueError):
                    continue
                decoded_rev[self._decode_piece(k)] = idx
            if decoded_rev:
                self.piece2id = decoded_rev

        progress_meta = state_dict.get("progress")
        if isinstance(progress_meta, Mapping):
            completed = progress_meta.get("completed_epochs")
            try:
                self._completed_epochs = int(completed)
            except (TypeError, ValueError):
                self._completed_epochs = 0
            history = progress_meta.get("history")
            if isinstance(history, list):
                coerced_history: list[dict[str, object]] = []
                for entry in history:
                    if isinstance(entry, Mapping):
                        coerced_history.append(copy.deepcopy(dict(entry)))
                self._epoch_history = coerced_history
            else:
                self._epoch_history = []
        else:
            self._completed_epochs = 0
            self._epoch_history = []

        logp = state_dict.get("logp")
        if isinstance(logp, torch.Tensor):
            self.logp = logp.to(self.device)
        else:
            vocab = max(len(self.id2piece), 1)
            fill_value = -math.log(vocab)
            if hasattr(torch, "full"):
                self.logp = torch.full(
                    (vocab,), fill_value, device=self.device, dtype=torch.float32
                )
            else:  # pragma: no cover - exercised by minimal torch stubs in tests
                try:
                    zeros = getattr(torch, "zeros", None)
                    if zeros is None:
                        raise AttributeError("zeros")
                    self.logp = zeros((vocab,), dtype=torch.float32)
                    if hasattr(self.logp, "to"):
                        self.logp = self.logp.to(self.device)
                    if vocab > 0:
                        if hasattr(self.logp, "fill_"):
                            self.logp.fill_(fill_value)
                        else:
                            self.logp += fill_value
                except Exception as exc:  # pragma: no cover - stub path
                    raise RuntimeError("torch tensor constructors unavailable") from exc

        self._mark_vocab_dirty()
        if self._completed_epochs < len(self._epoch_history):
            self._completed_epochs = len(self._epoch_history)
        if self.device == "cuda" and torch.cuda.is_available():
            self._rebuild_vocab_trie()
        return {"vocab": len(self.id2piece)}

    def import_sentencepiece(
        self,
        *,
        pieces: Mapping[int, bytes],
        scores: Sequence[float],
        source: str | None = None,
    ) -> None:
        """Import SentencePiece unigram pieces as the active vocabulary."""

        if not pieces:
            raise ValueError("pieces must contain at least one entry")

        ordered = sorted(((int(idx), bytes(piece)) for idx, piece in pieces.items()), key=lambda item: item[0])
        id2piece = {idx: piece for idx, piece in ordered}
        piece2id = {piece: idx for idx, piece in ordered}

        score_list = [float(value) for value in scores]
        target_len = len(id2piece)
        if len(score_list) < target_len:
            fill = score_list[-1] if score_list else -math.log(max(target_len, 1))
            score_list.extend([fill] * (target_len - len(score_list)))
        elif len(score_list) > target_len:
            score_list = score_list[:target_len]

        logp_tensor = torch.tensor(score_list, dtype=torch.float32, device=self.device)

        self.id2piece = id2piece
        self.piece2id = piece2id
        self.base_vocab = min(self.base_vocab, len(id2piece))
        self.target_vocab = max(self.target_vocab, len(id2piece))
        self.logp = logp_tensor
        self._completed_epochs = 0
        self._epoch_history = []
        self._mark_vocab_dirty()
        if self.device == "cuda" and torch.cuda.is_available():
            self._rebuild_vocab_trie()

    def metrics(self) -> Mapping[str, TrainerMetricsEWMA]:
        """Expose registered metrics trackers for telemetry consumers."""

        return self._metrics_mapping()

    @property
    def throughput_metrics(self) -> TrainerMetricsEWMA:
        """Return the primary throughput tracker."""

        return self._metrics

    def _ensure_base_powers(self) -> None:
        if self._base_powers is None or self._base_powers.device != torch.device(self.device):
            self._base_powers = (
                torch.pow(
                    torch.full((self.max_len,), 256, device=self.device, dtype=torch.int64),
                    torch.arange(self.max_len, device=self.device, dtype=torch.int64),
                )
            )

    def _mark_vocab_dirty(self) -> None:
        self._trie_dirty = True
        self._trie_initialized = False
        if self.device != "cuda":
            self._vocab_trie_next = None
            self._vocab_trie_terminal = None
            self._piece_lens_tensor = None
        self._cpu_piece_cache = None

    def _extend_candidates(self, sequences: torch.Tensor) -> None:
        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("GPU candidate extension requires CUDA availability")
            self._extend_candidates_gpu(sequences)
            return
        self._extend_candidates_cpu(sequences)

    def _reset_device_trie(self) -> None:
        if self.device != "cuda":
            return
        device = torch.device(self.device)
        self._vocab_trie_next = torch.full((1, 256), -1, dtype=torch.int32, device=device)
        self._vocab_trie_terminal = torch.full((1,), -1, dtype=torch.int32, device=device)
        self._trie_initialized = True

    def _append_piece_to_trie(self, piece: bytes, pid: int) -> None:
        if self.device != "cuda":
            return
        if self._vocab_trie_next is None or self._vocab_trie_terminal is None:
            self._reset_device_trie()
        assert self._vocab_trie_next is not None
        assert self._vocab_trie_terminal is not None
        state = 0
        for byte in piece:
            next_state = int(self._vocab_trie_next[state, byte].item())
            if next_state < 0:
                next_state = int(self._vocab_trie_next.size(0))
                pad_next = torch.full(
                    (1, 256), -1, dtype=torch.int32, device=self.device
                )
                pad_term = torch.full((1,), -1, dtype=torch.int32, device=self.device)
                self._vocab_trie_next = torch.cat((self._vocab_trie_next, pad_next), dim=0)
                self._vocab_trie_terminal = torch.cat(
                    (self._vocab_trie_terminal, pad_term), dim=0
                )
            self._vocab_trie_next[state, byte] = next_state
            state = next_state
        self._vocab_trie_terminal[state] = pid

    def _rebuild_vocab_trie(self) -> None:
        if self.device != "cuda":
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for GPU vocab trie construction")
        self._reset_device_trie()
        assert self._vocab_trie_next is not None
        assert self._vocab_trie_terminal is not None
        piece_lens = torch.empty(
            (len(self.id2piece),), dtype=torch.int32, device=self.device
        )
        for pid in range(len(self.id2piece)):
            piece = self.id2piece[pid]
            piece_lens[pid] = len(piece)
            if piece:
                self._append_piece_to_trie(piece, pid)
            else:
                self._vocab_trie_terminal[0] = pid
        self._piece_lens_tensor = piece_lens
        self._trie_dirty = False

    def _update_trie_after_extension(self, new_piece_ids: Sequence[int]) -> None:
        if self.device != "cuda" or not new_piece_ids:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for GPU vocab trie updates")
        rebuilt = False
        if self._vocab_trie_next is None or self._vocab_trie_terminal is None:
            self._rebuild_vocab_trie()
            rebuilt = True
        if self._piece_lens_tensor is None:
            self._rebuild_vocab_trie()
            rebuilt = True
        if rebuilt:
            return
        for pid in new_piece_ids:
            piece = self.id2piece[pid]
            self._append_piece_to_trie(piece, pid)
            lens_val = torch.full((1,), len(piece), dtype=torch.int32, device=self.device)
            self._piece_lens_tensor = torch.cat((self._piece_lens_tensor, lens_val), dim=0)
        self._trie_dirty = False
        self._trie_initialized = True

    def _extend_candidates_gpu(self, sequences: torch.Tensor) -> None:
        sequences = sequences.to(self.device)
        B, L = sequences.shape
        valid = sequences >= 0
        self._ensure_base_powers()
        assert self._base_powers is not None  # mypy guard
        all_keys: List[torch.Tensor] = []
        max_candidates = 1_000_000
        for n in range(2, self.max_len + 1):
            if L < n:
                break
            span_len = L - n + 1
            spans_valid = torch.ones((B, span_len), dtype=torch.bool, device=self.device)
            for k in range(n):
                spans_valid &= valid[:, k : k + span_len]
            if not spans_valid.any():
                continue
            idx = torch.nonzero(spans_valid, as_tuple=False)
            if idx.numel() == 0:
                continue
            if idx.size(0) > max_candidates:
                perm = torch.randperm(
                    idx.size(0), device=torch.device("cpu"), generator=self._rng
                ).to(self.device)
                idx = idx[perm[:max_candidates]]
            rows = idx[:, 0]
            cols = idx[:, 1]
            windows = torch.stack([sequences[rows, cols + k] for k in range(n)], dim=1)
            encoded = (windows.to(torch.int64) * self._base_powers[:n]).sum(dim=1)
            key = encoded * (self.max_len + 1) + n
            all_keys.append(key)
        if not all_keys:
            return
        key_tensor = torch.cat(all_keys)
        unique_keys = torch.unique(key_tensor, sorted=True)
        unique_lengths = (unique_keys % (self.max_len + 1)).to(torch.int32)
        unique_encoded = (unique_keys // (self.max_len + 1)).to(torch.int64)
        candidate_pieces: List[bytes] = []
        candidate_indices: List[int] = []
        for idx, (enc, length) in enumerate(zip(unique_encoded, unique_lengths)):
            length_int = int(length.item())
            if length_int < 2:
                continue
            value = int(enc.item())
            buf = [0] * length_int
            for pos in range(length_int - 1, -1, -1):
                buf[pos] = value & 0xFF
                value >>= 8
            piece = bytes(buf)
            if piece in self.piece2id:
                continue
            candidate_indices.append(idx)
            candidate_pieces.append(piece)
        if not candidate_pieces:
            return
        trie_children: List[dict[int, int]] = [dict()]
        terminal_ids: List[int] = [-1]
        terminal_pieces: List[bytes] = []
        for term_idx, (cand_idx, piece) in enumerate(zip(candidate_indices, candidate_pieces)):
            state = 0
            for byte in piece:
                nxt = trie_children[state].get(byte)
                if nxt is None:
                    nxt = len(trie_children)
                    trie_children[state][byte] = nxt
                    trie_children.append(dict())
                    terminal_ids.append(-1)
                state = nxt
            terminal_ids[state] = term_idx
            terminal_pieces.append(piece)
        num_states = len(trie_children)
        next_state = torch.full((num_states, 256), -1, dtype=torch.int32, device=self.device)
        for state, mapping in enumerate(trie_children):
            if not mapping:
                continue
            bytes_tensor = torch.tensor(list(mapping.keys()), dtype=torch.long, device=self.device)
            next_ids = torch.tensor(list(mapping.values()), dtype=torch.int32, device=self.device)
            next_state[state, bytes_tensor] = next_ids
        terminal_tensor = torch.tensor(terminal_ids, dtype=torch.int32, device=self.device)
        counts = torch.zeros((len(terminal_pieces),), dtype=torch.int32, device=self.device)
        if counts.numel() > 0:
            cuda_kernels.traverse_trie(
                sequences.to(torch.int32),
                valid.to(torch.uint8),
                next_state,
                terminal_tensor,
                counts,
                int(B),
                int(L),
                int(self.max_len),
            )
        if counts.numel() == 0:
            return
        order = torch.argsort(counts, descending=True)
        added_ids: List[int] = []
        for idx in order.tolist():
            cnt = int(counts[idx].item())
            if cnt <= 0:
                break
            piece = terminal_pieces[idx]
            if piece in self.piece2id:
                continue
            new_id = len(self.id2piece)
            self.id2piece[new_id] = piece
            self.piece2id[piece] = new_id
            added_ids.append(new_id)
            if len(self.id2piece) >= self.target_vocab:
                break
        V = len(self.id2piece)
        self.logp = torch.full((V,), -math.log(V), device=self.device)
        self._update_trie_after_extension(added_ids)

    def _extend_candidates_cpu(self, sequences: torch.Tensor) -> None:
        sequences = sequences.to(dtype=torch.int64, device=torch.device("cpu"))
        B, L = sequences.shape
        if B == 0 or L == 0:
            return
        valid = sequences >= 0
        self._ensure_base_powers()
        assert self._base_powers is not None
        base_powers = self._base_powers.to("cpu")
        candidate_counts: Dict[bytes, int] = {}
        candidate_keys: Dict[bytes, int] = {}
        max_len = self.max_len
        for n in range(2, max_len + 1):
            if L < n:
                break
            span_len = L - n + 1
            for row in range(B):
                row_tokens = sequences[row]
                row_valid = valid[row]
                for start in range(span_len):
                    if not bool(row_valid[start : start + n].all()):
                        continue
                    window = row_tokens[start : start + n]
                    piece = bytes(int(x) for x in window.tolist())
                    if piece in self.piece2id:
                        continue
                    candidate_counts[piece] = candidate_counts.get(piece, 0) + 1
                    if piece not in candidate_keys:
                        encoded = 0
                        window_list = window.tolist()
                        for pos, value in enumerate(window_list):
                            encoded += int(value) * int(base_powers[pos].item())
                        candidate_keys[piece] = encoded * (max_len + 1) + n
        if not candidate_counts:
            return
        ordered = sorted(
            candidate_counts.items(),
            key=lambda item: (-item[1], candidate_keys[item[0]]),
        )
        for piece, _ in ordered:
            if piece in self.piece2id:
                continue
            new_id = len(self.id2piece)
            self.id2piece[new_id] = piece
            self.piece2id[piece] = new_id
            if len(self.id2piece) >= self.target_vocab:
                break
        V = len(self.id2piece)
        device = torch.device(self.device)
        self.logp = torch.full((V,), -math.log(V), device=device)
        self._mark_vocab_dirty()

    def _ensure_vocab_trie(self) -> None:
        if self.device != "cuda":
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for GPU vocab trie construction")
        if (
            not self._trie_dirty
            and self._vocab_trie_next is not None
            and self._vocab_trie_terminal is not None
            and self._piece_lens_tensor is not None
            and self._trie_initialized
            and self._piece_lens_tensor.size(0) == len(self.id2piece)
        ):
            return
        self._rebuild_vocab_trie()

    def _ensure_cpu_piece_cache(self) -> None:
        if self._cpu_piece_cache is not None:
            return
        cache: List[torch.Tensor] = []
        for idx in range(len(self.id2piece)):
            piece = self.id2piece[idx]
            cache.append(torch.tensor(list(piece), dtype=torch.long))
        self._cpu_piece_cache = cache

    def _forward_backward_cpu(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_cpu_piece_cache()
        assert self._cpu_piece_cache is not None
        L = seq.size(0)
        seq = seq.to(torch.long)
        starts: list[list[int]] = [[] for _ in range(L)]
        for pid, piece in self.id2piece.items():
            plen = len(piece)
            if plen == 0 or plen > self.max_len or plen > L:
                continue
            piece_tensor = self._cpu_piece_cache[pid]
            for i in range(0, L - plen + 1):
                if torch.all(seq[i : i + plen] == piece_tensor):
                    starts[i].append(pid)
        logZ = torch.full((L + 1,), -1e9, device=self.device)
        logZ[0] = 0.0
        transitions: list[Tuple[int, int, int]] = []
        for i in range(L):
            zi = logZ[i]
            if zi < -1e8:
                continue
            for pid in starts[i]:
                plen = len(self.id2piece[pid])
                j = i + plen
                logZ[j] = torch.logaddexp(logZ[j], zi + self.logp[pid])
                transitions.append((i, pid, j))
        if logZ[L] < -1e8:
            exp = torch.zeros((len(self.id2piece),), device=self.device)
            for i in range(L):
                pid = int(seq[i].item())
                exp[pid] += 1
            return logZ[L], exp
        back = torch.full((L + 1,), -1e9, device=self.device)
        back[L] = 0.0
        for i, pid, j in reversed(transitions):
            back[i] = torch.logaddexp(back[i], back[j] + self.logp[pid])
        exp = torch.zeros((len(self.id2piece),), device=self.device)
        logZL = logZ[L]
        for i, pid, j in transitions:
            post = (logZ[i] + self.logp[pid] + back[j] - logZL).exp()
            exp[pid] += post
        return logZL, exp

    def _forward_backward_batch(
        self, sequences: torch.Tensor, valid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = sequences.to(self.device)
        valid = valid.to(self.device)
        B, L = sequences.shape
        V = len(self.id2piece)
        if B == 0:
            return torch.empty((0,), device=self.device), torch.zeros((0, V), device=self.device)
        if self.device != "cuda":
            logZ_list: List[torch.Tensor] = []
            exp_list: List[torch.Tensor] = []
            for row in range(B):
                length = int(valid[row].sum().item())
                if length == 0:
                    logZ_list.append(torch.tensor(float("-inf"), device=self.device))
                    exp_list.append(torch.zeros((V,), device=self.device))
                    continue
                seq = sequences[row, :length]
                logZ, exp = self._forward_backward_cpu(seq)
                exp_full = torch.zeros((V,), device=self.device)
                exp_full[: exp.numel()] = exp
                logZ_list.append(logZ)
                exp_list.append(exp_full)
            return torch.stack(logZ_list), torch.stack(exp_list)
        self._ensure_vocab_trie()
        assert self._vocab_trie_next is not None
        assert self._vocab_trie_terminal is not None
        assert self._piece_lens_tensor is not None
        sequences_i32 = sequences.to(torch.int32)
        valid_u8 = valid.to(torch.uint8)
        logp = self.logp.to(torch.float32)
        forward = cuda_kernels.forward_logz(
            sequences_i32,
            valid_u8,
            self.max_len,
            self._vocab_trie_next,
            self._vocab_trie_terminal,
            logp,
            self._piece_lens_tensor,
        )
        backward = cuda_kernels.backward_logz(
            sequences_i32,
            valid_u8,
            self.max_len,
            self._vocab_trie_next,
            self._vocab_trie_terminal,
            logp,
            self._piece_lens_tensor,
        )
        logZ = forward[:, -1]
        expectations = torch.zeros((B, V), device=self.device, dtype=torch.float32)
        cuda_kernels.accumulate_expectations(
            sequences_i32,
            valid_u8,
            self.max_len,
            self._vocab_trie_next,
            self._vocab_trie_terminal,
            logp,
            self._piece_lens_tensor,
            forward,
            backward,
            logZ,
            expectations,
        )
        fallback = logZ <= -1e29
        if fallback.any():
            rows = fallback.nonzero(as_tuple=False).squeeze(-1)
            for idx in rows.tolist():
                mask = valid[idx]
                tokens = sequences[idx][mask]
                if tokens.numel() == 0:
                    continue
                counts = torch.bincount(
                    tokens.to(torch.long), minlength=V
                ).to(expectations.dtype)
                expectations[idx].zero_()
                expectations[idx, : counts.numel()] = counts
        return logZ, expectations

    def _forward_backward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = seq.to(self.device)
        if seq.ndim != 1:
            raise ValueError("_forward_backward expects a 1D tensor")
        if seq.numel() == 0:
            return torch.tensor(float("-inf"), device=self.device), torch.zeros(
                (len(self.id2piece),), device=self.device
            )
        batch_seq = seq.unsqueeze(0)
        valid = torch.ones_like(batch_seq, dtype=torch.bool)
        logZ, exp = self._forward_backward_batch(batch_seq, valid)
        return logZ[0], exp[0]

    def _prune_vocab(self, expected_counts: torch.Tensor) -> None:
        V = len(self.id2piece)
        target = max(self.target_vocab, self.base_vocab)
        if V <= target:
            return
        counts_cpu = expected_counts.detach().to("cpu")
        candidate_indices = list(range(self.base_vocab, V))
        remove = V - target
        if remove <= 0 or not candidate_indices:
            return
        sorted_candidates = sorted(
            candidate_indices,
            key=lambda idx: (float(counts_cpu[idx].item()), idx),
        )
        drop_set = set(sorted_candidates[:remove])
        keep_indices = [idx for idx in range(V) if idx not in drop_set]
        if len(keep_indices) == V:
            return
        new_id2piece: Dict[int, bytes] = {}
        for new_id, old_id in enumerate(keep_indices):
            new_id2piece[new_id] = self.id2piece[old_id]
        self.id2piece = new_id2piece
        self.piece2id = {piece: idx for idx, piece in self.id2piece.items()}
        kept_counts = counts_cpu[keep_indices].to(torch.float32)
        smoothed = kept_counts + 1e-6
        probs = (smoothed / smoothed.sum()).clamp_min(1e-12)
        self.logp = probs.to(torch.device(self.device)).log()
        if self.device == "cuda" and torch.cuda.is_available():
            self._rebuild_vocab_trie()
        else:
            self._mark_vocab_dirty()

    def fit(
        self,
        batches: Iterable[torch.Tensor] | Sequence[torch.Tensor],
        *,
        epochs: int = 1,
    ) -> dict[str, object]:
        """Run one or more epochs of unigram training over ``batches``."""

        cached_batches = list(batches)
        try:
            epochs_int = int(epochs)
        except (TypeError, ValueError):
            epochs_int = 1
        if epochs_int <= 0:
            epochs_int = 1

        start_epoch = self._completed_epochs
        start_history_len = len(self._epoch_history)

        history: list[dict[str, object]] = []
        last_result: dict[str, object] | None = None
        for _ in range(epochs_int):
            epoch_result = self.fit_epoch(cached_batches)
            last_result = epoch_result

        if last_result is None:
            last_result = {"vocab": len(self.id2piece), "telemetry": {}}

        if len(self._epoch_history) > start_history_len:
            history = [
                copy.deepcopy(entry)
                for entry in self._epoch_history[start_history_len:]
            ]

        summary = dict(last_result)
        summary.setdefault("telemetry", {})
        summary["epochs_ran"] = self._completed_epochs - start_epoch
        summary["history"] = history
        return summary

    def fit_epoch(self, batches: List[torch.Tensor]) -> dict[str, object]:
        epoch_start = time.perf_counter()
        candidate_time = 0.0
        batch_count = 0
        token_count = 0
        sequence_count = 0
        forward_backward_time = 0.0
        accumulation_time = 0.0
        staging_time = 0.0
        metrics_tracker = self._metrics
        metrics_tracker.reset()
        metrics_enabled = metrics_tracker.enabled

        staged_batches: List[tuple[torch.Tensor, torch.Tensor]] | None = None
        if self.device == "cuda" and torch.cuda.is_available() and batches:
            staged_batches = []
            for batch in batches:
                t_stage = time.perf_counter()
                tokens = batch.to(device=self.device, dtype=torch.int32)
                valid = tokens >= 0
                staged_batches.append((tokens, valid))
                staging_time += time.perf_counter() - t_stage

        if batches:
            if len(self.id2piece) < self.target_vocab:
                t0 = time.perf_counter()
                self._extend_candidates(batches[0])
                candidate_time = time.perf_counter() - t0
            V = len(self.id2piece)
            exp_counts = torch.zeros((V,), device=self.device, dtype=torch.float32)
        else:
            V = len(self.id2piece)
            exp_counts = torch.zeros((V,), device=self.device, dtype=torch.float32)

        batch_iterable: Iterable[object]
        if staged_batches is not None:
            batch_iterable = staged_batches
        else:
            batch_iterable = batches

        for payload in batch_iterable:
            batch_count += 1
            if staged_batches is not None:
                x, valid = payload  # type: ignore[assignment]
            else:
                x = cast(torch.Tensor, payload).to(self.device)
                valid = x >= 0
            tokens_this_batch = int(valid.sum().item())
            token_count += tokens_this_batch
            sequence_count += int(x.shape[0])
            if x.numel() == 0:
                continue
            iteration_start = time.perf_counter()
            t_fb_start = iteration_start
            _, exp = self._forward_backward_batch(x, valid)
            forward_backward_time += time.perf_counter() - t_fb_start
            t_accum_start = time.perf_counter()
            exp_counts += exp.sum(dim=0)
            accumulation_time += time.perf_counter() - t_accum_start
            if metrics_enabled:
                batch_duration = time.perf_counter() - iteration_start
                metrics_tracker.record_tokens(tokens=tokens_this_batch, duration_s=batch_duration)

        t_update_start = time.perf_counter()
        smoothed = exp_counts + 1e-6
        logp = (smoothed / smoothed.sum()).clamp_min(1e-12).log()
        self.logp = logp
        self._prune_vocab(exp_counts)
        update_time = time.perf_counter() - t_update_start

        total_time = time.perf_counter() - epoch_start
        telemetry = {
            "wall_time_s": total_time,
            "candidate_extension_s": candidate_time,
            "forward_backward_s": forward_backward_time,
            "accumulation_s": accumulation_time,
            "update_s": update_time,
            "batches": batch_count,
            "sequences": sequence_count,
            "tokens": token_count,
        }

        telemetry["staging_s"] = staging_time
        if batch_count > 0:
            telemetry["iteration_latency_pre_ms"] = staging_time / batch_count * 1000.0
            telemetry["iteration_latency_post_ms"] = (
                forward_backward_time / batch_count * 1000.0
            )

        metrics_payload: dict[str, object] = {}
        for name, tracker in self.metrics().items():
            try:
                metrics_payload[name] = tracker.snapshot()
            except Exception:
                continue
        if metrics_payload:
            telemetry["metrics"] = metrics_payload
            throughput = metrics_payload.get("throughput")
            if isinstance(throughput, Mapping):
                tokens_rate = throughput.get("tokens_per_s")
                if tokens_rate is not None:
                    telemetry["tokens_per_s"] = float(tokens_rate)
                leases_rate = throughput.get("lease_per_s")
                if leases_rate is not None:
                    telemetry["lease_per_s"] = float(leases_rate)

        result = {"vocab": len(self.id2piece), "telemetry": telemetry}
        self._completed_epochs += 1
        epoch_entry = copy.deepcopy({"epoch": self._completed_epochs, **result})
        self._epoch_history.append(epoch_entry)
        return result

    def save_artifacts(self, path: str | PathLike[str]) -> dict[str, object]:
        """Persist the trained unigram model and return its location."""

        model_path = self.save(path)
        return {"model": str(model_path)}

    def save_checkpoint(
        self, path: str | PathLike[str], *args: Any, **kwargs: Any
    ) -> dict[str, object]:
        trainer_payload = {
            "model": {
                "base_vocab": int(self.base_vocab),
                "target_vocab": int(self.target_vocab),
                "max_len": int(self.max_len),
                "device": self.device,
            },
            "vocab": {
                "id2piece": {
                    str(idx): self._encode_piece(piece)
                    for idx, piece in sorted(self.id2piece.items())
                },
                "piece2id": {
                    self._encode_piece(piece): int(idx)
                    for piece, idx in self.piece2id.items()
                },
            },
        }
        tensors: dict[str, torch.Tensor] = {}
        if isinstance(self.logp, torch.Tensor):
            tensors["logp"] = self.logp.detach().clone().cpu()
        progress_section = {
            "completed_epochs": int(self._completed_epochs),
            "history": [copy.deepcopy(entry) for entry in self._epoch_history],
        }
        trainer_payload["progress"] = progress_section
        payload = CheckpointPayload(
            version=CheckpointPayload.CURRENT_VERSION,
            trainer=trainer_payload,
            rng=self._serialize_rng_state(),
        )
        os.makedirs(path, exist_ok=True)
        meta_path = os.path.join(path, "state.json")
        tensor_path = os.path.join(path, "tensors.pt")
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(payload.to_dict(), handle, indent=2, sort_keys=True)
        torch.save({name: tensor for name, tensor in tensors.items()}, tensor_path)
        return {"payload": payload.to_dict(), "tensors": tensors}

    def load_checkpoint(
        self, path: str | PathLike[str], *args: Any, **kwargs: Any
    ) -> dict[str, object]:
        meta_path = os.path.join(path, "state.json")
        tensor_path = os.path.join(path, "tensors.pt")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Checkpoint metadata missing at {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        tensors: dict[str, torch.Tensor] = {}
        if os.path.exists(tensor_path):
            loaded = torch.load(tensor_path, map_location="cpu")
            if isinstance(loaded, dict):
                tensors = loaded
        state = {"payload": payload, "tensors": tensors}
        self.load_state_dict(state)
        return state

    @property
    def completed_epochs(self) -> int:
        """Return the number of epochs completed across all runs."""

        return self._completed_epochs

    @property
    def epoch_history(self) -> list[dict[str, object]]:
        """Expose a shallow copy of the recorded epoch history."""

        return [copy.deepcopy(entry) for entry in self._epoch_history]

    def save(self, path: str | PathLike[str]) -> Path:
        """Serialize the unigram model to a SentencePiece ``.model`` file.

        The resulting artifact can be consumed directly by
        :class:`sentencepiece.SentencePieceProcessor`.  If ``path`` points to a
        directory, the model is written to ``unigram.model`` within that
        directory.  When ``path`` includes a filename, the parent directory is
        created automatically and the model is stored at the exact location.
        """

        try:
            from sentencepiece import sentencepiece_model_pb2 as sp_pb2  # type: ignore
        except Exception as exc:  # pragma: no cover - exercised only when dependency missing
            raise RuntimeError(
                "Saving a SentencePiece model requires the `sentencepiece` package"
            ) from exc

        output_path = Path(path)
        if output_path.suffix:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            model_path = output_path
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            model_path = output_path / "unigram.model"

        def _encode_piece(piece: bytes) -> str:
            text = piece.decode("latin-1")
            return text.replace(" ", "▁")

        log_probs = self.logp.detach().cpu().tolist()
        vocab_size = len(self.id2piece)

        model = sp_pb2.ModelProto()
        model.model_type = sp_pb2.ModelProto.UNIGRAM
        trainer_spec = model.trainer_spec
        trainer_spec.model_type = sp_pb2.TrainerSpec.UNIGRAM
        trainer_spec.vocab_size = vocab_size
        trainer_spec.character_coverage = 1.0
        trainer_spec.byte_fallback = True
        trainer_spec.max_sentencepiece_length = self.max_len
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
            piece = model.pieces.add()
            piece.piece = _encode_piece(self.id2piece[idx])
            piece.score = float(log_probs[idx])
            piece.type = sp_pb2.ModelProto.SentencePiece.NORMAL

        model_path.write_bytes(model.SerializeToString())
        print(f"Saved SentencePiece model → {model_path}")
        return model_path


__all__ = ["GPUUnigramTrainer"]
