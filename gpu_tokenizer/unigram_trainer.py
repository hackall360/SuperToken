"""GPU unigram tokenizer trainer prototype."""

from __future__ import annotations

import math
import time
from os import PathLike
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from . import cuda_kernels


class GPUUnigramTrainer:
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
        self.base_vocab = base_vocab
        self.target_vocab = vocab_size
        self.max_len = max_subword_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.id2piece: Dict[int, bytes] = {i: bytes([i]) for i in range(base_vocab)}
        self.piece2id: Dict[bytes, int] = {bytes([i]): i for i in range(base_vocab)}
        self.logp = torch.full((base_vocab,), -math.log(base_vocab), device=self.device)
        self._rng_template_state: torch.Tensor | None = None
        self._rng_template_device: torch.device | None = None
        self._base_powers: torch.Tensor | None = None
        self._trie_dirty = True
        self._vocab_trie_next: torch.Tensor | None = None
        self._vocab_trie_terminal: torch.Tensor | None = None
        self._piece_lens_tensor: torch.Tensor | None = None
        self.reset_rng(seed=seed, generator=generator)

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
        self._vocab_trie_next = None
        self._vocab_trie_terminal = None
        self._piece_lens_tensor = None

    def _extend_candidates(self, sequences: torch.Tensor) -> None:
        if self.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("GPU candidate extension requires CUDA availability")
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
            if len(self.id2piece) >= self.target_vocab:
                break
        V = len(self.id2piece)
        self.logp = torch.full((V,), -math.log(V), device=self.device)
        self._mark_vocab_dirty()

    def _ensure_vocab_trie(self) -> None:
        if self.device != "cuda":
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for GPU vocab trie construction")
        if not self._trie_dirty and self._vocab_trie_next is not None:
            return
        trie_children: List[dict[int, int]] = [dict()]
        terminal_ids: List[int] = [-1]
        for pid in range(len(self.id2piece)):
            piece = self.id2piece[pid]
            state = 0
            for byte in piece:
                nxt = trie_children[state].get(byte)
                if nxt is None:
                    nxt = len(trie_children)
                    trie_children[state][byte] = nxt
                    trie_children.append(dict())
                    terminal_ids.append(-1)
                state = nxt
            terminal_ids[state] = pid
        num_states = len(trie_children)
        next_state = torch.full((num_states, 256), -1, dtype=torch.int32, device=self.device)
        for state, mapping in enumerate(trie_children):
            if not mapping:
                continue
            keys = torch.tensor(list(mapping.keys()), dtype=torch.long, device=self.device)
            values = torch.tensor(list(mapping.values()), dtype=torch.int32, device=self.device)
            next_state[state, keys] = values
        terminal_tensor = torch.tensor(terminal_ids, dtype=torch.int32, device=self.device)
        piece_lens = torch.tensor(
            [len(self.id2piece[idx]) for idx in range(len(self.id2piece))],
            dtype=torch.int32,
            device=self.device,
        )
        self._vocab_trie_next = next_state
        self._vocab_trie_terminal = terminal_tensor
        self._piece_lens_tensor = piece_lens
        self._trie_dirty = False

    def _forward_backward_cpu(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        L = seq.size(0)
        starts: list[list[int]] = [[] for _ in range(L)]
        for pid, piece in self.id2piece.items():
            plen = len(piece)
            if plen == 0 or plen > self.max_len or plen > L:
                continue
            piece_tensor = torch.tensor(list(piece), dtype=torch.long, device=self.device)
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

    def fit_epoch(self, batches: List[torch.Tensor]) -> dict[str, object]:
        epoch_start = time.perf_counter()
        candidate_time = 0.0
        batch_count = 0
        token_count = 0
        sequence_count = 0
        forward_backward_time = 0.0
        accumulation_time = 0.0

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

        for x in batches:
            batch_count += 1
            x = x.to(self.device)
            valid = x >= 0
            token_count += int(valid.sum().item())
            sequence_count += int(x.shape[0])
            if x.numel() == 0:
                continue
            t_fb_start = time.perf_counter()
            _, exp = self._forward_backward_batch(x, valid)
            forward_backward_time += time.perf_counter() - t_fb_start
            t_accum_start = time.perf_counter()
            exp_counts += exp.sum(dim=0)
            accumulation_time += time.perf_counter() - t_accum_start

        t_update_start = time.perf_counter()
        smoothed = exp_counts + 1e-6
        logp = (smoothed / smoothed.sum()).clamp_min(1e-12).log()
        self.logp = logp
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

        return {"vocab": len(self.id2piece), "telemetry": telemetry}

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
