"""GPU unigram tokenizer trainer prototype."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch


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

    def _extend_candidates(self, sequences: torch.Tensor) -> None:
        B, L = sequences.shape
        valid = sequences >= 0
        grams: List[Tuple[bytes, int]] = []
        for n in range(2, self.max_len + 1):
            if L < n:
                break
            spans_valid = torch.ones((B, L - n + 1), dtype=torch.bool, device=sequences.device)
            for k in range(n):
                spans_valid &= valid[:, k : L - n + 1 + k]
            if not spans_valid.any():
                continue
            idx = torch.nonzero(spans_valid, as_tuple=False)
            if idx.numel() == 0:
                continue
            take = min(idx.size(0), 1_000_000)
            idx_cpu = idx.to("cpu")
            perm = torch.randperm(idx_cpu.size(0), device=torch.device("cpu"), generator=self._rng)
            idx_cpu = idx_cpu[perm[:take]]
            idx = idx_cpu.to(idx.device)
            rows = idx[:, 0]
            cols = idx[:, 1]
            ng = torch.stack([sequences[rows, cols + k] for k in range(n)], dim=1).to("cpu")
            from collections import Counter

            counts = Counter(map(tuple, ng.tolist()))
            for key, cnt in counts.items():
                grams.append((bytes(key), cnt))
        grams.sort(key=lambda x: -x[1])
        for piece, _ in grams:
            if piece not in self.piece2id:
                new_id = len(self.id2piece)
                self.id2piece[new_id] = piece
                self.piece2id[piece] = new_id
                if len(self.id2piece) >= self.target_vocab:
                    break
        V = len(self.id2piece)
        self.logp = torch.full((V,), -math.log(V), device=self.device)

    def _forward_backward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = torch.arange(len(self.id2piece), device=self.device)
        L = seq.size(0)
        starts: list[list[int]] = [[] for _ in range(L)]
        for pid, piece in self.id2piece.items():
            plen = len(piece)
            if plen == 0 or plen > self.max_len or plen > L:
                continue
            for i in range(0, L - plen + 1):
                if torch.all(seq[i : i + plen].cpu() == torch.tensor(list(piece), dtype=torch.long)):
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

    def fit_epoch(self, batches: List[torch.Tensor]) -> dict[str, int]:
        if len(self.id2piece) < self.target_vocab:
            self._extend_candidates(batches[0])
        V = len(self.id2piece)
        exp_counts = torch.zeros((V,), device=self.device)
        for x in batches:
            x = x.to(self.device)
            valid = x >= 0
            for row in range(x.size(0)):
                seq = x[row][valid[row]].to(self.device)
                _, exp = self._forward_backward(seq)
                exp_counts[: exp.numel()] += exp
        smoothed = exp_counts + 1e-6
        logp = (smoothed / smoothed.sum()).clamp_min(1e-12).log()
        self.logp = logp
        return {"vocab": len(self.id2piece)}


__all__ = ["GPUUnigramTrainer"]
