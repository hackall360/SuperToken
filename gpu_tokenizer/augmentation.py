"""Data augmentation helpers for tokenizer training pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random
from typing import Sequence


class AugmentationMode(str, Enum):
    """Enumeration of supported training-time augmentations."""

    NONE = "none"
    ENTROPY = "entropy"
    DIFFUSION = "diffusion"


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration describing the augmentation policy for a run."""

    mode: AugmentationMode = AugmentationMode.NONE
    strength: float = 0.0
    seed: int | None = None


class AugmentationPipeline:
    """Apply deterministic token-level augmentations to integer sequences."""

    def __init__(
        self,
        mode: AugmentationMode | str = AugmentationMode.NONE,
        *,
        strength: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if not isinstance(mode, AugmentationMode):
            mode = AugmentationMode(str(mode).lower())
        self.mode: AugmentationMode = mode
        self.strength: float = max(0.0, float(strength))
        self.seed: int | None = seed if seed is None else int(seed)
        self._counter = 0

    # ------------------------------------------------------------------
    # Helper constructors
    @classmethod
    def from_config(cls, config: AugmentationConfig) -> "AugmentationPipeline":
        return cls(config.mode, strength=config.strength, seed=config.seed)

    def fork(self) -> "AugmentationPipeline":
        """Spawn an identical pipeline with an independent call counter."""

        return AugmentationPipeline(
            self.mode,
            strength=self.strength,
            seed=self.seed,
        )

    # ------------------------------------------------------------------
    # Public API
    @property
    def enabled(self) -> bool:
        """Return ``True`` when the augmentation mutates sequences."""

        return self.mode is not AugmentationMode.NONE and self.strength > 0.0

    def summary(self) -> dict[str, object]:
        """Describe the current policy for configuration logging."""

        payload: dict[str, object] = {
            "mode": self.mode.value,
            "strength": float(self.strength),
            "enabled": self.enabled,
        }
        if self.seed is not None and self.enabled:
            payload["seed"] = int(self.seed)
        return payload

    def __call__(self, sequence: Sequence[int]) -> list[int]:
        """Augment *sequence* and return a materialised list of integers."""

        data = list(sequence)
        if not data or not self.enabled:
            return data

        rng = self._spawn_rng(self._counter)
        self._counter += 1

        if self.mode is AugmentationMode.ENTROPY:
            return self._apply_entropy(data, rng)
        if self.mode is AugmentationMode.DIFFUSION:
            return self._apply_diffusion(data, rng)
        return data

    # ------------------------------------------------------------------
    # Internal helpers
    def _spawn_rng(self, offset: int) -> random.Random:
        if self.seed is None:
            return random.Random()
        # Spread seeds apart by a large odd number to minimise collisions.
        step = 9_971
        return random.Random(self.seed + offset * step)

    def _apply_entropy(self, data: list[int], rng: random.Random) -> list[int]:
        drop_prob = max(0.0, min(1.0, self.strength))
        if drop_prob == 0.0:
            return data

        kept = [token for token in data if rng.random() >= drop_prob]
        if kept:
            return kept

        # Guarantee that at least one token survives when the drop probability
        # erases the entire sequence.
        return [data[rng.randrange(len(data))]]

    def _apply_diffusion(self, data: list[int], rng: random.Random) -> list[int]:
        if len(data) <= 1:
            return data

        swap_ratio = max(0.0, min(1.0, self.strength))
        swap_count = int(round(swap_ratio * len(data)))
        if swap_count <= 0:
            return data

        result = list(data)
        for _ in range(swap_count):
            left = rng.randrange(len(result))
            if len(result) == 1:
                break
            direction = -1 if rng.random() < 0.5 else 1
            right = max(0, min(len(result) - 1, left + direction))
            if left == right:
                continue
            result[left], result[right] = result[right], result[left]
        return result


def build_augmentation(
    mode: str | AugmentationMode,
    *,
    strength: float = 0.0,
    seed: int | None = None,
) -> AugmentationPipeline:
    """Factory that normalises CLI payloads into :class:`AugmentationPipeline`."""

    if not isinstance(mode, AugmentationMode):
        try:
            mode = AugmentationMode(str(mode).lower())
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported augmentation mode: {mode!r}") from exc
    return AugmentationPipeline(mode, strength=strength, seed=seed)


__all__ = [
    "AugmentationConfig",
    "AugmentationMode",
    "AugmentationPipeline",
    "build_augmentation",
]
