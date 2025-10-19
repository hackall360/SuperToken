import pytest

from gpu_tokenizer.augmentation import (
    AugmentationMode,
    AugmentationPipeline,
    build_augmentation,
)


def test_none_mode_leaves_sequence_untouched() -> None:
    pipeline = build_augmentation("none", strength=1.0, seed=42)
    seq = [1, 2, 3, 4]
    assert pipeline.summary()["enabled"] is False
    assert pipeline(seq) == seq


def test_entropy_mode_is_deterministic_with_seed() -> None:
    seq = list(range(10))
    pipeline_a = AugmentationPipeline(AugmentationMode.ENTROPY, strength=0.5, seed=11)
    pipeline_b = AugmentationPipeline(AugmentationMode.ENTROPY, strength=0.5, seed=11)
    out_a = pipeline_a(seq)
    out_b = pipeline_b(seq)
    assert out_a == out_b
    assert 0 < len(out_a) <= len(seq)
    assert set(out_a).issubset(set(seq))


def test_diffusion_preserves_token_multiset() -> None:
    seq = [0, 1, 2, 3, 4]
    pipeline = AugmentationPipeline(AugmentationMode.DIFFUSION, strength=0.75, seed=7)
    result = pipeline(seq)
    assert sorted(result) == sorted(seq)
    # Replaying the same configuration yields the same permutation.
    clone = AugmentationPipeline(AugmentationMode.DIFFUSION, strength=0.75, seed=7)
    assert clone(seq) == result


def test_fork_produces_equivalent_pipeline() -> None:
    base = build_augmentation("entropy", strength=0.4, seed=19)
    forked = base.fork()
    seq = [5, 6, 7, 8, 9]
    reference = build_augmentation("entropy", strength=0.4, seed=19)
    assert forked(seq) == reference(seq)


def test_build_augmentation_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError):
        build_augmentation("unknown")
