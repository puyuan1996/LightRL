from __future__ import annotations

import torch

from slime.world_model.train_result_transfer import (
    LatentResultProbe,
    _batches,
    _loss,
)


def test_result_probe_preserves_shape_and_normalizes() -> None:
    probe = LatentResultProbe(8)
    output = probe(torch.randn(5, 8))

    assert output.shape == (5, 8)
    assert torch.allclose(output.norm(dim=-1), torch.ones(5), atol=1e-6)


def test_result_probe_batches_are_seeded_and_complete() -> None:
    first = _batches(range(11), 4, seed=17, shuffle=True)
    second = _batches(range(11), 4, seed=17, shuffle=True)

    assert first == second
    assert sorted(value for batch in first for value in batch) == list(range(11))
    assert [len(batch) for batch in first] == [4, 4, 3]


def test_result_probe_loss_prefers_aligned_vectors() -> None:
    target = torch.randn(6, 8)
    aligned = _loss(target, target)
    reversed_direction = _loss(-target, target)

    assert aligned.item() < 1e-10
    assert reversed_direction.item() > 3.9
