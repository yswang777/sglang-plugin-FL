from types import SimpleNamespace

import pytest
import torch

from sglang_fl.dispatch.backends.vendor.mthreads.impl.mrotary_embedding import (
    mrotary_embedding_musa,
)


def test_mthreads_mrotary_expands_text_positions_and_restores_shapes() -> None:
    calls = []

    def forward_triton(positions, query, key):
        calls.append((positions, query.shape, key.shape))
        return query + 1, key + 2

    obj = SimpleNamespace(mrope_section=[2, 1, 1], forward_triton=forward_triton)
    positions = torch.tensor([3, 4])
    query = torch.randn(2, 3, 8)
    key = torch.randn(2, 1, 8)

    output_query, output_key = mrotary_embedding_musa(
        obj, positions, query, key
    )

    fused_positions, query_shape, key_shape = calls[0]
    assert fused_positions.shape == (3, 2)
    assert torch.equal(fused_positions[0], positions)
    assert torch.equal(fused_positions[1], positions)
    assert torch.equal(fused_positions[2], positions)
    assert query_shape == (2, 24)
    assert key_shape == (2, 8)
    assert output_query.shape == query.shape
    assert output_key.shape == key.shape
    torch.testing.assert_close(output_query, query + 1)
    torch.testing.assert_close(output_key, key + 2)


def test_mthreads_mrotary_preserves_multiaxis_positions() -> None:
    positions = torch.tensor([[0, 1], [2, 3], [4, 5]])
    seen = []

    def forward_triton(fused_positions, query, key):
        seen.append(fused_positions)
        return query, key

    obj = SimpleNamespace(mrope_section=[2, 1, 1], forward_triton=forward_triton)
    mrotary_embedding_musa(
        obj, positions, torch.randn(2, 8), torch.randn(2, 8)
    )
    assert seen[0] is positions


def test_mthreads_mrotary_rejects_invalid_position_shape() -> None:
    obj = SimpleNamespace(mrope_section=[2, 1, 1], forward_triton=lambda *args: args)
    with pytest.raises(ValueError, match=r"positions \[T\] or \[3, T\]"):
        mrotary_embedding_musa(
            obj,
            torch.zeros(2, 2, dtype=torch.long),
            torch.randn(2, 8),
            torch.randn(2, 8),
        )
