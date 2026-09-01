# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MUSA Triton paged absorbed-MLA decode kernel.

GLM-5.3 has no RoPE sub-dimension in its full-attention layers.  Its absorbed
MLA decode is therefore an MQA attention over the paged latent cache:

    scores[h, t] = dot(q_absorbed[h], latent_kv[t]) * scale
    out[h] = softmax(scores[h]) @ latent_kv

The key and value are the same 512-wide latent vector.  Reading that cache
directly avoids reconstructing and projecting full per-head K/V for every
historical token on every decode step.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_mla_decode_kernel(
    q_ptr,
    cache_ptr,
    page_table_ptr,
    seq_lens_ptr,
    out_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_cp: tl.constexpr,
    stride_ct: tl.constexpr,
    stride_ch: tl.constexpr,
    stride_cd: tl.constexpr,
    stride_ptb: tl.constexpr,
    stride_ptp: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
):
    program = tl.program_id(0)
    batch_idx = program // NUM_HEADS
    head_idx = program - batch_idx * NUM_HEADS

    d = tl.arange(0, HEAD_DIM)
    q = tl.load(
        q_ptr
        + batch_idx * stride_qb
        + head_idx * stride_qh
        + d * stride_qd
    ).to(tl.float32)

    seq_len = tl.load(seq_lens_ptr + batch_idx).to(tl.int32)
    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    # tl.range accepts a runtime upper bound, so graph replay reads the live
    # sequence length without compiling or padding to the maximum context.
    for start_n in tl.range(0, seq_len, BLOCK_N):
        n = start_n + tl.arange(0, BLOCK_N)
        valid_n = n < seq_len
        logical_page = n // PAGE_SIZE
        token_in_page = n - logical_page * PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr
            + batch_idx * stride_ptb
            + logical_page * stride_ptp,
            mask=valid_n,
            other=0,
        ).to(tl.int64)

        cache_offsets = (
            physical_page[:, None] * stride_cp
            + token_in_page[:, None] * stride_ct
            + d[None, :] * stride_cd
        )
        latent = tl.load(
            cache_ptr + cache_offsets,
            mask=valid_n[:, None],
            other=0.0,
        ).to(tl.float32)

        logits = tl.sum(latent * q[None, :], axis=1) * SOFTMAX_SCALE
        logits = tl.where(valid_n, logits, -float("inf"))
        block_max = tl.max(logits, axis=0)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(logits - new_max)

        accumulator = accumulator * old_scale + tl.sum(
            latent * probabilities[:, None], axis=0
        )
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=0)
        running_max = new_max

    output = accumulator / running_sum
    tl.store(
        out_ptr
        + batch_idx * stride_ob
        + head_idx * stride_oh
        + d * stride_od,
        output,
    )


def musa_paged_mla_decode(
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    softmax_scale: float,
    *,
    block_n: int = 16,
) -> torch.Tensor:
    """Decode one token per request from a paged latent MLA cache.

    Args:
        q: ``[batch, heads, latent_dim]`` absorbed query.
        latent_cache: ``[num_pages, page_size, 1, latent_dim]`` BF16 cache.
        page_table: ``[batch, max_pages]`` logical-to-physical page mapping.
        cache_seqlens: ``[batch]`` live sequence lengths.
        softmax_scale: attention scale.
    """
    if q.ndim != 3:
        raise ValueError(f"Expected q rank 3, got shape={tuple(q.shape)}")
    if latent_cache.ndim != 4 or latent_cache.shape[2] != 1:
        raise ValueError(
            "Expected latent_cache [pages, page_size, 1, dim], "
            f"got shape={tuple(latent_cache.shape)}"
        )
    if q.shape[-1] != latent_cache.shape[-1]:
        raise ValueError(
            f"Query/cache latent dimensions differ: {q.shape[-1]} vs "
            f"{latent_cache.shape[-1]}"
        )
    if q.shape[-1] & (q.shape[-1] - 1):
        raise ValueError("The initial MUSA kernel requires a power-of-two latent dim")
    if block_n not in (8, 16, 32):
        raise ValueError(f"Unsupported block_n={block_n}")

    q = q.contiguous()
    latent_cache = latent_cache.contiguous()
    page_table = page_table.contiguous()
    cache_seqlens = cache_seqlens.contiguous()
    output = torch.empty_like(q)
    batch, heads, head_dim = q.shape

    _paged_mla_decode_kernel[(batch * heads,)](
        q,
        latent_cache,
        page_table,
        cache_seqlens,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        latent_cache.stride(0),
        latent_cache.stride(1),
        latent_cache.stride(2),
        latent_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_HEADS=heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=latent_cache.shape[1],
        BLOCK_N=block_n,
        SOFTMAX_SCALE=softmax_scale,
        num_warps=4,
        num_stages=1,
    )
    return output
