"""Qwen4-Exp (internal Qwen3.8-Flash-Next) ops for sglang.

Ported from vllm-plugin-fl-qwen4-day0.patch
``vllm_fl/models/qwen3_8_flash_next/common/hyperconnection.py``.

HyperConnection residual scheme (https://arxiv.org/abs/2409.19606): hidden
states between layers have shape ``[..., HC*HS]`` with HC outer, HS inner
(checkpoint-native layout). Each decoder layer owns an ``attn_hyper_connection``
and a ``mlp_hyper_connection``; the model root owns a final
``hyper_connection_mixer`` (use_combine=False) that folds the HC streams back
into a single hidden state.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

_STATE_CAPACITY = int(os.environ.get("SGLANG_QWEN4_STATE_CAPACITY", "6208"))


@triton.jit
def _qsa_write_raw_and_pool_kernel(
    raw,
    src,
    pos_ptr,
    pooled,
    stride_raw,
    HEAD_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    cols = tl.arange(0, BLOCK)
    mask = cols < HEAD_DIM
    pos = tl.load(pos_ptr).to(tl.int64)
    values = tl.load(src + cols, mask=mask, other=0.0)
    tl.store(raw + pos * stride_raw + cols, values, mask=mask)
    group_start = (pos // RATIO) * RATIO
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for offset in range(RATIO):
        acc += tl.load(
            raw + (group_start + offset) * stride_raw + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    tl.store(pooled + cols, acc / RATIO, mask=mask)


@triton.jit
def _qsa_store_row_kernel(
    dst,
    src,
    row_ptr,
    stride_dst,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    cols = tl.arange(0, BLOCK)
    mask = cols < WIDTH
    row = tl.load(row_ptr).to(tl.int64)
    tl.store(
        dst + row * stride_dst + cols,
        tl.load(src + cols, mask=mask, other=0.0),
        mask=mask,
    )


@triton.jit
def _qsa_write_kv_kernel(
    key_cache,
    value_cache,
    key,
    value,
    pos_ptr,
    row_width,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < row_width
    pos = tl.load(pos_ptr).to(tl.int64)
    tl.store(
        key_cache + pos * row_width + offsets,
        tl.load(key + offsets, mask=mask, other=0.0),
        mask=mask,
    )
    tl.store(
        value_cache + pos * row_width + offsets,
        tl.load(value + offsets, mask=mask, other=0.0),
        mask=mask,
    )


@triton.jit
def _qsa_gather_kv_kernel(
    key_cache,
    value_cache,
    indices,
    gathered_key,
    gathered_value,
    width,
    row_width,
    total,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    selected_col = offsets // row_width
    inner = offsets % row_width
    row = tl.load(indices + selected_col, mask=mask, other=0).to(tl.int64)
    source = row * row_width + inner
    tl.store(
        gathered_key + offsets,
        tl.load(key_cache + source, mask=mask, other=0.0),
        mask=mask,
    )
    tl.store(
        gathered_value + offsets,
        tl.load(value_cache + source, mask=mask, other=0.0),
        mask=mask,
    )


class Qwen4ExpGroupedGemmaRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        group_size: int | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size})"
            )
        self.variance_epsilon = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        if self.group_size is None:
            variance = hidden_states.square().mean(dim=-1, keepdim=True)
            normalized = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        else:
            grouped = hidden_states.unflatten(
                -1, (hidden_states.shape[-1] // self.group_size, self.group_size)
            )
            variance = grouped.square().mean(dim=-1, keepdim=True)
            normalized = (
                grouped * torch.rsqrt(variance + self.variance_epsilon)
            ).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)


class Qwen4ExpGatedResidualSimple(nn.Module):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    ``mix()`` applies a grouped GemmaRMSNorm (per HC stream, group size =
    hidden_size) and projects through a low-rank silu/sigmoid gate to produce
    a single block input. ``combine()`` injects the block output back into each
    stream through a learned per-stream injection weight.

    Checkpoint weight names under a layer::

        <prefix>.input_mix_weight_down.weight [hc_lowrank, hc_count*hidden]
        <prefix>.input_mix_weight_up.weight   [hc_count*hidden, hc_lowrank]
        <prefix>.block_inject_weight.weight   [hc_count, hc_count*hidden]
        <prefix>.hc_norm.weight               [hc_count*hidden]  (zero-init, 1+w)

    M1 keeps these weights TP-replicated (plain ``nn.Linear``, no sharding).
    """

    def __init__(
        self,
        hc_count: int,
        hidden_size: int,
        hc_lowrank: int,
        rms_norm_eps: float = 1e-6,
        use_mix: bool = True,
        use_combine: bool = True,
        dtype: torch.dtype | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        self.hc_lowrank = hc_lowrank
        self.rms_norm_eps = rms_norm_eps
        self.prefix = prefix
        self.hyper_hidden_size = hc_count * hidden_size

        # Normalize each H-sized HC stream independently while retaining a
        # separate affine weight for every element of the HC*H layout.
        self.hc_norm = Qwen4ExpGroupedGemmaRMSNorm(
            self.hyper_hidden_size,
            eps=rms_norm_eps,
            group_size=hidden_size,
            dtype=dtype,
        )

        # -- raw Linear weights (checkpoint-compatible) ----------------------
        if use_mix:
            self.input_mix_weight_down = nn.Linear(
                self.hyper_hidden_size,
                hc_lowrank,
                bias=False,
                dtype=dtype,
            )
            self.input_mix_weight_up = nn.Linear(
                hc_lowrank,
                self.hyper_hidden_size,
                bias=False,
                dtype=dtype,
            )
        if use_combine:
            self.block_inject_weight = nn.Linear(
                self.hyper_hidden_size,
                hc_count,
                bias=False,
                dtype=dtype,
            )

    def _normalize(self, hyper_input: torch.Tensor) -> torch.Tensor:
        # GroupedGemmaRMSNorm already groups over the HS dim of each stream.
        return self.hc_norm(hyper_input)

    def mix(
        self, hyper_input: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Mix: RMSNorm -> low-rank gate -> gated mean."""
        import os as _os

        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        if not hasattr(self, "input_mix_weight_down"):
            raise RuntimeError("mix was disabled for this hyper-connection")
        T = hyper_input.shape[0]
        HS, HC = self.hidden_size, self.hc_count
        hyper = HC * HS
        use_triton = (
            hyper_input.device.type in ("cuda", "musa")
            and int(_os.environ.get("SGLANG_QWEN4_FORCE_TORCH_HC", "0")) == 0
            and HC in (2, 4)
            and self.hc_lowrank % 64 == 0
            and HS % 64 == 0
            and T > 0
        )
        if use_triton:
            gate = torch.empty(
                (T, self.hc_lowrank), dtype=hyper_input.dtype, device=hyper_input.device
            )
            normed = torch.empty_like(hyper_input)
            _hc_mix_down_kernel[(T, self.hc_lowrank // 64)](
                hyper_input, self.input_mix_weight_down.weight, self.hc_norm.weight,
                gate, normed, T, self.hc_lowrank,
                HC=HC, HS=HS, EPS=self.hc_norm.variance_epsilon,
                stride_wk=hyper, BLOCK_K=64, BLOCK_N=64,
            )
            out = torch.empty(
                (T, HS), dtype=hyper_input.dtype, device=hyper_input.device
            )
            _hc_mix_up_kernel[(T, HS // 64)](
                gate, self.input_mix_weight_up.weight, normed, out, T,
                HC=HC, HS=HS, LOWRANK=self.hc_lowrank,
                stride_wk=self.hc_lowrank, BLOCK_K=64, BLOCK_H=64,
            )
            return out, (hyper_input, normed)
        # Ascend's torch fallback creates fp32 RMSNorm/gate temporaries.  At a
        # 16K-token prefill the first one is ~314 MiB and eventually cannot be
        # satisfied from the fragmented allocator.  HC mixing is row-local, so
        # fill persistent outputs from bounded row chunks without changing any
        # per-token arithmetic.  Decode batches are small and keep the original
        # single-call path below.
        if hyper_input.device.type == "npu" and T > 256:
            normed = torch.empty_like(hyper_input)
            mixed = torch.empty(
                (T, HS), dtype=hyper_input.dtype, device=hyper_input.device
            )
            for start in range(0, T, 256):
                end = min(start + 256, T)
                chunk_normed = self._normalize(hyper_input[start:end])
                normed[start:end] = chunk_normed
                chunk_gate = F.silu(
                    F.linear(chunk_normed, self.input_mix_weight_down.weight) / HC
                )
                chunk_gate = torch.sigmoid(
                    F.linear(chunk_gate, self.input_mix_weight_up.weight)
                ).unflatten(-1, (HC, HS))
                mixed[start:end] = (
                    chunk_gate * chunk_normed.unflatten(-1, (HC, HS))
                ).mean(dim=-2)
            return mixed, (hyper_input, normed)
        hyper_input_normed = self._normalize(hyper_input)
        # Gate — original mix order: linear+silu then linear+sigmoid.
        gate = F.silu(
            F.linear(hyper_input_normed, self.input_mix_weight_down.weight)
            / self.hc_count
        )
        gate = torch.sigmoid(F.linear(gate, self.input_mix_weight_up.weight)).unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        mixed_input = (
            gate * hyper_input_normed.unflatten(-1, (self.hc_count, self.hidden_size))
        ).mean(dim=-2)
        return mixed_input.to(hyper_input.dtype), (hyper_input, hyper_input_normed)

    def combine(
        self,
        block_output: torch.Tensor,
        residuals: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        import os as _os

        if not hasattr(self, "block_inject_weight"):
            raise RuntimeError("combine was disabled for this hyper-connection")
        hyper_input, hyper_input_normed = residuals
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size
        T = block_output.shape[0]
        HS, HC = self.hidden_size, self.hc_count
        use_triton = (
            hyper_input.device.type in ("cuda", "musa")
            and int(_os.environ.get("SGLANG_QWEN4_FORCE_TORCH_HC", "0")) == 0
            and HC in (2, 4)
            and HS % 64 == 0
            and T > 0
        )
        if use_triton:
            out = torch.empty_like(hyper_input)
            _hc_combine_kernel[(T, HC)](
                hyper_input_normed, self.block_inject_weight.weight,
                hyper_input, block_output, out, T,
                HC=HC, HS=HS, stride_wi=HC * HS, BLOCK_K=64,
            )
            return out
        # Bound the torch fallback's full-hyper-width injection/residual
        # temporaries for long Ascend prefills.  Each output row is independent.
        if hyper_input.device.type == "npu" and T > 256:
            out = torch.empty_like(hyper_input)
            for start in range(0, T, 256):
                end = min(start + 256, T)
                chunk_residual = hyper_input[start:end].unflatten(-1, (HC, HS))
                chunk_injection = 2.0 * torch.sigmoid(
                    F.linear(
                        hyper_input_normed[start:end],
                        self.block_inject_weight.weight,
                    )
                    / HC
                )
                out[start:end] = (
                    chunk_residual
                    + block_output[start:end].unsqueeze(-2)
                    * chunk_injection.unsqueeze(-1)
                ).flatten(-2)
            return out
        residual = hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
        # The paired mix keeps its normalized hyper input so combine uses the
        # same HC module's injection weight.
        injection_weight = 2.0 * torch.sigmoid(
            F.linear(hyper_input_normed, self.block_inject_weight.weight)
            / self.hc_count
        )
        if int(_os.environ.get("SGLANG_QWEN4_DEBUG", "0")) and not getattr(
            self, "_dbg_done", False
        ):
            self._dbg_done = True
            print(
                f"[hc-c] prefix={self.prefix} "
                f"inject_w_max={self.block_inject_weight.weight.float().abs().max().item():.4f} "
                f"inject_val_max={injection_weight.float().max().item():.4f} "
                f"block_out_max={block_output.float().abs().max().item():.4f}",
                flush=True,
            )
        output = residual + block_output.unsqueeze(-2) * injection_weight.unsqueeze(-1)
        return output.flatten(-2).to(hyper_input.dtype)


# ---------------------------------------------------------------------------
# QSA (Query-Selective Attention): weight-free indexer + sparse GQA.
# M1 keeps a per-request contiguous state (no paged cache); the math matches
# tests/reference.py's qsa_*_reference functions exactly.
# ---------------------------------------------------------------------------


def apply_qsa_rope(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Apply the main attention's exact 1D/MRoPE composition to QSA heads.

    Ported from port-ref/indexer_qsa.py apply_qsa_rope (fast paths removed).
    ``positions`` is 1-D ``[T]`` or 2-D ``[3, T]`` (MRoPE axes).
    """
    rotary_dim = rotary_emb.rotary_dim
    rotary_emb._match_cos_sin_cache_dtype(tensor)  # noqa: SLF001
    cos_sin = rotary_emb.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        sections = rotary_emb.mrope_section
        if rotary_emb.mrope_interleaved:
            axis_cos, axis_sin = cos, sin
            channels = torch.arange(cos.shape[-1], device=cos.device)
            is_height = (channels % 3 == 1) & (channels < sections[1] * 3)
            is_width = (channels % 3 == 2) & (channels < sections[2] * 3)
            cos = torch.where(is_height, axis_cos[1], axis_cos[0])
            cos = torch.where(is_width, axis_cos[2], cos)
            sin = torch.where(is_height, axis_sin[1], axis_sin[0])
            sin = torch.where(is_width, axis_sin[2], sin)
        else:
            cos = torch.cat(
                [axis[index] for index, axis in enumerate(cos.split(sections, dim=-1))],
                dim=-1,
            )
            sin = torch.cat(
                [axis[index] for index, axis in enumerate(sin.split(sections, dim=-1))],
                dim=-1,
            )
    # Native (public operator) composition — same math as vLLM forward_native.
    rotated = rotary_emb._apply_rotary_emb_wrapped(  # noqa: SLF001
        tensor[..., :rotary_dim], cos, sin, rotary_emb.is_neox_style
    )
    return torch.cat((rotated, tensor[..., rotary_dim:]), dim=-1)


class Qwen4ExpQSAState:
    """Per-request contiguous buffers for raw and compressed indexer keys.

    M1 container: ``raw_key [max_tokens, 1, head_dim]`` (unnormalized token
    keys), ``compressed_key [max_blocks, 1, head_dim]`` (pooled group keys).
    No block table — direct indexing, but the pooling/position rules match
    ``qsa_compress_groups_reference``.
    """

    def __init__(
        self,
        max_tokens: int,
        index_head_dim: int,
        compress_ratio: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
    ) -> None:
        self.head_dim = index_head_dim
        self.compress_ratio = compress_ratio
        self.raw_len = 0
        self.compressed_len = 0
        max_blocks = (max_tokens + compress_ratio - 1) // compress_ratio
        self.raw_key = torch.zeros(
            (max_tokens, 1, index_head_dim), dtype=dtype, device=device
        )
        self.compressed_key = torch.zeros(
            (max_blocks, 1, index_head_dim), dtype=dtype, device=device
        )
        # Absolute position of each appended raw token (text: 1-D scalar).
        self.positions_buffer = torch.zeros(
            (max_tokens,), dtype=torch.int64, device=device
        )

    def append_raw(self, token_k: torch.Tensor, position: int = 0) -> None:
        assert self.raw_len < self.raw_key.shape[0]
        self.raw_key[self.raw_len].copy_(token_k.reshape(1, -1))
        self.positions_buffer[self.raw_len] = int(position)
        self.raw_len += 1

    def append_raw_batch(
        self, token_k: torch.Tensor, positions: torch.Tensor
    ) -> None:
        """Append a contiguous prefill chunk without per-token device syncs."""
        token_k = token_k.reshape(-1, 1, self.head_dim)
        count = token_k.shape[0]
        end = self.raw_len + count
        assert end <= self.raw_key.shape[0]
        flat_positions = positions.reshape(-1)
        if flat_positions.numel() != count:
            raise ValueError(
                "QSA state positions and token keys must have equal length"
            )
        self.raw_key[self.raw_len : end].copy_(token_k)
        self.positions_buffer[self.raw_len : end].copy_(
            flat_positions.to(dtype=torch.int64, device=self.raw_key.device)
        )
        self.raw_len = end

    def reset(self) -> None:
        self.raw_len = 0
        self.compressed_len = 0

    def graph_append_raw(
        self, token_k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append one decode row using only device-side indexed writes.

        The returned scalar position is also the logical row index.  Unlike
        ``raw_len`` it changes between graph replays without a host update.
        """
        pos = positions.reshape(-1)[:1].to(
            device=self.raw_key.device, dtype=torch.int64
        )
        pooled = torch.empty(
            (1, 1, self.head_dim), dtype=self.raw_key.dtype, device=self.raw_key.device
        )
        _qsa_write_raw_and_pool_kernel[(1,)](
            self.raw_key,
            token_k,
            pos,
            pooled,
            self.raw_key.stride(0),
            HEAD_DIM=self.head_dim,
            RATIO=self.compress_ratio,
            BLOCK=triton.next_power_of_2(self.head_dim),
        )
        return pos, pooled


class Qwen4ExpQSAKVState:
    """Per-request K/V history for QSA sparse attention.

    The indexer selects absolute (request-relative) token positions that may
    reach back into the request's history, so the sparse attention must gather
    from a running K/V buffer (not just the current forward's tokens).
    """

    def __init__(
        self,
        max_tokens: int,
        kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
    ) -> None:
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.k = torch.zeros(
            (max_tokens, kv_heads, head_dim), dtype=dtype, device=device
        )
        self.v = torch.zeros(
            (max_tokens, kv_heads, head_dim), dtype=dtype, device=device
        )
        self.len = 0

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        n = k.shape[0]
        self.k[self.len : self.len + n] = k
        self.v[self.len : self.len + n] = v
        self.len += n

    def reset(self) -> None:
        self.len = 0

    def graph_append(
        self, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Write one decode K/V row at its device-side absolute position."""
        pos = positions.reshape(-1)[:1].to(device=self.k.device, dtype=torch.int64)
        row_width = self.kv_heads * self.head_dim
        _qsa_write_kv_kernel[(1,)](
            self.k,
            self.v,
            k,
            v,
            pos,
            row_width,
            BLOCK=triton.next_power_of_2(row_width),
        )
        return pos


class Qwen4ExpQSAIndexer(nn.Module):
    """Replicated Q/K projection plus paged, weight-free QSA selection.

    Checkpoint weight names: ``index_qk_proj.weight [640, 2560]``,
    ``q_layernorm.weight [128]``, ``k_layernorm.weight [128]``.
    M1 keeps these replicated (plain ``nn.Linear``).
    """

    def __init__(
        self,
        hidden_size: int,
        indexer_n_heads: int = 4,
        indexer_kv_heads: int = 1,
        indexer_head_dim: int = 128,
        budget: int = 2048,
        compress_ratio: int = 4,
        rms_norm_eps: float = 1e-6,
        prefix: str = "",
        rotary_emb: nn.Module | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.indexer_n_heads = indexer_n_heads
        self.indexer_kv_heads = indexer_kv_heads
        self.indexer_head_dim = indexer_head_dim
        self.budget = budget
        self.compress_ratio = compress_ratio
        self.rotary_emb = rotary_emb
        self.prefix = prefix

        self.index_qk_proj = nn.Linear(
            hidden_size,
            (indexer_n_heads + indexer_kv_heads) * indexer_head_dim,
            bias=False,
            dtype=dtype,
        )
        self.q_layernorm = Qwen4ExpGroupedGemmaRMSNorm(
            indexer_head_dim, eps=rms_norm_eps, group_size=None, dtype=dtype
        )
        self.k_layernorm = Qwen4ExpGroupedGemmaRMSNorm(
            indexer_head_dim, eps=rms_norm_eps, group_size=None, dtype=dtype
        )

    @property
    def output_width(self) -> int:
        return self.budget + self.compress_ratio - 1

    def project_qk(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project replicated Q/K, normalize+rotate Q, and preserve raw K."""
        qk = self.index_qk_proj(hidden_states)
        q_raw, token_k = qk.split(
            (
                self.indexer_n_heads * self.indexer_head_dim,
                self.indexer_kv_heads * self.indexer_head_dim,
            ),
            dim=-1,
        )
        q = q_raw.reshape(-1, self.indexer_n_heads, self.indexer_head_dim)
        flat_q = q.reshape(-1, self.indexer_head_dim)
        normalized_q = self.q_layernorm(flat_q)
        q = normalized_q.reshape_as(q)
        if self.rotary_emb is not None:
            q = apply_qsa_rope(self.rotary_emb, positions, q)
        return q, token_k.reshape(-1, 1, self.indexer_head_dim)

    def manual_compress(
        self,
        st: Qwen4ExpQSAState,
        k_norm_noop: bool = False,
    ) -> None:
        """Pool each completed group of raw keys -> k_norm -> group-first rope."""
        ratio = self.compress_ratio
        head_dim = self.indexer_head_dim
        first_group = st.compressed_len
        end_group = st.raw_len // ratio
        if end_group <= first_group:
            return

        # Every group is independent. Batch all newly completed groups to
        # avoid thousands of tiny mean/norm/RoPE launches during a 4K prefill.
        raw = st.raw_key[first_group * ratio : end_group * ratio]
        groups = raw.reshape(-1, ratio, 1, head_dim).float().mean(dim=1)
        if not k_norm_noop:
            groups = self.k_layernorm(
                groups.reshape(-1, head_dim)
            ).reshape(-1, 1, head_dim)
        if self.rotary_emb is not None:
            # Group rope position = each group's first token absolute position.
            first_positions = st.positions_buffer[
                first_group * ratio : end_group * ratio : ratio
            ]
            groups = apply_qsa_rope(self.rotary_emb, first_positions, groups)
        st.compressed_key[first_group:end_group].copy_(
            groups.to(st.compressed_key.dtype)
        )
        st.compressed_len = end_group

    def manual_compress_batched(
        self, states: list[Qwen4ExpQSAState]
    ) -> None:
        """Compress newly completed decode groups from many requests together."""
        ratio = self.compress_ratio
        head_dim = self.indexer_head_dim
        pending: list[tuple[Qwen4ExpQSAState, int]] = []
        for state in states:
            end_group = state.raw_len // ratio
            pending.extend(
                (state, group)
                for group in range(state.compressed_len, end_group)
            )
        if not pending:
            return

        raw_groups = torch.stack(
            [
                state.raw_key[group * ratio : (group + 1) * ratio]
                for state, group in pending
            ],
            dim=0,
        )
        groups = raw_groups.float().mean(dim=1)
        groups = self.k_layernorm(
            groups.reshape(-1, head_dim)
        ).reshape(-1, 1, head_dim)
        if self.rotary_emb is not None:
            first_positions = torch.stack(
                [
                    state.positions_buffer[group * ratio]
                    for state, group in pending
                ]
            )
            groups = apply_qsa_rope(
                self.rotary_emb, first_positions, groups
            )
        groups = groups.to(pending[0][0].compressed_key.dtype)
        for row, (state, group) in enumerate(pending):
            state.compressed_key[group : group + 1].copy_(groups[row : row + 1])
            state.compressed_len = group + 1

    def update_compress(
        self,
        token_k: torch.Tensor,
        positions: torch.Tensor,
        request_state: Qwen4ExpQSAState,
    ) -> None:
        request_state.append_raw_batch(token_k, positions)
        self.manual_compress(request_state)

    def select_with_q(
        self,
        q: torch.Tensor,
        st: Qwen4ExpQSAState,
        positions: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return fixed-width request-relative token indices padded with ``-1``.

        Math matches ``qsa_mqa_paged_reference`` (scoring) +
        ``expand_qsa_indices_reference`` (expansion). ``positions`` carries the
        ABSOLUTE query position per row (decode rows sit at their true
        sequence position, not row index 0).
        """
        rows = q.shape[0]
        ratio = self.compress_ratio
        block_topk = self.budget // ratio
        width = self.output_width
        seq_len = st.raw_len
        num_groups = st.compressed_len
        device = q.device

        # The fixed-width selector uses several [rows, width] int64 tensors.
        # At 4096 rows each is roughly 54 MiB and the final nested where needs
        # another full-size allocation. Later scheduler waves can have enough
        # total memory but no contiguous block for that allocation. Selection
        # is row independent, so fill one contiguous output in bounded chunks.
        if device.type == "npu" and rows > 256:
            result = (
                out
                if out is not None
                else torch.empty((rows, width), dtype=torch.int64, device=device)
            )
            for start in range(0, rows, 256):
                end = min(start + 256, rows)
                self.select_with_q(
                    q[start:end],
                    st,
                    None if positions is None else positions[start:end],
                    result[start:end],
                )
            return result

        block_indices = torch.full(
            (rows, block_topk), -1, dtype=torch.int64, device=device
        )
        if num_groups > 0:
            # Vectorized scoring: [T, heads, M] dots -> clamp -> sum -> scale.
            qpos = (
                positions.to(torch.int64)
                if positions is not None
                else torch.arange(rows, dtype=torch.int64, device=device)
            )
            qpos = qpos.clamp_min(0)
            visible = torch.minimum(
                torch.minimum(
                    (qpos + 1) // ratio,
                    torch.full_like(qpos, seq_len // ratio),
                ),
                torch.full_like(qpos, num_groups),
            )
            mask = torch.arange(num_groups, device=device)[None, :] < visible[:, None]
            scores = torch.einsum(
                "rhd,cd->rhc", q.float(), st.compressed_key[:num_groups, 0].float()
            )
            scores = (
                scores.clamp_min(0).sum(dim=1) / (q.shape[-1] ** 0.5)
            )  # [T, M]
            scored = torch.where(
                mask, scores, torch.full_like(scores, float("-inf"))
            )
            k = min(block_topk, num_groups)
            topk_vals, topk_idx = torch.topk(scored, k, dim=-1)
            # Rows with visible < k pad with -inf; blank those slots. Also pad
            # up to block_topk when num_groups < block_topk.
            finite_mask = topk_vals != float("-inf")
            idx_padded = torch.full(
                (rows, block_topk), -1, dtype=torch.int64, device=device
            )
            idx_padded[:, :k] = torch.where(
                finite_mask, topk_idx.to(torch.int64), torch.full_like(topk_idx, -1)
            )
            block_indices = idx_padded
            import os as _os3

            if int(_os3.environ.get("SGLANG_QWEN4_DEBUG", "0")):
                print(
                    f"[qsa-sel] qpos0={int(qpos[0])} visible0={int(visible[0])} "
                    f"topk_blocks={block_indices[0].tolist()[:8]}",
                    flush=True,
                )

        if out is None:
            result = torch.full(
                (rows, width), -1, dtype=torch.int64, device=device
            )
        else:
            result = out
            result.fill_(-1)
        if rows == 0:
            return result
        # Vectorized block->token expansion (identical to
        # expand_qsa_indices_reference): each row expands its top-k compressed
        # blocks into 'ratio' tokens each, then appends the current partial
        # block's tail.
        qpos = (
            positions.to(torch.int64)
            if positions is not None
            else torch.arange(rows, dtype=torch.int64, device=device)
        )
        qpos = qpos.clamp_min(0)
        complete_blocks = torch.minimum(
            torch.minimum((qpos + 1) // ratio, torch.full_like(qpos, seq_len // ratio)),
            torch.full_like(qpos, block_topk),
        )
        expanded_count = complete_blocks * ratio  # [rows]
        tail_start = ((qpos + 1) // ratio) * ratio  # [rows]
        tail_count = (qpos + 1) - tail_start  # [rows]
        cols = torch.arange(width, device=device)  # [width]
        # Complete-block region: token = block_idx[rank]*ratio + col%ratio.
        cb_mask = cols[None, :] < expanded_count[:, None]  # [rows, width]
        block_rank = (cols[None, :] // ratio).clamp_max(block_topk - 1)
        gathered = block_indices.gather(1, block_rank.expand(rows, -1))  # [rows, width]
        token_cb = gathered * ratio + (cols[None, :] % ratio)
        valid_cb = cb_mask & (token_cb >= 0) & (token_cb < seq_len)
        # Tail region: current partial block after the top-k blocks.
        tail_offset = cols[None, :] - expanded_count[:, None]
        token_tail = tail_start[:, None] + tail_offset
        valid_tail = (
            (~cb_mask)
            & (tail_offset < tail_count[:, None])
            & (tail_offset < ratio - 1)
            & (token_tail >= 0)
            & (token_tail < seq_len)
        )
        torch.where(valid_tail, token_tail, result, out=result)
        torch.where(valid_cb, token_cb, result, out=result)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def select(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        request_state: Qwen4ExpQSAState,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full indexer pass: project -> update/compress -> select."""
        q, token_k = self.project_qk(hidden_states, positions)
        self.update_compress(token_k, positions, request_state)
        return self.select_with_q(q, request_state, positions, out)

    def select_graph_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        request_state: Qwen4ExpQSAState,
    ) -> torch.Tensor:
        """Fixed-shape, device-indexed QSA update and selection for graph replay."""
        q, token_k = self.project_qk(hidden_states, positions)
        scalar_positions = positions if positions.ndim == 1 else positions[0]
        pos, pooled = request_state.graph_append_raw(token_k, scalar_positions)
        ratio = self.compress_ratio
        group = torch.div(pos, ratio, rounding_mode="floor")
        pooled = self.k_layernorm(pooled.reshape(-1, self.indexer_head_dim)).reshape(
            -1, 1, self.indexer_head_dim
        )
        first_positions = group * ratio
        if self.rotary_emb is not None:
            pooled = apply_qsa_rope(self.rotary_emb, first_positions, pooled)
        _qsa_store_row_kernel[(1,)](
            request_state.compressed_key,
            pooled,
            group,
            request_state.compressed_key.stride(0),
            WIDTH=self.indexer_head_dim,
            BLOCK=triton.next_power_of_2(self.indexer_head_dim),
        )

        max_groups = request_state.compressed_key.shape[0]
        block_topk = self.budget // ratio
        width = self.output_width
        visible = (pos + 1) // ratio
        scores = torch.einsum(
            "bhd,cd->bhc", q.float(), request_state.compressed_key[:, 0].float()
        )
        scores = scores.clamp_min(0).sum(dim=1) / (q.shape[-1] ** 0.5)
        group_ids = torch.arange(max_groups, device=pos.device)[None, :]
        scores = torch.where(
            group_ids < visible[:, None],
            scores,
            torch.full_like(scores, float("-inf")),
        )
        vals, blocks = torch.topk(scores, block_topk, dim=-1)
        blocks = torch.where(
            torch.isfinite(vals), blocks.to(torch.int64), torch.full_like(blocks, -1)
        )
        complete_blocks = torch.minimum(
            visible, torch.full_like(visible, block_topk)
        )
        expanded_count = complete_blocks * ratio
        tail_start = visible * ratio
        tail_count = (pos + 1) - tail_start
        cols = torch.arange(width, device=pos.device)
        complete_mask = cols[None, :] < expanded_count[:, None]
        block_rank = (cols[None, :] // ratio).clamp_max(block_topk - 1)
        expanded = blocks.gather(1, block_rank.expand(q.shape[0], -1)) * ratio
        expanded = expanded + cols[None, :] % ratio
        tail_offset = cols[None, :] - expanded_count[:, None]
        tail = tail_start[:, None] + tail_offset
        valid_complete = complete_mask & (expanded >= 0) & (expanded <= pos[:, None])
        valid_tail = (
            (~complete_mask)
            & (tail_offset < tail_count[:, None])
            & (tail_offset < ratio - 1)
            & (tail <= pos[:, None])
        )
        return torch.where(
            valid_complete,
            expanded,
            torch.where(valid_tail, tail, torch.full_like(expanded, -1)),
        )

    def select_batched_decode(
        self,
        q: torch.Tensor,
        states: list[Qwen4ExpQSAState],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized QSA selection for one decode token per request."""
        batch = len(states)
        ratio = self.compress_ratio
        block_topk = self.budget // ratio
        width = self.output_width
        device = q.device
        # Keep the decode selector shape stable.  Slicing every request to the
        # current ``max(compressed_len)`` changes the NPU graph shape every
        # four generated tokens and can make the host schedulers spend nearly
        # all their time preparing new dynamic shapes while AICores sit idle.
        # Invalid capacity rows are masked below by ``compressed_lens``.
        max_groups = (
            states[0].compressed_key.shape[0] if states else 0
        )
        block_indices = torch.full(
            (batch, block_topk), -1, dtype=torch.int64, device=device
        )
        pos = positions.reshape(-1).to(torch.int64).clamp_min(0)
        raw_lens = torch.tensor(
            [state.raw_len for state in states], dtype=torch.int64, device=device
        )
        compressed_lens = torch.tensor(
            [state.compressed_len for state in states],
            dtype=torch.int64,
            device=device,
        )
        if max_groups:
            compressed = torch.stack(
                [state.compressed_key[:, 0] for state in states], dim=0
            )
            visible = torch.minimum(
                torch.minimum((pos + 1) // ratio, raw_lens // ratio),
                compressed_lens,
            )
            mask = (
                torch.arange(max_groups, device=device)[None, :]
                < visible[:, None]
            )
            scores = torch.einsum(
                "bhd,bcd->bhc", q.float(), compressed.float()
            )
            scores = scores.clamp_min(0).sum(dim=1) / (q.shape[-1] ** 0.5)
            scores = torch.where(
                mask, scores, torch.full_like(scores, float("-inf"))
            )
            k = min(block_topk, max_groups)
            vals, idx = torch.topk(scores, k, dim=-1)
            block_indices[:, :k] = torch.where(
                torch.isfinite(vals), idx.to(torch.int64), torch.full_like(idx, -1)
            )

        complete_blocks = torch.minimum(
            torch.minimum((pos + 1) // ratio, raw_lens // ratio),
            torch.full_like(pos, block_topk),
        )
        expanded_count = complete_blocks * ratio
        tail_start = ((pos + 1) // ratio) * ratio
        tail_count = (pos + 1) - tail_start
        cols = torch.arange(width, device=device)
        cb_mask = cols[None, :] < expanded_count[:, None]
        block_rank = (cols[None, :] // ratio).clamp_max(block_topk - 1)
        gathered = block_indices.gather(1, block_rank.expand(batch, -1))
        token_cb = gathered * ratio + (cols[None, :] % ratio)
        valid_cb = cb_mask & (token_cb >= 0) & (token_cb < raw_lens[:, None])
        tail_offset = cols[None, :] - expanded_count[:, None]
        token_tail = tail_start[:, None] + tail_offset
        valid_tail = (
            (~cb_mask)
            & (tail_offset < tail_count[:, None])
            & (tail_offset < ratio - 1)
            & (token_tail >= 0)
            & (token_tail < raw_lens[:, None])
        )
        return torch.where(
            valid_cb,
            token_cb,
            torch.where(
                valid_tail,
                token_tail,
                torch.full_like(token_cb, -1),
            ),
        )


class Qwen4ExpQSASparseAttention(nn.Module):
    """Sparse GQA over selected logical token positions (M1, torch).

    Math matches ``qsa_sparse_paged_attention_reference``. ``keys``/``values``
    are the main attention's per-token K/V ``[T, kv_heads, head_dim]``;
    ``indices`` is the indexer output ``[T, width]`` with ``-1`` padding.
    """

    def __init__(
        self,
        num_heads: int = 24,
        num_kv_heads: int = 2,
        head_dim: int = 256,
        softmax_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_heads // num_kv_heads
        self.scale = head_dim**-0.5 if softmax_scale is None else softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        indices: torch.Tensor,
        kv_len: int | None = None,
    ) -> torch.Tensor:
        rows, query_heads, head_dim = query.shape
        kv_heads = self.num_kv_heads
        group_size = self.group_size
        scale = self.scale

        # A 4096-token prefill can select 2048 keys per row. Materializing all
        # rows at once needs >16 GiB for gathered K/V before the fp32 scores.
        # Chunk query rows; attention is row-independent, so this preserves
        # the exact math while bounding temporary memory.
        # Keep the per-KV-head fp32 cast below ~64 MiB even when the selected
        # width reaches the full 4096-token prompt.  A 512-row chunk
        # requires ~514 MiB at a 2048-key selection width and can fail after
        # several prefill waves because the NPU allocator has little contiguous
        # headroom left, even though overall active memory is lower.
        row_chunk = 16
        for row_start in range(0, rows, row_chunk):
            row_end = min(row_start + row_chunk, rows)
            idx = indices[row_start:row_end]
            q = query[row_start:row_end].float()
            chunk_rows = row_end - row_start

            valid = idx >= 0
            if kv_len is not None:
                valid = valid & (idx < kv_len)
            counts = valid.sum(dim=1)
            if int(os.environ.get("SGLANG_QWEN4_GRAPH_CANARY", "0")):
                # Graph capture cannot synchronize to read counts.max().item()
                # and cannot record the dynamic output shapes created by
                # boolean indexing. Preserve the fixed selection width and
                # mask padded entries in the score tensor instead.
                maxn = idx.shape[1]
                comp = idx.clamp_min(0)
                col_mask = valid
            else:
                maxn = int(counts.max().item())
                if maxn == 0:
                    query[row_start:row_end].zero_()
                    continue
                flat_valid = valid.reshape(-1)
                ranks = torch.cumsum(valid, dim=1) - 1
                rows_id = torch.arange(
                    chunk_rows, device=indices.device
                )[:, None].expand_as(valid)
                dst = (
                    rows_id.reshape(-1)[flat_valid] * maxn
                    + ranks.reshape(-1)[flat_valid]
                )
                comp = torch.zeros(
                    (chunk_rows * maxn,), dtype=torch.long, device=indices.device
                )
                comp[dst] = idx.reshape(-1)[flat_valid]
                comp = comp.view(chunk_rows, maxn)
                col_mask = (
                    torch.arange(maxn, device=indices.device)[None, :]
                    < counts[:, None]
                )

            # Gather both KV heads together.  On Ascend, indexing the
            # ``keys[:, kvh, :]`` strided view directly can return numerically
            # incorrect data even though the equivalent contiguous/all-head
            # gather is correct.  The 16-row chunk keeps these temporary
            # buffers bounded (~32 MiB each at a 2048-token selection width).
            if (
                int(os.environ.get("SGLANG_QWEN4_GRAPH_CANARY", "0"))
                and chunk_rows == 1
            ):
                kk = torch.empty(
                    (1, maxn, kv_heads, head_dim),
                    dtype=keys.dtype,
                    device=keys.device,
                )
                vv = torch.empty_like(kk)
                row_width = kv_heads * head_dim
                total = maxn * row_width
                _qsa_gather_kv_kernel[(triton.cdiv(total, 512),)](
                    keys,
                    values,
                    comp,
                    kk,
                    vv,
                    maxn,
                    row_width,
                    total,
                    BLOCK=512,
                )
            else:
                kk = keys[comp]
                vv = values[comp]
            outs = []
            for kvh in range(kv_heads):
                h0, h1 = kvh * group_size, (kvh + 1) * group_size
                sc = torch.einsum(
                    "twd,thd->thw", kk[:, :, kvh].float(), q[:, h0:h1]
                ) * scale
                sc = torch.where(
                    col_mask[:, None, :],
                    sc,
                    torch.full_like(sc, float("-inf")),
                )
                pr = torch.softmax(sc, dim=-1)
                outs.append(
                    torch.einsum(
                        "thw,twd->thd", pr, vv[:, :, kvh].float()
                    )
                )
            # ``query`` is dead after sparse attention. Write each bounded row
            # chunk back into it instead of retaining all chunks and asking
            # ``cat`` for another full ~44 MiB output at the allocator's peak.
            query[row_start:row_end].copy_(
                torch.cat(outs, dim=1).to(query.dtype)
            )
        return query


# ---------------------------------------------------------------------------
# PLE (position-learning enhancement): ngram embedding + dilated short conv.
# Hash-id math mirrors tests/reference.py ple_hash_ids_reference exactly.
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PLE_LAYER_PRIME = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime_64(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not _is_prime_64(candidate):
            candidate += 2
        prime = candidate
    return prime


class _Qwen4ExpLazyShardEmbedding(nn.Module):
    """``padded_vocab_size x head_dim`` split into ``split_parts`` shards.

    Shards are registered as ``nn.Parameter`` (named ``shard_<i>``, matching
    the checkpoint ``ngram_embedding.shard_<i>.weight``) lazily on first
    access, so small unit tests do not materialize the ~100GB table. The
    weight loader (Task 8) calls :meth:`ensure_shard` for every local shard
    before copying checkpoint rows.

    TP sharding: rank ``r`` owns shards ``i % tp_size == r``, so each rank
    holds ~1/8 of the table (~12.5GB instead of ~100GB). The lookup gathers
    each rank's partial rows and all-gathers+sums (rows are disjoint).
    """

    def __init__(
        self,
        padded_vocab_size: int,
        head_dim: int,
        split_parts: int,
        dtype: torch.dtype,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        try:
            from sglang.srt.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
        except Exception:
            self.tp_rank = 0
            self.tp_size = 1
        self.padded_vocab_size = padded_vocab_size
        self.head_dim = head_dim
        self.split_parts = split_parts
        self.shard_size = (padded_vocab_size + split_parts - 1) // split_parts
        self.dtype = dtype
        self.device = device
        self.local_shards = [
            i for i in range(split_parts) if i % self.tp_size == self.tp_rank
        ]

    def is_local_shard(self, shard_index: int) -> bool:
        return shard_index % self.tp_size == self.tp_rank

    def ensure_shard(self, shard_index: int) -> nn.Parameter:
        if not self.is_local_shard(shard_index):
            raise ValueError(
                f"shard {shard_index} is not local to TP rank {self.tp_rank}"
            )
        name = f"shard_{shard_index}"
        param = getattr(self, name, None)
        if param is None:
            row_start = shard_index * self.shard_size
            rows = min(self.shard_size, self.padded_vocab_size - row_start)
            if rows <= 0:
                raise ValueError(
                    f"PLE embedding shard index {shard_index} exceeds "
                    f"split_parts={self.split_parts}"
                )
            param = nn.Parameter(
                torch.zeros(rows, self.head_dim, dtype=self.dtype, device=self.device)
            )
            setattr(self, name, param)
        return param

    def forward(
        self, ids: torch.Tensor, output_device: torch.device | str | None = None
    ) -> torch.Tensor:
        """Look up rows for ``ids [T, heads]`` -> ``[T, heads, head_dim]``.

        Local shards are gathered on CPU, the partial result moves to the ids
        device, and a TP all-gather+sum reconstructs the full rows (each id
        belongs to exactly one rank's shards).
        """
        orig_device = ids.device if output_device is None else torch.device(output_device)
        flat = ids.reshape(-1).to("cpu")
        partial = torch.zeros(
            flat.numel(), self.head_dim, dtype=torch.float32, device="cpu"
        )
        shard_idx = torch.div(flat, self.shard_size, rounding_mode="floor")
        within = flat % self.shard_size
        for i in torch.unique(shard_idx):
            i = int(i)
            if not self.is_local_shard(i):
                continue
            mask = shard_idx == i
            w = self.ensure_shard(i)
            partial[mask] = w.index_select(0, within[mask]).float()
        partial = partial.to(orig_device)
        if self.tp_size > 1:
            from sglang.srt.distributed import get_tp_group

            # Rows are disjoint across ranks, so an in-place sum all-reduce is
            # exactly equivalent to all-gather followed by a rank-axis sum.
            # This was A/B checked against the reference collective on Ascend
            # and avoids a world-size-times temporary buffer during prefill.
            partial = get_tp_group().all_reduce(partial)
        return (
            partial.to(self.dtype).reshape(*ids.shape, self.head_dim)
        )


class Qwen4ExpNGramEmbedding(nn.Module):
    """Hash-table n-gram embedding (checkpoint ``ple_embedding``).

    Checkpoint buffers/weights: ``layer_multipliers [3]`` (i64),
    ``ngram_heads_vocab_sizes [16]`` (i64), ``ngram_heads_offsets [16]``
    (i64), ``ngram_embedding.shard_{0..127}.weight [2500012, 160]``.
    """

    def __init__(
        self,
        ngram_size: int = 3,
        heads_per_ngram: int = 8,
        ngram_vocab_size_base: int = 20_000_000,
        make_divisible_by: int = 128,
        ple_dense_layer_id: int = 0,
        embed_dim: int = 2560,
        vocab_size: int = 248320,
        eos_token_id: int = 248044,
        split_ngram_parts: int = 128,
        seed: int = 1234,
        max_total_tokens: int = 1 << 20,
        max_num_reqs: int = 256,
        dtype: torch.dtype | None = None,
        prefix: str = "",
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.embedding_dim = embed_dim
        self.ngram_size = int(ngram_size)
        self.heads_per_ngram = int(heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(
                f"heads_per_ngram must be > 0, got {self.heads_per_ngram}"
            )
        if embed_dim % self.ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{embed_dim} % {self.ngram_heads} != 0"
            )
        self.head_dim = embed_dim // self.ngram_heads
        self.eos_token_id = int(eos_token_id)
        self.unigram_vocab_size = int(vocab_size)
        self.split_ngram_parts = int(split_ngram_parts)
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")

        max_multiplier = ((1 << 63) - 1) // self.unigram_vocab_size
        half_bound = max(1, max_multiplier // 2)
        base_seed = int(seed) + _PLE_LAYER_PRIME * int(ple_dense_layer_id)
        multipliers = []
        for index in range(self.ngram_size):
            value = base_seed + _SPLITMIX_GAMMA * (index + 1)
            multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
        self.register_buffer(
            "layer_multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )

        ngram_vocab_size_base = int(ngram_vocab_size_base)
        sizes: list[int] = []
        offsets: list[int] = []
        offset = 0
        for local_head in range(self.ngram_heads):
            global_head = int(ple_dense_layer_id) * self.ngram_heads + local_head
            size = _nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=True,
        )
        # Plain tensor attributes intentionally stay on CPU when Module.to(npu)
        # moves parameters and registered buffers. N-gram ids are consumed by
        # the CPU-sharded embedding, so computing them on NPU only creates a
        # long chain of tiny integer kernels followed by a forced D2H sync.
        # Keep host lookup metadata as Python values.  Some Ascend compatibility
        # hooks move even unregistered tensors during model placement.
        self._layer_multiplier_values = tuple(int(v) for v in multipliers)
        self._ngram_head_size_values = tuple(int(v) for v in sizes)
        self._ngram_head_offset_values = tuple(int(v) for v in offsets)
        divisor = int(make_divisible_by)
        padded_vocab_size = ((offset + divisor - 1) // divisor) * divisor
        self.padded_vocab_size = padded_vocab_size
        self.ngram_embedding = _Qwen4ExpLazyShardEmbedding(
            padded_vocab_size,
            self.head_dim,
            self.split_ngram_parts,
            dtype=torch.bfloat16 if dtype is None else dtype,
            device=device,
        )
        self.register_buffer(
            "graph_decode_embeddings",
            torch.zeros(
                int(os.environ.get("SGLANG_QWEN4_GRAPH_MAX_BS", "16")),
                embed_dim,
                dtype=torch.bfloat16 if dtype is None else dtype,
                device=device,
            ),
            persistent=False,
        )
        # During BS>1 graph capture each serialized row binds to its own
        # precomputed host-backed PLE embedding.  This is capture-time Python
        # metadata only; replay retains the selected tensor address.
        self._graph_decode_row = 0

        self.register_buffer(
            "positions_buffer",
            torch.arange(max_total_tokens, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "padded_buffer",
            torch.full(
                (max_num_reqs, max_total_tokens),
                self.eos_token_id,
                dtype=torch.int64,
            ),
            persistent=False,
        )

    @staticmethod
    def _shift_precompute(
        tokens: torch.Tensor, eos_token_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 2:
            raise ValueError("tokens must be a 2D tensor")
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.int64)
        eos_positions = torch.where(tokens == eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [
                eos_positions.new_full((batch_size, 1), -1),
                previous_eos_inclusive[:, :-1],
            ],
            dim=1,
        )
        return positions, positions.unsqueeze(0) - previous_eos - 1

    @staticmethod
    def _shift_apply(
        tokens: torch.Tensor,
        positions: torch.Tensor,
        position_in_segment: torch.Tensor,
        shift: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        if shift == 0:
            return tokens
        source = positions - shift
        gather_indices = source.clamp_min(0).unsqueeze(0).expand(tokens.shape[0], -1)
        shifted = tokens.gather(1, gather_indices)
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        return torch.where(valid, shifted, tokens.new_full((), eos_token_id))

    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        output_device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if int(os.environ.get("SGLANG_QWEN4_GRAPH_CANARY", "0")):
            try:
                capturing = bool(torch.npu.is_current_stream_capturing())
            except Exception:
                capturing = False
            if capturing:
                start = int(self._graph_decode_row)
                end = start + input_ids.numel()
                return self.graph_decode_embeddings[start:end]
        output_device = input_ids.device if output_device is None else output_device
        input_ids = input_ids.reshape(-1).to(device="cpu", dtype=torch.long)
        query_start_loc = query_start_loc.to(device="cpu", dtype=torch.long)
        num_reqs = query_start_loc.numel() - 1
        num_tokens = input_ids.shape[0]
        if num_tokens > self.positions_buffer.numel():
            raise ValueError(
                f"PLE received {num_tokens} tokens, but its workspace supports "
                f"at most {self.positions_buffer.numel()}"
            )
        if num_reqs > self.padded_buffer.shape[0]:
            raise ValueError(
                f"PLE received {num_reqs} requests, but its workspace supports "
                f"at most {self.padded_buffer.shape[0]}"
            )

        positions = torch.arange(num_tokens, dtype=torch.int64, device="cpu")
        request_lengths = query_start_loc[1:] - query_start_loc[:-1]
        max_request_len = max(int(request_lengths.max().item()), 1)
        packed = torch.full(
            (num_reqs, max_request_len),
            self.eos_token_id,
            dtype=torch.int64,
            device="cpu",
        )
        request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
        request_indices.clamp_(max=num_reqs - 1)
        columns = (positions - query_start_loc[request_indices]).clamp(
            0, packed.shape[1] - 1
        )
        packed[request_indices, columns] = input_ids
        ngram_context = ngram_context[:num_reqs].to(device="cpu", dtype=torch.long)

        context = torch.cat([ngram_context, packed], dim=-1)
        # new_tensor inherits the already-host input device and cannot be
        # redirected by model placement hooks.
        layer_multipliers_cpu = input_ids.new_tensor(
            self._layer_multiplier_values, dtype=torch.long
        )
        ngram_heads_vocab_sizes_cpu = input_ids.new_tensor(
            self._ngram_head_size_values, dtype=torch.long
        )
        ngram_heads_offsets_cpu = input_ids.new_tensor(
            self._ngram_head_offset_values, dtype=torch.long
        )
        positions_2d, position_in_segment = self._shift_precompute(
            context, self.eos_token_id
        )
        shifted = [context]
        for shift in range(1, self.ngram_size):
            shifted.append(
                self._shift_apply(
                    context,
                    positions_2d,
                    position_in_segment,
                    shift,
                    self.eos_token_id,
                )
            )
        adjusted_columns = columns + self.ngram_size - 1
        id_blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * layer_multipliers_cpu[0]
            for index in range(1, ngram):
                mixed = torch.bitwise_xor(
                    mixed, shifted[index] * layer_multipliers_cpu[index]
                )
            sizes = ngram_heads_vocab_sizes_cpu[start:end]
            offsets = ngram_heads_offsets_cpu[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes) + offsets
            id_blocks.append(ids[request_indices, adjusted_columns])
        ngram_ids = torch.cat(id_blocks, dim=-1)
        return self.ngram_embedding(
            ngram_ids, output_device=output_device
        ).flatten(-2)

    @torch.no_grad()
    def prepare_graph_decode(
        self,
        input_ids: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> None:
        """Populate the small static PLE input consumed by NPU graph replay."""
        batch = input_ids.numel()
        if batch > self.graph_decode_embeddings.shape[0]:
            raise ValueError(
                f"graph PLE batch {batch} exceeds "
                f"{self.graph_decode_embeddings.shape[0]}"
            )
        ids_cpu = input_ids.reshape(-1).to(device="cpu", dtype=torch.long)
        qsl_cpu = torch.arange(batch + 1, dtype=torch.long, device="cpu")
        context_cpu = ngram_context.to(device="cpu", dtype=torch.long)
        embedding = self.forward(
            ids_cpu,
            qsl_cpu,
            context_cpu,
            output_device=self.graph_decode_embeddings.device,
        )
        self.graph_decode_embeddings[:batch].copy_(embedding)


class Qwen4ExpPLELayer(nn.Module):
    """PLE layer: ngram embedding -> key/value -> gated similarity -> conv.

    Checkpoint weights: ``key_proj.weight [10240, 2560]``,
    ``value_proj.weight [2560, 2560]``, ``norm_key/norm_query/norm_conv.weight
    [10240]``, ``conv1d.weight [10240, 1, 4]``. M1 keeps everything replicated
    and the short-conv state explicit ``[channels, state_len]`` per request.
    """

    def __init__(
        self,
        hidden_size: int,
        hc_count: int,
        ple_embed_dim: int,
        ple_conv_kernel_size: int = 4,
        ngram_size: int = 3,
        heads_per_ngram: int = 8,
        ngram_vocab_size_base: int = 20_000_000,
        make_divisible_by: int = 128,
        vocab_size: int = 248320,
        eos_token_id: int = 248044,
        split_ngram_parts: int = 128,
        seed: int = 1234,
        rms_norm_eps: float = 1e-6,
        dtype: torch.dtype | None = None,
        max_total_tokens: int = 1 << 20,
        max_num_reqs: int = 256,
        prefix: str = "",
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.hc_count = int(hc_count)
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.conv_kernel_size = int(ple_conv_kernel_size)
        self.short_conv_dilation = int(ngram_size)
        self.conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.prefix = prefix

        self.ple_embedding = Qwen4ExpNGramEmbedding(
            ngram_size=ngram_size,
            heads_per_ngram=heads_per_ngram,
            ngram_vocab_size_base=ngram_vocab_size_base,
            make_divisible_by=make_divisible_by,
            ple_dense_layer_id=0,
            embed_dim=ple_embed_dim,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            split_ngram_parts=split_ngram_parts,
            seed=seed,
            max_total_tokens=max_total_tokens,
            max_num_reqs=max_num_reqs,
            dtype=dtype,
            prefix=f"{prefix}.ple_embedding",
            device=device,
        )
        self.key_proj = nn.Linear(ple_embed_dim, self.hc_hidden_size, bias=False, dtype=dtype)
        self.value_proj = nn.Linear(ple_embed_dim, self.hidden_size, bias=False, dtype=dtype)
        norm_args = (self.hc_hidden_size, rms_norm_eps, self.hidden_size, dtype)
        self.norm_key = Qwen4ExpGroupedGemmaRMSNorm(*norm_args)
        self.norm_query = Qwen4ExpGroupedGemmaRMSNorm(*norm_args)
        self.norm_conv = Qwen4ExpGroupedGemmaRMSNorm(*norm_args)
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            padding=self.conv_state_len,
            dilation=self.short_conv_dilation,
            bias=False,
            dtype=dtype,
        )
        nn.init.zeros_(self.conv1d.weight)

    def _apply_norm(
        self, norm: nn.Module, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-2] * shape[-1])
        # PLE's grouped RMSNorm promotes the whole row batch to fp32. A 4096
        # token prefill needs roughly 144 MiB for that temporary, which can
        # fail on later scheduler waves despite sufficient total free memory.
        # The operation is row independent, so bound the temporary while
        # retaining a contiguous bf16 output.
        if flat.device.type == "npu" and flat.shape[0] > 256:
            output = torch.empty_like(flat)
            for start in range(0, flat.shape[0], 256):
                end = min(start + 256, flat.shape[0])
                output[start:end] = norm(flat[start:end])
            return output.reshape(shape)
        return norm(flat).reshape(shape)

    def _short_conv(
        self, x: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal dilated depthwise short-conv (dilated_short_conv_reference).

        Vectorized with ``F.conv1d``: the full-kernel window for every new
        token always fits inside ``[state | x^T]`` because ``state_len ==
        (kernel - 1) * dilation``, so the conv output is exactly the last T
        causal positions (identical math to the reference's per-token loop).
        """
        if x.shape[0] == 0:
            return x.new_empty((0, x.shape[1])), state.clone()
        # [channels, state_len + T]; keep history in the conv weight dtype.
        w_dtype = self.conv1d.weight.dtype
        history = torch.cat(
            [state.to(w_dtype), x.transpose(0, 1).to(w_dtype)], dim=-1
        )
        if (
            int(os.environ.get("SGLANG_QWEN4_GRAPH_CANARY", "0"))
            and x.device.type == "npu"
            and x.shape[0] == 1
        ):
            # Exact T=1 dilated depthwise convolution without the legacy ACL
            # Conv2D lowering, which NPU graph cannot capture. With state_len
            # == (kernel-1)*dilation, the four causal taps are 0,3,6,9.
            taps = history[:, :: self.short_conv_dilation]
            out = (taps * self.conv1d.weight[:, 0, :]).sum(dim=-1)
            out = F.silu(out).unsqueeze(0)
            new_state = history[:, -self.conv_state_len :].contiguous()
            return out, new_state
        # conv1d (no padding, dilation d) -> [1, channels, state_len + T - (k-1)d]
        out = F.conv1d(
            history.unsqueeze(0),
            self.conv1d.weight,
            groups=self.conv1d.weight.shape[0],
            dilation=self.short_conv_dilation,
        ).squeeze(0)
        if x.device.type == "npu" and x.shape[0] > 256:
            # ``x`` is dead once it has been copied into ``history``. Reuse it
            # for the token-major activation output so a 4096-token prefill
            # does not request another contiguous ~72 MiB allocation after
            # request-state reuse has fragmented the allocator.
            token_major = out.transpose(0, 1)
            for start in range(0, x.shape[0], 256):
                end = min(start + 256, x.shape[0])
                x[start:end].copy_(F.silu(token_major[start:end]))
            out = x
        else:
            out = F.silu(out).transpose(0, 1)  # [T, channels]
        if self.conv_state_len:
            new_state = history[:, -self.conv_state_len :].contiguous()
        else:
            new_state = state.clone()
        return out, new_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = input_ids.reshape(-1)
        if input_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "PLE expects input_ids and hidden_states to have the same "
                f"token length, got {input_ids.shape[0]} and "
                f"{hidden_states.shape[0]}"
            )
        embeddings = self.ple_embedding(
            input_ids,
            query_start_loc,
            ngram_context,
            output_device=hidden_states.device,
        )
        import os as _os

        if int(_os.environ.get("SGLANG_QWEN4_DEBUG", "0")):
            print(
                f"[ple] ids={input_ids[:6].tolist()} "
                f"emb_max={embeddings.float().abs().max().item():.4f} "
                f"emb_mean={embeddings.float().mean().item():.4f} "
                f"emb_first4={[round(float(v), 4) for v in embeddings[0, :4].tolist()]}",
                flush=True,
            )
        key = self.key_proj(embeddings)
        value = self.value_proj(embeddings)
        token_count = hidden_states.shape[0]
        key = key.reshape(token_count, self.hc_count, self.hidden_size)
        query = hidden_states.reshape(token_count, self.hc_count, self.hidden_size)
        key = self._apply_norm(self.norm_key, key)
        query = self._apply_norm(self.norm_query, query)
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())
        gated_value = gate * value.unsqueeze(-2)
        normalized = self._apply_norm(self.norm_conv, gated_value).flatten(-2)
        conv_output, new_state = self._short_conv(normalized, conv_state)
        return gated_value.flatten(-2) + conv_output, new_state

    def forward_batched_decode(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        ngram_context: torch.Tensor,
        conv_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one PLE decode token for multiple requests as one batch."""
        batch = hidden_states.shape[0]
        query_start_loc = torch.arange(
            batch + 1, dtype=torch.int64, device=input_ids.device
        )
        embeddings = self.ple_embedding(
            input_ids.reshape(-1),
            query_start_loc,
            ngram_context,
            output_device=hidden_states.device,
        )
        key = self.key_proj(embeddings).reshape(
            batch, self.hc_count, self.hidden_size
        )
        value = self.value_proj(embeddings)
        query = hidden_states.reshape(batch, self.hc_count, self.hidden_size)
        key = self._apply_norm(self.norm_key, key)
        query = self._apply_norm(self.norm_query, query)
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())
        gated_value = gate * value.unsqueeze(-2)
        normalized = self._apply_norm(self.norm_conv, gated_value).flatten(-2)

        history = torch.cat(
            [
                conv_states.to(self.conv1d.weight.dtype),
                normalized.to(self.conv1d.weight.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        conv_output = F.silu(
            F.conv1d(
                history,
                self.conv1d.weight,
                groups=self.conv1d.weight.shape[0],
                dilation=self.short_conv_dilation,
            )[:, :, 0]
        )
        next_states = history[:, :, -self.conv_state_len :].contiguous()
        return gated_value.flatten(-2) + conv_output, next_states


# ---------------------------------------------------------------------------
# GDN (GatedDeltaNet) + QSA full attention — M1 torch implementations.
# GDN math mirrors sglang's FLA ``fused_sigmoid_gating_delta_rule_update``
# kernel; QSA attention wires Task-4 indexer/sparse modules with checkpoint
# weight names.
# ---------------------------------------------------------------------------

@triton.jit
def _moe_gate_up_kernel(
    x_ptr, w1_ptr, exp_idx_ptr, valid_ptr, out_ptr,
    K, stride_xg, stride_w1e, stride_w1i, stride_w1k, stride_outg,
    INTER, BLOCK_INTER: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Routed-expert up/gate GEMV: out[g, i] = silu(W_up[i]@x) * (W_gate[i]@x).

    ``w1`` is the TP-local ``[num_local_experts, 2*INTER, K]`` gate_up_proj;
    the gate half starts at ``INTER`` rows. Invalid slots (non-local expert)
    are skipped and their output stays zero.
    """
    pid_g = tl.program_id(0)
    pid_i = tl.program_id(1)
    if tl.load(valid_ptr + pid_g) == 0:
        return
    e = tl.load(exp_idx_ptr + pid_g)
    offs_i = pid_i * BLOCK_INTER + tl.arange(0, BLOCK_INTER)
    acc_up = tl.zeros((BLOCK_INTER,), tl.float32)
    acc_gate = tl.zeros((BLOCK_INTER,), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        w_ptrs = (
            w1_ptr + e * stride_w1e + offs_i[:, None] * stride_w1i
            + offs_k[None, :] * stride_w1k
        )
        w_up = tl.load(w_ptrs)
        w_gate = tl.load(w_ptrs + INTER * stride_w1i)
        xv = tl.load(x_ptr + pid_g * stride_xg + offs_k)
        acc_up += tl.sum(w_up * xv[None, :], axis=1)
        acc_gate += tl.sum(w_gate * xv[None, :], axis=1)
    out = acc_up * tl.sigmoid(acc_up) * acc_gate  # silu
    tl.store(out_ptr + pid_g * stride_outg + offs_i, out.to(out_ptr.dtype.element_ty))


@triton.jit
def _moe_down_kernel(
    intermed_ptr, w2_ptr, exp_idx_ptr, valid_ptr, out_ptr,
    INTER, stride_ig, stride_w2e, stride_w2h, stride_w2k, stride_oh,
    BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Routed-expert down GEMV: out[g, h] = W2[h] @ intermed[g].

    Invalid slots (non-local expert) are zeroed explicitly: ``out`` is a fresh
    ``torch.empty`` and the torch reference masks these slots to zero before
    the weighted sum, so an early return would leave uninitialized garbage in
    the final MoE output.
    """
    pid_g = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    if tl.load(valid_ptr + pid_g) == 0:
        tl.store(
            out_ptr + pid_g * stride_oh + offs_h,
            tl.zeros((BLOCK_H,), tl.float32),
            mask=offs_h < stride_oh,
        )
        return
    e = tl.load(exp_idx_ptr + pid_g)
    acc = tl.zeros((BLOCK_H,), tl.float32)
    for k0 in range(0, INTER, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        w_ptrs = (
            w2_ptr + e * stride_w2e + offs_h[:, None] * stride_w2h
            + offs_k[None, :] * stride_w2k
        )
        w2t = tl.load(w_ptrs)
        iv = tl.load(intermed_ptr + pid_g * stride_ig + offs_k)
        acc += tl.sum(w2t * iv[None, :], axis=1)
    tl.store(
        out_ptr + pid_g * stride_oh + offs_h,
        acc.to(out_ptr.dtype.element_ty),
        mask=offs_h < stride_oh,
    )


@triton.jit
def _gdn_delta_rule_kernel(
    q_ptr, k_ptr, v_ptr, a_ptr, b_ptr, A_log_ptr, dt_bias_ptr,
    h_ptr, o_ptr,
    T,
    stride_q_t, stride_q_kh,      # q: [T, KH, K]
    stride_k_t, stride_k_kh,      # k: [T, KH, K]
    stride_v_t, stride_v_hv,      # v: [T, HV, V]
    stride_a_t, stride_a_hv,      # a: [T, HV]
    stride_b_t, stride_b_hv,      # b: [T, HV]
    stride_h_hv,                  # h: [HV, K, V] contiguous
    stride_o_t, stride_o_hv,      # o: [T, HV, V]
    HV: tl.constexpr, KH: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BV: tl.constexpr,
    SCALE: tl.constexpr,
    SP_THRESHOLD: tl.constexpr,
):
    """Sigmoid-gated delta-rule recurrence, one program per (v-head, V-tile).

    Per token t and v-head hv (kv = hv // (HV // KH)):
        g     = -exp(A_log) * softplus(a + dt_bias)
        h    *= exp(g)
        vd    = (v - sum_K(h * k)) * sigmoid(b)
        h    += k ⊗ vd
        o     = sum_K(h * (q * scale))
    Matches ``Qwen4ExpGatedDeltaNet``'s torch reference exactly (fp32 state).
    """
    pid = tl.program_id(0)
    nv = V // BV
    hv = pid // nv
    vt = pid % nv
    kv = hv // (HV // KH)
    v_off = vt * BV
    rows = tl.arange(0, K)
    cols = v_off + tl.arange(0, BV)

    h = tl.load(h_ptr + hv * stride_h_hv + rows[:, None] * K + cols[None, :])
    A_log = tl.load(A_log_ptr + hv).to(tl.float32)
    dt = tl.load(dt_bias_ptr + hv).to(tl.float32)
    for t in range(T):
        a_t = tl.load(a_ptr + t * stride_a_t + hv * stride_a_hv).to(tl.float32)
        x = a_t + dt
        # tl.where may evaluate both branches. Clamp the exp branch so large
        # positive x cannot create an inf that poisons Ascend vector lanes.
        sp_exp = tl.log(1.0 + tl.exp(tl.minimum(x, SP_THRESHOLD)))
        sp = tl.where(x <= SP_THRESHOLD, sp_exp, x)
        g = -tl.exp(A_log) * sp
        h = h * tl.exp(g)
        b_t = tl.load(b_ptr + t * stride_b_t + hv * stride_b_hv).to(tl.float32)
        beta = tl.sigmoid(b_t)
        k_t = tl.load(k_ptr + t * stride_k_t + kv * stride_k_kh + rows).to(tl.float32)
        v_t = tl.load(v_ptr + t * stride_v_t + hv * stride_v_hv + cols).to(tl.float32)
        hk = tl.sum(h * k_t[:, None], axis=0)          # [BV]
        vd = (v_t - hk) * beta
        h = h + k_t[:, None] * vd[None, :]
        q_t = tl.load(q_ptr + t * stride_q_t + kv * stride_q_kh + rows).to(tl.float32)
        o_t = tl.sum(h * (q_t[:, None] * SCALE), axis=0)  # [BV]
        tl.store(o_ptr + t * stride_o_t + hv * stride_o_hv + cols, o_t)
    tl.store(h_ptr + hv * stride_h_hv + rows[:, None] * K + cols[None, :], h)


@triton.jit
def _hc_mix_down_kernel(
    x_ptr, w_down_ptr, w_norm_ptr, gate_ptr, normed_ptr,
    T, LOWRANK, HC: tl.constexpr, HS: tl.constexpr, EPS: tl.constexpr,
    stride_wk: tl.constexpr,  # w_down row stride == HC*HS
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Fused grouped-RMSNorm + low-rank down gate of HC mix.

    gate[t, n] = silu(Wd[n] @ normed(t) / HC). Also stores the normalized
    hyper row ``normed`` (with the (1+w) affine) for the up/mean and combine
    stages. One program per (row, 64-col block); the per-group row variance
    is recomputed redundantly per program (cheap vs. the GEMM traffic).
    """
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for g in range(HC):  # python-level unroll: per-group rms + dot pass
        vg = tl.zeros((), tl.float32)
        for i0 in range(g * HS, (g + 1) * HS, BLOCK_K):
            offs = i0 + tl.arange(0, BLOCK_K)
            xv = tl.load(x_ptr + pid_t * (HC * HS) + offs).to(tl.float32)
            vg += tl.sum(xv * xv)
        rms = tl.rsqrt(vg / HS + EPS)
        for i0 in range(g * HS, (g + 1) * HS, BLOCK_K):
            offs = i0 + tl.arange(0, BLOCK_K)
            xv = tl.load(x_ptr + pid_t * (HC * HS) + offs).to(tl.float32)
            wn = tl.load(w_norm_ptr + offs).to(tl.float32)
            nm = xv * rms * (1.0 + wn)
            tl.store(normed_ptr + pid_t * (HC * HS) + offs, nm.to(tl.bfloat16))
            wd = tl.load(
                w_down_ptr + offs_n[:, None] * stride_wk + offs[None, :]
            ).to(tl.float32)
            acc += tl.sum(wd * nm[None, :], axis=1)
    x = acc / HC
    tl.store(gate_ptr + pid_t * LOWRANK + offs_n,
             (x * tl.sigmoid(x)).to(tl.bfloat16))  # silu(acc/HC)


@triton.jit
def _hc_mix_up_kernel(
    gate_ptr, w_up_ptr, normed_ptr, out_ptr,
    T, HC: tl.constexpr, HS: tl.constexpr, LOWRANK: tl.constexpr,
    stride_wk: tl.constexpr,  # w_up row stride == LOWRANK
    BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
):
    """Up gate + gated mean: out[t,h] = mean_c sigmoid(Wu[c,h] @ gate(t)) * normed(t,c,h)."""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    acc = tl.zeros((BLOCK_H,), tl.float32)
    for g in range(HC):
        a = tl.zeros((BLOCK_H,), tl.float32)
        for k0 in range(0, LOWRANK, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            wu = tl.load(
                w_up_ptr + (g * HS + offs_h)[:, None] * stride_wk + offs_k[None, :]
            ).to(tl.float32)
            gk = tl.load(gate_ptr + pid_t * LOWRANK + offs_k).to(tl.float32)
            a += tl.sum(wu * gk[None, :], axis=1)
        nm = tl.load(normed_ptr + pid_t * (HC * HS) + g * HS + offs_h).to(tl.float32)
        acc += tl.sigmoid(a) * nm
    tl.store(out_ptr + pid_t * HS + offs_h, (acc / HC).to(tl.bfloat16))


@triton.jit
def _hc_combine_kernel(
    normed_ptr, w_inj_ptr, res_ptr, bo_ptr, out_ptr,
    T, HC: tl.constexpr, HS: tl.constexpr,
    stride_wi: tl.constexpr,  # w_inj row stride == HC*HS
    BLOCK_K: tl.constexpr,
):
    """Fused combine: inj[c] = 2*sigmoid(Wi[c]@normed(t)/HC); out = res + bo*inj.

    Wi is a full ``[HC, HC*HS]`` row (NOT block-diagonal — the checkpoint
    keeps ~75% of magnitude off-diagonal), so each program sweeps all
    HC*HS input elements (redundant per group, like _hc_mix_down_kernel).
    """
    pid_t = tl.program_id(0)
    pid_g = tl.program_id(1)
    base_t = pid_t * (HC * HS)
    a = tl.zeros((), tl.float32)
    for i0 in range(0, HC * HS, BLOCK_K):
        offs = i0 + tl.arange(0, BLOCK_K)
        nm = tl.load(normed_ptr + base_t + offs).to(tl.float32)
        wi = tl.load(w_inj_ptr + pid_g * stride_wi + offs).to(tl.float32)
        a += tl.sum(wi * nm)
    inj = 2.0 * tl.sigmoid(a / HC)
    base = base_t + pid_g * HS
    for i0 in range(0, HS, BLOCK_K):
        offs = i0 + tl.arange(0, BLOCK_K)
        res = tl.load(res_ptr + base + offs).to(tl.float32)
        bo = tl.load(bo_ptr + pid_t * HS + offs).to(tl.float32)
        tl.store(out_ptr + base + offs, (res + bo * inj).to(tl.bfloat16))


class _Qwen4ExpRMSNorm(nn.Module):
    """Standard RMSNorm (plain weight multiply) — GDN ``norm`` convention.

    The GDN checkpoint ``norm.weight`` is ~1.8 (NOT the Gemma 1+w convention
    used by HC/PLE/QSA norms, which store ~0).
    """

    def __init__(
        self, hidden_size: int, eps: float, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))

    def forward(self, x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        input_dtype = x.dtype
        if x.device.type == "npu" and x.shape[0] > 256:
            # GDN's recurrent output is dead after this normalization.  Let
            # that caller reuse it so a 4096-token prefill does not require
            # one more 44 MiB resident allocation at peak memory.  Each
            # calculation still happens in fp32 chunks before being copied
            # back, preserving the original numerical order.
            out = x if inplace else torch.empty_like(x)
            weight = self.weight.float()
            for start in range(0, x.shape[0], 256):
                end = min(start + 256, x.shape[0])
                chunk = x[start:end].float()
                variance = chunk.square().mean(dim=-1, keepdim=True)
                out[start:end] = (
                    chunk
                    * torch.rsqrt(variance + self.variance_epsilon)
                    * weight
                ).to(input_dtype)
            return out
        x = x.float()
        variance = x.square().mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.variance_epsilon) * self.weight.float()).to(input_dtype)


class Qwen4ExpGatedDeltaNet(nn.Module):
    """GatedDeltaNet linear attention (torch, M1).

    Checkpoint weights (exact names): ``in_proj_qkv [key_dim*2+value_dim,
    hidden]``, ``in_proj_z [value_dim, hidden]``, ``in_proj_a [num_v_heads,
    hidden]``, ``in_proj_b [num_v_heads, hidden]``, ``conv1d.weight [conv_dim,
    1, 4]``, ``A_log [num_v_heads]``, ``dt_bias [num_v_heads]``, ``norm.weight
    [head_v_dim]`` (standard RMSNorm), ``out_proj [hidden, value_dim]``.

    Recurrence (per v-head ``hv``, kv-head ``kv = hv // (V//K)``)::

        g    = -exp(A_log) * softplus(a + dt_bias)
        h   *= exp(g)
        vd   = (v - sum_K(h * k)) * sigmoid(b)
        h   += k ⊗ vd
        o    = sum_K(h * (q * scale)),  scale = head_k_dim ** -0.5
    """

    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel: int = 4,
        rms_norm_eps: float = 1e-6,
        dtype: torch.dtype | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_k_heads = int(num_k_heads)
        self.num_v_heads = int(num_v_heads)
        self.head_k_dim = int(head_k_dim)
        self.head_v_dim = int(head_v_dim)
        self.conv_kernel = int(conv_kernel)
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_state_len = self.conv_kernel - 1
        self.prefix = prefix

        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.conv_dim, bias=False, dtype=dtype
        )
        self.in_proj_z = nn.Linear(
            self.hidden_size, self.value_dim, bias=False, dtype=dtype
        )
        self.in_proj_a = nn.Linear(
            self.hidden_size, self.num_v_heads, bias=False, dtype=dtype
        )
        self.in_proj_b = nn.Linear(
            self.hidden_size, self.num_v_heads, bias=False, dtype=dtype
        )
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            self.conv_kernel,
            groups=self.conv_dim,
            bias=False,
            dtype=dtype,
        )
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_v_heads, dtype=dtype))
        self.norm = _Qwen4ExpRMSNorm(self.head_v_dim, eps=rms_norm_eps, dtype=dtype)
        self.out_proj = nn.Linear(
            self.value_dim, self.hidden_size, bias=False, dtype=dtype
        )

    @staticmethod
    def new_state(
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        device="cpu",
        dtype: torch.dtype = torch.float32,
    ):
        return torch.zeros(
            (num_v_heads, head_k_dim, head_v_dim), dtype=dtype, device=device
        )

    def _causal_conv1d(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Causal per-channel conv (kernel, dilation 1) with explicit state.

        Vectorized as one grouped conv1d over ``[state | x]`` (the PLE short
        conv does the same); matches the per-token loop exactly.
        """
        T = x.shape[0]
        w = self.conv1d.weight  # [C, 1, K]
        if x.device.type == "npu" and T == 1:
            from sgl_kernel_npu.mamba.causal_conv1d import (
                causal_conv1d_update_npu,
            )

            state_batch = state.unsqueeze(0)
            local_indices = torch.zeros(
                (1,), dtype=torch.int64, device=x.device
            )
            out = causal_conv1d_update_npu(
                x,
                state_batch,
                w[:, 0, :].contiguous(),
                activation="silu",
                conv_state_indices=local_indices,
            )
            return out, state
        # Avoid FlagGems ``cat`` materializing the transposed input as an
        # additional ~72 MiB contiguous tensor. Only the history output is
        # required; copy the two sources directly into that allocation.
        cat = torch.empty(
            (1, x.shape[1], self.conv_state_len + T),
            dtype=x.dtype,
            device=x.device,
        )
        if self.conv_state_len:
            cat[0, :, : self.conv_state_len].copy_(state)
        if x.device.type == "npu" and T > 256:
            for start in range(0, T, 256):
                end = min(start + 256, T)
                cat[0, :, self.conv_state_len + start : self.conv_state_len + end].copy_(
                    x[start:end].transpose(0, 1)
                )
        else:
            cat[0, :, self.conv_state_len :].copy_(x.transpose(0, 1))
        y = F.conv1d(cat, w, groups=x.shape[1])  # [1, C, T]
        # Never return a narrow view into ``cat`` here.  Persisting that view
        # in the per-request state keeps the entire ~72 MiB prefill backing
        # allocation alive for every request and GDN layer.  The logical
        # convolution history is tiny, so update its existing buffer in
        # place and let the full history tensor be released after this call.
        if self.conv_state_len:
            state.copy_(cat[0, :, -self.conv_state_len :])
        new_state = state
        # transpose(0,1) yields a non-contiguous view whose innermost stride
        # is 3 (channel-major storage); the Triton GDN kernel assumes inner
        # stride 1, so materialize a contiguous copy here.
        y_tokens = y[0].transpose(0, 1)
        if x.device.type == "npu" and T > 256:
            # Reuse the now-dead qkv input as the contiguous output buffer.
            # Allocating the full SiLU result here costs ~72 MiB for a 4096
            # token prefill and can fail after request-state reuse because the
            # allocator has enough aggregate free memory but no block that
            # large. Small temporaries keep the peak allocation bounded.
            # Store large NPU prefills as three consecutive token-major Q/K/V
            # regions. The maintained fused GDN kernel assumes that each
            # tensor is contiguous; the ordinary interleaved [T, Q+K+V]
            # layout otherwise forces ~66 MiB of late contiguous copies.
            flat = x.reshape(-1)
            dst_offset = 0
            src_offset = 0
            for width in (self.key_dim, self.key_dim, self.value_dim):
                dst = flat[dst_offset : dst_offset + T * width].view(T, width)
                for start in range(0, T, 256):
                    end = min(start + 256, T)
                    dst[start:end].copy_(
                        F.silu(y_tokens[start:end, src_offset : src_offset + width])
                    )
                dst_offset += T * width
                src_offset += width
            return x, new_state
        return F.silu(y_tokens).contiguous(), new_state

    def forward_batched(
        self,
        hidden_states: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        state_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run equal-length requests in a single fused recurrent launch.

        The ordinary ``forward`` path intentionally handles one variable-length
        sequence at a time. Keeping an equal-length request dimension lets the
        Ascend recurrent kernel, convolution, and all input projections process
        the whole prefill or decode batch at once.
        """
        if hidden_states.device.type != "npu":
            raise RuntimeError("batched decode is currently implemented for NPU only")
        squeeze_token_dim = hidden_states.ndim == 2
        if squeeze_token_dim:
            hidden_states = hidden_states.unsqueeze(1)
        if hidden_states.ndim != 3:
            raise ValueError("batched GDN input must have shape [B, T, H]")
        batch, tokens, _ = hidden_states.shape
        if conv_states.shape[0] != batch:
            raise ValueError("convolution state batch does not match hidden-state batch")
        if state_indices is None and ssm_states.shape[0] != batch:
            raise ValueError("recurrent state batch does not match hidden-state batch")

        qkv = self.in_proj_qkv(hidden_states)
        if tokens == 1:
            from sgl_kernel_npu.mamba.causal_conv1d import (
                causal_conv1d_update_npu,
            )

            local_indices = torch.arange(
                batch, dtype=torch.int64, device=hidden_states.device
            )
            qkv = causal_conv1d_update_npu(
                qkv[:, 0, :],
                conv_states,
                self.conv1d.weight[:, 0, :].contiguous(),
                activation="silu",
                conv_state_indices=local_indices,
            ).unsqueeze(1)
            next_conv_states = conv_states
        else:
            conv_cat = torch.cat((conv_states, qkv.transpose(1, 2)), dim=-1)
            qkv = F.silu(
                F.conv1d(conv_cat, self.conv1d.weight, groups=self.conv_dim)
            ).transpose(1, 2).contiguous()
            next_conv_states = conv_cat[:, :, -self.conv_state_len :]

        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.view(batch, tokens, self.num_k_heads, self.head_k_dim).contiguous()
        k = k.view(batch, tokens, self.num_k_heads, self.head_k_dim).contiguous()
        v = v.view(batch, tokens, self.num_v_heads, self.head_v_dim).contiguous()
        z = self.in_proj_z(hidden_states).view(
            batch, tokens, self.num_v_heads, self.head_v_dim
        )
        a = self.in_proj_a(hidden_states).view(
            batch, tokens, self.num_v_heads
        ).contiguous()
        b = self.in_proj_b(hidden_states).view(
            batch, tokens, self.num_v_heads
        ).contiguous()

        from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update_npu,
        )

        state_source = ssm_states.float().contiguous()
        if state_indices is None:
            state_indices = torch.arange(
                batch, dtype=torch.int64, device=hidden_states.device
            )
        else:
            state_indices = state_indices.to(
                dtype=torch.int64, device=hidden_states.device
            )
        outputs = fused_sigmoid_gating_delta_rule_update_npu(
            A_log=self.A_log,
            a=a,
            dt_bias=self.dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=state_source,
            initial_state_indices=state_indices,
            scale=self.head_k_dim**-0.5,
            # Qwen4 GDN normalizes Q/K per head before the delta-rule update.
            # The earlier no-normalization setting only matched the incomplete
            # CPU port and produces semantically invalid logits.
            use_qk_l2norm_in_kernel=True,
        )

        normed = self.norm(
            outputs.reshape(-1, self.head_v_dim), inplace=True
        ).reshape_as(outputs)
        # Qwen4 uses a sigmoid GDN output gate (unlike earlier variants that
        # use SiLU).  The wrong activation keeps tensors finite but destroys
        # the learned token distribution.
        out = self.out_proj(
            (normed * torch.sigmoid(z)).reshape(batch, tokens, self.value_dim)
        )
        if squeeze_token_dim:
            out = out[:, 0]
        return out, next_conv_states, state_source

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward over one request's tokens; returns (out, conv_state, ssm_state)."""
        import os as _os

        _dbg = int(_os.environ.get("SGLANG_QWEN4_DEBUG", "0"))
        _tn = int(_os.environ.get("SGLANG_QWEN4_TRACE_NAN", "0"))
        if _tn:
            _seen = globals().setdefault("_trace_nan_gdn_seen", set())
            if ("state",) not in _seen:
                if not (torch.isfinite(ssm_state).all() and torch.isfinite(conv_state).all()):
                    _seen.add(("state",))
                    print(
                        f"[GDN-STATE] ssm_nan={int(torch.isnan(ssm_state).sum())} "
                        f"ssm_inf={int(torch.isinf(ssm_state).sum())} "
                        f"conv_nan={int(torch.isnan(conv_state).sum())}",
                        flush=True,
                    )
        if _dbg:
            print(
                f"[gdn-in] hidden max={hidden_states.float().abs().max().item():.4f} "
                f"conv_state_max={conv_state.float().abs().max().item():.4f} "
                f"ssm_max={ssm_state.float().abs().max().item():.4f}",
                flush=True,
            )
        if _dbg and not hasattr(self, "_dbg_done"):
            self._dbg_done = True
            print(
                f"[gdn-w] A_log[:4]={[round(float(v), 3) for v in self.A_log[:4].tolist()]} "
                f"dt[:4]={[round(float(v), 3) for v in self.dt_bias[:4].tolist()]} "
                f"in_proj_a[0,:4]={[round(float(v), 4) for v in self.in_proj_a.weight[0, :4].tolist()]} "
                f"qkv[0,:4]={[round(float(v), 4) for v in self.in_proj_qkv.weight[0, :4].tolist()]}",
                flush=True,
            )
        T = hidden_states.shape[0]
        qkv = self.in_proj_qkv(hidden_states)  # [T, conv_dim]
        if _dbg:
            print(
                f"[gdn] in_proj_qkv finite={torch.isfinite(qkv).all().item()} "
                f"max={qkv.float().abs().max().item():.4f}",
                flush=True,
            )
        qkv, next_conv_state = self._causal_conv1d(qkv, conv_state)
        if _dbg:
            print(
                f"[gdn] conv finite={torch.isfinite(qkv).all().item()} "
                f"max={qkv.float().abs().max().item():.4f}",
                flush=True,
            )
        if qkv.device.type == "npu" and T > 256:
            flat_qkv = qkv.reshape(-1)
            q_end = T * self.key_dim
            k_end = q_end + T * self.key_dim
            q = flat_qkv[:q_end].view(T, self.key_dim)
            k = flat_qkv[q_end:k_end].view(T, self.key_dim)
            v = flat_qkv[k_end:].view(T, self.value_dim)
        else:
            q, k, v = qkv.split(
                [self.key_dim, self.key_dim, self.value_dim], dim=-1
            )
        z = self.in_proj_z(hidden_states)
        a = self.in_proj_a(hidden_states)
        b = self.in_proj_b(hidden_states)

        q = q.view(T, self.num_k_heads, self.head_k_dim)
        k = k.view(T, self.num_k_heads, self.head_k_dim)
        v = v.view(T, self.num_v_heads, self.head_v_dim)
        z = z.view(T, self.num_v_heads, self.head_v_dim)
        a = a.view(T, self.num_v_heads)
        b = b.view(T, self.num_v_heads)

        h = ssm_state.float()
        scale = self.head_k_dim**-0.5
        v_per_k = self.num_v_heads // self.num_k_heads
        # Triton fast path (CUDA/MUSA/Ascend NPU). K and V must be powers of two; the
        # fp32 recurrence math is identical to the torch loop below.
        is_pow2 = (
            self.head_k_dim & (self.head_k_dim - 1)
        ) == 0 and (self.head_v_dim & (self.head_v_dim - 1)) == 0
        _force_torch = getattr(self, "_force_torch", False) or int(
            _os.environ.get("SGLANG_QWEN4_FORCE_TORCH_GDN", "0")
        )
        if is_pow2 and hidden_states.device.type in ("cuda", "musa") and not _force_torch:
            outputs = torch.empty(
                (T, self.num_v_heads, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            BV = min(triton.next_power_of_2(self.head_v_dim), 32)
            nv = self.head_v_dim // BV
            grid = (self.num_v_heads * nv,)
            # Ascend's dynamic Triton loop becomes invalid beyond roughly 52
            # iterations. Carry the same in-place recurrence state across
            # 32-token launches; CUDA/MUSA retain the single-launch path.
            recurrence_chunk = 32 if hidden_states.device.type == "npu" else T
            for chunk_start in range(0, T, recurrence_chunk):
                chunk_end = min(chunk_start + recurrence_chunk, T)
                qc, kc = q[chunk_start:chunk_end], k[chunk_start:chunk_end]
                vc = v[chunk_start:chunk_end]
                ac, bc = a[chunk_start:chunk_end], b[chunk_start:chunk_end]
                oc = outputs[chunk_start:chunk_end]
                chunk_tokens = chunk_end - chunk_start
                _gdn_delta_rule_kernel[grid](
                    qc, kc, vc, ac, bc, self.A_log, self.dt_bias, h, oc,
                    chunk_tokens,
                    qc.stride(0), qc.stride(1),
                    kc.stride(0), kc.stride(1),
                    vc.stride(0), vc.stride(1),
                    ac.stride(0), ac.stride(1),
                    bc.stride(0), bc.stride(1),
                    h.stride(0),
                    oc.stride(0), oc.stride(1),
                    HV=self.num_v_heads, KH=self.num_k_heads,
                    K=self.head_k_dim, V=self.head_v_dim, BV=BV,
                    SCALE=scale, SP_THRESHOLD=20.0,
                    num_warps=4,
                )
            if _tn:
                _seen = globals().setdefault("_trace_nan_gdn_seen", set())
                if ("out",) not in _seen:
                    if not torch.isfinite(outputs).all():
                        _seen.add(("out",))
                        _nn = torch.isnan(outputs)
                        print(
                            f"[GDN-OUT] T={T} nan={int(_nn.sum())} "
                            f"inf={int(torch.isinf(outputs).sum())} "
                            f"rows={_nn.any(-1).nonzero().flatten()[:8].tolist()} "
                            f"h_max={float(torch.nan_to_num(h, nan=0.0).abs().max()):.4f}",
                            flush=True,
                        )
        elif hidden_states.device.type == "npu":
            # Use SGLang's maintained Ascend kernel. In addition to being
            # fused, it performs the required q/k L2 normalization that the
            # original day-0 torch port omitted (the omission makes the
            # delta-rule state diverge after ~52 tokens).
            from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
                fused_sigmoid_gating_delta_rule_update_npu,
            )

            state_source = h.unsqueeze(0)
            state_indices = torch.zeros(
                (1,), dtype=torch.int64, device=hidden_states.device
            )
            fused_out = fused_sigmoid_gating_delta_rule_update_npu(
                A_log=self.A_log,
                a=a.contiguous().unsqueeze(0),
                dt_bias=self.dt_bias,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=q.contiguous().unsqueeze(0),
                k=k.contiguous().unsqueeze(0),
                v=v.contiguous().unsqueeze(0),
                b=b.contiguous().unsqueeze(0),
                initial_state_source=state_source,
                initial_state_indices=state_indices,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
            )
            outputs = fused_out[0]
            h = state_source[0]
        else:
            outputs = torch.empty(
                (T, self.num_v_heads, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            for t in range(T):
                g = -torch.exp(self.A_log.float()) * F.softplus(
                    a[t].float() + self.dt_bias.float()
                )
                h = h * torch.exp(g)[:, None, None]
                beta = torch.sigmoid(b[t].float())
                for hv in range(self.num_v_heads):
                    kv = hv // v_per_k
                    krow = k[t, kv].float()          # [K]
                    vd = v[t, hv].float() - torch.einsum("kd,k->d", h[hv], krow)
                    vd = vd * beta[hv]
                    h[hv] = h[hv] + krow[:, None] * vd[None, :]
                    outputs[t, hv] = (
                        torch.einsum("kd,k->d", h[hv], q[t, kv].float() * scale)
                    ).to(outputs.dtype)

        normed = self.norm(
            outputs.reshape(-1, self.head_v_dim), inplace=True
        ).reshape_as(outputs)
        # Both ``z`` and ``normed`` are dead after gating.  The checkpoint
        # requires sigmoid(z), not SiLU(z).
        z.sigmoid_()
        normed.mul_(z)
        out = normed
        out = self.out_proj(out.reshape(T, -1))
        if _dbg:
            print(
                f"[gdn] T={T} out finite={torch.isfinite(out).all().item()} "
                f"max={out.float().abs().max().item():.4f} "
                f"nan_count={torch.isnan(out).sum().item()} "
                f"first6={[round(float(v), 4) for v in out[0, :6].tolist()]} "
                f"ssm_max={h.float().abs().max().item():.4f}",
                flush=True,
            )
        return out, next_conv_state, h


class Qwen4ExpMoe(nn.Module):
    """Sparse MoE (torch, M1) with checkpoint-exact weight names and TP sharding.

    Checkpoint weights: ``gate.weight [512, 2560]`` (router, replicated),
    ``experts.gate_up_proj [512, 1280, 2560]`` / ``experts.down_proj [512,
    2560, 640]`` (3-D per-expert, TP-sharded: each rank holds experts whose
    global id % tp_size == tp_rank), ``shared_expert.gate_proj.weight [640,
    2560]`` / ``shared_expert.up_proj.weight`` / ``shared_expert.down_proj.weight
    [2560, 640]``, ``shared_expert_gate.weight [1, 2560]`` (sigmoid gate,
    replicated).

    sglang's default ``Qwen2MoeSparseMoeBlock`` uses a Triton ``FusedMoE``
    kernel that produces NaN on CPU (the M1 unit-test platform), so M1 ships a
    plain-torch implementation with identical math. M3 can swap in the fused
    block for real serving.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        shared_intermediate_size: int,
        hidden_act: str = "silu",
        norm_topk_prob: bool = True,
        dtype: torch.dtype | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        from sglang.srt.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.intermediate_size = int(intermediate_size)
        self.shared_intermediate_size = int(shared_intermediate_size)
        self.hidden_act = hidden_act
        self.norm_topk_prob = norm_topk_prob
        self.prefix = prefix
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.local_expert_ids = [
            e for e in range(self.num_experts) if e % self.tp_size == self.tp_rank
        ]
        self.num_local_experts = len(self.local_expert_ids)
        # Precomputed global-expert-id -> local index (-1 = not on this rank).
        # Built as a plain tensor on CPU in __init__; moved by _apply / to().
        local_map = torch.full((self.num_experts,), -1, dtype=torch.long)
        for i, e in enumerate(self.local_expert_ids):
            local_map[e] = i
        self.register_buffer("_local_expert_map", local_map, persistent=False)

        self.gate = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype)
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                2 * self.intermediate_size,
                hidden_size,
                dtype=dtype,
            )
        )
        self.experts.down_proj = nn.Parameter(
            torch.empty(
                self.num_local_experts, hidden_size, self.intermediate_size, dtype=dtype
            )
        )
        self.shared_expert = nn.Module()
        self.shared_expert.gate_proj = nn.Linear(
            hidden_size, shared_intermediate_size, bias=False, dtype=dtype
        )
        self.shared_expert.up_proj = nn.Linear(
            hidden_size, shared_intermediate_size, bias=False, dtype=dtype
        )
        self.shared_expert.down_proj = nn.Linear(
            shared_intermediate_size, hidden_size, bias=False, dtype=dtype
        )
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False, dtype=dtype)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.experts.gate_up_proj, std=0.02)
        nn.init.normal_(self.experts.down_proj, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        import os as _os

        _dbg = int(_os.environ.get("SGLANG_QWEN4_DEBUG", "0"))
        from sglang.srt.distributed import tensor_model_parallel_all_reduce

        T = hidden_states.shape[0]
        gate_logits = self.gate(hidden_states)  # [T, E]
        if int(_os.environ.get("SGLANG_QWEN4_TRACE_NAN", "0")):
            _seen = globals().setdefault("_trace_nan_moe_seen", set())
            for _name, _t in (("moe_in", hidden_states), ("gate", gate_logits)):
                if (self.prefix, _name) in _seen or torch.isfinite(_t).all():
                    continue
                _seen.add((self.prefix, _name))
                _nan = torch.isnan(_t)
                print(
                    f"[NAN-TRACE] layer={self.prefix} stage={_name} "
                    f"nan={int(_nan.sum())} inf={int(torch.isinf(_t).sum())} "
                    f"rows={_nan.any(-1).nonzero().flatten()[:6].tolist()} "
                    f"absmax={float(torch.nan_to_num(_t, nan=0.0, posinf=0.0, neginf=0.0).abs().max()):.4f}",
                    flush=True,
                )
        if _dbg and not getattr(self, "_dbg_done", False):
            self._dbg_done = True
            print(
                f"[moe-w] local_experts={self.num_local_experts} "
                f"gate_max={gate_logits.float().abs().max().item():.4f} "
                f"expert_w_max={self.experts.gate_up_proj.float().abs().max().item():.4f}",
                flush=True,
            )
        topk_logits, topk_experts = torch.topk(gate_logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = torch.softmax(topk_logits, dim=-1)
        else:
            topk_weights = topk_logits

        act_fn = F.silu if self.hidden_act == "silu" else F.gelu

        T = hidden_states.shape[0]
        G = T * self.top_k
        local_idx = self._local_expert_map.to(hidden_states.device)[
            topk_experts.reshape(-1)
        ]
        valid = local_idx >= 0
        safe_idx = local_idx.clamp_min(0)
        valid_i32 = valid.to(torch.int32)

        # Triton fast path (MUSA/CUDA): fused per-slot GEMV over the TP-local
        # expert weights, skipping the materialized [G, 2*inter, K] gather.
        _force_torch = getattr(self, "_force_torch", False) or int(
            _os.environ.get("SGLANG_QWEN4_FORCE_TORCH_MOE", "0")
        )
        inter = self.intermediate_size
        K = self.hidden_size
        use_triton = (
            hidden_states.device.type in ("cuda", "musa")
            and not _force_torch
            and inter % 64 == 0
            and K % 64 == 0
            and G > 0
        )
        if use_triton:
            x_exp = hidden_states.unsqueeze(1).expand(T, self.top_k, K).reshape(G, K)
            intermed = torch.empty(
                (G, inter), dtype=hidden_states.dtype, device=hidden_states.device
            )
            contrib = torch.empty(
                (G, K), dtype=hidden_states.dtype, device=hidden_states.device
            )
            BI, BK1 = 128, 64
            if int(_os.environ.get("SGLANG_QWEN4_MOE_BISECT", "0")):
                import torch as _t
                _t.musa.synchronize()
                topk_cpu = topk_experts.cpu()
                map_dev = self._local_expert_map.to("cpu")
                print(
                    f"[moe-bisect] layer={self.prefix} G={G} inter={inter} K={K} "
                    f"e_max={int(safe_idx.max()) if G else -1} "
                    f"e_min={int(safe_idx.min()) if G else -1} "
                    f"valid={int(valid_i32.sum())}/{G} "
                    f"topk0={topk_cpu[0].tolist() if T else []} "
                    f"map_dev0={map_dev.tolist()[:4]} "
                    f"map_cpu0={self._local_expert_map.data.cpu().tolist()[:4]} "
                    f"safe_contig={safe_idx.is_contiguous()} "
                    f"safe_dev={safe_idx.device} "
                    f"ptrs=(topk_logits,{topk_logits.data_ptr()},{topk_experts.data_ptr()},{topk_weights.data_ptr()},{gate_logits.data_ptr()},{gate_logits.stride()}) "
                    f"gate_fin={bool(torch.isfinite(gate_logits).all())} "
                    f"gate_max={float(gate_logits.abs().max()):.4f}",
                    flush=True,
                )
            _moe_gate_up_kernel[(G, (inter + BI - 1) // BI)](
                x_exp, self.experts.gate_up_proj, safe_idx, valid_i32,
                intermed, K,
                x_exp.stride(0), self.experts.gate_up_proj.stride(0),
                self.experts.gate_up_proj.stride(1), self.experts.gate_up_proj.stride(2),
                intermed.stride(0),
                INTER=inter, BLOCK_INTER=BI, BLOCK_K=BK1,
            )
            if int(_os.environ.get("SGLANG_QWEN4_MOE_BISECT", "0")):
                torch.musa.synchronize()
                print("[moe-bisect] gate_up OK", flush=True)
            BH, BK2 = 128, 64
            _moe_down_kernel[(G, (K + BH - 1) // BH)](
                intermed, self.experts.down_proj, safe_idx, valid_i32, contrib, inter,
                intermed.stride(0), self.experts.down_proj.stride(0),
                self.experts.down_proj.stride(1), self.experts.down_proj.stride(2),
                contrib.stride(0),
                BLOCK_H=BH, BLOCK_K=BK2,
            )
            if int(_os.environ.get("SGLANG_QWEN4_MOE_BISECT", "0")):
                torch.musa.synchronize()
                print("[moe-bisect] down OK", flush=True)
            output = (
                contrib.reshape(T, self.top_k, K) * topk_weights.unsqueeze(-1)
            ).sum(dim=1)
            if _dbg:
                print(
                    f"[moe] triton pre-ar max={output.float().abs().max().item():.4f} "
                    f"topk_exp0={topk_experts[0].tolist()[:10]}",
                    flush=True,
                )
            output = tensor_model_parallel_all_reduce(output)
            if _dbg:
                print(
                    f"[moe] triton post-ar max={output.float().abs().max().item():.4f}",
                    flush=True,
                )
        elif T >= 8:
            # Ascend-native dropless MoE.  The prior bounded-bmm path still
            # evaluated every global routed slot on every TP rank and masked
            # the 7/8 non-local results afterwards.  Permute only this rank's
            # routes, run the two expert projections as grouped matmuls, then
            # unpermute/weight them.  Besides removing the per-slot weight
            # gathers, this reduces expert FLOPs by approximately tp_size.
            import torch_npu

            routing_idx = torch.where(
                valid,
                local_idx,
                torch.full_like(local_idx, self.num_local_experts),
            ).reshape(T, self.top_k).to(torch.int64)
            num_valid = int(valid_i32.sum().item())
            if num_valid:
                permuted, sorted_indices = torch_npu.npu_moe_token_permute(
                    hidden_states,
                    routing_idx,
                    num_out_tokens=num_valid,
                )
                counts = torch.bincount(
                    routing_idx[routing_idx < self.num_local_experts],
                    minlength=self.num_local_experts,
                ).to(torch.int64)
                gate_up = torch_npu.npu_grouped_matmul(
                    [permuted],
                    [self.experts.gate_up_proj.transpose(1, 2)],
                    group_list=counts,
                    split_item=3,
                    group_type=0,
                    group_list_type=1,
                )[0]
                intermed = act_fn(gate_up[:, :inter]) * gate_up[:, inter:]
                expert_out = torch_npu.npu_grouped_matmul(
                    [intermed],
                    [self.experts.down_proj.transpose(1, 2)],
                    group_list=counts,
                    split_item=3,
                    group_type=0,
                    group_list_type=1,
                )[0]
                output = torch_npu.npu_moe_token_unpermute(
                    expert_out,
                    sorted_indices,
                    probs=topk_weights,
                )
            else:
                output = torch.zeros_like(hidden_states)
            output = tensor_model_parallel_all_reduce(output)
        elif T >= 64:
            # Long-prefill NPU path. Gathering a complete expert matrix for
            # every routed slot scales as O(T * top_k * expert_weights) and a
            # 4096-token request would transiently allocate about 250 GiB.
            # Apply the already-validated vectorized torch reference in
            # bounded routed-slot chunks instead. This deliberately preserves
            # its gather/bmm ordering and numerical behavior.
            x_exp = hidden_states.unsqueeze(1).expand(T, self.top_k, K).reshape(G, K)
            contrib_chunks = []
            slot_chunk = 512
            for slot_start in range(0, G, slot_chunk):
                slot_end = min(slot_start + slot_chunk, G)
                chunk_idx = safe_idx[slot_start:slot_end]
                chunk_x = x_exp[slot_start:slot_end]
                gu = self.experts.gate_up_proj[chunk_idx]
                dp = self.experts.down_proj[chunk_idx]
                up = act_fn(
                    torch.bmm(
                        gu[:, :inter], chunk_x.unsqueeze(-1)
                    ).squeeze(-1)
                )
                gate = torch.bmm(
                    gu[:, inter:], chunk_x.unsqueeze(-1)
                ).squeeze(-1)
                chunk_out = torch.bmm(
                    dp, (up * gate).unsqueeze(-1)
                ).squeeze(-1)
                chunk_out = torch.where(
                    valid[slot_start:slot_end, None],
                    chunk_out,
                    torch.zeros_like(chunk_out),
                )
                contrib_chunks.append(chunk_out)
            contrib = torch.cat(contrib_chunks, dim=0)
            output = (
                contrib.reshape(T, self.top_k, K) * topk_weights.unsqueeze(-1)
            ).sum(dim=1)
            output = tensor_model_parallel_all_reduce(output)
        else:
            # Vectorized routed experts: gather per-(token,slot) expert weights
            # and batch the matmuls (identical math to the per-token loop,
            # non-local experts masked out before the all-reduce).
            gu = self.experts.gate_up_proj[safe_idx].reshape(
                G, 2 * inter, K
            )
            dp = self.experts.down_proj[safe_idx].reshape(
                G, K, inter
            )
            x_exp = (
                hidden_states.unsqueeze(1)
                .expand(T, self.top_k, K)
                .reshape(G, K)
            )
            up = act_fn(
                torch.bmm(gu[:, :inter], x_exp.unsqueeze(-1)).squeeze(-1)
            )
            gate = torch.bmm(gu[:, inter:], x_exp.unsqueeze(-1)).squeeze(-1)
            intermed = (up * gate).unsqueeze(-1)  # [G, inter, 1]
            contrib = torch.bmm(dp, intermed).squeeze(-1)  # [G, H]
            contrib = contrib.reshape(T, self.top_k, K)
            contrib = torch.where(
                valid.reshape(T, self.top_k, 1), contrib, torch.zeros_like(contrib)
            )
            output = (contrib * topk_weights.unsqueeze(-1)).sum(dim=1)
            if _dbg:
                print(
                    f"[moe] pre-ar max={output.float().abs().max().item():.4f} "
                    f"topk_exp0={topk_experts[0].tolist()[:10]}",
                    flush=True,
                )
            output = tensor_model_parallel_all_reduce(output)
            if _dbg:
                print(
                    f"[moe] post-ar max={output.float().abs().max().item():.4f}",
                    flush=True,
                )

        # Shared expert with sigmoid gate (Qwen3-style).
        shared = act_fn(
            self.shared_expert.gate_proj(hidden_states)
        ) * self.shared_expert.up_proj(hidden_states)
        shared = self.shared_expert.down_proj(shared)
        shared = shared * torch.sigmoid(self.shared_expert_gate(hidden_states))
        if _dbg:
            print(
                f"[moe] shared max={shared.float().abs().max().item():.4f}",
                flush=True,
            )
        return output + shared


class Qwen4ExpQSAAttention(nn.Module):
    """Full attention with QSA sparse selection (torch, M1).

    Checkpoint weights under a layer: ``q_proj [24*2*256, 2560]``,
    ``k_proj [512, 2560]``, ``v_proj [512, 2560]``, ``q_norm/k_norm [256]``
    (Gemma 1+w), ``o_proj [2560, 6144]``, ``indexer.*``.
    """

    def __init__(
        self,
        config: Any,
        layer_id: int,
        quant_config: Any = None,
        prefix: str = "",
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        del quant_config
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", None) or (self.hidden_size // self.num_heads))
        self.attn_output_gate = bool(getattr(config, "attn_output_gate", True))
        self.prefix = prefix
        self.max_position_embeddings = int(getattr(config, "max_position_embeddings", 8192))

        gate_mult = 2 if self.attn_output_gate else 1
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * gate_mult * self.head_dim,
            bias=False,
            dtype=dtype,
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False, dtype=dtype
        )
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.q_norm = Qwen4ExpGroupedGemmaRMSNorm(self.head_dim, eps, None, dtype)
        self.k_norm = Qwen4ExpGroupedGemmaRMSNorm(self.head_dim, eps, None, dtype)

        from sglang.srt.layers.rotary_embedding import get_rope
        from sglang.srt.utils.hf_transformers_utils import get_rope_config

        self.rope_theta, rope_scaling = get_rope_config(config)
        if rope_scaling and not (
            "rope_type" in rope_scaling or "type" in rope_scaling
        ):
            rope_scaling = None
        self.partial_rotary_factor = float(getattr(config, "partial_rotary_factor", 1.0))
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            rope_scaling=rope_scaling,
            base=self.rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )

        self.indexer = Qwen4ExpQSAIndexer(
            hidden_size=self.hidden_size,
            indexer_n_heads=int(getattr(config, "indexer_n_heads", 4)),
            indexer_kv_heads=int(getattr(config, "indexer_kv_heads", 1)),
            indexer_head_dim=int(getattr(config, "indexer_head_dim", 128)),
            budget=int(getattr(config, "indexer_budget", 2048)),
            compress_ratio=int(getattr(config, "indexer_compress_ratio", 4)),
            rms_norm_eps=eps,
            prefix=f"{prefix}.indexer",
            rotary_emb=self.rotary_emb,
            dtype=dtype,
        )
        self.sparse_attn = Qwen4ExpQSASparseAttention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )
        # Per-request K/V history for sparse attention (M1, keyed by req pool
        # index). reset_request_state() clears it for brand-new requests.
        self.kv_states: dict[int, Qwen4ExpQSAKVState] = {}

    def reset_request_state(self, req_key: int) -> None:
        # Request-pool indices are reused. Preserve the large K/V backing and
        # clear only its logical length to avoid allocator fragmentation across
        # benchmark waves.
        state = self.kv_states.get(req_key)
        if state is not None:
            state.reset()

    @staticmethod
    def _apply_qk_norm(
        norm: nn.Module, tensor: torch.Tensor, head_dim: int
    ) -> torch.Tensor:
        """Normalize independent Q/K rows with bounded Ascend temporaries."""
        shape = tensor.shape
        flat = tensor.reshape(-1, head_dim)
        if flat.device.type == "npu" and flat.shape[0] > 4096:
            # ``flat`` may already be the contiguous materialization of a
            # strided q/gate view. Reuse it as output instead of allocating a
            # second full-size bf16 tensor, and keep fp32 RMSNorm temporaries
            # bounded to one 4096-row slice.
            for start in range(0, flat.shape[0], 4096):
                end = min(start + 4096, flat.shape[0])
                flat[start:end].copy_(norm(flat[start:end]))
            return flat.reshape(shape)
        return norm(flat).reshape(shape)

    def forward_batched_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        qsa_states: list[Qwen4ExpQSAState],
        req_keys: list[int],
    ) -> torch.Tensor:
        """Vectorized one-token QSA decode over independent request states."""
        batch = hidden_states.shape[0]
        if batch != len(qsa_states) or batch != len(req_keys):
            raise ValueError("QSA decode batch and state lists must match")

        qg = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states).view(
            batch, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            batch, self.num_kv_heads, self.head_dim
        )
        if self.attn_output_gate:
            qg = qg.view(batch, self.num_heads, 2 * self.head_dim)
            q, gate = torch.chunk(qg, 2, dim=-1)
        else:
            q = qg.view(batch, self.num_heads, self.head_dim)
            gate = None
        q = self._apply_qk_norm(self.q_norm, q, self.head_dim)
        k = self._apply_qk_norm(self.k_norm, k, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)

        index_q, token_k = self.indexer.project_qk(hidden_states, positions)
        kv_states = []
        for i, (req_key, state) in enumerate(zip(req_keys, qsa_states)):
            # Decode position is the next request-relative token. Avoid
            # ``int(npu_tensor)`` here: that would force one host/device sync
            # per request in every full-attention layer.
            state.append_raw(token_k[i], state.raw_len)
            if req_key not in self.kv_states:
                self.kv_states[req_key] = Qwen4ExpQSAKVState(
                    max_tokens=_STATE_CAPACITY,
                    kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
            kv_state = self.kv_states[req_key]
            kv_state.append(k[i : i + 1], v[i : i + 1])
            kv_states.append(kv_state)

        self.indexer.manual_compress_batched(qsa_states)

        scalar_positions = positions if positions.ndim == 1 else positions[0]
        indices = self.indexer.select_batched_decode(
            index_q, qsa_states, scalar_positions
        )
        safe_indices = indices.clamp_min(0)
        gathered_k = torch.stack(
            [state.k[safe_indices[i]] for i, state in enumerate(kv_states)], dim=0
        )
        gathered_v = torch.stack(
            [state.v[safe_indices[i]] for i, state in enumerate(kv_states)], dim=0
        )
        kv_lens = torch.tensor(
            [state.len for state in kv_states],
            dtype=torch.int64,
            device=hidden_states.device,
        )
        valid = (indices >= 0) & (indices < kv_lens[:, None])
        head_outputs = []
        group_size = self.num_heads // self.num_kv_heads
        for kv_head in range(self.num_kv_heads):
            h0 = kv_head * group_size
            h1 = (kv_head + 1) * group_size
            scores = torch.einsum(
                "bwd,bhd->bhw",
                gathered_k[:, :, kv_head].float(),
                q[:, h0:h1].float(),
            ) * self.sparse_attn.scale
            scores = torch.where(
                valid[:, None, :],
                scores,
                torch.full_like(scores, float("-inf")),
            )
            probs = torch.softmax(scores, dim=-1)
            head_outputs.append(
                torch.einsum(
                    "bhw,bwd->bhd",
                    probs,
                    gathered_v[:, :, kv_head].float(),
                )
            )
        attn_output = torch.cat(head_outputs, dim=1).to(hidden_states.dtype)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output.reshape(batch, -1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        qsa_state: Qwen4ExpQSAState,
        req_key: int = 0,
        is_new: bool = False,
    ) -> torch.Tensor:
        if is_new:
            kv_state = self.kv_states.get(req_key)
            if kv_state is not None:
                kv_state.reset()
            qsa_state.reset()
        if req_key not in self.kv_states:
            self.kv_states[req_key] = Qwen4ExpQSAKVState(
                # SGLang can enter one final decode forward before enforcing
                # the requested output limit. Keep a small boundary margin
                # above the exact 4096 + 1024 token workload.
                max_tokens=_STATE_CAPACITY,
                kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        kv_state = self.kv_states[req_key]
        T = hidden_states.shape[0]
        qg = self.q_proj(hidden_states)  # [T, H*gate_mult*d]
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if self.attn_output_gate:
            qg = qg.view(T, self.num_heads, 2 * self.head_dim)
            q, gate = torch.chunk(qg, 2, dim=-1)
        else:
            q = qg.view(T, self.num_heads, self.head_dim)
            gate = None
        q = q.view(T, self.num_heads, self.head_dim)
        k = k.view(T, self.num_kv_heads, self.head_dim)
        v = v.view(T, self.num_kv_heads, self.head_dim)

        q = self._apply_qk_norm(self.q_norm, q, self.head_dim)
        k = self._apply_qk_norm(self.k_norm, k, self.head_dim)
        import os as _os

        _dbg = int(_os.environ.get("SGLANG_QWEN4_DEBUG", "0"))
        if _dbg:
            print(
                f"[qsa{self.layer_id}] pre-rope q_fin={torch.isfinite(q).all().item()} "
                f"k_fin={torch.isfinite(k).all().item()} "
                f"positions={positions[:6].tolist()}",
                flush=True,
            )
        q, k = self.rotary_emb(positions, q, k)
        if _dbg:
            print(
                f"[qsa{self.layer_id}] post-rope q_fin={torch.isfinite(q).all().item()} "
                f"k_fin={torch.isfinite(k).all().item()} "
                f"q_nan={torch.isnan(q).sum().item()} k_nan={torch.isnan(k).sum().item()}",
                flush=True,
            )

        graph_decode = (
            bool(int(_os.environ.get("SGLANG_QWEN4_GRAPH_CANARY", "0")))
            and bool(int(_os.environ.get("SGLANG_QWEN4_GRAPH_QSA", "1")))
            and T == 1
            and hidden_states.device.type == "npu"
        )
        if graph_decode:
            indices = self.indexer.select_graph_decode(
                hidden_states, positions, qsa_state
            )
            scalar_positions = positions if positions.ndim == 1 else positions[0]
            kv_state.graph_append(k, v, scalar_positions)
        else:
            indices = self.indexer.select(hidden_states, positions, qsa_state)
            # Append this forward's K/V to the running history BEFORE attention so
            # decode can reach back into prior tokens.
            kv_state.append(k, v)
        if _dbg:
            print(
                f"[qsa{self.layer_id}] indices shape={tuple(indices.shape)} "
                f"valid={int((indices >= 0).sum())} kv_len={kv_state.len}",
                flush=True,
            )
        import os as _os2

        if int(_os2.environ.get("SGLANG_QWEN4_NOQSA", "0")):
            attn_output = torch.zeros_like(q)
        elif int(_os2.environ.get("SGLANG_QWEN4_FULLATTN", "0")):
            # A/B: dense attention over ALL history (ignore indexer selection).
            all_idx = torch.arange(kv_state.len, device=q.device).unsqueeze(0).expand(
                q.shape[0], -1
            )
            attn_output = self.sparse_attn(q, kv_state.k, kv_state.v, all_idx, kv_state.len)
        else:
            attn_output = self.sparse_attn(
                q,
                kv_state.k,
                kv_state.v,
                indices,
                None if graph_decode else kv_state.len,
            )
        if _dbg:
            print(
                f"[qsa{self.layer_id}] attn_out fin={torch.isfinite(attn_output).all().item()} "
                f"nan={torch.isnan(attn_output).sum().item()}",
                flush=True,
            )
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        output = self.o_proj(attn_output.reshape(T, -1))
        if _dbg:
            print(
                f"[qsa{self.layer_id}] o_proj fin={torch.isfinite(output).all().item()} "
                f"nan={torch.isnan(output).sum().item()}",
                flush=True,
            )
        return output


__all__ = [
    "Qwen4ExpGatedResidualSimple",
    "Qwen4ExpGroupedGemmaRMSNorm",
    "Qwen4ExpQSAIndexer",
    "Qwen4ExpQSASparseAttention",
    "Qwen4ExpQSAState",
    "apply_qsa_rope",
    "Qwen4ExpNGramEmbedding",
    "Qwen4ExpPLELayer",
    "Qwen4ExpGatedDeltaNet",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpMoe",
]
