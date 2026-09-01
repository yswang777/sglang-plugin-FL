"""Qwen4-Exp (internal Qwen3.8-Flash-Next) decoder layer for sglang.

Ported from vllm-plugin-fl-qwen4-day0.patch
``vllm_fl/models/qwen3_8_flash_next/model.py``
(``Qwen3_8FlashNextDecoderLayer``). M1 keeps everything TP-replicated
(no sharding) except the MoE/GDN modules that sglang's own layers handle.

Layer structure (per vLLM forward)::

    if self.ple is not None:            # layer_id == ple_layer_ids-1
        hidden = hidden + ple(...)
    mixed, hc_res = attn_hyper_connection.mix(hidden)
    out = linear_attn(mixed) | qsa_attn(mixed, positions)
    hidden = attn_hyper_connection.combine(out, hc_res)
    mixed, hc_res = mlp_hyper_connection.mix(hidden)
    hidden = mlp(mixed)                 # Qwen2MoeSparseMoeBlock
    hidden = mlp_hyper_connection.combine(hidden, hc_res)
"""

from __future__ import annotations

import os as _os
from typing import Any

import torch
from torch import nn

from sglang.srt.models.qwen4_ops import (
    Qwen4ExpGatedDeltaNet,
    Qwen4ExpGatedResidualSimple,
    Qwen4ExpMoe,
    Qwen4ExpPLELayer,
    Qwen4ExpQSAAttention,
    Qwen4ExpQSAState,
)

# --- TEMP NaN-trace instrumentation (SGLANG_QWEN4_TRACE_NAN=1) ---
_TRACE_NAN = int(_os.environ.get("SGLANG_QWEN4_TRACE_NAN", "0"))
_STATE_CAPACITY = int(_os.environ.get("SGLANG_QWEN4_STATE_CAPACITY", "6208"))
_trace_nan_seen: set[tuple[int, str]] = set()


def _trace_finite(layer_id: int, tag: str, x: torch.Tensor) -> None:
    """Print the FIRST time a stage output goes non-finite (per layer/tag)."""
    if not _TRACE_NAN:
        return
    if torch.isfinite(x).all():
        return
    key = (layer_id, tag)
    if key in _trace_nan_seen:
        return
    _trace_nan_seen.add(key)
    nan = torch.isnan(x)
    print(
        f"[NAN-TRACE] layer={layer_id} stage={tag} "
        f"nan={int(nan.sum())} inf={int(torch.isinf(x).sum())} "
        f"nan_rows={nan.any(-1).nonzero().flatten()[:6].tolist()} "
        f"absmax={float(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).abs().max()):.4f}",
        flush=True,
    )


# --- end TEMP NaN-trace ---


def _ple_layer(config: Any, layer_id: int, prefix: str) -> Qwen4ExpPLELayer:
    """Build the PLE layer with checkpoint-config defaults (M1 replicated)."""
    return Qwen4ExpPLELayer(
        hidden_size=int(config.hidden_size),
        hc_count=int(config.hc_count),
        ple_embed_dim=int(
            getattr(config, "ple_embed_dim", None) or config.hidden_size
        ),
        ple_conv_kernel_size=int(getattr(config, "ple_conv_kernel_size", 4)),
        ngram_size=int(getattr(config, "ngram_size", 3)),
        heads_per_ngram=int(getattr(config, "heads_per_ngram", 8)),
        ngram_vocab_size_base=int(getattr(config, "ngram_vocab_size_base", 20_000_000)),
        make_divisible_by=int(getattr(config, "make_ngram_vocab_size_divisible_by", 128)),
        vocab_size=int(getattr(config, "vocab_size", 0)),
        eos_token_id=int(getattr(config, "eos_token_id", 0)),
        split_ngram_parts=int(getattr(config, "split_ngram_parts", 128)),
        seed=int(getattr(config, "seed", 1234)),
        rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        dtype=torch.bfloat16 if getattr(config, "dtype", None) == "bfloat16" else None,
        max_total_tokens=_STATE_CAPACITY,
        max_num_reqs=64,
        prefix=prefix,
    )


class Qwen4ExpDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        layer_id: int,
        quant_config: Any = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = int(layer_id)
        self.layer_type = config.layer_types[self.layer_id]
        self.prefix = prefix
        self.hidden_size = int(config.hidden_size)
        self.hc_count = int(config.hc_count)
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.hc_lowrank = int(getattr(config, "hc_lowrank", 320))
        eps = float(getattr(config, "rms_norm_eps", 1e-6))

        # PLE injected before the HC mix on ple_layer_ids (1-based).
        ple_layer_ids = getattr(config, "ple_layer_ids", []) or []
        self.ple: Qwen4ExpPLELayer | None = None
        if (self.layer_id + 1) in {int(i) for i in ple_layer_ids}:
            self.ple = _ple_layer(config, self.layer_id, f"{prefix}.ple")

        self.attn_hyper_connection = Qwen4ExpGatedResidualSimple(
            self.hc_count,
            self.hidden_size,
            self.hc_lowrank,
            rms_norm_eps=eps,
            prefix=f"{prefix}.attn_hyper_connection",
        )

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen4ExpGatedDeltaNet(
                hidden_size=self.hidden_size,
                num_k_heads=int(config.linear_num_key_heads),
                num_v_heads=int(config.linear_num_value_heads),
                head_k_dim=int(config.linear_key_head_dim),
                head_v_dim=int(config.linear_value_head_dim),
                conv_kernel=int(config.linear_conv_kernel_dim),
                rms_norm_eps=eps,
                prefix=f"{prefix}.linear_attn",
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen4ExpQSAAttention(
                config,
                self.layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        self.mlp = Qwen4ExpMoe(
            hidden_size=self.hidden_size,
            num_experts=int(config.num_experts),
            top_k=int(config.num_experts_per_tok),
            intermediate_size=int(config.moe_intermediate_size),
            shared_intermediate_size=int(
                getattr(config, "shared_expert_intermediate_size", 0)
            ),
            hidden_act=str(getattr(config, "hidden_act", "silu")),
            norm_topk_prob=bool(getattr(config, "norm_topk_prob", True)),
            prefix=f"{prefix}.mlp",
        )
        self.mlp_hyper_connection = Qwen4ExpGatedResidualSimple(
            self.hc_count,
            self.hidden_size,
            self.hc_lowrank,
            rms_norm_eps=eps,
            prefix=f"{prefix}.mlp_hyper_connection",
        )

        # M1 per-request state containers, keyed by request pool index.
        # reset_request_state() clears them when a request is brand new.
        self.ple_conv_states: dict[int, torch.Tensor] = {}
        self.gdn_conv_states: dict[int, torch.Tensor] = {}
        self.gdn_conv_sources: dict[int, tuple[torch.Tensor, int]] = {}
        self.gdn_ssm_states: dict[int, torch.Tensor] = {}
        self.gdn_ssm_sources: dict[int, tuple[torch.Tensor, int]] = {}
        self.qsa_states: dict[int, Qwen4ExpQSAState] = {}

    def reset_request_state(self, req_key: int) -> None:
        """Drop M1 per-request state for a new request (fresh prefill)."""
        graph_canary = (
            bool(int(_os.environ.get("SGLANG_QWEN4_GRAPH_CANARY", "0")))
            and bool(int(_os.environ.get("SGLANG_QWEN4_GRAPH_INPLACE_STATE", "1")))
        )
        # NPU graph replay retains the exact tensor addresses observed during
        # capture.  Keep those allocations alive when a request-pool slot is
        # reused; clearing their contents is equivalent to dropping eager
        # state, while preserving the addresses consumed by the graph.
        if graph_canary:
            ple_state = self.ple_conv_states.get(req_key)
            if ple_state is not None:
                ple_state.zero_()
            conv_state = self.gdn_conv_states.get(req_key)
            if conv_state is not None:
                conv_state.zero_()
            ssm_state = self.gdn_ssm_states.get(req_key)
            if ssm_state is not None:
                ssm_state.zero_()
        else:
            self.ple_conv_states.pop(req_key, None)
            self.gdn_conv_states.pop(req_key, None)
            self.gdn_ssm_states.pop(req_key, None)
        self.gdn_conv_sources.pop(req_key, None)
        self.gdn_ssm_sources.pop(req_key, None)
        # QSA histories are the largest per-request allocations. Request-pool
        # slots are reused across scheduler waves, so keep their backing and
        # reset lengths in place instead of freeing/reallocating large blocks.
        # Reallocation fragmented the Ascend caching allocator on wave two.
        qsa_state = self.qsa_states.get(req_key)
        if qsa_state is not None:
            qsa_state.reset()
        if self.layer_type == "full_attention":
            self.self_attn.reset_request_state(req_key)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        positions: torch.Tensor,
        req_keys: list[int],
        is_new: list[bool],
    ) -> torch.Tensor:
        """Process stateful attention per request and stateless blocks batched.

        sglang feeds the whole batch (T rows, R requests) to the model in one
        forward call. M1 processed every row under ``req_keys[0]``'s state,
        which mixed independent sequences into one GDN/QSA/PLE state — over
        decode steps that state overflows fp32 (h_max -> 3.4e38) and the
        next MoE gate sees NaN. PLE remains split by request; GDN and decode
        QSA batch their math while retaining separate request-state buffers.
        HyperConnection and MoE operate on the concatenated token batch. This
        preserves state isolation and restores multi-request throughput.
        """
        graph_serial_rows = (
            bool(int(_os.environ.get("SGLANG_QWEN4_GRAPH_CANARY", "0")))
            and len(req_keys) > 1
            and hidden_states.shape[0] == len(req_keys)
        )
        if graph_serial_rows:
            # NPU graph replay retains tensor addresses from capture.  Run
            # every decode row through the already validated T=1 graph path,
            # binding row r to logical state slot r.  This is deliberately a
            # correctness-first BS>1 canary; once it passes churn tests these
            # row-wise recurrences can be replaced by batched Triton kernels.
            outputs = []
            for r, req_key in enumerate(req_keys):
                if self.ple is not None:
                    self.ple.ple_embedding._graph_decode_row = r
                pos_r = (
                    positions[..., r : r + 1]
                    if positions.ndim > 1
                    else positions[r : r + 1]
                )
                qsl_r = torch.tensor(
                    [0, 1],
                    dtype=query_start_loc.dtype,
                    device=query_start_loc.device,
                )
                outputs.append(
                    self._forward_req(
                        hidden_states[r : r + 1],
                        input_ids[r : r + 1],
                        qsl_r,
                        ngram_context[r : r + 1],
                        pos_r,
                        req_key,
                        is_new[r],
                    )
                )
            return torch.cat(outputs, dim=0)
        if len(req_keys) == 1:
            return self._forward_req(
                hidden_states, input_ids, query_start_loc, ngram_context,
                positions, req_keys[0], is_new[0],
            )
        # Decode contributes most of the work in the target workload.  In that
        # path every request contributes exactly one row, so reading each qsl
        # element with ``int(npu_tensor)`` would force two device synchronizes
        # per request *per layer* (6144 synchronizes/token at B=64, L=48).
        # Build those trivial slices on the host without touching the NPU.
        if hidden_states.shape[0] == len(req_keys):
            segments = [(r, r, r + 1) for r in range(len(req_keys))]
        else:
            # Prefill is much less frequent.  Transfer the small boundary
            # vector once instead of synchronizing once for every scalar.
            qsl_host = query_start_loc.reshape(-1).tolist()
            segments = []
            for r in range(len(req_keys)):
                start, end = qsl_host[r], qsl_host[r + 1]
                if end <= start:
                    continue
                segments.append((r, start, end))

        for r, _, _ in segments:
            if is_new[r]:
                self.reset_request_state(req_keys[r])

        # Only one configured layer owns PLE. Avoid rebuilding/catenating all
        # request slices in the other 47 layers, and batch its one-token decode
        # convolution when PLE is present.
        if self.ple is not None and not int(
            _os.environ.get("SGLANG_QWEN4_NOPLE", "0")
        ):
            ple_hidden = torch.empty_like(hidden_states)
            decode_segments = []
            for r, start, end in segments:
                req_key = req_keys[r]
                hs_r = hidden_states[start:end]
                if req_key not in self.ple_conv_states:
                    self.ple_conv_states[req_key] = torch.zeros(
                        self.hc_hidden_size,
                        self.ple.conv_state_len,
                        dtype=hs_r.dtype,
                        device=hs_r.device,
                    )
                if end - start == 1:
                    decode_segments.append((r, start, end))
                    continue
                qsl_r = torch.tensor(
                    [0, end - start],
                    dtype=query_start_loc.dtype,
                    device=query_start_loc.device,
                )
                ple_out, next_ple = self.ple(
                    hs_r,
                    input_ids[start:end],
                    qsl_r,
                    ngram_context[r : r + 1],
                    self.ple_conv_states[req_key],
                )
                self.ple_conv_states[req_key] = next_ple
                ple_hidden[start:end] = hs_r + ple_out
            if len(decode_segments) == 1:
                r, start, end = decode_segments[0]
                req_key = req_keys[r]
                qsl_r = torch.tensor(
                    [0, 1], dtype=query_start_loc.dtype,
                    device=query_start_loc.device,
                )
                ple_out, next_ple = self.ple(
                    hidden_states[start:end], input_ids[start:end], qsl_r,
                    ngram_context[r : r + 1], self.ple_conv_states[req_key],
                )
                self.ple_conv_states[req_key] = next_ple
                ple_hidden[start:end] = hidden_states[start:end] + ple_out
            elif decode_segments:
                row_indices_host = [start for _, start, _ in decode_segments]
                request_indices_host = [r for r, _, _ in decode_segments]
                row_indices = torch.tensor(
                    row_indices_host,
                    dtype=torch.int64,
                    device=hidden_states.device,
                )
                decode_states = torch.stack(
                    [
                        self.ple_conv_states[req_keys[r]]
                        for r, _, _ in decode_segments
                    ],
                    dim=0,
                )
                ple_out, next_states = self.ple.forward_batched_decode(
                    hidden_states[row_indices],
                    input_ids[row_indices_host],
                    ngram_context[request_indices_host],
                    decode_states,
                )
                ple_hidden[row_indices] = hidden_states[row_indices] + ple_out
                for i, (r, _, _) in enumerate(decode_segments):
                    self.ple_conv_states[req_keys[r]] = next_states[i]
            hidden_states = ple_hidden
        _trace_finite(self.layer_id, "ple_batch", hidden_states)

        mixed, attn_residual = self.attn_hyper_connection.mix(hidden_states)
        _trace_finite(self.layer_id, "attn_mix_batch", mixed)

        # The concatenated layout is unchanged by the PLE/mix steps, so the
        # original request offsets still delimit mixed rows.
        # GDN groups requests by their current token count. This preserves
        # batching even when SGLang mixes one-token decode rows with fixed-size
        # chunked-prefill rows in the same forward batch.
        if self.layer_type == "linear_attention":
            attn_output = torch.empty(
                (mixed.shape[0], self.hidden_size),
                dtype=mixed.dtype,
                device=mixed.device,
            )
            length_groups: dict[int, list[tuple[int, int, int]]] = {}
            for segment in segments:
                _, start, end = segment
                length_groups.setdefault(end - start, []).append(segment)
            for group in length_groups.values():
                conv_states = []
                ssm_states = []
                for r, start, end in group:
                    req_key = req_keys[r]
                    mixed_r = mixed[start:end]
                    if req_key not in self.gdn_conv_states:
                        self.gdn_conv_states[req_key] = torch.zeros(
                            self.linear_attn.conv_dim,
                            self.linear_attn.conv_state_len,
                            dtype=mixed_r.dtype,
                            device=mixed_r.device,
                        )
                    if req_key not in self.gdn_ssm_states:
                        self.gdn_ssm_states[req_key] = Qwen4ExpGatedDeltaNet.new_state(
                            self.linear_attn.num_v_heads,
                            self.linear_attn.head_k_dim,
                            self.linear_attn.head_v_dim,
                            device=mixed_r.device,
                        )
                    conv_states.append(self.gdn_conv_states[req_key])
                    ssm_states.append(self.gdn_ssm_states[req_key])
                if len(group) == 1:
                    r, start, end = group[0]
                    req_key = req_keys[r]
                    out, next_conv, next_ssm = self.linear_attn(
                        mixed[start:end], conv_states[0], ssm_states[0]
                    )
                    attn_output[start:end] = out
                    self.gdn_conv_states[req_key] = next_conv
                    self.gdn_ssm_states[req_key] = next_ssm
                else:
                    full_decode_group = (
                        len(group) == len(req_keys)
                        and mixed.shape[0] == len(req_keys)
                    )
                    mixed_batch = (
                        mixed.unsqueeze(1)
                        if full_decode_group
                        else torch.stack(
                            [mixed[start:end] for _, start, end in group], dim=0
                        )
                    )
                    active_keys = [req_keys[r] for r, _, _ in group]
                    source_entries = [
                        self.gdn_ssm_sources.get(req_key)
                        for req_key in active_keys
                    ]
                    shared_source = (
                        source_entries[0][0]
                        if source_entries[0] is not None
                        and all(
                            entry is not None
                            and entry[0] is source_entries[0][0]
                            for entry in source_entries
                        )
                        else None
                    )
                    if shared_source is None:
                        state_source = torch.stack(ssm_states, dim=0)
                        state_indices = None
                        result_indices = list(range(len(group)))
                    else:
                        result_indices = [entry[1] for entry in source_entries]
                        # The stable decode batch preserves request order, so
                        # the shared backing is already in the required order.
                        # Avoid constructing an NPU index tensor in every layer.
                        if result_indices == list(range(len(group))):
                            # A final partial wave can reuse rows 0..B-1 from
                            # the previous full-wave backing (for example
                            # backing=18, current B=12).  Keep backing and
                            # hidden-state batch dimensions equal.
                            state_source = shared_source[: len(group)]
                            state_indices = None
                        else:
                            state_source = shared_source
                            state_indices = torch.tensor(
                                result_indices,
                                dtype=torch.int64,
                                device=mixed.device,
                            )
                    conv_entries = [
                        self.gdn_conv_sources.get(req_key)
                        for req_key in active_keys
                    ]
                    shared_conv = (
                        conv_entries[0][0]
                        if conv_entries[0] is not None
                        and all(
                            entry is not None
                            and entry[0] is conv_entries[0][0]
                            for entry in conv_entries
                        )
                        and [entry[1] for entry in conv_entries]
                        == list(range(len(group)))
                        else None
                    )
                    conv_source = (
                        shared_conv[: len(group)]
                        if shared_conv is not None
                        else torch.stack(conv_states, dim=0)
                    )
                    out, next_conv, next_ssm = self.linear_attn.forward_batched(
                        mixed_batch,
                        conv_source,
                        state_source,
                        state_indices=state_indices,
                    )
                    if full_decode_group:
                        attn_output = out.squeeze(1)
                    for i, (r, start, end) in enumerate(group):
                        req_key = req_keys[r]
                        if not full_decode_group:
                            attn_output[start:end] = out[i]
                        self.gdn_conv_states[req_key] = next_conv[i]
                        self.gdn_conv_sources[req_key] = (next_conv, i)
                        source_index = result_indices[i]
                        self.gdn_ssm_states[req_key] = next_ssm[source_index]
                        self.gdn_ssm_sources[req_key] = (next_ssm, source_index)
        elif self.layer_type == "full_attention":
            attn_output = torch.empty(
                (mixed.shape[0], self.hidden_size),
                dtype=mixed.dtype,
                device=mixed.device,
            )
            decode_segments = []
            for r, start, end in segments:
                req_key = req_keys[r]
                mixed_r = mixed[start:end]
                if req_key not in self.qsa_states:
                    self.qsa_states[req_key] = Qwen4ExpQSAState(
                        # SGLang overlap scheduling can retain roughly 512
                        # internal decode rows beyond input+requested output.
                        max_tokens=_STATE_CAPACITY,
                        index_head_dim=self.self_attn.indexer.indexer_head_dim,
                        compress_ratio=self.self_attn.indexer.compress_ratio,
                        dtype=mixed_r.dtype,
                        device=mixed_r.device,
                    )
                if end - start == 1:
                    decode_segments.append((r, start, end))
                    continue
                pos_r = (
                    positions[..., start:end]
                    if positions.ndim > 1
                    else positions[start:end]
                )
                attn_output[start:end] = self.self_attn(
                    mixed_r,
                    pos_r,
                    self.qsa_states[req_key],
                    req_key=req_key,
                    is_new=is_new[r],
                )
            if len(decode_segments) == 1:
                r, start, end = decode_segments[0]
                req_key = req_keys[r]
                pos_r = positions[..., start:end]
                attn_output[start:end] = self.self_attn(
                    mixed[start:end],
                    pos_r,
                    self.qsa_states[req_key],
                    req_key=req_key,
                    is_new=is_new[r],
                )
            elif decode_segments:
                decode_keys = [req_keys[r] for r, _, _ in decode_segments]
                decode_states = [self.qsa_states[key] for key in decode_keys]
                full_decode_batch = (
                    len(decode_segments) == len(req_keys)
                    and mixed.shape[0] == len(req_keys)
                )
                row_indices = (
                    None
                    if full_decode_batch
                    else torch.tensor(
                        [start for _, start, _ in decode_segments],
                        dtype=torch.int64,
                        device=mixed.device,
                    )
                )
                decode_output = self.self_attn.forward_batched_decode(
                    mixed if full_decode_batch else mixed[row_indices],
                    positions if full_decode_batch else positions[..., row_indices],
                    decode_states,
                    decode_keys,
                )
                if full_decode_batch:
                    attn_output = decode_output
                else:
                    attn_output[row_indices] = decode_output
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")
        _trace_finite(self.layer_id, "attn_out_batch", attn_output)
        hidden_states = self.attn_hyper_connection.combine(
            attn_output, attn_residual
        )
        mixed, mlp_residual = self.mlp_hyper_connection.mix(hidden_states)
        hidden_states = self.mlp(mixed)
        hidden_states = self.mlp_hyper_connection.combine(
            hidden_states, mlp_residual
        )
        _trace_finite(self.layer_id, "out_batch", hidden_states)
        return hidden_states

    def _forward_req(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        positions: torch.Tensor,
        req_key: int,
        is_new: bool,
    ) -> torch.Tensor:
        import os as _os

        _dbg = int(_os.environ.get("SGLANG_QWEN4_DEBUG", "0"))
        if is_new:
            self.reset_request_state(req_key)

        if _dbg:
            print(
                f"[layer{self.layer_id}] in finite={torch.isfinite(hidden_states).all().item()} "
                f"nan={torch.isnan(hidden_states).sum().item()} "
                f"max={hidden_states.float().abs().max().item():.4f}",
                flush=True,
            )
        _trace_finite(self.layer_id, "in", hidden_states)

        if self.ple is not None:
            if not int(_os.environ.get("SGLANG_QWEN4_NOPLE", "0")):
                if req_key not in self.ple_conv_states:
                    self.ple_conv_states[req_key] = torch.zeros(
                        self.hc_hidden_size,
                        self.ple.conv_state_len,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    )
                ple_out, new_state = self.ple(
                    hidden_states,
                    input_ids,
                    query_start_loc,
                    ngram_context,
                    self.ple_conv_states[req_key],
                )
                if (
                    int(_os.environ.get("SGLANG_QWEN4_GRAPH_CANARY", "0"))
                    and int(_os.environ.get("SGLANG_QWEN4_GRAPH_INPLACE_STATE", "1"))
                ):
                    self.ple_conv_states[req_key].copy_(new_state)
                else:
                    self.ple_conv_states[req_key] = new_state
                hidden_states = hidden_states + ple_out
        _trace_finite(self.layer_id, "ple", hidden_states)

        mixed, hc_residual = self.attn_hyper_connection.mix(hidden_states)
        if _dbg:
            print(
                f"[layer{self.layer_id}] mixed finite={torch.isfinite(mixed).all().item()} "
                f"nan={torch.isnan(mixed).sum().item()} max={mixed.float().abs().max().item():.4f}",
                flush=True,
            )
        _trace_finite(self.layer_id, "attn_mix", mixed)
        if self.layer_type == "linear_attention":
            if req_key not in self.gdn_conv_states:
                self.gdn_conv_states[req_key] = torch.zeros(
                    self.linear_attn.conv_dim,
                    self.linear_attn.conv_state_len,
                    dtype=mixed.dtype,
                    device=mixed.device,
                )
            if req_key not in self.gdn_ssm_states:
                self.gdn_ssm_states[req_key] = Qwen4ExpGatedDeltaNet.new_state(
                    self.linear_attn.num_v_heads,
                    self.linear_attn.head_k_dim,
                    self.linear_attn.head_v_dim,
                    device=mixed.device,
                )
            attn_output, next_conv, next_ssm = self.linear_attn(
                mixed,
                self.gdn_conv_states[req_key],
                self.gdn_ssm_states[req_key],
            )
            if (
                int(_os.environ.get("SGLANG_QWEN4_GRAPH_CANARY", "0"))
                and int(_os.environ.get("SGLANG_QWEN4_GRAPH_INPLACE_STATE", "1"))
            ):
                self.gdn_conv_states[req_key].copy_(next_conv)
                self.gdn_ssm_states[req_key].copy_(next_ssm)
            else:
                self.gdn_conv_states[req_key] = next_conv
                self.gdn_ssm_states[req_key] = next_ssm
        elif self.layer_type == "full_attention":
            if req_key not in self.qsa_states:
                self.qsa_states[req_key] = Qwen4ExpQSAState(
                    # 4096 input + 1024 output + scheduler boundary margin.
                    max_tokens=_STATE_CAPACITY,
                    index_head_dim=self.self_attn.indexer.indexer_head_dim,
                    compress_ratio=self.self_attn.indexer.compress_ratio,
                    dtype=mixed.dtype,
                    device=mixed.device,
                )
            attn_output = self.self_attn(
                mixed, positions, self.qsa_states[req_key],
                req_key=req_key, is_new=is_new,
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")
        _trace_finite(self.layer_id, "attn_out", attn_output)

        hidden_states = self.attn_hyper_connection.combine(attn_output, hc_residual)
        if _dbg:
            print(
                f"[layer{self.layer_id}] attn-combined max={hidden_states.float().abs().max().item():.4f}",
                flush=True,
            )
        _trace_finite(self.layer_id, "attn_comb", hidden_states)
        mixed, hc_residual = self.mlp_hyper_connection.mix(hidden_states)
        if _dbg:
            print(
                f"[layer{self.layer_id}] mlp-mixed max={mixed.float().abs().max().item():.4f}",
                flush=True,
            )
        _trace_finite(self.layer_id, "mlp_mix", mixed)
        hidden_states = self.mlp(mixed)
        _trace_finite(self.layer_id, "mlp_out", hidden_states)
        hidden_states = self.mlp_hyper_connection.combine(hidden_states, hc_residual)
        if _dbg:
            print(
                f"[layer{self.layer_id}] out max={hidden_states.float().abs().max().item():.4f}",
                flush=True,
            )
        _trace_finite(self.layer_id, "mlp_comb", hidden_states)
        return hidden_states


__all__ = ["Qwen4ExpDecoderLayer"]
