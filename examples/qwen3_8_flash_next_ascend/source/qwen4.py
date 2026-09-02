"""Qwen4-Exp (internal Qwen3.8-Flash-Next) causal language model for sglang.

Ported from vllm-plugin-fl-qwen4-day0.patch
``vllm_fl/models/qwen3_8_flash_next/model.py`` (``Qwen3_8FlashNextModel`` +
``Qwen3_8FlashNextForCausalLM``) and the sglang ``Qwen3_5ForCausalLM``
template. M1: torch-first, TP-replicated HC/indexer/PLE, text-only.

Checkpoint layout (weights live under ``model.language_model.*`` in the VLM
checkpoint; ``lm_head.weight`` sits at root). ``hyper_connection_mixer`` is
the final 4->1 stream fold (use_combine=False).
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.configs.qwen4 import Qwen4ExpConfig, Qwen4ExpTextConfig
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.models.qwen4_layer import Qwen4ExpDecoderLayer
from sglang.srt.models.qwen4_ops import Qwen4ExpGatedResidualSimple
from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.utils.common import make_layers
from sglang.srt.distributed import get_pp_group

logger = logging.getLogger(__name__)


class Qwen4ExpForCausalLM(nn.Module):
    """Text backbone: HC streams -> 48 layers -> final mixer -> logits."""

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = int(config.hidden_size)          # 2560
        self.hc_count = int(config.hc_count)                # 4
        self.hc_hidden_size = self.hidden_size * self.hc_count  # 10240
        self.vocab_size = int(config.vocab_size)
        self.ngram_context_len = max(int(getattr(config, "ngram_size", 3)) - 1, 0)
        self.prefix = prefix
        self.pp_group = get_pp_group()

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            self.hidden_size,
            org_num_embeddings=self.vocab_size,
        )

        def get_layer(idx: int, prefix: str) -> Qwen4ExpDecoderLayer:
            return Qwen4ExpDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            get_layer,
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=f"{prefix}.layers",
        )

        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.hyper_connection_mixer = Qwen4ExpGatedResidualSimple(
            self.hc_count,
            self.hidden_size,
            int(getattr(config, "hc_lowrank", 320)),
            rms_norm_eps=eps,
            use_combine=False,
            prefix=f"{prefix}.hyper_connection_mixer",
        )

        # Standalone causal-LM path (no VLM wrapper) owns lm_head + logits
        # processor. The VLM wrapper path (prefix="model.language_model")
        # provides its own, so skip creation here to avoid a duplicate head.
        if prefix == "":
            self.lm_head = ParallelLMHead(
                self.vocab_size,
                self.hidden_size,
                quant_config=quant_config,
                prefix="lm_head",
            )
            self.logits_processor = LogitsProcessor(config)

        # M1 per-request state: ngram context history, keyed by req pool index.
        self.ngram_history: dict[int, list[int]] = {}

    def _prepare_ngram_context(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        req_keys: list[int],
    ) -> torch.Tensor:
        """Build [R, ngram_size-1] left-context per request (EOS-padded)."""
        qsl = query_start_loc.reshape(-1).long()
        num_reqs = qsl.numel() - 1
        eos = int(getattr(self.config, "eos_token_id", 0))
        needed = self.ngram_context_len
        capturing = False
        if int(os.environ.get("SGLANG_QWEN4_GRAPH_CANARY", "0")):
            try:
                capturing = bool(torch.npu.is_current_stream_capturing())
            except Exception:
                capturing = False
        if capturing:
            # Capture-only path: PLE is disabled by the canary launcher, so its
            # history must not force an NPU->host synchronization. This is not
            # a production accuracy path.
            return torch.full(
                (num_reqs, needed),
                eos,
                dtype=torch.long,
                device=input_ids.device,
            )
        context = torch.full(
            (num_reqs, needed), eos, dtype=torch.long, device=input_ids.device
        )
        for r in range(num_reqs):
            start, end = int(qsl[r]), int(qsl[r + 1])
            history = self.ngram_history.get(req_keys[r], [])
            # Left context = last `needed` tokens before this batch, EOS-padded.
            if len(history) < needed:
                pad = [eos] * (needed - len(history)) + history
            else:
                pad = history[-needed:]
            for i in range(needed):
                context[r, i] = pad[i]
            # Keep only the trailing window for the next forward.
            if needed:
                self.ngram_history[req_keys[r]] = (
                    history + input_ids[start:end].tolist()
                )[-needed:]
            else:
                self.ngram_history[req_keys[r]] = []
        return context

    def _req_info(
        self, forward_batch: ForwardBatch
    ) -> tuple[list[int], list[bool], torch.Tensor]:
        """Per-request pool indices, new-request flags, and query_start_loc."""
        graph_canary = int(os.environ.get("SGLANG_QWEN4_GRAPH_CANARY", "0"))
        graph_fixed_keys = int(
            os.environ.get("SGLANG_QWEN4_GRAPH_FIXED_REQ_KEYS", "1")
        )
        graph_max_bs = int(os.environ.get("SGLANG_QWEN4_GRAPH_MAX_BS", "1"))
        if graph_canary and graph_fixed_keys and (
            graph_max_bs == 1 or forward_batch.forward_mode.is_decode()
        ):
            # BS=1 binds both eager prefill and graph decode to slot zero; this
            # is the validated request-churn-safe mapping.  Experimental BS>1
            # keeps eager prefill physical and binds graph rows at capture.
            req_keys = list(range(forward_batch.batch_size))
        else:
            req_keys = forward_batch.req_pool_indices.reshape(-1).tolist()
        if forward_batch.forward_mode.is_decode():
            # Decode: each request contributes exactly one token; PLE treats
            # each row as its own length-1 request with its own left context.
            qsl = torch.arange(
                len(req_keys) + 1,
                dtype=torch.int64,
                device=forward_batch.input_ids.device,
            )
            is_new = [False] * len(req_keys)
        else:
            qsl = getattr(forward_batch, "query_start_loc", None)
            if qsl is None:
                extend_lens = getattr(forward_batch, "extend_seq_lens", None)
                if extend_lens is not None:
                    # SGLang 0.5.11 does not expose query_start_loc on this
                    # ForwardBatch, but does provide each request's flattened
                    # extend length.  arange(R+1) is only valid for decode and
                    # silently drops all but one token per prefill request.
                    qsl = torch.zeros(
                        len(req_keys) + 1,
                        dtype=torch.int64,
                        device=forward_batch.input_ids.device,
                    )
                    qsl[1:] = torch.cumsum(
                        extend_lens.to(device=qsl.device, dtype=torch.int64),
                        dim=0,
                    )
                else:
                    qsl = torch.arange(
                        len(req_keys) + 1,
                        dtype=torch.int64,
                        device=forward_batch.input_ids.device,
                    )
            prefix_lens = getattr(forward_batch, "extend_prefix_lens", None)
            if prefix_lens is not None:
                is_new = [int(p) == 0 for p in prefix_lens.tolist()]
            else:
                is_new = [True] * len(req_keys)
        return req_keys, is_new, qsl

    @torch.no_grad()
    def prepare_qwen4_graph_inputs(self, forward_batch: ForwardBatch) -> None:
        """Prepare the host-backed PLE lookup before NPU graph replay."""
        if not forward_batch.forward_mode.is_decode():
            return
        req_keys = forward_batch.req_pool_indices.reshape(-1).tolist()
        ids_cpu = forward_batch.input_ids.reshape(-1).to(
            device="cpu", dtype=torch.long
        )
        qsl_cpu = torch.arange(ids_cpu.numel() + 1, dtype=torch.long)
        context = self._prepare_ngram_context(ids_cpu, qsl_cpu, req_keys)
        for layer in self.layers:
            ple = getattr(layer, "ple", None)
            if ple is not None:
                ple.ple_embedding.prepare_graph_decode(ids_cpu, context)
                return
        raise RuntimeError("Qwen4 graph PLE layer was not found")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if int(os.environ.get("SGLANG_QWEN4_DEBUG", "0")):
            req_keys, is_new, _qsl = self._req_info(forward_batch)
            print(
                f"[fwd] mode={forward_batch.forward_mode.name} "
                f"T={input_ids.shape[0]} ids={input_ids[:8].tolist()} "
                f"pos={positions[:8].tolist() if positions.ndim == 1 else positions.shape} "
                f"req_keys={req_keys} is_new={is_new}",
                flush=True,
            )
        if input_embeds is None:
            if int(os.environ.get("SGLANG_QWEN4_DEBUG", "0")):
                import torch as _t

                print(
                    f"[embed-pre] tp={getattr(self.embed_tokens, 'tp_size', '?')} "
                    f"shape={tuple(self.embed_tokens.weight.shape)} "
                    f"w_finite={_t.isfinite(self.embed_tokens.weight).all().item()} "
                    f"w_nan={_t.isnan(self.embed_tokens.weight).sum().item()} "
                    f"ids={input_ids[:6].tolist()}",
                    flush=True,
                )
            hidden_states = self.embed_tokens(input_ids)
            if int(os.environ.get("SGLANG_QWEN4_DEBUG", "0")):
                import torch as _t

                _r0 = [round(float(v), 4) for v in hidden_states[0, :4].tolist()]
                _r1 = (
                    [round(float(v), 4) for v in hidden_states[1, :4].tolist()]
                    if hidden_states.shape[0] > 1
                    else []
                )
                print(
                    f"[embed] ids={input_ids[:6].tolist()} "
                    f"finite={_t.isfinite(hidden_states).all().item()} "
                    f"nan={_t.isnan(hidden_states).sum().item()} "
                    f"max={hidden_states.float().abs().max().item():.4f} "
                    f"row0_first4={_r0} row1_first4={_r1}",
                    flush=True,
                )
        else:
            hidden_states = input_embeds
        # HC layout: [T, HC*HS] with HC outer, HS inner — repeat produces
        # exactly that [..., HC, HS] arrangement (verified against HC mix).
        hidden_states = hidden_states.repeat(1, self.hc_count)

        req_keys, is_new, qsl = self._req_info(forward_batch)
        # Fresh request: clear its ngram history so PLE starts from EOS.
        for rk, new in zip(req_keys, is_new):
            if new:
                self.ngram_history.pop(rk, None)

        ngram_context = self._prepare_ngram_context(input_ids, qsl, req_keys)

        # PLE consumes token ids and request boundaries on the host.  Convert
        # them once per model forward instead of once per PLE layer; the latter
        # forces a blocking NPU stream synchronization in every layer/rank.
        if int(os.environ.get("SGLANG_QWEN4_GRAPH_CANARY", "0")):
            ple_input_ids = input_ids.reshape(-1)
            ple_qsl = qsl
            ple_ngram_context = ngram_context
        else:
            ple_input_ids = input_ids.reshape(-1).to(device="cpu", dtype=torch.long)
            ple_qsl = qsl.to(device="cpu", dtype=torch.long)
            ple_ngram_context = ngram_context.to(device="cpu", dtype=torch.long)

        _dump_layers = os.environ.get("SGLANG_QWEN4_DUMP_LAYERS", "")
        _layers_buf = []
        for layer_idx in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_idx]
            hidden_states = layer(
                hidden_states,
                ple_input_ids,
                ple_qsl,
                ple_ngram_context,
                positions,
                req_keys=req_keys,
                is_new=is_new,
            )
            if (
                _dump_layers
                and input_ids.shape[0] >= 3
                and (input_ids[:3].tolist() == [96328, 102357, 96878])
            ):
                _layers_buf.append(hidden_states.float().cpu())
        if _dump_layers and _layers_buf:
            try:
                from sglang.srt.distributed import get_tensor_model_parallel_rank

                if get_tensor_model_parallel_rank() == 0:
                    torch.save(_layers_buf, _dump_layers)
                    print(
                        f"[e2e-layers] saved {len(_layers_buf)} layers -> {_dump_layers}",
                        flush=True,
                    )
            except Exception as _e:
                print(f"[e2e-layers] failed: {_e}", flush=True)
            if int(os.environ.get("SGLANG_QWEN4_DEBUG", "0")):
                import torch as _t

                hf = _t.isfinite(hidden_states).all().item()
                print(
                    f"[qwen4-debug r{get_pp_group().rank_in_group} "
                    f"l{layer_idx}] hidden mean={hidden_states.float().mean().item():.4f} "
                    f"max={hidden_states.float().abs().max().item():.4f} finite={hf}",
                    flush=True,
                )

        mixed, _ = self.hyper_connection_mixer.mix(hidden_states)
        _dump = os.environ.get("SGLANG_QWEN4_DUMP_FINAL", "")
        if _dump and input_ids.shape[0] >= 3 and (
            input_ids[:3].tolist() == [96328, 102357, 96878]
        ):
            try:
                from sglang.srt.distributed import get_tensor_model_parallel_rank

                if get_tensor_model_parallel_rank() == 0:
                    torch.save(mixed.float().cpu(), _dump)
                    print(f"[e2e-dump] saved final hidden to {_dump}", flush=True)
            except Exception as _e:
                print(f"[e2e-dump] failed: {_e}", flush=True)
        if int(os.environ.get("SGLANG_QWEN4_DEBUG", "0")):
            import torch as _t

            print(
                f"[mixer] down_max={self.hyper_connection_mixer.input_mix_weight_down.weight.float().abs().max().item():.4f} "
                f"up_max={self.hyper_connection_mixer.input_mix_weight_up.weight.float().abs().max().item():.4f} "
                f"hc_norm_max={self.hyper_connection_mixer.hc_norm.weight.float().abs().max().item():.4f} "
                f"in_max={hidden_states.float().abs().max().item():.4f} "
                f"out_max={mixed.float().abs().max().item():.4f}",
                flush=True,
            )
        if hasattr(self, "logits_processor"):
            # Standalone path: compute logits on the last PP rank.
            logits = self.logits_processor(
                input_ids, mixed, self.lm_head, forward_batch, []
            )
            if int(os.environ.get("SGLANG_QWEN4_DEBUG", "0")):
                import torch as _t

                lt = getattr(logits, "next_token_logits", None)
                if lt is not None:
                    top5 = _t.topk(lt.float(), 5, dim=-1)
                    print(
                        f"[logits] top5_ids={top5.indices.tolist()} "
                        f"top5_val={[[round(float(x), 2) for x in row] for row in top5.values.tolist()]}",
                        flush=True,
                    )
            return logits
        # VLM-wrapper path: return the pre-logits hidden states; the wrapper
        # owns lm_head + logits processor.
        return mixed

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loaded_params = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        skipped: list[str] = []
        if int(os.environ.get("SGLANG_QWEN4_DEBUG", "0")):
            print(
                f"[load] params_dict sample: {list(params_dict.keys())[:8]}",
                flush=True,
            )
        # Name normalization:
        # The module tree registers params WITHOUT any "model." / "model.language_model."
        # prefix (sglang model classes own bare names under their layer prefix),
        # while checkpoint keys carry "model.language_model." (LM) / "model.visual."
        # (vision) / root "lm_head." prefixes. Always strip the LM wrapper prefixes;
        # the PLE/expert branches resolve modules relative to self.
        strip_prefix = ("model.language_model.", "model.")

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name.startswith(("model.visual", "visual", "mtp")):
                continue
            if strip_prefix:
                for pfx in strip_prefix:
                    if name.startswith(pfx):
                        name = name[len(pfx) :]
                        break

            # PLE ngram embedding shards live in a lazy sharded table.
            if "ple_embedding.ngram_embedding.shard_" in name and name.endswith(
                ".weight"
            ):
                shard_text = name.rsplit("shard_", 1)[1].split(".")[0]
                module_name = name[: name.rfind("ngram_embedding")]
                # Resolve relative to self (strip the wrapper's own prefix).
                if self.prefix and module_name.startswith(self.prefix + "."):
                    module_name = module_name[len(self.prefix) + 1 :]
                embedding = self._resolve_module(module_name + "ngram_embedding")
                shard_index = int(shard_text)
                if not embedding.is_local_shard(shard_index):
                    # Non-local TP shard: skip (this rank does not hold it).
                    continue
                param = embedding.ensure_shard(shard_index)
                if param.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected {tuple(param.shape)}, "
                        f"got {tuple(loaded_weight.shape)}"
                    )
                param.data.copy_(
                    loaded_weight.to(device=param.device, dtype=param.dtype)
                )
                loaded_params.add(name)
                continue

            # MoE expert tables are TP-sharded: the checkpoint holds all
            # experts, the local param only a slice. Slice by local ids.
            if name.endswith("experts.gate_up_proj") or name.endswith(
                "experts.down_proj"
            ):
                expert_name = name.rsplit(".", 1)[-1]
                module_name = name[: name.rfind("experts")].rstrip(".")
                if self.prefix and module_name.startswith(self.prefix + "."):
                    module_name = module_name[len(self.prefix) + 1 :]
                module = self._resolve_module(module_name)
                param = getattr(module.experts, expert_name)
                local_ids = module.local_expert_ids
                param.data.copy_(
                    loaded_weight[local_ids].to(
                        device=param.device, dtype=param.dtype
                    )
                )
                loaded_params.add(name)
                continue

            # TP vocab-parallel tables (embed / lm_head) are sharded; route
            # through the module's weight_loader which slices by rank.
            if name.endswith("embed_tokens.weight"):
                from sglang.srt.distributed import (
                    get_tensor_model_parallel_rank,
                    get_tensor_model_parallel_world_size,
                )

                param = self.embed_tokens.weight
                tp_rank = get_tensor_model_parallel_rank()
                tp_size = get_tensor_model_parallel_world_size()
                per = loaded_weight.shape[0] // tp_size
                sl = loaded_weight[tp_rank * per : (tp_rank + 1) * per]
                import torch as _t

                if int(os.environ.get("SGLANG_QWEN4_DEBUG", "0")):
                    print(
                        f"[load-embed] rank={tp_rank} param={tuple(param.shape)} "
                        f"slice={tuple(sl.shape)} sl_finite={_t.isfinite(sl).all().item()} "
                        f"params_has_embed={any('embed_tokens' in k for k in params_dict)}",
                        flush=True,
                    )
                param.data.copy_(sl)
                loaded_params.add(name)
                continue

            if name not in params_dict:
                skipped.append(name)
                if int(os.environ.get("SGLANG_QWEN4_DEBUG", "0")):
                    logger.warning(f"Parameter {name} not found in params_dict")
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        if int(os.environ.get("SGLANG_QWEN4_DEBUG", "0")) and skipped:
            print(
                f"[load] SKIPPED {len(skipped)} params (first 20): {skipped[:20]}",
                flush=True,
            )
        if int(os.environ.get("SGLANG_QWEN4_DEBUG", "0")):
            import torch as _t

            total_nan = sum(
                _t.isnan(p).sum().item() for p in self.parameters() if p.numel()
            )
            total_inf = sum(
                _t.isinf(p).sum().item() for p in self.parameters() if p.numel()
            )
            print(f"[load-done] total_nan={total_nan} total_inf={total_inf}", flush=True)
        if int(os.environ.get("SGLANG_QWEN4_GRAPH_CANARY", "0")):
            for layer in self.layers:
                ple = getattr(layer, "ple", None)
                if ple is not None:
                    embedding = ple.ple_embedding
                    embedding.graph_decode_embeddings = (
                        embedding.graph_decode_embeddings.to(
                            device=ple.key_proj.weight.device,
                            dtype=ple.key_proj.weight.dtype,
                        )
                    )
        return loaded_params

    def _resolve_module(self, name: str) -> nn.Module:
        module: nn.Module = self
        for part in name.rstrip(".").split("."):
            module = getattr(module, part)
        return module


class Qwen4ExpForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """VLM wrapper for the Qwen4-Exp checkpoint (arch: Qwen4ExpForConditionalGeneration).

    M1 is text-only: the vision tower is constructed (so the checkpoint's
    vision_config validates) but its weights are skipped during loading, and
    ``forward`` raises when image inputs are present.
    """

    hf_to_sglang_mapper = None

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen4ExpForCausalLM,
    ):
        Qwen3VLForConditionalGeneration.__init__(
            self, config, quant_config, prefix, language_model_cls
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
        **kwargs,
    ) -> torch.Tensor:
        del pp_proxy_tensors
        if forward_batch.contains_image_inputs():
            raise NotImplementedError(
                "Qwen4Exp vision inputs are out of M1 scope (text-only)"
            )
        input_embeds = kwargs.get("input_embeds", None)
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )
        if get_embedding:
            return hidden_states
        if self.pp_group.is_last_rank:
            lp_out = self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, []
            )
            _dump_logits = os.environ.get("SGLANG_QWEN4_DUMP_LOGITS", "")
            if _dump_logits and input_ids.shape[0] >= 3 and (
                input_ids[:3].tolist() == [96328, 102357, 96878]
            ):
                try:
                    from sglang.srt.distributed import get_tensor_model_parallel_rank

                    if get_tensor_model_parallel_rank() == 0:
                        lt = getattr(lp_out, "next_token_logits", None)
                        if lt is not None:
                            torch.save(lt.float().cpu(), _dump_logits)
                            print(
                                f"[e2e-logits] saved {tuple(lt.shape)} -> {_dump_logits}",
                                flush=True,
                            )
                except Exception as _e:
                    print(f"[e2e-logits] failed: {_e}", flush=True)
            if int(os.environ.get("SGLANG_QWEN4_DEBUG", "0")):
                import torch as _t

                lt = getattr(lp_out, "next_token_logits", None)
                if lt is not None:
                    top5 = _t.topk(lt.float(), 5, dim=-1)
                    # rank of the expected first token 98116 ("北")
                    exp = 98116
                    rank_exp = (
                        int((lt.float() > lt[0, exp]).sum().item()) if lt.shape[0] else -1
                    )
                    print(
                        f"[wlogits] top5_ids={top5.indices.tolist()} "
                        f"top5_val={[[round(float(x), 2) for x in row] for row in top5.values.tolist()]} "
                        f"rank_98116={rank_exp} logit_98116={round(float(lt[0, exp]), 2) if lt.shape[0] else None}",
                        flush=True,
                    )
            # Long Ascend prefills leave several GiB of inactive blocks in the
            # caching allocator.  Release them once per completed model forward,
            # after logits have been materialized and all layer temporaries are
            # dead.  Do not touch the latency-sensitive decode path.
            if input_ids.shape[0] > 2048 and hasattr(torch, "npu"):
                torch.npu.synchronize()
                torch.npu.empty_cache()
            return lp_out
        return hidden_states

    def prepare_qwen4_graph_inputs(self, forward_batch: ForwardBatch) -> None:
        self.model.prepare_qwen4_graph_inputs(forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """M1 text-only loader: skip vision/MTP, delegate LM to the backbone."""
        model_weights = []
        head_weight = None
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name or "mtp" in name:
                continue
            if name.startswith(("model.visual", "visual")):
                continue
            if name == "lm_head.weight":
                head_weight = loaded_weight
                continue
            model_weights.append((name, loaded_weight))
        self.model.load_weights(model_weights)
        if head_weight is not None and self.pp_group.is_last_rank:
            head = getattr(self, "lm_head", None)
            if head is not None and hasattr(head, "weight"):
                # ParallelLMHead is TP-sharded. slice by the MODEL TP rank
                # (matching embed's VocabParallelEmbedding slice) so the logits
                # all-gather order lines up with the loaded rows.
                from sglang.srt.distributed import (
                    get_tensor_model_parallel_rank,
                    get_tensor_model_parallel_world_size,
                )

                tp_rank = get_tensor_model_parallel_rank()
                tp_size = get_tensor_model_parallel_world_size()
                per = head_weight.shape[0] // tp_size
                sl = head_weight[tp_rank * per : (tp_rank + 1) * per]
                head.weight.data.copy_(
                    sl.to(device=head.weight.device, dtype=head.weight.dtype)
                )
                if int(os.environ.get("SGLANG_QWEN4_DEBUG", "0")):
                    print(
                        f"[load-lmhead] rank={tp_rank} shape={tuple(head.weight.shape)} "
                        f"row0_first4={[round(float(v), 4) for v in head.weight[0, :4].tolist()]}",
                        flush=True,
                    )
        return set()


# Registry: the checkpoint's architectures=[Qwen4ExpForConditionalGeneration]
# resolves here; the standalone text backbone is also exposed.
EntryClass = [Qwen4ExpForCausalLM, Qwen4ExpForConditionalGeneration]


__all__ = ["Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"]
