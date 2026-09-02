# GLM-5 Next Ascend operator adaptation RP

## Summary

This proposal tracks the operator work required to run GLM-5 Next with
SGLang-FL on Ascend NPU. The target checkpoint used for the static analysis is
`/models/GLM-5-Next-0808/hf/`.

The model is not a conventional decoder-only MoE model. Its text network has
45 decoder layers:

- 34 KDA linear-attention layers;
- 11 DSA/MLA sparse-attention layers;
- mHC residual mixing with `hc_mult=4`;
- 288 routed experts, top-8 routing and one shared expert;
- one NextN/MTP layer;
- a GLM-OCR vision encoder.

The proposal has two parts:

1. ATen operators currently present in the SGLang-FL Ascend blacklist used by
   the target container;
2. model-specific fused operators needed by the KDA, DSA, mHC and MoE paths.

The latency column is deliberately left as `TBD (profile)`. No model service or
NPU profiler was started during this static analysis. The values must be filled
from a fixed end-to-end workload and must be reported separately for prefill and
decode. Fused operators and their constituent ATen operators overlap, so row
percentages must not be added directly.

## Environment and source baseline

| Component | Baseline |
|---|---|
| SGLang | 0.5.11, container checkout `600df8dd498955f5aef4417383ea9ae5d2c238f7` |
| SGLang-FL | Ascend dispatch configuration from the target container |
| PyTorch | 2.8.0+cpu package paired with torch_npu 2.8.0.post2 |
| CANN | 8.5.0 |
| FlagGems | v5.3.0, `98fae44cdf2898f39c7f24f080d7c88b83d7c593` |
| Model hidden size | 4096 |
| KDA | 64 heads, head dim 128, short convolution kernel 4, gate lower bound -5.0 |
| DSA | 32 index heads, index head dim 128, token top-k 2048, kpool 4 |
| mHC | hc_mult 4, Sinkhorn iterations 20, epsilon 1e-6 |
| MoE | 288 routed experts, top-8, expert intermediate size 2048 |

`T` below means the number of tokens in the current forward pass, `B` means
batch size and `H` means attention heads. Head counts and projection dimensions
are sharded by TP/CP where applicable.

## P0 fused-operator requests

| Operator name | Kernel name | Operator interface | Interface source | E2E latency (%) | Input shapes and attributes | FlagGems operator | Request |
|---|---|---|---|---:|---|---|---|
| `mhc_pre` | `_mhc_pre_fused_kernel_hc_mult_4_impl` | `mhc_pre` | GLM-5 Next `hc_attn_pre` / `hc_ffn_pre` | TBD (profile) | residual `[T,4,4096]`; fn `[24,16384]`; base `[24]`; scale `[3]`; outputs `[T,4096]`, `[T,4,4]`, `[T,4]`; FP32 parameters and BF16 activations | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/fused/mhc/mhc_pre.py) | Validate Ascend correctness and performance for decode and long prefill. |
| `mhc_post` | `mhc_post_kernel_hc_mult_4` | `mhc_post` | GLM-5 Next decoder mHC post stage | TBD (profile) | x `[T,4096]`; residual `[T,4,4096]`; post mix `[T,4]`; combination mix `[T,4,4]` | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/fused/mhc/mhc_post.py) | Called after attention and FFN in every layer. |
| `mhc_split_sinkhorn` | `mhc_split_sinkhorn_kernel_hcmult_4` | `hc_split_sinkhorn` | GLM-5 Next mHC mixing | TBD (profile) | mixes `[T,24]`; scale `[3]`; base `[24]`; hc_mult=4; iterations=20; eps=1e-6 | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/fused/mhc/hc_split_sinkhorn.py) | Use FP32 accumulation and verify numerical stability. |
| `kda_recurrent_decode` | `fused_recurrent_gated_delta_rule_fwd_kernel` | `fused_recurrent_gated_delta_rule_fwd` with a GLM5 KDA adapter | SGLang GLM5 KDA recurrent path | TBD (profile) | q/k/v `[B,1,64/TP,128]`; state `[B,64/TP,128,128]`; BF16 input, FP32 state/accumulation recommended | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/runtime/backend/_ascend/fla/fused_recurrent.py) | Confirm GLM5 KDA state and lower-bound semantics; existing kernel is gated-delta-rule. |
| `kda_sigmoid_gate_update` | `fused_sigmoid_gating_delta_rule_update_kernel` | `fused_sigmoid_gating_delta_rule_update` | SGLang GLM5 KDA decode path | TBD (profile) | q/k/v `[B,T,H,128]`; beta `[B,T,H]`; gate `[B,T,H,128]`; state `[B,H,128,128]` | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/runtime/backend/_ascend/fla/sigmoid_gating.py) | Add/verify gate lower bound -5.0 and packed decode. |
| `chunk_kda_prefill` | `chunk_kda_fwd` kernel family | `chunk_kda` / `chunk_kda_fwd` | SGLang GLM5 KDA prefill path | TBD (profile) | q/k/v `[B,T,64/TP,128]`; dynamic/long T; state `[B,64/TP,128,128]`; typical chunk 64 | No | Missing in the inspected FlagGems baseline. This is the primary 34/45-layer prefill gap. |
| `fused_rmsnorm_gate` | fused RMSNorm + sigmoid gate kernel | `FusedRMSNormGated` | GLM5 linear-attention output normalization | TBD (profile) | output `[1,T,H,128]`; gate `[T,H,128]`; weight `[128]`; eps=1e-5; activation sigmoid | No | Fuse RMSNorm, sigmoid and multiply. |
| `dsa_bucket_topk` | `kernel_bucket_sort_topk` / TLE bucket sort | `bucket_sort_topk` | GLM5 DSA indexer | TBD (profile) | pooled score `[N,L/4]` FP32; starts/ends `[N]`; group top-k 512; expanded token top-k 2048 | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/fused/DSA/bin_topk.py) | Verify pool-granularity selection and tail-token semantics. |
| `dsa_kpool_topk_transform` | `kpool_topk_transform_kernel` | `fast_kpool_topk_transform_fused` | GLM5 DSA kpool indexer | TBD (profile) | scores `[N,L/4]`; group top-k 512; pool size 4; page table `[B,max_pages]`; output `[N,2048(+tail)]` | No | Fuse pool selection, token expansion and page-table/ragged-index transformation. |
| `dsa_sparse_mla` | `triton_sparse_mla_fwd` | `triton_sparse_mla_fwd_interface` | DeepSeek/GLM5 sparse MLA backend | TBD (profile) | q `[B,SQ,H,DT]`; kv `[B,SK,VG,DT]`; indices `[B,SQ,VG,2048(+tail)]` | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/fused/DSA/sparse_mla.py) | Integrate with Ascend cache layout and kpool indices for prefill/decode. |
| `fused_moe` | Ascend `fused_moe_kernel` | `fused_experts_impl` | [SGLang fused MoE](https://github.com/sgl-project/sglang/blob/master/python/sglang/srt/layers/moe/fused_moe_triton/layer.py) | TBD (profile) | a `[T,4096]`; w1 `[288,4096,4096]` logically gate/up for expert I=2048; w2 `[288,4096,2048]`; ids/weights `[T,8]`; one shared expert; BF16/FP8 | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/runtime/backend/_ascend/fused/fused_moe.py) | Verify 288 experts, top-8 sigmoid/noaux_tc routing, shared expert and FP8 scales. |
| `silu_and_mul_with_clamp` | `silu_and_mul_with_clamp_kernel` | `silu_and_mul_with_clamp` | GLM5 capped SwiGLU (`swiglu_limit=10.0`) | TBD (profile) | gate/up `[...,I]`; dense I=12288; routed-expert I=2048; vision projection I=10240; BF16/FP8 | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/fused/silu_and_mul_with_clamp.py) | Cover large rows, multi-block launch and quantized post-processing. |
| `flash_mla_sparse` | `triton_flash_mla_sparse_fwd` | `flash_mla_sparse_fwd` | SGLang DSA/MLA attention backend | TBD (profile) | decode q `[B,1,H,latent_dim]`; paged KV cache; top-k `[B,2048(+tail)]`; BF16/FP8 cache | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/fused/flashmla_sparse.py) | Ensure sparse implementation is selected instead of dense flash MLA. |
| `fused_add_rms_norm` | fused add/RMSNorm kernel | `fused_add_rms_norm` | SGLang residual/norm path | TBD (profile) | x/residual `[T,4096]`; weight `[4096]`; eps=1e-5; BF16 | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/runtime/backend/_ascend/fused/fused_add_rms_norm.py) | P1 for GLM5 text because mHC cannot be replaced by ordinary add+norm; still useful for vision/fallback paths. |

## Ascend ATen blacklist requests

The target container has the following 21 FlagGems exclusions. “FlagGems
operator” means an implementation exists at the pinned FlagGems commit; it does
not mean that the implementation is currently correct or performant on the
target Ascend workload.

| Operator name | Kernel name | Operator interface | Interface source | E2E latency (%) | Input shapes and attributes | FlagGems operator | Priority |
|---|---|---|---|---:|---|---|---|
| `stack` | `stack_copy_func_kernel_4` / Ascend stack | `aten::stack` / `torch.stack` | [PyTorch](https://pytorch.org/docs/2.8/generated/torch.stack.html) | TBD (profile) | Tensor list; typical item `[B]`, `[B+1]` or `[T,4096]`; variable dim; BF16/INT32 | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/runtime/backend/_ascend/ops/stack.py) | P1 |
| `sub` | `sub_func` / tensor-scalar variant | `aten::sub.Tensor`, `aten::sub.Scalar` | [PyTorch](https://pytorch.org/docs/2.8/generated/torch.sub.html) | TBD (profile) | `[T,4096]`, `[T,64,128]`, `[B]`; broadcasting; BF16/FP32/INT32 | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/ops/sub.py) | P1 |
| `lt_scalar` | `lt_func_scalar` | `aten::lt.Scalar` | [PyTorch](https://pytorch.org/docs/2.8/generated/torch.lt.html) | TBD (profile) | mask `[B]`, `[T]` or `[T,32,L]`; numeric to BOOL | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/ops/lt.py) | P1 |
| `clamp` | clamp min/max kernels | `aten::clamp` | [PyTorch](https://pytorch.org/docs/2.8/generated/torch.clamp.html) | TBD (profile) | KDA gate `[T,64,128]`, lower bound -5.0; capped SwiGLU, limit 10.0 | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/ops/clamp.py) | P0 |
| `add` | `add_func` / tensor-scalar variant | `aten::add.Tensor`, `aten::add.Scalar` | [PyTorch](https://pytorch.org/docs/2.8/generated/torch.add.html) | TBD (profile) | `[T,4096]`, `[T,4,4096]`, `[T,64,128]`; broadcasting; BF16/FP32 | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/ops/add.py) | P0 |
| `lt` | `lt_func` | `aten::lt.Tensor` | [PyTorch](https://pytorch.org/docs/2.8/generated/torch.lt.html) | TBD (profile) | DSA mask and metadata `[B]`, `[T]`, `[T,32,L]` | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/ops/lt.py) | P1 |
| `mul` | `mul_func` / scalar variant | `aten::mul.Tensor`, `aten::mul.Scalar` | [PyTorch](https://pytorch.org/docs/2.8/generated/torch.mul.html) | TBD (profile) | `[T,4096]`, `[T,4,4096]`, `[T,64,128]`; BF16/FP32 | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/ops/mul.py) | P0 |
| `full_like` | Ascend `full_like` | `aten::full_like` | [PyTorch](https://pytorch.org/docs/2.8/generated/torch.full_like.html) | TBD (profile) | graph/cache metadata `[B]`, `[B+1]`, `[T]`, `[T,2048]`; INT32/BOOL/FP32 | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/runtime/backend/_ascend/ops/full_like.py) | P1 |
| `index_copy_` | generated index-copy kernel | `aten::index_copy_` | [PyTorch](https://pytorch.org/docs/2.8/generated/torch.Tensor.index_copy_.html) | TBD (profile) | cache dst `[num_slots,...]`; index `[T]`; src `[T,...]`; dim=0 | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/ops/index_copy_.py) | P0 |
| `gather` | Ascend fixed flat gather / generated gather | `aten::gather` | [PyTorch](https://pytorch.org/docs/2.8/generated/torch.gather.html) | TBD (profile) | score/cache tensors; indices `[T,2048]` or `[T]`; dim 0/1 | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/runtime/backend/_ascend/ops/gather.py) | P0 |
| `count_nonzero` | reduction and combine kernels | `aten::count_nonzero` | [PyTorch](https://pytorch.org/docs/2.8/generated/torch.count_nonzero.html) | TBD (profile) | graph-padding mask `[B]`; BOOL/INT32; small decode shapes | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/runtime/backend/_ascend/ops/count_nonzero.py) | P1 |
| `index_put_` | generated index-put kernel | `aten::index_put_` | [PyTorch](https://pytorch.org/docs/2.8/generated/torch.Tensor.index_put_.html) | TBD (profile) | state/cache dst; Tensor[] indices; values `[T,...]`; accumulate false | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/ops/index_put.py) | P0 |
| `_index_put_impl_` | generated index-put kernel | `aten::_index_put_impl_` | [PyTorch schema](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/native_functions.yaml) | TBD (profile) | same as index_put_; includes unsafe and accumulate flags | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/ops/index_put.py) | P0 |
| `cumsum` | scan-part and base-sum kernels | `aten::cumsum` | [PyTorch](https://pytorch.org/docs/2.8/generated/torch.cumsum.html) | TBD (profile) | sequence lengths `[B]`; KDA/DSA metadata `[B,T,H]` or flattened `[N]` | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/runtime/backend/_ascend/ops/cumsum.py) | P0 |
| `fill_scalar_` | `fill_scalar_func` | `aten::fill_.Scalar` | [PyTorch](https://pytorch.org/docs/2.8/generated/torch.Tensor.fill_.html) | TBD (profile) | graph/cache buffers `[B]`, `[B+1]`, `[T]` or cache slices | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/ops/fill.py) | P1 |
| `argmax` | Ascend argmax kernel | `aten::argmax` | [PyTorch](https://pytorch.org/docs/2.8/generated/torch.argmax.html) | TBD (profile) | greedy logits `[B,154880]` or router/index scores; BF16/FP32 | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/runtime/backend/_ascend/ops/argmax.py) | P1 |
| `index` | `index_kernel_func` / wrapper | `aten::index.Tensor` | [PyTorch schema](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/native_functions.yaml) | TBD (profile) | cache/state advanced indexing; DSA indices `[T,2048]`; INT32/INT64 | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/runtime/backend/_ascend/ops/index.py) | P0 |
| `add_` | add in-place kernels | `aten::add_.Tensor`, `aten::add_.Scalar` | [PyTorch](https://pytorch.org/docs/2.8/generated/torch.Tensor.add_.html) | TBD (profile) | residual/cache `[T,4096]`, `[T,4,4096]`, `[T,64,128]`; alias-sensitive | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/ops/add.py) | P0 |
| `conv1d` | generic matmul-based conv1d | `aten::conv1d` | [PyTorch](https://pytorch.org/docs/2.8/generated/torch.conv1d.html) | TBD (profile) | KDA depthwise causal conv; qkv channels 24576 before TP; kernel 4; prefill `[B,C,T]`; decode state `[B,C,3]` | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/ops/conv1d.py) | P0 |
| `floor_divide` | division kernel, floor mode | `aten::floor_divide` | [PyTorch](https://pytorch.org/docs/2.8/generated/torch.floor_divide.html) | TBD (profile) | DSA kpool/page metadata `[B]`, `[T]`; divisors 4 and 64 | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/ops/div.py) | P1 |
| `true_divide_` | division in-place kernel | `aten::div_.Tensor`, `aten::div_.Scalar` | [PyTorch](https://pytorch.org/docs/2.8/generated/torch.Tensor.div_.html) | TBD (profile) | normalization/scale `[T,4,4]`, `[T,64,128]` or scalar broadcast; FP32/BF16 | [Yes](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/ops/div.py) | P1 |

## Adaptation blockers outside the operator list

The operator work alone is not sufficient to boot the model. The following
framework integration is also required:

1. install the matching Transformers GLM5 Next configuration/model patch;
2. port GLM5 config, registry, model and weight-loader code to the SGLang 0.5.11
   API used by SGLang-FL;
3. route CUDA-only imports and `torch.cuda` graph/stream code through the
   SGLang-FL platform layer;
4. integrate mHC communicators and KDA state-cache allocation;
5. add DSA kpool tail/compressed-cache metadata and page-table transformations;
6. validate BF16 first, then FP8, vision, NextN/MTP and distributed modes.

## Proposed delivery order

1. Transformers config and SGLang model construction without weight loading;
2. GLM5 weight mapping and CPU/static shape validation;
3. mHC correctness kernels;
4. KDA decode and chunked-prefill kernels;
5. existing DSA sparse MLA without kpool compression;
6. kpool top-k, tail handling and compressed cache;
7. 288-expert top-8 MoE and capped SwiGLU;
8. single-NPU text prefill/decode;
9. vision, FP8, graph, TP/PP/DP/CP and NextN/MTP.

## Profiling and acceptance criteria

For latency attribution, collect at least the following workloads with identical
software and launch parameters:

- prefill: batch 1 and representative production batch, input lengths 128,
  1K and 8K;
- decode: batch 1 and representative production batch, one token per step;
- record each operator's total device time, invocation count, top shapes and
  percentage of the matching end-to-end phase;
- report warm and cold/JIT runs separately.

An operator can be removed from the blacklist only when:

- numerical output matches torch_npu/reference within an agreed tolerance;
- supported dtypes, overloads, broadcasting and in-place alias semantics match
  PyTorch for the observed GLM5 shapes;
- prefill and decode performance do not regress against the current fallback;
- no AICore exception, out-of-bounds access or unstable autotuning occurs;
- a focused regression test is added to SGLang-FL or FlagGems.
