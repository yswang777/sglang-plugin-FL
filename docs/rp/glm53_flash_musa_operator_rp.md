# GLM-5.3-Flash MUSA operator adaptation RP

## Summary

This proposal records the operator work observed while adapting
`GLM-5.3-Flash` to SGLang-FL on eight Moore Threads MTT S5000 devices. It is
based on a running TP8 service and an eight-rank MUSA profiler trace rather
than static model inspection.

The target text model contains:

- 45 decoder layers: 34 KDA linear-attention layers and 11 DSA/MLA layers;
- hidden size 4096 and mHC residual mixing with `hc_mult=4`;
- 288 routed experts, top-8 routing and one shared expert;
- KDA with 64 heads and head dimension 128 before TP sharding;
- DSA index top-k 2048, index pool size 4 and latent KV dimension 512;
- FP8 expert weights with BF16 activations.

The request covers two groups:

1. ATen operators that currently have to stay in
   `SGLANG_FL_FLAGOS_BLACKLIST` on MUSA;
2. model-specific fused kernels on the KDA, mHC, MoE and paged MLA paths.

The "FlagGems operator" column means that an implementation exists at the
pinned FlagGems revision. It does not mean that the implementation is selected
or is correct for the MUSA shapes in this report. The active trace still uses
SGLang or torch-MUSA fallback kernels for the blacklisted/fused paths shown
below.

## Adaptation inventory

The validated container adaptation is intentionally split into framework and
operator work:

1. the GLM-5.3 Transformers config/model implementation and SGLang model
   registration, weight mapping and FP8 MoE loading were aligned first;
2. MUSA-specific DSA metadata, zero-RoPE handling and KDA state-cache handling
   were added to the SGLang overlay;
3. GLM-5.3 decode was changed from reconstructing and projecting all historical
   KV tokens to a paged/absorbed MLA kernel that directly reads the 512-wide
   latent cache through the live page table;
4. mHC pre/post operations use dedicated Triton-MUSA kernels for the observed
   skinny shapes;
5. KDA decode uses a fused sigmoid-gating delta-rule update kernel, while KDA
   prefill uses chunked kernels;
6. the service uses full decode graph at maximum batch size 16. No EAGLE, MTP
   or other speculative decoding is enabled.

The SGLang model and cache-layout changes are maintained in the SGLang overlay;
this RP does not copy that source into SGLang-FL. It identifies the operator
interfaces that should be implemented or stabilized in FlagGems/SGLang-FL so
the overlay can be reduced over time.

## Environment and profile methodology

| Component | Baseline |
|---|---|
| Host | `worker32081`, 8 x MTT S5000 |
| Container | `glm-5-next` |
| Model | `/data/models/GLM-5.3-Flash` |
| SGLang | `0.0.0.dev0+d20260826`, source baseline `f2ef826f0caefc40e33aa676124bba80b9092887` |
| SGLang-FL | `0.1.0` |
| FlagGems | `5.3.0rc2`, revision `98fae44cdf2898f39c7f24f080d7c88b83d7c593` |
| PyTorch / torch-MUSA | `2.9.0` / `2.9.0` |
| Triton | `3.1.0` |
| Runtime | TP8, batch 16, full decode graph, paged MLA |

The source trace is
`glm53_paged_mla_bs16-1787821741.8142612-TP-{0..7}.trace.json.gz` and contains
20 profiled steps with shape recording. The latency value in the tables is the
median across all eight TP ranks of cumulative GPU-kernel time divided by
effective GPU-kernel time.

Two profiler-only artifacts were excluded from the denominator:

- `zeros_kernel` clearing a 67,108,864-element profiler buffer;
- `repeat_kernel_2d` emitted by profiler bookkeeping.

They reduced the same batch-16/64-output workload from 137.83 output tok/s
outside the profiler to 54.31 tok/s under the profiler. The reported values are
therefore suitable for operator prioritization, but are not strict wall-clock
critical-path percentages. Percentages for fused operators and their
constituent ATen operations overlap and must not be summed.

## Blacklisted ATen operators

The required runtime blacklist is:

```text
count_nonzero,cumsum,mm,unique,_unique2,unique_dim,unique_consecutive,index_put,index_put_,_index_put_impl_
```

| Operator name | Kernel name | Operator interface | Interface source | Model E2E latency share (%) | Shapes and other inputs | FlagGems operator |
|---|---|---|---|---:|---|---|
| `mm` | `musa_asm_sgemm_nt_128_128x64_epilogue_robust64`; BF16 TCE GEMM; small-N GEMM/GEMV | `aten::mm` / `torch.mm` | [PyTorch](https://pytorch.org/docs/stable/generated/torch.mm.html) | 16.49 | 674 calls/rank. BF16: `[31/465,4096] x [4096,128/288/3072/8]`, `[31/465,128] x [128,1024]`, `[31/465,1024] x [1024,4096]`, `[31/465,512] x [512,4096]`. FP32 hot shape: `[465,16384] x [16384,24]` (90 calls). | [Yes: MThreads mm](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/runtime/backend/_mthreads/ops/mm.py#L433) |
| `index_put_` / `_index_put_impl_` | `GPUIndexKernel`; `IndexVectorizedKernel`; `IndexElementwiseKernel` | `aten::index_put_`; `aten::_index_put_impl_` | [PyTorch index_put_](https://pytorch.org/docs/stable/generated/torch.Tensor.index_put_.html) | 0.10 | 84 calls/rank. Main cases: destination `[535]`, index/source `[16]`; destination `[535,1048580]`, index `[16]`; KV destination `[863104,1,512]`, source `[31/465,1,512]`. BF16, FP32, INT32 and INT64. | [Yes: MThreads index_put](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/runtime/backend/_mthreads/ops/index_put.py#L277) |
| `cumsum` | `musa::dnn::ScanPartSumShflKernel` | `aten::cumsum` / `torch.cumsum` | [PyTorch](https://pytorch.org/docs/stable/generated/torch.cumsum.html) | 0.004 | 8 calls/rank: `[1]` x3, `[15]` x3, `[2]` x1 and `[16]` x1; dim 0; BOOL, INT32 and INT64. | [Yes: FlagGems cumsum](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/ops/cumsum.py#L368) |
| `count_nonzero` | Not hit in this profile window | `aten::count_nonzero` / `torch.count_nonzero` | [PyTorch](https://pytorch.org/docs/stable/generated/torch.count_nonzero.html) | 0.00 | Compatibility and graph-capture blacklist item. A focused input is still required to capture its failing MUSA shape. | [Yes: FlagGems count_nonzero](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/ops/count_nonzero.py#L76) |
| `unique` / `_unique2` | Not hit in this profile window | `aten::unique`; `aten::_unique2`; `torch.unique` | [PyTorch](https://pytorch.org/docs/stable/generated/torch.unique.html) | 0.00 | Compatibility and graph-capture blacklist item. | [Yes: MThreads _unique2](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/runtime/backend/_mthreads/ops/unique.py#L769) |
| `unique_dim` | Not hit in this profile window | `aten::unique_dim` | [PyTorch unique](https://pytorch.org/docs/stable/generated/torch.unique.html) | 0.00 | No standalone MThreads `unique_dim` implementation was found at the pinned FlagGems revision. |  |
| `unique_consecutive` | Not hit in this profile window | `aten::unique_consecutive` / `torch.unique_consecutive` | [PyTorch](https://pytorch.org/docs/stable/generated/torch.unique_consecutive.html) | 0.00 | Compatibility and graph-capture blacklist item. | [Yes: FlagGems unique_consecutive](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/ops/unique_consecutive.py#L452) |
| `index_put` (out of place) | Not hit in this profile window | `aten::index_put` / `Tensor.index_put` | [PyTorch](https://pytorch.org/docs/stable/generated/torch.Tensor.index_put.html) | 0.00 | Only the in-place `index_put_/_index_put_impl_` path was observed. | [Yes: MThreads index_put](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/runtime/backend/_mthreads/ops/index_put.py#L277) |

## Fused and model-specific operators

| Operator name | Kernel name | Operator interface | Interface source | Model E2E latency share (%) | Shapes and other inputs | FlagGems operator |
|---|---|---|---|---:|---|---|
| KDA decode fused update | `fused_sigmoid_gating_delta_rule_update_kernel` | `fused_sigmoid_gating_delta_rule_update` | [SGLang](https://github.com/sgl-project/sglang/blob/f2ef826f0caefc40e33aa676124bba80b9092887/python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py#L330) | 8.57 | Decode q/k/v/a approximately `[16,1,8,128]`; state `[num_slots,8,128,128]`; grid `[1,4,128]`; BF16. Shapes are reconstructed from model config and kernel launch. |  |
| mHC pre fusion | `_mhc_skinny_mm_splitk_kernel`; `_mhc_skinny_mm_reduce_kernel`; `_mhc_pre_finalize_triton_kernel` | `_mhc_pre_dispatch` / `mhc_pre` | [SGLang mHC interface](https://github.com/sgl-project/sglang/blob/f2ef826f0caefc40e33aa676124bba80b9092887/python/sglang/kernels/ops/layernorm/mhc.py#L782) | 5.55 | residual `[T,4,4096]` BF16; mix `[T,24]` FP32; post `[T,4]`; combination `[T,4,4]`; output `[T,4096]`; observed `T=31/465`. | [Yes: FlagGems mhc_pre](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/fused/mhc/mhc_pre.py) |
| fused MoE | `fused_moe_kernel` | `sglang::inplace_fused_experts` / `fused_experts_impl` | [SGLang](https://github.com/sgl-project/sglang/blob/f2ef826f0caefc40e33aa676124bba80b9092887/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py#L931) | 3.91 | hidden `[31/465,4096]` BF16; w1 `[288,512,4096]` FP8; w2 `[288,4096,256]` FP8; top-k weight/id `[31/465,8]`; scales `[288,4,32]` and `[288,32,2]`; 84 calls/rank. | [Yes: FlagGems fused_moe](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/fused/fused_moe.py) |
| KDA prefill chunk fusion | `chunk_kda_fwd_kernel_inter_solve_fused` | `chunk_kda_fwd_intra` / `chunk_kda` | [SGLang](https://github.com/sgl-project/sglang/blob/f2ef826f0caefc40e33aa676124bba80b9092887/python/sglang/kernels/ops/attention/fla/chunk_intra.py#L47) | 1.68 | Packed q/k/v primary dimensions `[T,8,128]`; main observed token count `T=465`; cu_seqlens for batch 16; BF16; 68 calls/rank. |  |
| per-token group FP8 quantization | `per_token_group_quant_8bit_kernel<...,128,8,...>`; `_per_token_group_quant_8bit` | `sgl_kernel::sgl_per_token_group_quant_8bit_v2` | [SGLang](https://github.com/sgl-project/sglang/blob/f2ef826f0caefc40e33aa676124bba80b9092887/python/sglang/kernels/ops/quantization/per_token_group_quant_8bit_v2.py#L108) | 0.95 | BF16 to FP8: `[31/465,4096]`, scale `[31/465,32]`; `[31/465,256]`, scale `[31/465,2]`; `[31/465,1536]`, scale `[31/465,12]`; `[31/465,2048]`, scale `[31/465,16]`; group size 128. | [Yes: FlagGems per-token group FP8](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/ops/per_token_group_quant_fp8.py) |
| MUSA paged/absorbed MLA decode | `_paged_mla_decode_kernel` | `musa_paged_mla_decode` | Container overlay: `python/sglang/srt/hardware_backend/musa/attention/paged_mla.py` | 0.90 | q `[16,8,512]`; latent cache `[13486,64,1,512]` BF16 (863104 token slots); page table `[16,max_pages]`; cache lengths `[16]`; grid `[128,1,1]`. |  |
| mHC post fusion | `_mhc_post_triton_kernel` | `_mhc_post_dispatch` / `mhc_post` | [SGLang mHC interface](https://github.com/sgl-project/sglang/blob/f2ef826f0caefc40e33aa676124bba80b9092887/python/sglang/kernels/ops/layernorm/mhc.py#L1039) | 0.80 | x `[T,4096]`; residual `[T,4,4096]`; post `[T,4]`; combination `[T,4,4]`; output `[T,4,4096]`; observed `T=31/465`; FP32 accumulation. | [Yes: FlagGems mhc_post](https://github.com/flagos-ai/FlagGems/blob/98fae44cdf2898f39c7f24f080d7c88b83d7c593/src/flag_gems/fused/mhc/mhc_post.py) |

## Framework and communication observation

`mcclKernel_AllReduce_RING_SIMPLE_Sum___mt_bfloat16` accounts for a median
48.00% of effective recorded GPU-kernel time across the eight ranks (range
24.70%-50.27%, 182 calls/rank). This is not submitted as a FlagGems operator
request because profiler synchronization and rank wait time can inflate the
number. It should be followed up with an unprofiled communication timeline and
experiments on computation/communication overlap, bucket size, stream
dependencies and rank imbalance.

## Requested priority

1. **P0 - MThreads FlagGems `mm`:** cover the FP32
   `[465,16384] x [16384,24]` case and the recorded BF16 skinny/tall shapes.
   This is the only blacklisted ATen operator with a double-digit share.
2. **P0 - KDA decode fused update:** optimize batch 16 with TP-sharded
   `[16,1,8,128]` tensors and reduce recurrent-state traffic and launch cost.
3. **P1 - mHC pre:** fuse or pipeline skinny-mm, reduction and finalization,
   while preserving FP32 Sinkhorn/reduction numerics.
4. **P1 - fused MoE plus quantization:** use the actual 288-expert, top-8,
   FP8-block-scale shapes instead of square synthetic GEMMs.
5. **P2 - paged MLA, mHC post and index_put:** focus on graph-capture and cache
   correctness as well as throughput.
6. **P3 - cumsum/count_nonzero/unique:** prioritize correctness, dynamic-shape
   handling and graph capture. Their measured performance contribution is
   negligible or absent in this workload.

## Current validation status

- The paged MLA kernel standalone test at batch 16, 8 heads, latent dimension
  512 and sequence length 4096 reached max absolute error `0.000488281` and
  passed `allclose` against the reference.
- Full-graph batch-16/256-output trials measured 183.24 and 185.09 output tok/s.
- A four-concurrent GPQA run at 4096 maximum tokens completed without request
  errors: two cases produced the correct option and stopped naturally; two hit
  the token limit before an extractable final option.
- A previous 16-concurrent long-output run was not stable and ended with 15
  disconnected requests. This proposal is not claiming final model-accuracy or
  long-run stability acceptance.
- No speculative, EAGLE or MTP path was used for these measurements.

## Acceptance criteria

An operator should be removed from the MUSA blacklist or selected as the
default fused implementation only after all of the following pass:

1. numerical comparison against the current torch-MUSA/SGLang reference for
   every observed dtype and shape;
2. dynamic token counts `1/15/16/31/465`, plus batch-16 decode and representative
   long-prefill coverage;
3. eager and full-graph execution without graph-break, aliasing or cache-state
   regressions;
4. at least three identical 16-concurrent end-to-end runs reporting output
   tok/s, P50/P95 latency, finish reason and response-token consistency;
5. GPQA short-answer checks with thinking enabled and maximum 4096 generated
   tokens;
6. a focused SGLang-FL or FlagGems regression test for the accepted MUSA path.
