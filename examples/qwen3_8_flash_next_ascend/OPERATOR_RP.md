# Operator follow-up for Qwen3.8-Flash-Next

This list separates model-scoped ATen fallbacks from fused/native kernels that
need operator-team profiling. Latency shares are deliberately left unfilled:
the retained evidence contains serving throughput and accuracy results, but no
operator timeline from which a defensible end-to-end percentage can be derived.

## Fused and communication kernels

| Operator | Kernel | Interface | Source | E2E latency share | Representative decode shape | FlagGems |
|---|---|---|---|---:|---|---|
| GDN recurrent update | `fused_sigmoid_gating_delta_rule_update_npu` | `fused_sigmoid_gating_delta_rule_update_npu` | [sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/sgl_kernel_npu/fla/fused_sigmoid_gating_recurrent.py) | pending profile | q/k/v `[1, BS, 48, 128]`; state `[slot, 48, 128, 128]` | |
| MoE token permutation | `npu_moe_token_permute` | `torch_npu.npu_moe_token_permute` | [vLLM Ascend call site](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/device/device_op.py) | pending profile | hidden `[T, 2560]`; expert ids `[T, 10]` | |
| MoE gate/up GEMM | `npu_grouped_matmul` | `torch_npu.npu_grouped_matmul` | [torch_npu](https://github.com/Ascend/pytorch/blob/master/torch_npu/contrib/function/_grouped_matmul.py) | pending profile | A `[sum(tokens_e), 2560]`; W `[local_experts, 2560, 1280]` | |
| MoE down GEMM | `npu_grouped_matmul` | `torch_npu.npu_grouped_matmul` | [torch_npu](https://github.com/Ascend/pytorch/blob/master/torch_npu/contrib/function/_grouped_matmul.py) | pending profile | A `[sum(tokens_e), 640]`; W `[local_experts, 640, 2560]` | |
| MoE token unpermutation | `npu_moe_token_unpermute` | `torch_npu.npu_moe_token_unpermute` | [vLLM Ascend call site](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/device/device_op.py) | pending profile | tokens `[T * 10, 2560]`; probs `[T, 10]` | |
| Tensor-parallel reduction | `HCCL AllReduce` | `tensor_model_parallel_all_reduce` | [SGLang](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/parallel_state.py) | pending profile | BF16 `[T, 2560]`; TP=8 | |

`BS`/`T` is the active decode batch and must be measured at 1, 8, 16 and 32.
The suggested stable window is prompt=128 tokens, output=256 tokens,
temperature=0, thinking enabled, after three warmup requests. Report NPU
self-time, call count, P50/P95 and critical-path time separately.

## Model-scoped ATen blacklist

The exact overload list lives in `qwen38.env`. The main operator families are:

| Family | Representative interfaces | Typical input | FlagGems implementation |
|---|---|---|---|
| convolution | `conv1d`, `conv2d` | state convolution kernel=4; vision NCHW | [conv1d](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/ops/conv1d.py), [conv2d](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/ops/conv2d.py) |
| indexed read/write | `index`, `index_put*`, `index_copy*`, `gather*` | request slot index `[BS]`; indexer budget=2048 | [index](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/ops/index.py), [index_put](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/ops/index_put.py), [gather](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/ops/gather.py) |
| pointwise arithmetic | `add*`, `sub*`, `mul*`, `remainder*`, `floor_divide*` | `[T, 2560]` or `[T, 48, 128]` with broadcasting | [FlagGems ops](https://github.com/flagos-ai/FlagGems/tree/master/src/flag_gems/ops) |
| comparison/masking | `ge*`, `lt*`, `bitwise_and*`, `bitwise_not*`, `where*` | boolean/int masks `[T]` or `[T, heads]` | [FlagGems ops](https://github.com/flagos-ai/FlagGems/tree/master/src/flag_gems/ops) |
| reduction | `sum*`, `mean*`, `max*`, `argmax`, `bincount` | `[T, H, D]`, router `[T, 512]`, or logits shard | [FlagGems ops](https://github.com/flagos-ai/FlagGems/tree/master/src/flag_gems/ops) |
| activation/matmul | `silu*`, `bmm*` | MoE gate `[tokens_e, 1280]`; head matrices with D=128 | [silu](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/ops/silu.py), [bmm](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/ops/bmm.py) |

An implementation existing in FlagGems does not mean this service dispatches to
it. These overloads are blacklisted precisely because the validated service
routes them through the reference/torch_npu path.
