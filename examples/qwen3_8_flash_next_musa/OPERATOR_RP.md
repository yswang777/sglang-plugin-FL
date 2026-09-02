# Qwen3.8-Flash-Next MUSA operator RP

## Summary

This proposal records the remaining operator work for BF16
Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`) on eight Moore Threads
MTT S5000 devices. The model has 48 layers, hidden size 2560, Hyper-Connection
width 4, PLE/ngram state, hybrid GDN/QSA attention, 512 routed experts, and a
shared expert.

The model and request-state implementation is being aligned with the official
SGLang Qwen4 implementation. This RP does not upstream the experimental model
overlay into SGLang-FL.

## Required FlagOS blacklist

```text
count_nonzero,cumsum,mm,unique,_unique2,unique_dim,unique_consecutive
```

These operators remain on torch-MUSA/SGLang paths because the tested FlagGems
implementations were either incompatible with the serving shapes or had not
passed concurrent accuracy and graph-capture validation. `mm` is performance
critical; the reduction/unique family is primarily a correctness and dynamic
shape requirement.

## Operator inventory

| Priority | Area | Observed interface / shape | Required work |
|---|---|---|---|
| P0 | Routed MoE | hidden 2560, 512 experts, intermediate 640, top-k routing, TP8/EP8 | Replace per-token/per-slot GEMV with grouped GEMM and real expert all-to-all; preserve BF16 parity. |
| P0 | TP communication | one model-wide reduction per MoE layer | Fuse or overlap reduction without breaking MUSA graph capture. |
| P1 | MRoPE | positions `[T]` and `[3,T]`; Q/K may be head-shaped | Use SGLang's fused Triton path on MUSA and preserve shape/in-place semantics. Included in this change. |
| P1 | GDN | request-scoped convolution and SSM state | Validate fused prefill/decode kernels with mixed requests and slot reuse. |
| P1 | QSA | sparse indexer plus paged KV state | Add a parity-verified MUSA kernel; experimental Triton selection was slower in the measured short run. |
| P2 | Hyper-Connection/PLE | HC width 4; ngram history and conv state | Keep state keyed by request-pool slot and make graph buffers static. |

## Correctness constraints

PLE, GDN convolution/SSM, QSA, and ngram history are request state. A batched
implementation must key them by request-pool index, split prefill by
`query_start_loc`, concatenate per-request results in token order, and clear all
state when a slot is reused. A single `req_keys[0]` for a mixed batch corrupts
recurrent state and eventually MoE routing.

## Measured status

- deterministic Beijing smoke: pass with `temperature=0`, natural stop;
- TP8/EP1 plus fast all-reduce: 5.36 aggregate output tok/s at concurrency 16;
- TP8/EP8: 6.48 aggregate output tok/s at concurrency 16;
- QSA Triton experiment: 6.01 aggregate output tok/s;
- CUDA Graph plus MCCL: initialization failed because the MUSA operation was
  not permitted while the stream was capturing.

These measurements identify bottlenecks; they do not claim production
acceptance.

## Acceptance gates

1. MUSA numerical comparison for every fused module and recorded shape.
2. Deterministic complete-token parity against the reference path.
3. Three consecutive 16-concurrent runs with 16/16 fixed-case accuracy.
4. Request-slot reuse and mixed-prefill/decode stability with no non-finite state.
5. Long-running soak and output-throughput A/B with P50/P95 latency.
6. Only after all gates pass may CUDA Graph or NEXTN/MTP be considered for a
   production launcher.
