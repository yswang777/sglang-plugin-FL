# Qwen3.8-Flash-Next on Moore Threads MUSA

This directory records the model-scoped SGLang-FL settings used to bring up
the BF16 `Qwen4ExpForConditionalGeneration` checkpoint on eight MTT S5000
devices. The model implementation and checkpoint mappings belong in SGLang;
this bundle contains only the SGLang-FL/MUSA integration surface.

## Validated scope

- model path: `/data/models/Qwen3.8-Flash-Next`;
- precision: BF16;
- topology: TP8 and EP8;
- serving: eager decode, maximum 16 resident requests;
- MRoPE: SGLang fused Triton path selected through the MThreads backend;
- deterministic smoke: the capital-of-China prompt finishes with `北京`.

Run:

```bash
examples/qwen3_8_flash_next_musa/launch.sh
examples/qwen3_8_flash_next_musa/verify-smoke.sh
```

The blacklist is deliberately model-scoped. Do not remove entries until the
FlagGems implementation passes numerical, eager, graph, and concurrent-serving
tests for the shapes listed in `OPERATOR_RP.md`.

## Current limitations

- CUDA Graph is disabled. MUSA graph capture and MCCL all-reduce were not
  compatible in the tested stack.
- The current custom MoE path is decode-bound. A 16-concurrent 32-token run
  reached only 6.48 aggregate output tok/s; this is not a performance-complete
  configuration.
- A deterministic capital smoke passed, but the full 16-concurrent 16/16
  accuracy, token-parity, and soak acceptance has not been completed.
- NEXTN/MTP, EAGLE, draft, and other speculative options must remain disabled
  until those acceptance gates pass.
