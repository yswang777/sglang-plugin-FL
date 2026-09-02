# Qwen3.8-Flash-Next Ascend adaptation bundle

This directory contains the source overlay and launch configuration used to
adapt Qwen3.8-Flash-Next to SGLang-FL on Ascend 910C. It is intentionally kept
separate from plugin-wide Ascend defaults: this model's observed ATen failures
must not impose fallbacks on unrelated models.

## Compatibility

- SGLang base: `600df8dd498955f5aef4417383ea9ae5d2c238f7`
- Accelerator: one 16-card Ascend 910C host
- Parallelism: TP8 x DP2
- Precision and attention: BF16 with the `ascend` backend
- Checkpoint architecture: `Qwen4ExpForConditionalGeneration`

Run `install-source.sh` only on the exact SGLang base above. The installer
checks the revision, applies the registration/fallback patch, installs the six
overlay modules, and compiles them before returning success.

Target topology: one 16-card Ascend 910C host, TP8 x DP2. Model path defaults
to `/models/Qwen3.8-Flash-Next/`; served model name defaults to
`Qwen3.8-Flash-Next`; port defaults to 30100.

For the benchmark host, optionally run `host-tuning.sh` once as root before
starting the server. It selects the performance CPU governor, disables swap
pressure and NUMA balancing, and raises scheduler migration cost. It is not
called automatically because these are host-wide settings.

## Adaptation components

- `source/qwen4.py`: model assembly and checkpoint mapping.
- `source/qwen4_layer.py`: Hyper-Connection, PLE and request state handling.
- `source/qwen4_ops.py`: QSA, GDN and native Ascend grouped-MoE execution.
- `source/qwen4_config.py` and `source/qwen4_exp.py`: composite configs.
- `source/npu_graph_runner.py`: NPU Graph capture/replay support.
- `sglang-600df8-registration.patch`: model registration, request-pool sizing,
  native PyTorch n-gram fallback and compatibility changes.
- `qwen38.env`: model-scoped FlagGems blacklist and communication settings.
- `OPERATOR_RP.md`: fused-kernel and blacklist follow-up for the operator team.

## Default: best validated concurrent profile

Run `/opt/qwen38-adaptation/current/launch.sh`. This is the validated eager
profile: eight resident requests per DP (16 total across DP2), FlagGems enabled
with only observed bad operators blacklisted, chunked prefill 8192, and a
140,000-token pool. The prior 16-request multidomain suite passed 16/16; repeat
4096-input/32-output/concurrency-16 runs measured 265.77 and 264.82 total tok/s.
This metric is prefill-dominated and must not be reported as decode throughput;
the same retained run reported about 2.06 output tok/s.

## Optional: accuracy-verified Graph profile

Run `/opt/qwen38-adaptation/current/launch-graph-bs1.sh`. It captures NPU Graph
at BS=1 and binds all stateful PLE/GDN/QSA buffers to logical slot zero. It is
accuracy-verified but intentionally limits each DP scheduler to one resident
request. Do not increase the graph batch argument in this launcher: BS=2
capture was tested and exhausted the remaining per-rank HBM.

The Graph profile completed a 16-case GPQA run with 13/16 scored correct; all
13 cases that emitted an explicit final answer were correct, while three used
the full 34,000-token output budget without an explicit answer. Short replay
and request-slot reuse checks returned `Beijing` and `1073`, with no non-finite
value, traceback, 507015, or OOM in the restored service.

`--skip-server-warmup` skips the generic HTTP warmup request only. The Graph
launcher still performs NPU Graph capture during initialization. It is retained
because this exact command is the verified configuration.

## Known limitations

- The high-concurrency baseline uses eager decode; it is not a Graph result.
- The Graph launcher captures BS=1 only. BS=2 capture exhausted remaining
  per-rank HBM in the validated TP8 x DP2 topology.
- GDN/PLE/QSA state is request-slot scoped. Batching changes require slot-reuse
  and multi-request accuracy tests, not only health checks.
- Remove blacklist entries only after operator accuracy and sustained-serving
  A/B validation on the recorded shapes.

## Verification and source recovery

After the server is ready, run
`/opt/qwen38-adaptation/current/verify-smoke.sh`. The `source/` directory is the
exact active config/model/Graph-runner source snapshot at image-finalization time. `install-source.sh`
restores that snapshot into the SGLang tree. `SHA256SUMS` verifies the bundle.
