# XingChen4-29B-A4B 在 MUSA TP4 上的适配与算子 RP

本文记录 XingChen4-29B-A4B 在 4 张 MUSA S5000 上的已验证适配结论，并给出后续算子优化 RP。模型适配以 XingChen4 命名；参考的 TeleChat4 上游实现只作为代码来源，不作为模型名称。

本文只把真实服务和端到端压测中观察到的结果写成结论。尚未完成 shape-aware profiler 归因的算子统一标记为“待单独 profile”，不使用其他模型或不同数据类型的结果代替。

## 验证环境

| 项目 | 配置 |
|---|---|
| 模型 | XingChen4-29B-A4B |
| 模型结构 | 40 layers，hidden size 3584，32 attention heads |
| MLA | q-lora rank 768，kv-lora rank 512，qk-nope 128，qk-rope 64，v-head 128 |
| MoE | 64 routed experts，top-k=4，1 shared expert，MoE intermediate size 1024 |
| mHC | 4 residual streams，20 次 Sinkhorn iteration |
| 上下文长度 | 262144 |
| 硬件 | 4 x MUSA S5000 80G |
| 并行 | TP=4，单节点 |
| 数据类型 | BF16 activation/weights；mHC projection 使用 FP16 |
| 服务 | SGLang 0.5.6，OpenAI-compatible endpoint |

## 适配范围

### 1. 模型结构与权重加载

XingChen4 不是仅修改 `architectures` 名称即可支持的标准 DeepSeek 模型。适配需要同时处理：

- XingChen4 config 中的 full-attention、linear-attention 和 next-token prediction layer ID；
- MLA、64-expert MoE、shared expert 和 top-k routing；
- mHC pre/post 的 4 路 residual stream、clamp、Sinkhorn 和权重映射；
- Transformers 5 对 `rope_parameters` 的兼容，保留 checkpoint 的 `rope_theta` 与 `rope_scaling`；
- SGLang 外部模型注册和 checkpoint 参数名到 fused mHC 参数的映射。

已验证的权重审计结果为 795/795 参数加载，缺失参数 0。

### 2. mHC 数值契约

mHC pre 包含 projection、clamp、sigmoid gating、20 次 Sinkhorn normalization 和 residual mixing；mHC post 将 attention/MLP 输出按 post gate 写回 4 路 residual stream。需要保持以下语义：

- pre gate 使用原始 sigmoid 值，构造 layer input 时再加 epsilon；
- post gate 使用 `2 * sigmoid`；
- residual mixing 的归一化维度、clamp 范围 `[-30, 30]` 和 FP32 累加顺序与参考实现一致；
- 当前环境关闭不可用的 DeepGEMM/TileLang mHC 路径，使用已验证的 Triton kernel。

### 3. MoE 与图模式

XingChen4 的 routed MoE 形状为 `E=64`，主要输出维度 `N=256`。已有 GLM 的 `E=288` FP8 配置在 expert 数量、shape 和数据类型上均不匹配，不能直接复用。

已为 MUSA S5000 离线调优 BF16 Triton fused-MoE 配置，覆盖 batch/token bucket 1 到 64。服务采用：

- decode full graph，最大 graph batch size 64；
- prefill graph disabled，chunked prefill size 16384；
- Triton MoE runner；
- shared-experts fusion disabled；
- radix cache disabled（仅对应本次性能基线）。

### 4. FlagGems 兼容性边界

已验证启动配置对以下 ATen 算子使用 reference/vendor fallback：

`count_nonzero`、`cumsum`、`mm`、`unique`、`_unique2`、`unique_dim`、`unique_consecutive`、`index_put`、`index_put_`、`_index_put_impl_`。

该 blacklist 是当前 MUSA 运行时下的正确性和稳定性措施，不代表 FlagGems 没有对应实现。逐项解除前需要使用相同 checkpoint、相同 prompt/token IDs 和多 batch shape 做数值 A/B。

## 端到端验证结果

固定负载为 4096 input tokens、1024 output tokens、64 concurrent requests：

| 指标 | 结果 |
|---|---:|
| 成功请求 | 64/64 |
| output throughput | 495.59 tok/s |
| mean TPOT | 100.59 ms |
| mean TTFT | 29.24 s |

该结果是服务端到端基线，不是单算子 profile。各 kernel 的模型端到端时延占比需要在相同配置下重新采集带 shape/correlation ID 的 trace，详见 [算子 RP 表](xingchen4-29b-a4b-operator-rp.tsv)。

## 精度与协议验收

仅检查 `/health` 或单个 HTTP 200 不足以验收。已完成的验证包括：

1. 真实 MUSA TP4 权重加载和生成；
2. 64 并发、每请求 1024 output tokens 的完整压测；
3. reasoning/content 分离的 OpenAI chat 协议验证；
4. 原先 97 个空 content 的 GSM8K case 使用原评测参数重放：97/97 content 非空，93/97 答案正确，HTTP error 0，length 截断 0；
5. 保留 thinking；没有通过关闭 reasoning 或永久合并 reasoning/content 来伪装修复。

XingChen4 有时在 EOS 前不输出 `</think>`，且评测请求中的精确 stop `Question:` 可能在 reasoning 阶段误触发。服务端兼容逻辑只对 XingChen4、启用 reasoning parser 且 `separate_reasoning=true` 的请求生效：移除精确的 `Question:` stop，并启用 non-empty-content fallback；正常存在 `</think>` 时仍保持 reasoning/content 分离。

## 算子优化优先级

1. **P0：fused MoE 全链路。** 在 `E=64, top-k=4, hidden=3584, intermediate=1024` 的真实 shape 上联合优化 route/top-k、dispatch、两次 grouped GEMM、SwiGLU 和 reorder，并把已调优配置纳入回归。
2. **P0：mHC pre/post。** 优先减少 projection、clamp、sigmoid、Sinkhorn、residual mixing 之间的中间 tensor 和 launch 次数；必须保留 FP32 归一化/累加语义。
3. **P1：MLA decode。** 对 q/kv LoRA projection、RoPE、paged KV cache 和 attention decode 做 shape-aware profile，再评估 GEMM/attention 融合。
4. **P1：TP4 communication。** 采集每层 all-reduce 的消息大小和 stream overlap，评估 residual/RMSNorm 融合及通信计算 overlap。
5. **P1：RMSNorm 与 SwiGLU。** 保留当前 fused path，补齐不同 token bucket 的正确性和性能回归。
6. **P2：逐项解除 blacklist。** 每次只启用一个 FlagGems ATen 算子，完成数值、稳定性和吞吐 A/B 后再保留。

## 建议复测矩阵

- decode token bucket：1、2、4、8、16、32、48、64；
- prefill length：1K、4K、16K；
- 并发：1、16、32、64；
- 每项至少记录 output throughput、TPOT、TTFT、每卡显存、请求成功率和最新服务日志；
- profiler 结果需区分 GPU wall span 与各 stream kernel 累计时间，重叠 kernel 百分比不可直接相加。
