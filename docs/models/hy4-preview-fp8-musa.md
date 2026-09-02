# Hy4-preview-FP8 在 MUSA TP16 上的适配记录

本文记录 Hy4-preview-FP8 通过 sglang-plugin-FL 在两节点 MUSA S5000（每节点 8 卡，TP=16）上的适配结论，并给出后续算子优化 RP。本文只描述已经观察到的行为；历史 profiler 数据不等同于当前版本的性能承诺。

## 验证环境

| 项目 | 配置 |
|---|---|
| 模型 | Hy4-preview-FP8（Hy v4，78 layers） |
| 关键维度 | hidden=6144，attention heads=64，KV heads=8 |
| MoE | 256 experts，top-k=8，MoE intermediate=2048 |
| 稀疏注意力 | DSA，index top-k=2048 |
| 硬件 | 2 x 8 MUSA S5000 80G |
| 并行 | TP=16，跨两节点 |
| 权重/激活 | FP8 expert weights，BF16 activation，per-token-group FP8 quant |

## 主要适配难点

### 1. 模型与 SGLang 的数学契约

Hy4 不是标准 dense Transformer。适配需要同时保持以下语义：

- HC pre/post 的 channel 维和 residual 更新顺序；HC post 使用 FP32 累加后写回 BF16。
- MoE router top-k、routed scaling factor、expert dispatch/reorder 和 FP8 block scale。
- DSA indexer checkpoint 参数必须完整加载，尤其 `index_topk=2048`，不能被运行时补丁覆盖为 0。
- 稀疏 attention 的 selected KV gather、FP8 KV dequant、page-table transform 和 TP head 切分必须一致。
- 服务能够返回 JSON 不代表精度正确，需用确定性算术、raw completion 和连续多请求验证。

### 2. FlagGems ATen fallback/blacklist

部分 ATen 算子在 MUSA 上使用 FlagGems 实现会出现精度、shape 或稳定性问题，需要按算子回退到 reference/vendor。实际排查涉及：

`count_nonzero`、`index_put_`、`index_select`、`mm`、`add`、`mul`、`sum`、`mean`、`rsqrt`、`sigmoid`、`unique`、`bmm`、`floor_divide`、`index`、`topk`、`copy_` 和 `_to_copy` 等。

blacklist 是兼容性措施，不代表 FlagGems 仓库没有该算子。对应实现及性能占比见 [算子 RP 表](hy4-preview-fp8-operator-rp.tsv)。

### 3. 融合算子和 MUSA kernel

- HC post：融合 `residual + x * post`，避免多个 broadcast elementwise kernel 和中间 tensor。
- SwiGLU：融合 clamp、sigmoid 和乘法，保持模型指定的 clamp 范围。
- fused MoE：端到端链路包含 router、top-k、dispatch/scatter、FP8 quant、grouped GEMM、SwiGLU 和 reorder，不能只优化 GEMM。
- DSA：prefill sparse attention、top-k/transform、selected KV gather+dequant 是三个独立热点。
- TP16：跨节点 all-reduce 是历史 trace 中最大单项，需要通信/计算 overlap 和小包融合。

### 4. 并发和 KV cache 稳定性

高并发下出现过 KV cache retract 后 worker/detokenizer 退出。服务验收不能只看 `/health`，还需要持续请求、worker 存活、日志 traceback/OOM 和每卡显存。精度评测宜先使用经过稳定性验证的并发上限，再单独扩大容量边界。

## 验收方法

至少覆盖：

1. `/health` 返回 200，模型 endpoint 正常。
2. 1-token 和 8-token raw completion 无乱码。
3. 保留 reasoning 的 chat 请求可正确回答 `17 * 27 = 459`。
4. 连续多请求不导致 detokenizer、scheduler 或 TP worker 退出。
5. 日志没有新的 missing indexer weights、OOM 或 traceback。
6. 记录启动参数、每卡显存和并发配置。

## 历史性能剖析

数据来自既有 TP0 profiler trace。GPU wall span 为 4.127400 s；GPU kernel 可能跨 stream 重叠，因此各行占比不可直接相加为 100%。该数据用于确定优化优先级，不是当前主分支 benchmark 成绩。

| 优先级 | Kernel family | 模型端到端时延占比 |
|---|---|---:|
| P0 | TP AllReduce | 25.013% |
| P0 | fused MoE 全链 | 8.635% |
| P0 | DSA prefill | 5.240% |
| P1 | `_to_copy` | 2.408% |
| P1 | `copy_` | 2.182% |
| P1 | `mul` | 1.877% |
| P1 | `bmm` | 1.683% |
| P1 | reduction family | 1.655% |
| P1 | HC post | 1.343% |
| P2 | DSA top-k sort | 0.524% |

## 建议工作顺序

1. 通信团队优化 TP16 all-reduce 及 overlap。
2. 算子团队整体优化 fused MoE，优先减少 activation quant 和 layout reorder。
3. 完善 DSA prefill、top-k transform、FP8 KV gather+dequant 的 MUSA fused path。
4. 优先消除 `_to_copy/copy_`，再融合 `mul + reduction + rsqrt`、small-N `bmm`。
5. 对当前 trace 无法归因的算子使用 input-shape + correlation-id profiler 复采，不填写推测数值。
