这是在 FlashInfer 基础上新增的 SM120 FP8×FP8 CUDA 分支，支持 decode、causal/noncausal prefill 和增量 prefill。代码位于 `flashinfer/experimental/mla_fp8_sm120/`；没有修改已安装的 FlashInfer，也没有接入或停止现有服务。

性能表来自 FlashInfer 0.6.15.post1 环境；当前目标分支为 0.7.2。提交版使用仓库头文件的验证单独记录在 [checkout_validation.json](checkout_validation.json)，不能把旧版计时当成新版性能。

与 FlashInfer 的关系：复用 MLAPlan 的分页、变长请求和 work/merge 元数据格式；在本地副本中参数化 Q tile、worker 数及 KV 分段粒度。QK/PV 调用 FlashInfer 的 FP8 MMA helper，加载使用其 cp.async/swizzle 工具，合并沿用 state_t 的稳定归约公式并改为跨 warp 并行读取 KV 分段。它是实验性新增实现，不是官方现成后端，也不是对 Triton 实验改名。

设备为 RTX PRO 5000 72GB Blackwell（SM120），CUDA 12.9。GLM 形状为 20 heads、吸收后 Q/K=512+64、latent 输出=512、page=16、softmax scale=1/16。完整 576 维 KV 使用 E4M3，每 token 一个 FP32 scale；Q 每 head 一个 FP32 scale；P 按 tile 和行量化。两次乘法均使用原生 FP8 MMA，softmax/累加/merge 使用 FP32，partial 和最终输出为 BF16。

计时使用与前次相同的 CUDA graph 外部事件，cold 样本前以 256 MiB 缓冲区驱逐 L2，驱逐不计时；另保留 hot 结果。所有方法随机交错计时，每方法 30 次，开始前等待 GPU/显存控制器连续 5 秒利用率不超过 2%；其他进程活动或显存变化污染的整轮数据丢弃。下表为 cold median，比较的是算子耗时，不是模型 tokens/s。

Q 从 BF16 量化的时间计入 native/Triton FP8 主表；历史 KV 已在 cache 中。完整 prefill 另测把 KV 量化计入的路径。计划、首次 JIT、投影、页分配、MoE、通信不计入算子时间。

| 场景 | B | Q/请求 | KV/请求 | 原 FlashInfer BF16 μs | 新增 CUDA FP8 μs | 相对 BF16 加速 | 既有 Triton FP8 μs |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode | 1 | 1 | 1024 | 22.53 | 18.43 | 1.22× | 16.38 |
| decode | 1 | 1 | 8192 | 55.30 | 30.72 | 1.80× | 24.58 |
| decode | 1 | 1 | 32768 | 102.40 | 61.44 | 1.67× | 61.44 |
| decode | 4 | 1 | 8192 | 94.21 | 53.25 | 1.77× | 59.39 |
| decode | 4 | 1 | 32768 | 231.42 | 130.05 | 1.78× | 149.50 |
| decode | 16 | 1 | 8192 | 296.96 | 137.22 | 2.16× | 151.55 |
| decode | 16 | 1 | 32768 | 1254.26 | 415.74 | 3.02× | 492.53 |
| 增量 prefill | 1 | 16 | 8192 | 96.26 | 77.82 | 1.24× | — |
| 完整 prefill | 1 | 128 | 128 | 28.67 | 34.82 | 0.82× | — |
| 完整 prefill | 1 | 512 | 512 | 81.92 | 116.74 | 0.70× | — |
| 增量 prefill | 1 | 128 | 8192 | 686.08 | 358.40 | 1.91× | — |
| 完整 prefill | 1 | 2048 | 2048 | 821.25 | 714.75 | 1.15× | — |
| 完整 prefill | 1 | 4096 | 4096 | 3129.34 | 2477.06 | 1.26× | — |

上述配置经过有限范围搜索，所有候选和跳过原因保留在 results.json 的 trials_file 所指向的 tuning_trials/ 文件中；并不代表任何实现的性能上限。很短 kernel 的 CUDA event 时间有约微秒级量化与波动。

| 完整 prefill 长度 | BF16 μs | FP8：含 Q 和 KV 量化 μs | 含量化后的加速 |
|---:|---:|---:|---:|
| 128 | 28.67 | 32.77 | 0.88× |
| 512 | 81.92 | 102.40 | 0.80× |
| 2048 | 821.25 | 651.26 | 1.26× |
| 4096 | 3129.34 | 2365.44 | 1.32× |

增量 prefill 的新 KV 量化另列在原始数据中，使用连续行写出，不包含任意页 scatter。KV 载荷仍由 1152 B/token/layer 降为 580 B，减少 49.65%。

计入 KV 量化的完整 prefill 路径会先写出 FP8 cache，因而预热后续 attention 的 KV 数据。它的 cold 总时间可能反而小于“已存在但冷”的 FP8 cache 路径；两者的差值不能作为 KV 量化本身的成本。原始数据另提供独立量化时间与连续重放的 hot 时间。

数值误差来自合成输入，不能替代真实 GLM 任务准确率。以下是包含 Q/KV/P 量化后的输出相对原 BF16 输出的 L2 误差：

| 场景 | B | Q | KV | 相对 L2 |
|---|---:|---:|---:|---:|
| decode_b1_1k | 1 | 1 | 1024 | 5.094% |
| decode_b1_8k | 1 | 1 | 8192 | 4.085% |
| decode_b1_32k | 1 | 1 | 32768 | 3.096% |
| decode_b4_8k | 4 | 1 | 8192 | 3.769% |
| decode_b4_32k | 4 | 1 | 32768 | 3.044% |
| decode_b16_8k | 16 | 1 | 8192 | 3.946% |
| decode_b16_32k | 16 | 1 | 32768 | 3.089% |
| extend_16_8k | 1 | 16 | 8192 | 3.777% |
| prefill_128 | 1 | 128 | 128 | 5.026% |
| prefill_512 | 1 | 512 | 512 | 5.262% |
| extend_128_8k | 1 | 128 | 8192 | 3.849% |
| prefill_2048 | 1 | 2048 | 2048 | 5.311% |
| prefill_4096 | 1 | 4096 | 4096 | 5.139% |

独立数值校验还将 FP8 Q/KV 反量化后交给 CPU FP64 参考计算，用于区分 Q/KV 量化误差和 kernel/P 量化误差；覆盖非整页尾部、打乱物理页、causal 边界、完整/增量 prefill、空 KV、10/5 heads、分离/融合 merge 和 CUDA graph 重放。完整记录见 [validation_summary.json](validation_summary.json)：28 个数值配置和 4 个 Compute Sanitizer memcheck 配置，memcheck 均为 0 errors。

Prefill 为什么可以做：将 query token 和 head 打包为行，保持每行自己的 causal 截止位置即可；FP8 MMA 不要求 query length 为 1。Prefill 查询较多，Q tile 的有效行比例更高，但 FP8 量化和分段合并的成本仍需实测。

本实验针对吸收后的 MLA prefill。模型也可能选择展开 K/V 后的普通 attention：GLM 原始 QK=256、V=256，相比吸收后 576/512 有不同的计算量与展开成本。不能从本报告推断此分支就是整模型最优 prefill 路径。

使用示例（先确认 GPU 空闲）：

```python
from native_fp8.wrapper import NativeMLA
runner = NativeMLA(workspace, qo_indptr, kv_indptr, kv_indices, kv_lengths,
                   heads=20, page_size=16, causal=True,
                   bm=32, bn=32, workers=110)
out = runner.run(q_bf16, kv_fp8, kv_scales)
# q_bf16: [总 query tokens, 20, 576]
# kv_fp8: [物理 pages, 16, 576]; kv_scales: [物理 pages, 16]
# 已有 FP8 Q 时可调用 run_prequantized(q_fp8, kv_fp8, q_scales, kv_scales)。
```

复现：`python native_fp8/check.py`；`python native_fp8/bench.py --samples 30`。测试脚本会等待 GPU 空闲；直接调用 runner 的应用自行负责资源调度。当前只支持本机 SM120、E4M3、连续的 576 维输入、BF16 输出；有效页索引和分页生命周期由调用者管理。

文件：

- [CUDA 分支](../../../flashinfer/experimental/mla_fp8_sm120/mla_fp8.cu) 与 [本地 scheduler](../../../flashinfer/experimental/mla_fp8_sm120/scheduler_fp8.cuh)
- [Python plan/run 接口](wrapper.py)
- [数值测试](check.py) 与 [性能测试](bench.py)
- [全部样本和配置](results.json)、[汇总 CSV](summary.csv)、[延迟图](latency.png)
- [原文件基线哈希](source_provenance.json)、[已安装包未改动审计](post_optimization_audit.json)
