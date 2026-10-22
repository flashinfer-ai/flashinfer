这次优化将完整 2K prefill 从 **716.80 μs 降至 557.06 μs（1.29×）**，4K 从 **2478.08 μs 降至 1897.47 μs（1.31×）**。主要变化是复用 QK、softmax 和 FP8 P，而非移除整块 KV 反量化；原生 FP8 路径本来就不生成 BF16 KV cache。128/512 也有改善，但冷缓存下仍未超过 BF16。

本报告比较上一提交 `8f02f40243e9b1f09712d72584a5103768f1cc2f` 与本次源码。两份原生 FP8 内核均重新使用当前仓库的 0.7.2 核心头文件编译；BF16 对照仍是已安装的 FlashInfer **0.6.15.post1 / fa2**，不能表述为“官方 0.7.2 BF16 对照”。GPU 为 RTX PRO 5000 72GB Blackwell（SM120），CUDA 12.9，PyTorch 2.11.0+cu129。

形状保持 GLM-4.7-Flash 的 20 heads、Q/K=576、输出 latent=512、page=16、softmax scale=1/16。Q/KV/P 使用 E4M3，QK 和 PV 都是 FP8×FP8 MMA，累加和 softmax 为 FP32，输出及 partial output 为 BF16。[SASS 证据](sass_evidence.json)保留了 `QMMA.16832.F32.E4M3.E4M3` 指令。

计时是 CUDA graph 算子延迟，不是模型 tokens/s。每方法 30 次随机交错测量，候选调优每项 5 次后另做独立比较；保存全部 cold/hot 样本。Cold 样本前用 256 MiB 缓冲区驱逐 L2，驱逐不计时。开始前等待 GPU 和显存控制器连续 5 秒利用率 ≤2%，外部活动或显存变化污染的轮次丢弃。本轮共丢弃 6 个轮次，见 [rejected_rounds.json](rejected_rounds.json)。未停止或修改任何服务，也未重装已安装的 FlashInfer；已安装相关源码的哈希仍与前次审计一致，见 [installed_source_audit.json](installed_source_audit.json)。

下表均为 cold median。原生 FP8 主表包含 BF16 Q→FP8 的量化，KV cache 已存在；计划、首次编译、投影、页分配和其他模型层不计入。

| 场景 | BF16 μs | 上一版 FP8 μs | 本版 FP8 μs | 相对上一版 | 相对 BF16 |
|---|---:|---:|---:|---:|---:|
| decode，B=1，KV=8K | 55.30 | 32.77 | 32.77 | 1.00× | 1.69× |
| decode，B=16，KV=32K | 1255.42 | 417.79 | 393.22 | 1.06× | 3.19× |
| 完整 prefill，Q=128 | 28.67 | 34.82 | 32.77 | 1.06× | 0.88× |
| 完整 prefill，Q=512 | 83.97 | 118.78 | 108.54 | 1.09× | 0.77× |
| 增量 prefill，Q=128，KV=8K | 690.18 | 360.45 | 294.91 | 1.22× | 2.34× |
| 完整 prefill，Q=2048 | 825.34 | 716.80 | 557.06 | 1.29× | 1.48× |
| 完整 prefill，Q=4096 | 3137.02 | 2478.08 | 1897.47 | 1.31× | 1.65× |

完整 prefill 从 BF16 KV 开始、同时计入 Q 和 KV 量化时：

| 长度 | BF16 μs | 上一版含量化 μs | 本版含量化 μs | 相对 BF16 |
|---:|---:|---:|---:|---:|
| 128 | 28.67 | 32.77 | 30.72 | 0.93× |
| 512 | 83.97 | 106.50 | 94.21 | 0.89× |
| 2048 | 825.34 | 653.31 | 522.24 | 1.58× |
| 4096 | 3137.02 | 2363.39 | 1849.34 | 1.70× |

KV 量化会预热后续 attention 的 cache，所以这一表的 cold 总时间可能小于“已存在但冷的 FP8 cache”路径。两表差值不等于 KV 量化成本。增量 prefill 主表不计新 KV 的量化或 paged scatter。

隔离 Q 量化后，短 prefill 仍有内核内部成本：

| 长度 | 上一版预量化 Q μs | 本版预量化 Q μs | 本版含 Q 量化 μs |
|---:|---:|---:|---:|
| 128 | 30.72 | 28.67 | 32.77 |
| 512 | 106.50 | 96.26 | 108.54 |
| 2048 | 659.46 | 497.66 | 557.06 |
| 4096 | 2361.20 | 1777.66 | 1897.47 |

512 的预量化 Q 路径仍需 96.26 μs，而 BF16 是 83.97 μs，说明不能只把问题归结为输入量化。很短的 kernel 还受约微秒级事件量化和调度波动影响；128 的 2.05 μs 差值需谨慎解读。候选搜索是有限范围调优，并非性能上限。

本次计算组织的变化：

- 原实现按输出维度划分多个 group，各 group 重复做 QK、online softmax、P scale 和 FP8 打包。`share_p=True` 只由 `dg==0` 计算这些结果；共享内存保存 packed P 及行状态，CTA 同步后所有 group 各自计算一段 PV 输出。
- KV scale 仍在 QK 的 score 缩放中使用；PV 前把每 token 的 V scale 乘入 P，再量化 P。无需逐元素生成 BF16 K/V。省掉的是重复工作，scale 本身并未被删除。
- `groups=4` 把每线程负责的输出累加器减小，长 prefill 能容纳更多活动 warp。共享 P 会增加一次 CTA barrier 和共享内存读写，因此保留 `share_p=False`，且 B=1 decode 实测选择原路径。
- 非共享路径保留原来的 accumulator rescale 顺序。2K/4K 选中配置在本次合成输入上与上一版输出逐位相同；512 改了 BN，P 量化分组不同，与上一版的输出相对 L2 约 1.040%。

Nsight Compute 对预量化 Q、一次 warmup 后的 2K attention 单次 launch 给出的辅助证据如下；它没有采用主表的逐样本冷缓存协议，计时不可直接混入主表。使用 `--clock-control none --cache-control none`，未锁定 GPU 时钟。原始输出见 [previous](profile_previous.csv) / [optimized](profile_optimized.csv)。

| 项目 | 上一版 BM64/BN64/groups2 | 本版 BM64/BN64/groups4/shared P |
|---|---:|---:|
| 每 CTA 线程数 | 256 | 512 |
| 寄存器 / 线程 | 254 | 128 |
| 共享内存 / CTA | 73,984 B | 79,104 B |
| Achieved occupancy | 16.66% | 33.29% |
| 单次 attention launch | 639.49 μs | 484.67 μs |
| DRAM throughput | 4.06% | 5.93% |
| Compute (SM) throughput | 39.06% | 33.75% |

SM throughput 百分比降低并不代表变慢，这次减少了重复计算。新版本仍有 barrier 等待和寄存器 spill：2K/4K 配置每线程有 88 B local memory，且仍使用单 stage。低 DRAM 利用率来自这一预热后的具体 profile，不能外推为所有 decode/prefill 都不受带宽限制。本次没有尝试改变模型的 absorbed MLA 算法；展开 K/V 的普通 attention prefill 有不同的计算量和投影成本。

选中配置（仅适用于本报告的形状与设备）：

| 场景 | BM / BN | stages | groups | workers | fused merge | share_p | 寄存器 / shared B / local B |
|---|---:|---:|---:|---:|---|---|---|
| decode，B=1，KV=8K | 32 / 32 | 1 | 4 | 110 | True | False | 166 / 36992 / 0 |
| decode，B=16，KV=32K | 32 / 32 | 1 | 4 | 440 | False | True | 128 / 38528 / 8 |
| 完整 prefill，Q=128 | 32 / 32 | 1 | 4 | 55 | False | True | 146 / 38528 / 0 |
| 完整 prefill，Q=512 | 32 / 64 | 2 | 4 | 110 | False | True | 175 / 95232 / 0 |
| 增量 prefill，Q=128，KV=8K | 64 / 64 | 1 | 4 | 440 | False | True | 128 / 79104 / 88 |
| 完整 prefill，Q=2048 | 64 / 64 | 1 | 4 | 110 | False | True | 128 / 79104 / 88 |
| 完整 prefill，Q=4096 | 64 / 64 | 1 | 4 | 220 | False | True | 128 / 79104 / 88 |

2K/4K 完整 prefill 的选中 schedule 分别有 320/640 个 query-cluster work，没有 split-KV；它们的提速不能归因于 split merge。增量 prefill 和 decode 则仍可能分段。

**118 个数值/graph 配置通过，12 个 memcheck 配置为 0 errors，4 个 racecheck 配置为 0 errors / 0 warnings**，记录见 [validation.json](validation.json)。相对反量化后 CPU FP64 参考的最大 L2 误差为 1.044%，最大 LSE 绝对误差为 2.16e-6。整仓库 `pre-commit run -a` 通过。CPU FP64 参考先反量化 Q/KV，以区分输入量化与内核/P 量化误差；测试覆盖尾页、打乱页表、ragged、causal/noncausal、增量与完整 prefill、257-token 多 tile、空 KV、5/10/20 heads、不同输出 group、单/双缓冲、融合/独立 merge 和 CUDA graph。完整量化链相对 BF16 的合成输出 L2 误差约 **3.09%～5.31%**；这不是模型任务准确率，尚未验证真实 GLM 激活、长文本质量或下游任务。

复现，均从仓库根目录执行：

```bash
python benchmarks/mla_fp8_sm120/native_fp8/example.py --prefill
python benchmarks/mla_fp8_sm120/native_fp8/bench_revision.py \
  --cases decode_b1_8k decode_b16_32k prefill_128 prefill_512 extend_128_8k prefill_2048 prefill_4096 \
  --samples 30 --output /tmp/my-shared-p-results.json
python tests/experimental/mla_fp8_sm120/check.py --bm 64 --bn 64 --groups 4 --share-p
python tests/experimental/mla_fp8_sm120/check.py --bm 32 --bn 64 --stages 2 --groups 4 --share-p
compute-sanitizer --tool racecheck --error-exitcode 99 \
  python tests/experimental/mla_fp8_sm120/check.py \
  --bm 32 --bn 64 --stages 2 --groups 4 --share-p --only prefill_tiles
```

Profile 命令（旧版追加 `--previous --groups 2`）：

```bash
ncu --clock-control none --cache-control none \
  --kernel-name regex:BatchMLAPagedAttentionFP8SM120 --launch-skip 1 --launch-count 1 \
  --section LaunchStats --section Occupancy --section SpeedOfLight --section WarpStateStats \
  --csv --log-file /tmp/my-mla-profile.csv \
  python benchmarks/mla_fp8_sm120/native_fp8/profile_revision.py
```

调用示例里的 2K 配置为 `NativeMLA(..., causal=True, bm=64, bn=64, groups=4, workers=110, fused=False, share_p=True)`；4K 改为 `workers=220`。`run(q_bf16, kv_fp8, kv_scales)` 包含 Q 量化，`run_prequantized(q_fp8, kv_fp8, q_scales, kv_scales)` 接受预量化 Q。

实现仍位于 `flashinfer/experimental/mla_fp8_sm120`，没有修改 BF16 内核，也未接入 SGLang 或标准 MLA wrapper 自动选择。新开关需要显式传入；输入布局、精度及缓存生命周期约束沿用[后端说明](../../../../flashinfer/experimental/mla_fp8_sm120/README.md)。所有延迟样本见 [results.json](results.json)，全部候选见其 `trials_file`，便于表格处理的版本见 [summary.csv](summary.csv)。
