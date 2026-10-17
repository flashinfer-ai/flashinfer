"""Create a CSV, plot and Chinese report from the completed benchmark JSON."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
r = json.loads((ROOT / "kv_bench_results.json").read_text())
rows = []
for c in r["cases"]:
    m = c["methods"]
    get = lambda key: m[key]["cold"]["median_us"]
    row = dict(
        batch=c["batch"],
        kv_len=c["kv_len"],
        flashinfer_bf16_us=get("flashinfer_bf16"),
        triton_bf16_us=get("triton_bf16"),
        fp8_cache_fixed_tile_bf16_mma_us=get("fp8_cache_same_tile_bf16_mma"),
        fp8_cache_tuned_bf16_mma_us=get("fp8_cache_bf16_mma"),
        fp8_cache_fp8_mma_us=get("fp8_cache_fp8_mma"),
        fp8_widen_then_flashinfer_us=get("fp8_cache_external_dequant_flashinfer"),
        native_fp8_speedup_vs_flashinfer=get("flashinfer_bf16")
        / get("fp8_cache_fp8_mma"),
        cache_only_fixed_tile_speedup=get("triton_bf16")
        / get("fp8_cache_same_tile_bf16_mma"),
        fp8_cache_bytes=c["cache_bytes_fp8_including_scales"],
        bf16_cache_bytes=c["cache_bytes_bf16"],
        new_kv_quant_us=c["new_kv_quantization"]["median_us"],
    )
    rows.append(row)
with (ROOT / "kv_bench_summary.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), layout="constrained")
for ax, batch in zip(axes, (1, 4, 16), strict=False):
    selected = [x for x in rows if x["batch"] == batch]
    x = list(range(len(selected)))
    specs = [
        ("flashinfer_bf16_us", "Original FlashInfer BF16", "#576b80"),
        ("fp8_cache_tuned_bf16_mma_us", "Triton: FP8 cache, BF16 compute", "#dc8e38"),
        ("fp8_cache_fp8_mma_us", "Triton: FP8 cache + FP8 compute", "#268c76"),
    ]
    for j, (key, label, color) in enumerate(specs):
        ax.bar(
            [v + (j - 1) * 0.25 for v in x],
            [row[key] for row in selected],
            width=0.23,
            label=label,
            color=color,
        )
    ax.set_xticks(x, [f"{row['kv_len'] // 1024}K" for row in selected])
    ax.set_title(f"Batch {batch}, 20 heads")
    ax.set_xlabel("KV length / request")
    ax.set_ylabel("Decode latency (us, log scale)")
    ax.set_yscale("log")
    ax.grid(axis="y", which="major", alpha=0.2)
axes[0].legend(fontsize=8)
fig.suptitle(
    "SM120 MLA: cold-cache CUDA-graph timing; experimental FP8 kernels", fontsize=12
)
fig.savefig(ROOT / "kv_bench_latency.png", dpi=180)
plt.close(fig)

native_gains = [x["native_fp8_speedup_vs_flashinfer"] for x in rows]
cache_gains = [x["cache_only_fixed_tile_speedup"] for x in rows]
lines = [
    f"本机 {len(rows)} 组 GLM-4.7-Flash 形状的 SM120 decode 实测：FP8 KV＋FP8 计算相对原 FlashInfer BF16 的算子加速为 **{min(native_gains):.2f}×～{max(native_gains):.2f}×**。保持 tile/split/warp 和 BF16 计算不变时，仅 FP8 cache 的加速比为 **{min(cache_gains):.2f}×～{max(cache_gains):.2f}×**，没有稳定的时延收益；cache 容量减少 49.65%。",
    "",
    "本次没有修改原 FlashInfer BF16 kernel。报告中的“原 FlashInfer BF16”调用已安装的 FA2 kernel；“新增 Triton BF16”及各条 FP8 路径均为独立目录中的实验实现。新增 Triton BF16 用于固定分块、调度和计算精度的对照，不能与原 FlashInfer BF16 混称。相关 FlashInfer 源文件在实验前后的哈希一致，见[校验记录](post_benchmark_flashinfer_audit.json)。本实验没有修改或停止 SGLang 服务。",
    "",
    "GPU：RTX PRO 5000 72GB Blackwell，SM120；20 heads，QK=512+64，latent 输出=512，page size=16，scale=1/16。所有输出为 BF16。实验把 cache 的全部 512+64 维量化为 FP8 E4M3，每个 token 有一个 FP32 scale；它不是保留 BF16 RoPE 的 FlashMLA 稀疏 cache 格式。FP8 计算路径还对每个 head 的 Q 量化，并对每个 KV tile 的 P 量化。",
    "",
    "计时前等待 GPU/显存控制器利用率连续 5 秒不高于 2%。期间用 NVML 每 250 ms 收集其他进程的计算/显存活动；发现干扰则丢弃该轮。计时使用 CUDA graph 中的 external CUDA events，方法执行顺序随机交错，每种形状每种方法取 30 次。主表的每次调用前清理 256 MiB L2 驱逐缓冲区，清理时间不计入结果；另记录了重复访问热 cache 的结果。",
    "",
    "FP8 计算的耗时包含当前 Q 的量化、attention 分段和结果合并。历史 KV 已在 cache 中，初次生成历史 cache 的量化不包含在 decode 计时内。新增一条 KV 的行量化另行计时；不含页分配、请求调度、模型投影、MoE、通信或整模型执行。",
    "",
    "| Batch | 每请求 KV | 原 FlashInfer BF16 / μs | 新增 Triton：FP8 KV＋BF16 计算 / μs | 新增 Triton：FP8 KV＋FP8 计算 / μs | 后者相对 FlashInfer 加速 |",
    "|---:|---:|---:|---:|---:|---:|",
]
for x in rows:
    lines.append(
        f"| {x['batch']} | {x['kv_len']} | {x['flashinfer_bf16_us']:.1f} | {x['fp8_cache_tuned_bf16_mma_us']:.1f} | {x['fp8_cache_fp8_mma_us']:.1f} | {x['native_fp8_speedup_vs_flashinfer']:.2f}× |"
    )
lines += [
    "",
    "上表将完整可运行路径与当前 FlashInfer 基线比较，包含 kernel 分块、调度与精度变化的综合收益。下面保持新增 Triton BF16/FP8-cache 两条路径的 Q tile、KV tile、warp 数和 split 数相同，单独观察存储变为 FP8、加载后在 kernel 内反量化回 BF16 的效果。",
    "",
    "| Batch | KV | 新增 Triton BF16 / μs | 同分块 Triton FP8 KV、BF16 计算 / μs | 固定分块 cache 加速 |",
    "|---:|---:|---:|---:|---:|",
]
for x in rows:
    lines.append(
        f"| {x['batch']} | {x['kv_len']} | {x['triton_bf16_us']:.1f} | {x['fp8_cache_fixed_tile_bf16_mma_us']:.1f} | {x['cache_only_fixed_tile_speedup']:.2f}× |"
    )
lines += [
    "",
    "两种 FP8-cache/BF16-compute 表分别是固定分块控制组和独立小范围调优组，因此数值可能不同。调优只搜索脚本列出的少量 tile/split 组合，不能视为各方案的性能上限。",
    "",
    "每层每 token：BF16 cache 为 1152 B；FP8 数据加 FP32 scale 为 580 B，容量减少 **49.65%**。容量收益不等于时延收益：反量化、寄存器与 shared-memory 压力仍可能限制 BF16 计算路径。",
    "",
    "| Batch | KV | FP8 cache 整体展开后再调原 FlashInfer / μs | 新增 B 条 KV 的量化 / μs |",
    "|---:|---:|---:|---:|",
]
for x in rows:
    lines.append(
        f"| {x['batch']} | {x['kv_len']} | {x['fp8_widen_then_flashinfer_us']:.1f} | {x['new_kv_quant_us']:.2f} |"
    )
lines += [
    "",
    "整体反量化控制组只展开本次测试的 cache；真实服务若展开更大的预分配 pool，成本可能更高。新增 KV 量化使用连续行写出，未计入任意页位置 scatter 或页分配的附加成本。短 kernel 的 CUDA event 计时存在约微秒级量化/波动，不应放大解读很小的差值。",
    "",
    "数值检查：新增 Triton BF16 路径与原 FlashInfer 相对 L2 误差小于 1%；另用 CPU FP64 校验了 ragged 分页和三种计算路径。FP8 路径会引入额外量化误差，下面列出相对原 FlashInfer BF16 输出的误差。这些是合成激活，不是 GLM 任务准确率测评。",
    "",
    "| Batch | KV | FP8 KV＋BF16 计算相对 L2 | FP8 KV＋FP8 计算相对 L2 |",
    "|---:|---:|---:|---:|",
]
for c in r["cases"]:
    lines.append(
        f"| {c['batch']} | {c['kv_len']} | {c['errors']['fp8_cache_bf16_mma']['relative_l2_vs_flashinfer']:.4%} | {c['errors']['fp8_cache_fp8_mma']['relative_l2_vs_flashinfer']:.4%} |"
    )
lines += [
    "",
    "原始文件：",
    "",
    "- [融合 kernel](kv_bench_kernels.py)",
    "- [测试和计时脚本](bench_fp8_kv.py)",
    "- [全部样本、调优配置、误差和 GPU 监测记录](kv_bench_results.json)",
    "- [汇总 CSV](kv_bench_summary.csv)",
    "- [延迟图](kv_bench_latency.png)",
    "- [FlashMLA 与 FlashInfer 的区别及本机可用性](flashmla_vs_flashinfer.zh.md)",
    "",
    "复现：`python bench_fp8_kv.py --samples 30`；该命令会等待 GPU 空闲，不停止现有进程。",
]
(ROOT / "fp8_kv_performance.zh.md").write_text("\n".join(lines) + "\n")
print(json.dumps(rows, indent=2))
print("Wrote CSV, plot, and fp8_kv_performance.zh.md")
