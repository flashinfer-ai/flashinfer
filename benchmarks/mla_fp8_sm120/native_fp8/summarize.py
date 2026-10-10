"""Generate the final native FP8 report from accepted, uncontaminated samples."""

import csv
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
r = json.loads((ROOT / "results.json").read_text())
cases = r["cases"]
rows = []
for c in cases:
    m = c["methods"]
    us = lambda k: m[k]["cold"]["median_us"] if k in m else None
    rows.append(
        dict(
            name=c["name"],
            batch=c["batch"],
            qlen=c["qlen"],
            kv_len=c["kv_len"],
            flashinfer_bf16_us=us("flashinfer_bf16"),
            native_fp8_us=us("native_fp8"),
            triton_fp8_us=us("triton_fp8"),
            native_prequantized_q_us=us("native_fp8_prequantized_q"),
            native_including_kv_quant_us=us("native_fp8_including_kv_quantization"),
            speedup_vs_bf16=us("flashinfer_bf16") / us("native_fp8"),
            speedup_vs_triton=us("triton_fp8") / us("native_fp8")
            if "triton_fp8" in m
            else None,
            relative_l2=c["errors"]["native_fp8"]["relative_l2_vs_flashinfer"],
            **c["selected"]["config"],
        )
    )
with (ROOT / "summary.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
for ax, prefill in zip(axes, (False, True), strict=False):
    group = [x for x in rows if (x["qlen"] > 1) == prefill]
    x = list(range(len(group)))
    specs = [
        ("flashinfer_bf16_us", "Original FlashInfer BF16", "#65798b"),
        ("native_fp8_us", "FlashInfer-derived CUDA FP8", "#258773"),
    ]
    if not prefill:
        specs.append(("triton_fp8_us", "Previous Triton FP8", "#d88c3e"))
    for j, (key, label, color) in enumerate(specs):
        ax.bar(
            [v + (j - (len(specs) - 1) / 2) * 0.25 for v in x],
            [v[key] for v in group],
            width=0.23,
            label=label,
            color=color,
        )
    labels = [
        f"B{v['batch']}\n{v['kv_len'] // 1024}K"
        if not prefill
        else f"Q{v['qlen']}\nKV{v['kv_len']}"
        for v in group
    ]
    ax.set_xticks(x, labels)
    ax.set_yscale("log")
    ax.set_ylabel("Latency (us, log scale)")
    ax.set_title("Causal prefill / extend" if prefill else "Decode")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
fig.suptitle(
    "SM120, GLM-4.7-Flash shapes: cold-cache operator latency, Q quantization included"
)
fig.savefig(ROOT / "latency.png", dpi=180)
plt.close(fig)

lines = [
    "这是在 FlashInfer 基础上新增的 SM120 FP8×FP8 CUDA 分支，支持 decode、causal/noncausal prefill 和增量 prefill。代码位于 `flashinfer/experimental/mla_fp8_sm120/`；没有修改已安装的 FlashInfer，也没有接入或停止现有服务。",
    "",
    "性能表来自 FlashInfer 0.6.15.post1 环境；当前目标分支为 0.7.2。提交版使用仓库头文件的验证单独记录在 [checkout_validation.json](checkout_validation.json)，不能把旧版计时当成新版性能。",
    "",
    "与 FlashInfer 的关系：复用 MLAPlan 的分页、变长请求和 work/merge 元数据格式；在本地副本中参数化 Q tile、worker 数及 KV 分段粒度。QK/PV 调用 FlashInfer 的 FP8 MMA helper，加载使用其 cp.async/swizzle 工具，合并沿用 state_t 的稳定归约公式并改为跨 warp 并行读取 KV 分段。它是实验性新增实现，不是官方现成后端，也不是对 Triton 实验改名。",
    "",
    "设备为 RTX PRO 5000 72GB Blackwell（SM120），CUDA 12.9。GLM 形状为 20 heads、吸收后 Q/K=512+64、latent 输出=512、page=16、softmax scale=1/16。完整 576 维 KV 使用 E4M3，每 token 一个 FP32 scale；Q 每 head 一个 FP32 scale；P 按 tile 和行量化。两次乘法均使用原生 FP8 MMA，softmax/累加/merge 使用 FP32，partial 和最终输出为 BF16。",
    "",
    "计时使用与前次相同的 CUDA graph 外部事件，cold 样本前以 256 MiB 缓冲区驱逐 L2，驱逐不计时；另保留 hot 结果。所有方法随机交错计时，每方法 30 次，开始前等待 GPU/显存控制器连续 5 秒利用率不超过 2%；其他进程活动或显存变化污染的整轮数据丢弃。下表为 cold median，比较的是算子耗时，不是模型 tokens/s。",
    "",
    "Q 从 BF16 量化的时间计入 native/Triton FP8 主表；历史 KV 已在 cache 中。完整 prefill 另测把 KV 量化计入的路径。计划、首次 JIT、投影、页分配、MoE、通信不计入算子时间。",
    "",
    "| 场景 | B | Q/请求 | KV/请求 | 原 FlashInfer BF16 μs | 新增 CUDA FP8 μs | 相对 BF16 加速 | 既有 Triton FP8 μs |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for x in rows:
    tr = "—" if x["triton_fp8_us"] is None else f"{x['triton_fp8_us']:.2f}"
    kind = (
        "decode"
        if x["qlen"] == 1
        else ("完整 prefill" if x["qlen"] == x["kv_len"] else "增量 prefill")
    )
    lines.append(
        f"| {kind} | {x['batch']} | {x['qlen']} | {x['kv_len']} | {x['flashinfer_bf16_us']:.2f} | {x['native_fp8_us']:.2f} | {x['speedup_vs_bf16']:.2f}× | {tr} |"
    )
lines += [
    "",
    "上述配置经过有限范围搜索，所有候选和跳过原因保留在 results.json 的 trials_file 所指向的 tuning_trials/ 文件中；并不代表任何实现的性能上限。很短 kernel 的 CUDA event 时间有约微秒级量化与波动。",
    "",
    "| 完整 prefill 长度 | BF16 μs | FP8：含 Q 和 KV 量化 μs | 含量化后的加速 |",
    "|---:|---:|---:|---:|",
]
for x in rows:
    if x["native_including_kv_quant_us"] is not None:
        v = x["native_including_kv_quant_us"]
        lines.append(
            f"| {x['qlen']} | {x['flashinfer_bf16_us']:.2f} | {v:.2f} | {x['flashinfer_bf16_us'] / v:.2f}× |"
        )
lines += [
    "",
    "增量 prefill 的新 KV 量化另列在原始数据中，使用连续行写出，不包含任意页 scatter。KV 载荷仍由 1152 B/token/layer 降为 580 B，减少 49.65%。",
    "",
    "计入 KV 量化的完整 prefill 路径会先写出 FP8 cache，因而预热后续 attention 的 KV 数据。它的 cold 总时间可能反而小于“已存在但冷”的 FP8 cache 路径；两者的差值不能作为 KV 量化本身的成本。原始数据另提供独立量化时间与连续重放的 hot 时间。",
    "",
    "数值误差来自合成输入，不能替代真实 GLM 任务准确率。以下是包含 Q/KV/P 量化后的输出相对原 BF16 输出的 L2 误差：",
    "",
    "| 场景 | B | Q | KV | 相对 L2 |",
    "|---|---:|---:|---:|---:|",
]
for x in rows:
    lines.append(
        f"| {x['name']} | {x['batch']} | {x['qlen']} | {x['kv_len']} | {x['relative_l2']:.3%} |"
    )
lines += [
    "",
    "独立数值校验还将 FP8 Q/KV 反量化后交给 CPU FP64 参考计算，用于区分 Q/KV 量化误差和 kernel/P 量化误差；覆盖非整页尾部、打乱物理页、causal 边界、完整/增量 prefill、空 KV、10/5 heads、分离/融合 merge 和 CUDA graph 重放。完整记录见 [validation_summary.json](validation_summary.json)：28 个数值配置和 4 个 Compute Sanitizer memcheck 配置，memcheck 均为 0 errors。",
    "",
    "Prefill 为什么可以做：将 query token 和 head 打包为行，保持每行自己的 causal 截止位置即可；FP8 MMA 不要求 query length 为 1。Prefill 查询较多，Q tile 的有效行比例更高，但 FP8 量化和分段合并的成本仍需实测。",
    "",
    "本实验针对吸收后的 MLA prefill。模型也可能选择展开 K/V 后的普通 attention：GLM 原始 QK=256、V=256，相比吸收后 576/512 有不同的计算量与展开成本。不能从本报告推断此分支就是整模型最优 prefill 路径。",
    "",
    "使用示例（先确认 GPU 空闲）：",
    "",
    "```python",
    "from native_fp8.wrapper import NativeMLA",
    "runner = NativeMLA(workspace, qo_indptr, kv_indptr, kv_indices, kv_lengths,",
    "                   heads=20, page_size=16, causal=True,",
    "                   bm=32, bn=32, workers=110)",
    "out = runner.run(q_bf16, kv_fp8, kv_scales)",
    "# q_bf16: [总 query tokens, 20, 576]",
    "# kv_fp8: [物理 pages, 16, 576]; kv_scales: [物理 pages, 16]",
    "# 已有 FP8 Q 时可调用 run_prequantized(q_fp8, kv_fp8, q_scales, kv_scales)。",
    "```",
    "",
    "复现：`python native_fp8/check.py`；`python native_fp8/bench.py --samples 30`。测试脚本会等待 GPU 空闲；直接调用 runner 的应用自行负责资源调度。当前只支持本机 SM120、E4M3、连续的 576 维输入、BF16 输出；有效页索引和分页生命周期由调用者管理。",
    "",
    "文件：",
    "",
    "- [CUDA 分支](../../../flashinfer/experimental/mla_fp8_sm120/mla_fp8.cu) 与 [本地 scheduler](../../../flashinfer/experimental/mla_fp8_sm120/scheduler_fp8.cuh)",
    "- [Python plan/run 接口](wrapper.py)",
    "- [数值测试](check.py) 与 [性能测试](bench.py)",
    "- [全部样本和配置](results.json)、[汇总 CSV](summary.csv)、[延迟图](latency.png)",
    "- [原文件基线哈希](source_provenance.json)、[已安装包未改动审计](post_optimization_audit.json)",
]
(ROOT / "README.zh.md").write_text("\n".join(lines) + "\n")
print(json.dumps(rows, indent=2))
