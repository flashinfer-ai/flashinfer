#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path


FLOPS_RE = re.compile(r"FLOPs:\s*([0-9.]+)\s*TFLOPs/s")
TIME_RE = re.compile(r"execution time:\s*([0-9.]+)\s*ms")
BW_RE = re.compile(r"memory bandwidth:\s*([0-9.]+)\s*GB/s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TRTLLM-GEN MLA batch-split A/B benchmarks and plot TFLOPs/s"
    )
    parser.add_argument("--fixed-lut", type=Path, default=Path("fixed_data.json"))
    parser.add_argument("--varlen-lut", type=Path, default=Path("varlen_data.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("batch_split_plots"))
    parser.add_argument("--batch-sizes", type=str, default="38")
    parser.add_argument(
        "--all-lut-batches",
        action="store_true",
        help="Benchmark all batch sizes present in each LUT's splits object",
    )
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--q-len-per-request", type=int, default=None)
    parser.add_argument("--num-iters", type=int, default=None)
    parser.add_argument("--dry-run-iters", type=int, default=None)
    return parser.parse_args()


def load_lut(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def normalize_dtype(dtype):
    return {
        "float8_e4m3fn": "fp8_e4m3",
        "bfloat16": "bfloat16",
    }.get(dtype, dtype)


def batch_sizes_from_arg(value):
    result = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return sorted(dict.fromkeys(result))


def batch_sizes_for_lut(lut, args):
    if not args.all_lut_batches:
        return batch_sizes_from_arg(args.batch_sizes)
    splits = lut.get("splits", {})
    return sorted(int(batch_size) for batch_size in splits)


def env_from_lut(lut, fixed_seq_len, batch_size, lut_path, optimized, args):
    config = lut.get("config", {})
    env = os.environ.copy()
    env.update(
        {
            "BACKEND": "trtllm-gen",
            "BATCH_SIZE": str(batch_size),
            "Q_LEN_PER_REQUEST": str(
                args.q_len_per_request or config.get("q_len_per_request", 2)
            ),
            "SEQ_LEN": str(args.seq_len or config.get("seq_len", 20480)),
            "FIXED_SEQ_LEN": "1" if fixed_seq_len else "0",
            "PAGE_SIZE": str(config.get("page_size", 32)),
            "NUM_QO_HEADS": str(config.get("num_q_heads", 64)),
            "HEAD_DIM_CKV": str(config.get("kv_lora_rank", 512)),
            "HEAD_DIM_KPE": str(config.get("qk_rope_head_dim", 64)),
            "QK_NOPE_HEAD_DIM": str(config.get("qk_nope_head_dim", 128)),
            "DTYPE": normalize_dtype(config.get("dtype", "fp8_e4m3")),
        }
    )
    if args.num_iters is not None:
        env["NUM_ITERS"] = str(args.num_iters)
    elif "repeat_iters" in config:
        env["NUM_ITERS"] = str(config["repeat_iters"])
    if args.dry_run_iters is not None:
        env["DRY_RUN_ITERS"] = str(args.dry_run_iters)
    elif "dry_run_iters" in config:
        env["DRY_RUN_ITERS"] = str(config["dry_run_iters"])
    if optimized:
        env["LUT_CONFIG_ENV_VAL"] = str(lut_path)
    else:
        env.pop("LUT_CONFIG_ENV_VAL", None)
        env.pop("FLASHINFER_TRTLLM_GEN_MLA_BATCH_SPLIT_LUT", None)
    return env


def parse_metric(pattern, stdout, name):
    match = pattern.search(stdout)
    if match is None:
        raise RuntimeError(f"Could not parse {name} from output:\n{stdout}")
    return float(match.group(1))


def run_case(repo_root, out_dir, dataset, lut, lut_path, fixed_seq_len, batch_size, optimized, args):
    label = "optimized" if optimized else "baseline"
    env = env_from_lut(lut, fixed_seq_len, batch_size, lut_path, optimized, args)
    print(f"[{dataset}] batch={batch_size} {label}", flush=True)
    completed = subprocess.run(
        ["./benchmarks/run_trtllm_gen_mla_batch_split.sh"],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed for {dataset} batch={batch_size} {label}:\n"
            f"{completed.stdout}"
        )
    return {
        "dataset": dataset,
        "batch_size": batch_size,
        "optimized": int(optimized),
        "mode": label,
        "lut_path": str(lut_path if optimized else ""),
        "fixed_seq_len": int(fixed_seq_len),
        "tflops": parse_metric(FLOPS_RE, completed.stdout, "TFLOPs/s"),
        "execution_time_ms": parse_metric(TIME_RE, completed.stdout, "execution time"),
        "memory_bandwidth_gbps": parse_metric(BW_RE, completed.stdout, "memory bandwidth"),
    }


def write_csv(path, rows):
    fieldnames = [
        "dataset",
        "batch_size",
        "optimized",
        "mode",
        "lut_path",
        "fixed_seq_len",
        "tflops",
        "execution_time_ms",
        "memory_bandwidth_gbps",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_bar_dataset(path, dataset, rows):
    import matplotlib.pyplot as plt

    batch_sizes = sorted({row["batch_size"] for row in rows})
    baseline = {
        row["batch_size"]: row["tflops"] for row in rows if row["mode"] == "baseline"
    }
    optimized = {
        row["batch_size"]: row["tflops"] for row in rows if row["mode"] == "optimized"
    }

    x = list(range(len(batch_sizes)))
    width = 0.38
    fig, axis = plt.subplots(figsize=(max(7, len(batch_sizes) * 0.45), 4.5))
    axis.bar([value - width / 2 for value in x], [baseline[b] for b in batch_sizes], width, label="without LUT")
    axis.bar([value + width / 2 for value in x], [optimized[b] for b in batch_sizes], width, label="with LUT")
    axis.set_title(f"TRTLLM-GEN MLA batch split: {dataset}")
    axis.set_xlabel("batch size")
    axis.set_ylabel("TFLOPs/s")
    axis.set_xticks(x, [str(batch_size) for batch_size in batch_sizes], rotation=45)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_line_dataset(path, dataset, rows):
    import matplotlib.pyplot as plt

    batch_sizes = sorted({row["batch_size"] for row in rows})
    baseline = {
        row["batch_size"]: row["tflops"] for row in rows if row["mode"] == "baseline"
    }
    optimized = {
        row["batch_size"]: row["tflops"] for row in rows if row["mode"] == "optimized"
    }

    fig, axis = plt.subplots(figsize=(max(7, len(batch_sizes) * 0.45), 4.5))
    axis.plot(
        batch_sizes,
        [baseline[batch_size] for batch_size in batch_sizes],
        marker="o",
        linewidth=1.8,
        label="without LUT",
    )
    axis.plot(
        batch_sizes,
        [optimized[batch_size] for batch_size in batch_sizes],
        marker="o",
        linewidth=1.8,
        label="with LUT",
    )
    axis.set_title(f"TRTLLM-GEN MLA batch split: {dataset}")
    axis.set_xlabel("batch size")
    axis.set_ylabel("TFLOPs/s")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("fixed", args.fixed_lut, True),
        ("varlen", args.varlen_lut, False),
    ]
    all_rows = []
    for dataset, lut_path, fixed_seq_len in datasets:
        lut_path = lut_path.resolve()
        lut = load_lut(lut_path)
        rows = []
        for batch_size in batch_sizes_for_lut(lut, args):
            rows.append(
                run_case(repo_root, args.out_dir, dataset, lut, lut_path, fixed_seq_len, batch_size, False, args)
            )
            rows.append(
                run_case(repo_root, args.out_dir, dataset, lut, lut_path, fixed_seq_len, batch_size, True, args)
            )
        csv_path = args.out_dir / f"{dataset}_batch_split.csv"
        write_csv(csv_path, rows)
        bar_plot_path = args.out_dir / f"{dataset}_batch_split_bar.png"
        line_plot_path = args.out_dir / f"{dataset}_batch_split_line.png"
        plot_bar_dataset(bar_plot_path, dataset, rows)
        plot_line_dataset(line_plot_path, dataset, rows)
        print(f"wrote {csv_path}")
        print(f"wrote {bar_plot_path}")
        print(f"wrote {line_plot_path}")
        all_rows.extend(rows)

    combined_csv_path = args.out_dir / "batch_split_summary.csv"
    write_csv(combined_csv_path, all_rows)
    print(f"wrote {combined_csv_path}")


if __name__ == "__main__":
    main()