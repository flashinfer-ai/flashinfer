from __future__ import annotations

import argparse
import os
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import time
import flashinfer
from flashinfer.testing.utils import bench_gpu_time
import matplotlib.pyplot as plt

NUM_LAYERS = 36  # QWen3 8b

save_dir = "bench_plots/persistent"


def plot_original_comparison(df_all):
    """Generate the original comparison plots."""
    cases = {"1": "chunked"}

    # Clean up potential duplicated header rows after appends and ensure numeric types
    if "scheduler" in df_all.columns:
        df_all = df_all[df_all["scheduler"] != "scheduler"]
    df_all["seq_cfg_id"] = pd.to_numeric(df_all["seq_cfg_id"], errors="coerce")
    df_all["bandwidth_GB_s"] = pd.to_numeric(df_all["bandwidth_GB_s"], errors="coerce")
    df_all = df_all.dropna(
        subset=["seq_cfg_id", "bandwidth_GB_s"]
    )  # keep only valid rows
    df_all["seq_cfg_id"] = df_all["seq_cfg_id"].astype(int)
    repeats = df_all["num_repeats"].unique()

    # Pick an available case (prefer 4: hybrid-chunked). Avoid empty plots when requested case is missing
    available_cases = sorted(df_all["seq_cfg_id"].dropna().unique().tolist())
    if not available_cases:
        raise ValueError("No rows found in bench_batch_attention.csv")
    preferred_case = 1 if 1 in available_cases else available_cases[0]
    selected_case_label = cases.get(str(preferred_case), f"case {preferred_case}")

    for repeat in repeats:
        # Collect data for all cases
        case_names = []
        batch_prefill_values = []
        persistent_original_values = []
        decode_prefill_values = []
        decode_len = df_all["decode_len"].unique()[0] // 1024
        prefill_chunk_size = df_all["prefill_chunk_size"].unique()[0] // 1024
        for case in [str(preferred_case)]:
            # Compute averages for this specific repeat count
            batch_prefill_avg = df_all[
                (df_all["scheduler"] == "BatchPrefillWithPagedKVCacheWrapper")
                & (df_all["seq_cfg_id"] == int(case))
                & (df_all["num_repeats"] == repeat)
            ]["bandwidth_GB_s"].mean()
            decode_prefill_avg = df_all[
                (df_all["scheduler"] == "Decode + Prefill")
                & (df_all["seq_cfg_id"] == int(case))
                & (df_all["num_repeats"] == repeat)
            ]["bandwidth_GB_s"].mean()
            persistent_original_avg = df_all[
                (df_all["scheduler"] == "BatchAttentionWrapper")
                & (df_all["seq_cfg_id"] == int(case))
                & (df_all["num_repeats"] == repeat)
            ]["bandwidth_GB_s"].mean()

            case_names.append(selected_case_label)
            # Replace NaNs with zeros to ensure bars render even if some schedulers are missing
            batch_prefill_values.append(np.nan_to_num(batch_prefill_avg, nan=0.0))
            decode_prefill_values.append(np.nan_to_num(decode_prefill_avg, nan=0.0))
            persistent_original_values.append(
                np.nan_to_num(persistent_original_avg, nan=0.0)
            )

        # Create grouped bar plot
        x = np.arange(len(case_names))
        width = 0.2
        # Group persistent schedules together on the left, others on the right
        offsets = np.array([0.5, 1.5, 2.5]) * width

        plt.figure(figsize=(12, 8))  # Increased height for better spacing
        bars1 = plt.bar(
            x + offsets[0],
            persistent_original_values,
            width,
            label="Persistent",
            color="#2ca02c",
        )
        bars2 = plt.bar(
            x + offsets[1],
            batch_prefill_values,
            width,
            label="Batch Prefill",
            color="#ff7f0e",
        )
        bars3 = plt.bar(
            x + offsets[2],
            decode_prefill_values,
            width,
            label="Decode + Prefill",
            color="#9467bd",
        )

        plt.ylabel("Average Bandwidth (GB/s)")
        plt.title(
            f"Average Bandwidth ({selected_case_label}, {repeat} repeats, {prefill_chunk_size}k prefill, {decode_len}k decode)"
        )

        # Add more space above the highest bar for legend and value labels
        max_value = max(
            max(batch_prefill_values),
            max(decode_prefill_values),
            max(persistent_original_values),
        )
        plt.ylim(0, max_value * 1.08)  # 8% more space above the highest bar

        plt.legend(fontsize=8, loc="upper right")

        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if height == 0:
                    continue
                # Use proportional offset based on data range
                offset = max_value * 0.02  # 2% of max value as offset
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + offset,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        add_value_labels(bars1)  # Persistent (original)
        add_value_labels(bars2)  # Batch Prefill
        add_value_labels(bars3)  # Decode + Prefill

        plt.tight_layout()
        plt.savefig(
            f"{save_dir}/persistent_comparison_{repeat}_repeats_{prefill_chunk_size}k_prefill_{decode_len}k_decode.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()


def run_bench(
    decode_kv_lens: Sequence[int],
    decode_qo_lens: Sequence[int],
    prefill_kv_lens: Sequence[int],
    prefill_qo_lens: Sequence[int],
    *,
    page_block_size: int,
    num_kv_heads: int,
    num_qo_heads: int,
    head_dim: int,
    device: int = 0,
    causal: bool = True,
    repeats: int = 50,
) -> Tuple[float, float, float, float, float, float, float]:
    kv_lens = list(decode_kv_lens) + list(prefill_kv_lens)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32)
    q_lens = torch.tensor(
        list(decode_qo_lens) + list(prefill_qo_lens), dtype=torch.int32
    )
    seq_lens_blocks = torch.ceil(seq_lens / page_block_size).int()

    q_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens, 0)], dim=0).int()
    kv_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0
    ).int()
    num_blocks = kv_indptr[-1].item()

    q = torch.rand(
        q_indptr[-1].item(), num_qo_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    kv_data = torch.randn(
        num_blocks,
        2,
        page_block_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    # old
    wrapper_old = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        kv_layout="NHD",
        backend="fa2",
    )
    last_page_len = (seq_lens - 1) % page_block_size + 1

    def old_plan():
        wrapper_old.plan(
            q_indptr.to(device),
            kv_indptr.to(device),
            torch.arange(num_blocks, dtype=torch.int32, device=device),
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_block_size,
            causal=causal,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )

    old_plan()  # warmup module loading
    start_time = time.perf_counter()
    old_plan()
    end_time = time.perf_counter()
    measurements_old = bench_gpu_time(
        lambda: wrapper_old.run(q, kv_data), repeat_iters=repeats
    )
    ms_old = np.mean(measurements_old) + (end_time - start_time) * 1000 / NUM_LAYERS

    # Fused kernel
    wrapper = flashinfer.BatchAttention(kv_layout="NHD")

    def persistent_plan():
        wrapper.plan(
            q_indptr.to(device),
            kv_indptr.to(device),
            torch.arange(num_blocks, dtype=torch.int32, device=device),
            seq_lens.to(device),
            num_qo_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            page_block_size,
            causal=causal,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )

    persistent_plan()  # warmup module loading
    start_time = time.perf_counter()
    persistent_plan()
    end_time = time.perf_counter()
    measurements_new_normal = bench_gpu_time(
        lambda: wrapper.run(q, kv_data),
        repeat_iters=repeats,
    )
    ms_new_normal = (
        np.mean(measurements_new_normal) + (end_time - start_time) * 1000 / NUM_LAYERS
    )
    o, _ = wrapper.run(q, kv_data)

    # Separate prefill and decode wrappers
    q_lens_d = torch.tensor(decode_qo_lens, dtype=torch.int32, device=device)
    q_indptr_d = torch.cat(
        [torch.tensor([0], device=device), torch.cumsum(q_lens_d, 0)], dim=0
    ).int()
    seq_lens_d = torch.tensor(decode_kv_lens, dtype=torch.int32)
    seq_lens_blocks_d = torch.ceil(seq_lens_d / page_block_size).int()
    kv_indptr_d = torch.cat(
        [torch.tensor([0]), torch.cumsum(seq_lens_blocks_d, 0)], dim=0
    ).int()
    num_blocks_d = kv_indptr_d[-1].item()
    if len(decode_qo_lens) > 0:
        wrapper_decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
            kv_layout="NHD",
            backend="fa2",
            use_tensor_cores=True,
        )

        last_page_len_d = (seq_lens_d - 1) % page_block_size + 1

        q_d = q[: q_indptr_d[-1].item()]
        kv_data_d = kv_data[:num_blocks_d]
        start_time_d = time.perf_counter()
        wrapper_decode.plan(
            kv_indptr_d.to(device),
            torch.arange(num_blocks_d, dtype=torch.int32, device=device),
            last_page_len_d.to(device),
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_block_size,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        end_time_d = time.perf_counter()
        measurements_decode = bench_gpu_time(
            lambda: wrapper_decode.run(q_d, kv_data_d), repeat_iters=repeats
        )
        ms_decode = (
            np.mean(measurements_decode)
            + (end_time_d - start_time_d) * 1000 / NUM_LAYERS
        )
    else:
        ms_decode = 0

    wrapper_prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        kv_layout="NHD",
        backend="fa2",
    )
    q_lens_p = torch.tensor(prefill_qo_lens, dtype=torch.int32, device=device)
    q_indptr_p = torch.cat(
        [torch.tensor([0], device=device), torch.cumsum(q_lens_p, 0)], dim=0
    ).int()
    seq_lens_p = torch.tensor(prefill_kv_lens, dtype=torch.int32, device=device)
    seq_lens_blocks_p = torch.ceil(seq_lens_p / page_block_size).int()
    kv_indptr_p = torch.cat(
        [torch.tensor([0], device=device), torch.cumsum(seq_lens_blocks_p, 0)],
        dim=0,
    ).int()
    num_blocks_p = kv_indptr_p[-1].item()
    q_p = q[q_indptr_d[-1].item() :]
    kv_data_p = kv_data[num_blocks_d:]
    last_page_len_p = (seq_lens_p - 1) % page_block_size + 1

    if len(prefill_qo_lens) > 0:
        start_time_p = time.perf_counter()
        wrapper_prefill.plan(
            q_indptr_p.to(device),
            kv_indptr_p.to(device),
            torch.arange(num_blocks_p, dtype=torch.int32, device=device),
            last_page_len_p.to(device),
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_block_size,
            causal=causal,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        end_time_p = time.perf_counter()
        measurements_prefill = bench_gpu_time(
            lambda: wrapper_prefill.run(q_p, kv_data_p), repeat_iters=repeats
        )
        ms_prefill = (
            np.mean(measurements_prefill)
            + (end_time_p - start_time_p) * 1000 / NUM_LAYERS
        )
    else:
        ms_prefill = 0

    ms_separate = ms_prefill + ms_decode

    total_bytes = (
        q.numel() * q.element_size() + kv_data.numel() * kv_data.element_size()
    )
    mem_MB = total_bytes / 1024**2
    bw_old = total_bytes / (ms_old * 1e-3) / 1024**3
    bw_new_normal = total_bytes / (ms_new_normal * 1e-3) / 1024**3
    bw_separate = total_bytes / (ms_separate * 1e-3) / 1024**3

    return (
        ms_old,
        ms_new_normal,
        ms_separate,
        mem_MB,
        bw_old,
        bw_new_normal,
        bw_separate,
    )  # type: ignore


def synthesize_seq_len_configs(
    decode_len, prefill_len, prefill_chunk_size, num_prefill_reqs, num_decode_reqs
) -> Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]:
    decode_lens: List[List[Tuple[int, int]]] = [
        [(decode_len, 1)] * num_decode_reqs,
    ]
    prefill_lens: List[List[Tuple[int, int]]] = [
        [(prefill_len, prefill_chunk_size)] * num_prefill_reqs,
    ]

    return decode_lens, prefill_lens


def main(args: argparse.Namespace) -> None:
    # If plotting mode, load existing data and plot
    if args.plot:
        if not os.path.exists("bench_batch_attention.csv"):
            print("Error: bench_batch_attention.csv not found. Run benchmark first.")
            return

        df_all = pd.read_csv("bench_batch_attention.csv")
        os.makedirs(save_dir, exist_ok=True)

        plot_original_comparison(df_all)

        return

    # Benchmark mode
    np.random.seed(42)
    torch.random.manual_seed(42)
    decode_len = args.decode_len
    prefill_len = args.prefill_len
    prefill_chunk_size = args.prefill_chunk_size
    num_prefill_reqs = args.num_prefill_reqs
    num_decode_reqs = args.num_decode_reqs

    decode_lens, prefill_lens = synthesize_seq_len_configs(
        decode_len, prefill_len, prefill_chunk_size, num_prefill_reqs, num_decode_reqs
    )
    if num_prefill_reqs == 0:
        prefill_chunk_size = 0
        prefill_len = 0
    if num_decode_reqs == 0:
        decode_len = 0

    combinations = [
        {
            "page_block_size": 1,
            "head_dim": 128,
            "num_kv_heads": 8,
            "num_qo_heads": 32,
            "model_name": "Qwen-8B",
        },
        {
            "page_block_size": 1,
            "head_dim": 128,
            "num_kv_heads": 8,
            "num_qo_heads": 64,
            "model_name": "Llama-3.1-70B",
        },
        {
            "page_block_size": 1,
            "head_dim": 64,
            "num_kv_heads": 4,
            "num_qo_heads": 64,
            "model_name": "Qwen-MoE-235B",
        },
    ]
    records_old = []
    records_new = []
    records_separate = []
    for cfg_id, (decode_case, prefill_case) in enumerate(
        zip(decode_lens, prefill_lens), start=1
    ):
        prefill_kv_lens = [p[0] for p in prefill_case]
        prefill_qo_lens = [p[1] for p in prefill_case]
        decode_kv_lens = [p[0] for p in decode_case]
        decode_qo_lens = [p[1] for p in decode_case]
        for param in combinations:
            pbs, hd, n_kv, n_qo, model_name = (
                param["page_block_size"],  # type: ignore
                param["head_dim"],  # type: ignore
                param["num_kv_heads"],  # type: ignore
                param["num_qo_heads"],  # type: ignore
                param["model_name"],
            )
            (
                ms_old,
                ms_new_normal,
                ms_separate,
                mem_MB,
                bw_old,
                bw_new_normal,
                bw_separate,
            ) = run_bench(
                decode_kv_lens,
                decode_qo_lens,
                prefill_kv_lens,
                prefill_qo_lens,
                page_block_size=pbs,  # type: ignore
                num_kv_heads=n_kv,  # type: ignore
                num_qo_heads=n_qo,  # type: ignore
                head_dim=hd,  # type: ignore
                device=0,
                causal=True,
                repeats=args.repeats,
            )
            records_old.extend(
                [
                    {
                        "scheduler": "BatchPrefillWithPagedKVCacheWrapper",
                        "seq_cfg_id": cfg_id,
                        "page_size": pbs,
                        "head_dim": hd,
                        "num_kv_heads": n_kv,
                        "num_qo_heads": n_qo,
                        "model_name": model_name,
                        "time_ms": ms_old,
                        "memory_MB": mem_MB,
                        "bandwidth_GB_s": bw_old,
                        "num_repeats": args.repeats,
                        "decode_len": decode_len,
                        "prefill_len": prefill_len,
                        "prefill_chunk_size": prefill_chunk_size,
                        "num_decode_reqs": num_decode_reqs,
                    },
                ]
            )
            records_new.extend(
                [
                    {
                        "scheduler": "BatchAttentionWrapper",
                        "seq_cfg_id": cfg_id,
                        "page_size": pbs,
                        "head_dim": hd,
                        "num_kv_heads": n_kv,
                        "num_qo_heads": n_qo,
                        "model_name": model_name,
                        "time_ms": ms_new_normal,
                        "memory_MB": mem_MB,
                        "bandwidth_GB_s": bw_new_normal,
                        "num_repeats": args.repeats,
                        "decode_len": decode_len,
                        "prefill_len": prefill_len,
                        "prefill_chunk_size": prefill_chunk_size,
                        "num_decode_reqs": num_decode_reqs,
                    },
                ]
            )
            records_separate.extend(
                [
                    {
                        "scheduler": "Decode + Prefill",
                        "seq_cfg_id": cfg_id,
                        "page_size": pbs,
                        "head_dim": hd,
                        "num_kv_heads": n_kv,
                        "num_qo_heads": n_qo,
                        "model_name": model_name,
                        "time_ms": ms_separate,
                        "memory_MB": mem_MB,
                        "bandwidth_GB_s": bw_separate,
                        "num_repeats": args.repeats,
                        "decode_len": decode_len,
                        "prefill_len": prefill_len,
                        "prefill_chunk_size": prefill_chunk_size,
                        "num_decode_reqs": num_decode_reqs,
                    },
                ]
            )
    df = pd.DataFrame(
        records_old + records_new + records_separate,
        columns=[
            "scheduler",
            "seq_cfg_id",
            "page_size",
            "head_dim",
            "num_kv_heads",
            "num_qo_heads",
            "model_name",
            "time_ms",
            "memory_MB",
            "bandwidth_GB_s",
            "num_repeats",
            "decode_len",
            "prefill_len",
            "prefill_chunk_size",
            "num_decode_reqs",
        ],
    )
    file_name = "bench_batch_attention.csv"
    if os.path.exists(file_name) and args.overwrite:
        os.remove(file_name)

    # Append if file exists; write header only on first write
    append_mode = "a" if os.path.exists(file_name) else "w"
    write_header = append_mode == "w"
    df.to_csv(file_name, index=False, mode=append_mode, header=write_header)

    # Only drop columns for printing, not for CSV
    df_print = df.drop(
        columns=[
            "page_size",
            "num_repeats",
            "decode_len",
            "prefill_len",
            "prefill_chunk_size",
            "num_decode_reqs",
        ]
    )
    print(df_print.to_markdown(index=False, floatfmt=".2f"))


if __name__ == "__main__":
    # Benchmark different attention schedulers and optionally plot results

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--prefill_chunk_size", type=int, default=4096)
    parser.add_argument("--num_prefill_reqs", type=int, default=1)
    parser.add_argument("--num_decode_reqs", type=int, default=128)
    parser.add_argument("--decode_len", type=int, default=8192)
    parser.add_argument("--prefill_len", type=int, default=8192)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot existing benchmark results instead of running benchmark",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing CSV file"
    )
    args = parser.parse_args()
    main(args)
