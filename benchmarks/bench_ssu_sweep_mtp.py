#!/usr/bin/env python3
"""
Benchmark selective_state_update (MTP mode) across different batch sizes.

Compares FlashInfer's CUDA kernel against the Triton reference implementation.
Runs with fixed state dtype (bf16) and mtp=6, collecting results into a pandas dataframe.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from flashinfer.testing import bench_gpu_time

# Add tests directory to path for create_test_inputs and triton reference
sys.path.insert(0, str(Path(__file__).parent.parent / "tests" / "mamba"))

from utils import create_test_inputs, clone_preserving_strides
from triton_reference.selective_state_update import selective_state_update_triton
from flashinfer.mamba import selective_state_update as flashinfer_selective_state_update


def create_benchmark_inputs(
    batch_size,
    nheads,
    dim,
    dstate,
    ngroups,
    state_dtype,
    device="cuda",
    mtp=0,
    generate_intermediate_states_buffer=False,
):
    """Create test inputs for benchmarking."""
    cache_steps = None if mtp == 0 else mtp
    return create_test_inputs(
        batch_size=batch_size,
        nheads=nheads,
        dim=dim,
        dstate=dstate,
        ngroups=ngroups,
        input_dtype=torch.bfloat16,
        weight_dtype=torch.float32,
        state_dtype=state_dtype,
        matrixA_dtype=torch.float32,
        generate_z=False,
        generate_intermediate_states_buffer=generate_intermediate_states_buffer,
        cache_steps=cache_steps,
        device=device,
        seed=0,
    )


def benchmark_kernel(name, kernel_fn, inputs, cache_steps=0, ncu=False):
    """Benchmark a single kernel and return median time in ms."""
    print(f"\n  Benchmarking {name}...")

    state = clone_preserving_strides(inputs["state_cache"])
    out = torch.empty_like(inputs["x"])

    intermediate_states_buffer = inputs.get("intermediate_states_buffer", None)
    intermediate_slot_idx = inputs.get("intermediate_slot_idx", None)

    kwargs = dict(
        state=state,
        x=inputs["x"],
        dt=inputs["dt"],
        A=inputs["A"],
        B=inputs["B"],
        C=inputs["C"],
        D=inputs["D"],
        z=None,
        dt_bias=inputs["dt_bias"],
        dt_softplus=True,
        state_batch_indices=inputs["slot_idx"],
        pad_slot_id=-1,
        out=out,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_slot_idx,
        cache_steps=cache_steps,
    )

    if ncu:
        kernel_fn(**kwargs)
        torch.cuda.synchronize()
        print(f"    Single invocation done (ncu mode)")
        return 0.0

    try:
        measurements = bench_gpu_time(
            lambda: kernel_fn(**kwargs),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
        )
    except RuntimeError as e:
        print(f"    Kernel failed: {e}")
        return float("inf")

    median_time = np.median(measurements)
    print(f"    Median time: {median_time:.3f} ms")
    return median_time


def make_triton_wrapper():
    """Wrap the triton reference to match the common benchmark interface."""

    def wrapper(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        dt_bias,
        dt_softplus,
        state_batch_indices,
        pad_slot_id,
        out,
        intermediate_states_buffer,
        intermediate_state_indices,
        cache_steps,
    ):
        selective_state_update_triton(
            state=state,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            pad_slot_id=pad_slot_id,
            out=out,
            intermediate_states_buffer=intermediate_states_buffer,
            cache_steps=cache_steps,
            intermediate_state_indices=intermediate_state_indices,
        )

    return wrapper


def make_flashinfer_wrapper(algorithm="auto"):
    """Wrap FlashInfer's selective_state_update to match the common benchmark interface."""

    def wrapper(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        dt_bias,
        dt_softplus,
        state_batch_indices,
        pad_slot_id,
        out,
        intermediate_states_buffer,
        intermediate_state_indices,
        cache_steps,
    ):
        flashinfer_selective_state_update(
            state=state,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            pad_slot_id=pad_slot_id,
            out=out,
            intermediate_states_buffer=intermediate_states_buffer,
            cache_steps=cache_steps,
            intermediate_state_indices=intermediate_state_indices,
            algorithm=algorithm,
        )

    return wrapper


def run_measurement(
    batch_size,
    nheads,
    dim,
    ngroups,
    dstate,
    state_dtype,
    mtp=0,
    generate_intermediate_states_buffer=False,
    ncu=False,
):
    """Run benchmarks on all kernels and return results dict."""
    inputs = create_benchmark_inputs(
        batch_size=batch_size,
        nheads=nheads,
        dim=dim,
        dstate=dstate,
        ngroups=ngroups,
        state_dtype=state_dtype,
        mtp=mtp,
        generate_intermediate_states_buffer=generate_intermediate_states_buffer,
    )

    cache_steps = inputs.get("cache_steps", 0)

    kernels = {
        "triton_reference": make_triton_wrapper(),
        "flashinfer_simple": make_flashinfer_wrapper(algorithm="simple"),
        "flashinfer_vertical": make_flashinfer_wrapper(algorithm="vertical"),
    }

    results = {}
    for name, fn in kernels.items():
        median_time = benchmark_kernel(
            name, fn, inputs, cache_steps=cache_steps, ncu=ncu
        )
        results[name] = median_time

    return results


# -- Main --

parser = argparse.ArgumentParser(
    description="Benchmark selective_state_update MTP mode"
)
parser.add_argument(
    "--ncu",
    action="store_true",
    help="NCU profiling mode: single invocation per kernel, no warmup or timing",
)
parser.add_argument(
    "-b",
    "--batch",
    type=int,
    nargs="+",
    default=None,
    help="Batch size(s) to benchmark (default: powers of 2 from 1 to 2048)",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=None,
    help="Output image file path (default: auto-generated in benchmarks/img/)",
)
args = parser.parse_args()

# Powers of two from 1 to 2048
batch_sizes = args.batch if args.batch is not None else [2**i for i in range(12)]

state_dtype_name = "bf16"
state_dtype_torch = torch.bfloat16
mtp_value = 6

all_results = []

print("=" * 80)
print("COLLECTING BENCHMARK RESULTS (MTP ENABLED)")
print("=" * 80)
print(f"Batch sizes to test: {batch_sizes}")
print(f"State dtype: {state_dtype_name}")
print(f"MTP (cache_steps): {mtp_value}")
if args.ncu:
    print("NCU mode: single invocation, no warmup/timing")
print("=" * 80)

for batch_size in batch_sizes:
    print(f"\n  Running benchmark for batch_size={batch_size}, mtp={mtp_value}")

    results = run_measurement(
        batch_size=batch_size,
        nheads=64,
        dim=64,
        ngroups=8,
        dstate=128,
        state_dtype=state_dtype_torch,
        mtp=mtp_value,
        generate_intermediate_states_buffer=True,
        ncu=args.ncu,
    )

    if not results:
        print(f"  Warning: No results returned for batch_size={batch_size}")
        continue

    for kernel_name, avg_time in results.items():
        all_results.append(
            {
                "batch_size": batch_size,
                "kernel": kernel_name,
                "avg_time_ms": avg_time,
            }
        )

# Create DataFrame
df = pd.DataFrame(all_results)

print("\n" + "=" * 80)
print("BENCHMARK RESULTS SUMMARY")
print("=" * 80)

if args.ncu or df.empty:
    if df.empty:
        print("No results collected!")
    sys.exit(0)
else:
    gpu_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown GPU"
    )
    print(f"\nGPU: {gpu_name}")
    print(f"State dtype: {state_dtype_name}")
    print(f"MTP (cache_steps): {mtp_value}")

    df_pivot = df.pivot(index="batch_size", columns="kernel", values="avg_time_ms")

    print(f"\nAverage execution time (ms) by batch size and kernel:")
    print(df_pivot.to_csv())

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    baseline_col = "triton_reference"

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    default_colors = prop_cycle.by_key()["color"]

    df_plot = df_pivot.reset_index()

    speedup_columns = [
        col for col in df_plot.columns if col != "batch_size" and col != baseline_col
    ]

    x_ticks = df_plot["batch_size"].values
    x_tick_labels = [f"{x}" for x in x_ticks]

    num_speedup_cols = len(speedup_columns)
    x_positions = np.arange(len(df_plot["batch_size"]))
    bar_width = 0.8 / max(num_speedup_cols, 1)

    for idx, col in enumerate(speedup_columns):
        if baseline_col not in df_plot.columns or col not in df_plot.columns:
            continue
        speedup = df_plot[baseline_col] / df_plot[col]
        offset = (idx - num_speedup_cols / 2 + 0.5) * bar_width
        bars = ax.bar(
            x_positions + offset,
            speedup,
            bar_width,
            color=default_colors[idx % len(default_colors)],
            label=col,
            alpha=0.7,
        )
        for bar, y in zip(bars, speedup):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{y:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Speedup factor")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_tick_labels)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title(
        rf"State dtype: {state_dtype_name}, MTP={mtp_value} — Speedup = Runtime$_{{triton}}$ / Runtime$_{{flashinfer}}$"
    )
    ax.set_ylim([0, None])
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"Selective State Update Benchmark (MTP={mtp_value}) [{gpu_name}]",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        gpu_name_clean = gpu_name.replace(" ", "_").replace("/", "_")
        output_filename = f"runtime_vs_batch_size_mtp{mtp_value}_{gpu_name_clean}.png"
        img_dir = Path(__file__).parent / "img"
        img_dir.mkdir(exist_ok=True)
        output_path = img_dir / output_filename
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to: {output_path}")
