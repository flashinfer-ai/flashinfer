#!/usr/bin/env python3
"""
Benchmark selective_state_update (MTP mode) — % of Speed-of-Light (SOL).

Measures FlashInfer kernel achieved memory bandwidth as a percentage of the
GPU's peak HBM bandwidth. This is the right metric for memory-bound kernels.

Methodology follows benchmarks/routines/mamba.py: problem_bytes (read + write)
divided by kernel time gives achieved TB/s, then SOL% = achieved / peak * 100.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from flashinfer.testing import bench_gpu_time

# Add tests directory to path for create_test_inputs
sys.path.insert(0, str(Path(__file__).parent.parent / "tests" / "mamba"))

from utils import create_test_inputs, clone_preserving_strides
from flashinfer.mamba import selective_state_update as flashinfer_selective_state_update


# Peak HBM bandwidth in TB/s for known GPUs (bidirectional)
# Source: NVIDIA product specs
_PEAK_BW_TB_S = {
    "H100 SXM": 3.35,
    "H100 PCIe": 2.0,
    "H100 NVL": 3.35,
    "H200": 4.8,
    "A100 SXM": 2.0,
    "A100 PCIe": 1.555,
    "A100-SXM4-80GB": 2.0,
    "A100-SXM4-40GB": 1.555,
    "B200": 8.0,
    "B100": 8.0,
    "L40S": 0.864,
    "L40": 0.864,
    "A10": 0.6,
}

# Peak SIMT (non-tensor-core) FP32 throughput in TFLOPS
# Source: NVIDIA product specs — these are the CUDA core (SIMT) numbers,
# NOT tensor core numbers.
_PEAK_SIMT_FP32_TFLOPS = {
    "H100 SXM": 67.0,
    "H100 PCIe": 51.2,
    "H100 NVL": 67.0,
    "H200": 67.0,  # same die as H100
    "A100 SXM": 19.5,
    "A100 PCIe": 19.5,
    "A100-SXM4-80GB": 19.5,
    "A100-SXM4-40GB": 19.5,
    "B200": 90.0,
    "B100": 60.0,
    "L40S": 36.6,
    "L40": 36.6,
    "A10": 31.2,
}


def _lookup_gpu(table, gpu_name, override, unit_name):
    """Look up a GPU spec from a table, with optional override."""
    if override is not None:
        return override
    for key, val in table.items():
        if key.lower() in gpu_name.lower():
            return val
    raise ValueError(
        f"Unknown GPU '{gpu_name}'. Please specify the override flag. "
        f"Known GPUs: {list(table.keys())}"
    )


def get_peak_bandwidth_tb_s(gpu_name, override=None):
    """Return peak HBM bandwidth in TB/s."""
    return _lookup_gpu(_PEAK_BW_TB_S, gpu_name, override, "TB/s")


def get_peak_simt_fp32_tflops(gpu_name, override=None):
    """Return peak SIMT FP32 throughput in TFLOPS."""
    return _lookup_gpu(_PEAK_SIMT_FP32_TFLOPS, gpu_name, override, "TFLOPS")


def tensor_size_bytes(t):
    """Return the number of physical bytes backing tensor *t*.

    Dimensions with stride 0 are broadcasts — the actual storage is only 1
    element along that axis, so we count those dimensions as size 1.
    """
    n_elems = 1
    for size, stride in zip(t.shape, t.stride(), strict=True):
        n_elems *= size if stride != 0 else 1
    return n_elems * t.element_size()


def compute_problem_bytes(inputs):
    """Compute total bytes read + written from actual tensors.

    The kernel reads only the state_cache rows selected by slot_idx.
    When intermediate_states_buffer is present, the kernel writes intermediate
    states (indexed by intermediate_slot_idx) instead of writing back to
    state_cache.  Otherwise it writes state_cache back in-place.

    Read:  state_cache[slot_idx], x, dt, A, B, C, D, dt_bias,
           slot_idx, intermediate_slot_idx (if present), z (if present)
    Write: output (same shape as x),
           + intermediate_states_buffer[intermediate_slot_idx] if present,
           + state_cache[slot_idx] otherwise (written back in-place)
    """
    state_cache = inputs["state_cache"]
    slot_idx = inputs["slot_idx"]
    # Only the rows selected by slot_idx are accessed (not the full cache).
    state_accessed = state_cache[slot_idx]  # (batch_size, nheads, dim, dstate)
    state_read_bytes = tensor_size_bytes(state_accessed)

    read_bytes = state_read_bytes
    for k in ["x", "dt", "A", "B", "C", "D", "dt_bias", "slot_idx"]:
        read_bytes += tensor_size_bytes(inputs[k])
    if inputs.get("z") is not None:
        read_bytes += tensor_size_bytes(inputs["z"])
    if inputs.get("intermediate_slot_idx") is not None:
        read_bytes += tensor_size_bytes(inputs["intermediate_slot_idx"])

    write_bytes = tensor_size_bytes(inputs["x"])  # output (same shape/dtype as x)
    if inputs.get("intermediate_states_buffer") is not None:
        # Kernel writes to intermediate_states_buffer[intermediate_slot_idx],
        # not back to state_cache.
        istate = inputs["intermediate_states_buffer"]
        islot = inputs["intermediate_slot_idx"]
        write_bytes += tensor_size_bytes(istate[islot])
    else:
        # No intermediate buffer: state is written back in-place
        write_bytes += state_read_bytes

    return read_bytes + write_bytes


def compute_problem_flops(inputs):
    """Count FP32 FLOPs for the SSU kernel (SIMT, not tensor-core).

    Equations (per batch, step, head, dim_row):
      Pre-compute:
        dt_val  = dt + dt_bias                           1 add
        dt_val  = softplus(dt_val)                        3 ops (exp, add, log)
        dA      = exp(A * dt_val)                         1 mul + 1 exp = 2
        dtx     = dt_val * x                              1 mul
                                                   total: 7 per (B,T,H,D)

      State update (per dstate element):
        h = h * dA + B * dtx                              2 mul + 1 add = 3
                                                   total: 3 per (B,T,H,D,N)

      Output reduction:
        y += C[n] * h[n]  for n in dstate                 1 mul + 1 add = 2 per N
        y += D * x                                        1 mul + 1 add = 2
                                                   total: 2*N + 2 per (B,T,H,D)

      Optional gating:
        sig_z = 1 / (1 + exp(-z))                         3 (exp, add, div)
        y = y * z * sig_z                                 2 mul
                                                   total: 5 per (B,T,H,D)
    """
    x = inputs["x"]
    has_z = inputs.get("z") is not None

    # x shape: (batch, [T,] nheads, dim) — T dimension present in MTP mode
    if x.dim() == 4:
        batch_size, T_val, nheads, dim = x.shape
    else:
        batch_size, nheads, dim = x.shape
        T_val = 1

    state_cache = inputs["state_cache"]
    dstate = state_cache.shape[-1]

    outer = batch_size * T_val * nheads * dim  # (B, T, H, D) iterations

    flops_precompute = 7 * outer
    flops_state = 3 * outer * dstate
    flops_output = (2 * dstate + 2) * outer
    flops_gating = 5 * outer if has_z else 0

    return flops_precompute + flops_state + flops_output + flops_gating


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


def benchmark_kernel(
    name,
    kernel_fn,
    inputs,
    cache_steps=0,
    ncu=False,
    rand_seed=None,
    philox_rounds=10,
    repeat_time_ms=1000,
):
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
        rand_seed=rand_seed,
        philox_rounds=philox_rounds,
        disable_state_update=True,
    )

    if ncu:
        kernel_fn(**kwargs)
        torch.cuda.synchronize()
        print("    Single invocation done (ncu mode)")
        return 0.0

    try:
        measurements = bench_gpu_time(
            lambda: kernel_fn(**kwargs),
            dry_run_time_ms=100,
            repeat_time_ms=repeat_time_ms,
        )
    except RuntimeError as e:
        print(f"    Kernel failed: {e}")
        return float("inf")

    median_time = np.median(measurements)
    print(f"    Median time: {median_time:.3f} ms")
    return median_time


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
        rand_seed,
        philox_rounds,
        disable_state_update,
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
            rand_seed=rand_seed,
            philox_rounds=philox_rounds,
            disable_state_update=disable_state_update,
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
    philox_rounds=None,
    repeat_time_ms=1000,
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

    # Stochastic rounding: create rand_seed tensor when philox_rounds is set
    rand_seed = None
    effective_philox_rounds = 10
    if philox_rounds is not None:
        rand_seed = torch.tensor(42, dtype=torch.int64, device="cuda")
        effective_philox_rounds = philox_rounds

    kernels = {
        "flashinfer_simple": make_flashinfer_wrapper(algorithm="simple"),
        "flashinfer_vertical": make_flashinfer_wrapper(algorithm="vertical"),
        "flashinfer_horizontal": make_flashinfer_wrapper(algorithm="horizontal"),
        "flashinfer_auto": make_flashinfer_wrapper(algorithm="auto"),
    }

    results = {}
    for name, fn in kernels.items():
        median_time = benchmark_kernel(
            name,
            fn,
            inputs,
            cache_steps=cache_steps,
            ncu=ncu,
            rand_seed=rand_seed,
            philox_rounds=effective_philox_rounds,
            repeat_time_ms=repeat_time_ms,
        )
        results[name] = median_time

    return results, inputs


# -- Main --

parser = argparse.ArgumentParser(
    description="Benchmark selective_state_update — % of Speed-of-Light (SOL)"
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
parser.add_argument(
    "--dtype",
    type=str,
    nargs="+",
    default=["bf16", "f32"],
    help=(
        "State dtype(s) to benchmark (default: bf16 f32). "
        "Supported: bf16, f16, f32, or f16-philox-N for stochastic rounding "
        "with N philox rounds (e.g. f16-philox-5)"
    ),
)
parser.add_argument(
    "--dstate",
    type=int,
    default=128,
    help="State dimension (default: 128)",
)
parser.add_argument(
    "--mtp",
    type=int,
    default=6,
    help="Number of MTP (cache) steps (default: 6)",
)
parser.add_argument(
    "-r",
    "--repeat",
    type=int,
    default=1000,
    help="Repeat time in milliseconds for benchmarking (default: 1000)",
)
parser.add_argument(
    "--peak-bw",
    type=float,
    default=None,
    help="GPU peak HBM bandwidth in TB/s (auto-detected from GPU name if omitted)",
)
parser.add_argument(
    "--peak-flops",
    type=float,
    default=None,
    help="GPU peak SIMT FP32 throughput in TFLOPS (auto-detected from GPU name if omitted)",
)
args = parser.parse_args()

# Powers of two from 1 to 2048
batch_sizes = args.batch if args.batch is not None else [2**i for i in range(12)]

_dtype_name_to_torch = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32,
}


def parse_dtype_spec(spec):
    """Parse a dtype spec like 'bf16', 'f16', or 'f16-philox-5'.

    Returns (display_name, torch_dtype, philox_rounds_or_None).
    """
    m = re.match(r"^(bf16|f16|f32)-philox-(\d+)$", spec)
    if m:
        base, rounds = m.group(1), int(m.group(2))
        return spec, _dtype_name_to_torch[base], rounds
    if spec not in _dtype_name_to_torch:
        raise ValueError(
            f"Unknown dtype spec '{spec}'. "
            "Expected bf16, f16, f32, or <dtype>-philox-<rounds>"
        )
    return spec, _dtype_name_to_torch[spec], None


state_dtypes = [parse_dtype_spec(s) for s in args.dtype]
mtp_value = args.mtp

# Fixed kernel parameters (matching bench_ssu_sweep_mtp.py)
NHEADS = 64
DIM = 64
NGROUPS = 8

# Resolve peak specs
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown GPU"
peak_bw_tb_s = get_peak_bandwidth_tb_s(gpu_name, args.peak_bw)
peak_flops_tflops = get_peak_simt_fp32_tflops(gpu_name, args.peak_flops)

all_results = []

print("=" * 80)
print("COLLECTING BENCHMARK RESULTS — SOL% (MTP ENABLED)")
print("=" * 80)
print(f"GPU: {gpu_name}")
print(f"Peak HBM bandwidth: {peak_bw_tb_s:.2f} TB/s")
print(f"Peak SIMT FP32: {peak_flops_tflops:.1f} TFLOPS")
print(f"Batch sizes to test: {batch_sizes}")
print(f"State dtypes: {[name for name, _, _ in state_dtypes]}")
print(f"MTP (cache_steps): {mtp_value}, dstate: {args.dstate}")
if args.ncu:
    print("NCU mode: single invocation, no warmup/timing")
print("=" * 80)

for state_dtype_name, state_dtype_torch, philox_rounds in state_dtypes:
    for batch_size in batch_sizes:
        print(
            f"\n  Running benchmark for batch_size={batch_size}, "
            f"state_dtype={state_dtype_name}, mtp={mtp_value}, dstate={args.dstate}"
        )

        results, inputs = run_measurement(
            batch_size=batch_size,
            nheads=NHEADS,
            dim=DIM,
            ngroups=NGROUPS,
            dstate=args.dstate,
            state_dtype=state_dtype_torch,
            mtp=mtp_value,
            generate_intermediate_states_buffer=True,
            ncu=args.ncu,
            philox_rounds=philox_rounds,
            repeat_time_ms=args.repeat,
        )

        if not results:
            print(f"  Warning: No results returned for batch_size={batch_size}")
            continue

        problem_bytes = compute_problem_bytes(inputs)
        problem_flops = compute_problem_flops(inputs)

        # SOL time = memory time + compute time
        # Memory and compute are not overlapped in this kernel, so total
        # ideal time is the sum of both.
        sol_mem_time_ms = problem_bytes / (peak_bw_tb_s * 1e9)  # TB/s → bytes/ms
        sol_compute_time_ms = problem_flops / (
            peak_flops_tflops * 1e9
        )  # TFLOPS → FLOPs/ms
        sol_time_ms = sol_mem_time_ms + sol_compute_time_ms

        for kernel_name, median_time_ms in results.items():
            if median_time_ms > 0 and median_time_ms != float("inf"):
                sol_pct = sol_time_ms / median_time_ms * 100.0
            else:
                sol_pct = 0.0

            all_results.append(
                {
                    "batch_size": batch_size,
                    "state_dtype": state_dtype_name,
                    "kernel": kernel_name,
                    "avg_time_ms": median_time_ms,
                    "sol_pct": sol_pct,
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
    print(f"\nGPU: {gpu_name}")
    print(f"Peak HBM bandwidth: {peak_bw_tb_s:.2f} TB/s")
    print(f"Peak SIMT FP32: {peak_flops_tflops:.1f} TFLOPS")
    print(f"State dtypes: {[name for name, _, _ in state_dtypes]}")
    print(f"MTP (cache_steps): {mtp_value}")

    unique_dtypes = df["state_dtype"].unique()
    num_dtypes = len(unique_dtypes)
    fig, axes = plt.subplots(num_dtypes, 1, figsize=(10, 5 * num_dtypes), squeeze=False)

    for dtype_idx, dtype_name in enumerate(unique_dtypes):
        df_dtype = df[df["state_dtype"] == dtype_name]

        # Print time table
        df_time_pivot = df_dtype.pivot(
            index="batch_size", columns="kernel", values="avg_time_ms"
        )
        print(
            f"\nMedian time (ms) by batch size and kernel (state_dtype={dtype_name}):"
        )
        print(df_time_pivot.to_csv())

        # Print SOL% table
        df_sol_pivot = df_dtype.pivot(
            index="batch_size", columns="kernel", values="sol_pct"
        )
        print(f"\nSOL% by batch size and kernel (state_dtype={dtype_name}):")
        print(df_sol_pivot.to_csv())

        ax = axes[dtype_idx, 0]
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        default_colors = prop_cycle.by_key()["color"]

        df_plot = df_sol_pivot.reset_index()
        kernel_columns = [col for col in df_plot.columns if col != "batch_size"]

        x_positions = np.arange(len(df_plot["batch_size"]))
        x_tick_labels = [f"{x}" for x in df_plot["batch_size"].values]
        num_cols = len(kernel_columns)
        bar_width = 0.8 / max(num_cols, 1)

        for idx, col in enumerate(kernel_columns):
            sol_vals = df_plot[col]
            offset = (idx - num_cols / 2 + 0.5) * bar_width
            bars = ax.bar(
                x_positions + offset,
                sol_vals,
                bar_width,
                color=default_colors[idx % len(default_colors)],
                label=col,
                alpha=0.7,
            )
            for bar, y in zip(bars, sol_vals, strict=True):
                if y > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        y,
                        f"{y:.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=0,
                    )

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("% SOL")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_tick_labels)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=100, color="red", linestyle="--", alpha=0.5, label="100% SOL")
        dstate_subtitle = f", dstate={args.dstate}" if args.dstate != 128 else ""
        ax.set_title(
            f"State dtype: {dtype_name}, MTP={mtp_value}{dstate_subtitle} "
            f"— Peak BW: {peak_bw_tb_s:.2f} TB/s, Peak SIMT FP32: {peak_flops_tflops:.0f} TFLOPS"
        )
        ax.set_ylim([0, None])
        ax.legend(loc="best", fontsize=8)

    dstate_title = f", dstate={args.dstate}" if args.dstate != 128 else ""
    fig.suptitle(
        f"SSU % of Speed-of-Light (MTP={mtp_value}{dstate_title}) [{gpu_name}]",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        gpu_name_clean = gpu_name.replace(" ", "_").replace("/", "_")
        dtype_str = "_".join(name for name, _, _ in state_dtypes)
        output_filename = (
            f"sol_vs_batch_size_mtp{mtp_value}_{dtype_str}_{gpu_name_clean}.png"
        )
        img_dir = Path(__file__).parent / "img"
        img_dir.mkdir(exist_ok=True)
        output_path = img_dir / output_filename
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to: {output_path}")
