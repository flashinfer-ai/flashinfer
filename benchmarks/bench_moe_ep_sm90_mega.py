"""SM90 (Hopper) pull-style FP8 mega-MoE token-sweep benchmark.

Reproduces the kernel drop's Hopper P03 multirank token sweep
(``moe_hopper_fp8/run_token_sweep_benchmark.py`` at kernel commit ``1275b8b``)
through the FlashInfer ``MoEEpLayer`` mega path, so results are directly
comparable with the drop's reference CSVs
(``moe_hopper_fp8/benchmark_data/20260720/
20260720_multirank_{pertensor|blockwise}_{nonswapab|swapab}_TileM{M}_TileN{N}.csv``
— same 4xH100 hardware, same vendored kernel).

Geometry defaults (the drop's DSV4 P03 case; all are CLI flags):
tokens/rank sweep 512..32768 (powers of two), topk=6, 384 total experts
(EP4 -> 96 local), hidden=7168, intermediate=3072 (FI post-SwiGLU convention;
the drop's ``INTERMEDIATE_GATEUP=6144`` is 2x), gate_up_clamp=10.0,
kind=fp8_e4m3, 1xacc, load_balance_mode=atomic_counter and
token-back=reuse_dispatch_warps (both the drop's P03 perf-run settings),
warmup=3, iters=20, tile K=128.

Default tiles per layout (== the shim's per-layout defaults):
  * non_swap_ab: M64 N128  -> compare against ``..._nonswapab_TileM64_TileN128.csv``
  * swap_ab:     M256 N32  -> compare against ``..._swapab_TileM256_TileN32.csv``

Two timed series per point, CUDA events per rank around each call:
  * ``e2e``     — ``MoEEpLayer.forward`` (validation + bf16->fp8 staging +
    kernel + output copy).  This is the FI production path; it has NO drop
    counterpart column (the drop times the bare kernel launch).
  * ``compute`` — the backend's supported plugin API (``stage_inputs`` once,
    then repeated ``MegaKernelBackend.compute(output=None)``: bare fused
    launch + in-kernel/standalone top-k reduce, zero-copy output).  This is
    the closest FI analogue of the drop's per-rank ``mega_us`` + ``topk_us``
    (its ``reported_min_total_us``); the drop's ``*_mega_us`` columns exclude
    the standalone TopkReduce, so expect FI ``compute`` ~= drop ``mega + topk``.
  (``MoEEpMegaLayer`` has no per-stage timing hook — ``enable_timing`` /
  ``last_timings_ms`` are split-layer only — so the compute series drives the
  documented ``MegaKernelBackend`` API directly; no private internals.)

Launch (one process per GPU, 4xH100; srun+torchrun safe, no interactivity):

    torchrun --nproc_per_node=4 benchmarks/bench_moe_ep_sm90_mega.py

Rank 0 prints one ``BENCH_CSV`` row per (scale_mode, layout, tokens) point
(header once), each carrying the matching drop reference CSV filename.  A
point that OOMs prints a SKIP row and the sweep continues.  Between points
the layer/session and symmetric-heap buffers are destroyed before the next
allocation (the 32768-token workspace needs the heap to itself: the combine
plane alone is ~2.7 GB).
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from dataclasses import dataclass
from statistics import fmean, median

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here]

# Drop parity (run_perf_test.sh): multirank Hopper needs NVLS off unless the
# environment has a working NCCL/NVSHMEM NVLS setup. setdefault so users can
# override.
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
os.environ.setdefault("NVSHMEM_DISABLE_NVLS", "1")

DEFAULT_TOKENS = (512, 1024, 2048, 4096, 8192, 16384, 32768)
E4M3_MAX = 448.0
# Static per-tensor calibration scalars (identical on every EP rank by the
# kernel's dequant contract) — same derivation as the multirank parity test:
# randn bf16 activations and 1/sqrt(K)-normalized weights keep |x| and the
# SwiGLU outputs within 8, with the reference's 0.95 headroom margin.
FC1_ACT_SCALE = 8.0 / (0.95 * E4M3_MAX)
FC2_ACT_SCALE = 8.0 / (0.95 * E4M3_MAX)

# Shim per-layout default tiles (K fixed at 128 = Fp8DispatchScaleAtomK), and
# the drop reference CSV each default maps to (see module docstring).
DEFAULT_TILE = {"non_swap_ab": (64, 128), "swap_ab": (256, 32)}
REF_DATE = "20260720"  # Vincent's reference run under benchmark_data/<date>/

CSV_HEADER = (
    "BENCH_CSV,kernel,scale_mode,operand_order,tile_m,tile_n,tile_k,"
    "tokens_per_rank,topk,world_size,total_experts,local_experts,hidden,"
    "intermediate_downproj,intermediate_gateup,warmup,iters,status,"
    "e2e_min_us,e2e_max_us,e2e_mean_us,e2e_median_us,"
    "compute_min_us,compute_max_us,compute_mean_us,compute_median_us,"
    "fc1_flops_per_rank,fc2_flops_per_rank,total_flops_per_rank,"
    "critical_tflops_compute,critical_tflops_e2e,tok_s_e2e,ref_csv"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--tokens",
        type=str,
        default=",".join(str(t) for t in DEFAULT_TOKENS),
        help="comma-separated tokens-per-rank sweep",
    )
    p.add_argument(
        "--scale-mode",
        choices=["per_tensor", "blockwise", "both"],
        default="both",
        help="FP8 scale ABI(s) to sweep",
    )
    order = p.add_mutually_exclusive_group()
    order.add_argument(
        "--swap-ab",
        dest="operand_order",
        action="store_const",
        const="swap_ab",
        help="swap-AB layout only",
    )
    order.add_argument(
        "--no-swap-ab",
        dest="operand_order",
        action="store_const",
        const="non_swap_ab",
        help="native (non-swap) layout only",
    )
    p.set_defaults(operand_order="both")
    p.add_argument(
        "--mma-tiler",
        type=str,
        default=None,
        metavar="M,N",
        help="override the mma tile (M,N; K fixed at 128). Default: the "
        "shim's per-layout default (non-swap 64,128 / swap-AB 256,32).",
    )
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--num-experts", type=int, default=384)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument(
        "--intermediate",
        type=int,
        default=3072,
        help="post-SwiGLU (downproj) width; gate+up is 2x (drop's 6144)",
    )
    p.add_argument("--gate-up-clamp", type=float, default=10.0)
    p.add_argument("--kind", choices=["fp8_e4m3", "fp8_e5m2"], default="fp8_e4m3")
    p.add_argument(
        "--fp8-accum-mode", choices=["1xacc", "2xacc"], default="1xacc"
    )
    p.add_argument(
        "--load-balance-mode",
        choices=["static", "atomic_counter"],
        default="atomic_counter",
        help="atomic_counter matches the drop's perf-run setting",
    )
    p.add_argument(
        "--token-back",
        choices=["reuse_dispatch_warps", "epi_warps"],
        default="reuse_dispatch_warps",
        help="fc2 token-back path. reuse_dispatch_warps matches the drop's "
        "P03 perf runs (mega_runner non-ikr default); epi_warps is the "
        "FI-multirank-validated default — fall back to it if the dispatch-"
        "warp path misbehaves.",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    return p.parse_args()


def _flops_per_rank(tokens_per_rank: int, topk: int, hidden: int, inter: int):
    """Drop FLOP formula (run_token_sweep_benchmark.compute_gemm_flops_per_rank)."""
    routed = tokens_per_rank * topk
    gateup = 2 * inter
    fc1 = 2 * routed * hidden * gateup
    fc2 = 2 * routed * hidden * inter
    return fc1, fc2, fc1 + fc2


def _tflops(flops: int, time_us: float) -> float:
    return flops / time_us / 1e6 if time_us > 0 else float("nan")


@dataclass
class PointResult:
    status: str  # "pass" | "skip_oom" | "failed"
    e2e_us: list[float]  # cross-rank per-rank mean e2e us (len == world)
    e2e_median_us: list[float]
    compute_us: list[float]
    compute_median_us: list[float]
    error: str = ""


def _balanced_routing(
    num_tokens: int,
    topk: int,
    num_experts: int,
    rank: int,
    world_size: int,
    device,
):
    """Deterministic balanced routing (the drop's 'balanced' distribution spirit).

    Consecutive experts per row (distinct within a row since topk <
    num_experts), globally even per-expert counts, rotated per rank so every
    rank exercises cross-rank dispatch to every peer.
    """
    import torch

    flat = torch.arange(num_tokens * topk, device=device, dtype=torch.int64)
    ids = (flat + rank * (num_experts // world_size)) % num_experts
    return ids.view(num_tokens, topk)


def _make_point_inputs(args, tokens: int, rank: int, world_size: int, device):
    import torch

    g = torch.Generator(device="cuda").manual_seed(42 + rank)
    hidden_states = torch.randn(
        tokens, args.hidden, dtype=torch.bfloat16, device=device, generator=g
    )
    topk_ids = _balanced_routing(
        tokens, args.top_k, args.num_experts, rank, world_size, device
    )
    topk_weights = torch.softmax(
        torch.randn(tokens, args.top_k, device=device, generator=g), dim=-1
    )
    return hidden_states, topk_ids, topk_weights.to(torch.float32)


def _make_transformed_weights(args, scale_mode: str, local_experts: int, rank, device):
    """bf16 pack -> kernel-ready FP8 tuples, releasing the bf16 source."""
    import torch

    from flashinfer.moe_ep import preprocess_sm90_pull_fp8_mega_weights
    from flashinfer.moe_ep.weights import MoEWeightPack

    g = torch.Generator(device="cuda").manual_seed(13 + rank)
    # 1/sqrt(K) normalization keeps the fp8 dynamic range sane for the static
    # per-tensor calibration above (perf benchmark: shapes/dtypes are what
    # matter, but everything stays finite / unsaturated).
    w13 = torch.randn(
        local_experts,
        2 * args.intermediate,
        args.hidden,
        dtype=torch.bfloat16,
        device=device,
        generator=g,
    ) * (args.hidden**-0.5)
    w2 = torch.randn(
        local_experts,
        args.hidden,
        args.intermediate,
        dtype=torch.bfloat16,
        device=device,
        generator=g,
    ) * (args.intermediate**-0.5)
    transformed = preprocess_sm90_pull_fp8_mega_weights(
        MoEWeightPack(w13=w13, w2=w2),
        intermediate_size=args.intermediate,
        hidden_size=args.hidden,
        kind=args.kind,
        fp8_scale_mode=scale_mode,
        fc1_activation_dequant_scale=FC1_ACT_SCALE,
        fc2_activation_dequant_scale=FC2_ACT_SCALE,
    )
    del w13, w2  # release the bf16 source before the big workspaces come up
    return transformed


def _megakernel_config(args, scale_mode: str, operand_order: str, tile):
    from flashinfer.moe_ep import Sm90PullFp8MegaMoeConfig

    swap_ab = operand_order == "swap_ab"
    return Sm90PullFp8MegaMoeConfig(
        intermediate_size=args.intermediate,
        top_k=args.top_k,
        kind=args.kind,
        fp8_scale_mode=scale_mode,
        fp8_accum_mode=args.fp8_accum_mode,
        swap_ab=swap_ab,
        mma_tiler_mnk=(tile[0], tile[1], 128),
        load_balance_mode=args.load_balance_mode,
        gate_up_clamp=args.gate_up_clamp,
        in_kernel_fc2_reduce=False,
        token_back_by_dispatch=(args.token_back == "reuse_dispatch_warps"),
        fc1_activation_dequant_scale=FC1_ACT_SCALE,
        fc2_activation_dequant_scale=FC2_ACT_SCALE,
    )


def _time_calls(call, *, warmup: int, iters: int) -> list[float]:
    """Per-rank CUDA-event timings (us) of ``call``, barrier-aligned per iter.

    Mirrors bench_moe_ep's discipline (barrier + sync fencing each sample) so
    per-rank numbers are comparable with the drop's per-rank profiler means.
    """
    import torch
    import torch.distributed as dist

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    dist.barrier()

    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        dist.barrier()
        torch.cuda.synchronize()
        start.record()
        call()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop) * 1e3)  # ms -> us
    return samples


def _is_oom(exc: BaseException) -> bool:
    import torch

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "oom" in msg or "nvshmem_malloc" in msg


def _run_point(args, scale_mode: str, operand_order: str, tile, tokens: int) -> PointResult:
    """One (scale_mode, layout, tokens) point: build layer, time e2e + compute.

    Collective status agreement after the fallible phase keeps ranks in
    lockstep when one OOMs (best-effort: a failure inside a symmetric-heap
    collective typically raises on all ranks together).
    """
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    local_experts = args.num_experts // world_size

    layer = None
    bench_backend = None
    bench_workspace = None
    result: PointResult | None = None
    error = ""
    try:
        kcfg = _megakernel_config(args, scale_mode, operand_order, tile)
        transformed = _make_transformed_weights(
            args, scale_mode, local_experts, rank, device
        )
        hidden_states, topk_ids, topk_weights = _make_point_inputs(
            args, tokens, rank, world_size, device
        )
        fleet_params = FleetParams(
            num_experts=args.num_experts,
            max_tokens_per_rank=tokens,
            token_hidden_size=args.hidden,
        )
        bootstrap = BootstrapConfig(
            world_size=world_size, rank=rank, auto_bootstrap=False
        )
        # Weights are preprocessed once above and shared by both series
        # (transformed_weights path: the layer never touches the bf16 pack).
        layer = MoEEpLayer(
            bootstrap=bootstrap,
            fleet_params=fleet_params,
            weights=None,
            backend=MegaConfig(
                megakernel=kcfg,
                quantize_input=True,
                preprocess_weights=False,
                transformed_weights=transformed,
            ),
        )
        t = MoEEpTensors(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

        # --- series 1: FI e2e (validate + stage + kernel + output copy) ---
        e2e = _time_calls(
            lambda: layer.forward(t), warmup=args.warmup, iters=args.iters
        )

        # --- series 2: compute-only via the documented backend plugin API.
        # prepare_workspace hits the process pool with the same key as the
        # layer's, so this shares the layer's symm buffer AND compiled kernel
        # (no second compile / allocation). Stage once, then time bare
        # compute(output=None) launches (zero-copy output) — the closest FI
        # analogue of the drop's mega+topk timed region.
        bench_backend = create_mega_kernel(kcfg)
        bench_backend.bind_ep_bootstrap(bootstrap)
        bench_workspace = bench_backend.prepare_workspace(bootstrap, fleet_params)
        bench_backend.stage_inputs(t, bench_workspace, quantize_input=True)
        compute = _time_calls(
            lambda: bench_backend.compute(
                bench_workspace, transformed, output=None
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        torch.cuda.synchronize()

        my_stats = (
            "pass",
            fmean(e2e),
            median(e2e),
            fmean(compute),
            median(compute),
            "",
        )
    except Exception as exc:  # noqa: BLE001 - sweep must survive one bad point
        status = "skip_oom" if _is_oom(exc) else "failed"
        error = f"{type(exc).__name__}: {exc}"
        my_stats = (status, float("nan"), float("nan"), float("nan"), float("nan"), error)
    finally:
        # Free THIS point's session before the next allocation (the 32k-token
        # workspace needs the symmetric heap to itself). Backend release first
        # (drops the pool refcount), then the layer's (last release frees).
        if bench_backend is not None and bench_workspace is not None:
            try:
                bench_backend.destroy(bench_workspace)
            except Exception:  # noqa: BLE001
                pass
        if layer is not None:
            try:
                layer.destroy()
            except Exception:  # noqa: BLE001
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Collective agreement: gather every rank's stats; any non-pass rank turns
    # the whole point into a SKIP row so the sweep stays in lockstep.
    all_stats: list = [None] * world_size
    dist.all_gather_object(all_stats, my_stats)
    dist.barrier()

    statuses = [s[0] for s in all_stats]
    if all(s == "pass" for s in statuses):
        result = PointResult(
            status="pass",
            e2e_us=[s[1] for s in all_stats],
            e2e_median_us=[s[2] for s in all_stats],
            compute_us=[s[3] for s in all_stats],
            compute_median_us=[s[4] for s in all_stats],
        )
    else:
        worst = "skip_oom" if "skip_oom" in statuses else "failed"
        errors = "; ".join(
            f"rank{i}:{s[5]}" for i, s in enumerate(all_stats) if s[5]
        )
        result = PointResult(
            status=worst,
            e2e_us=[],
            e2e_median_us=[],
            compute_us=[],
            compute_median_us=[],
            error=errors,
        )
    return result


def _ref_csv_name(scale_mode: str, operand_order: str, tile) -> str:
    scale_tag = "pertensor" if scale_mode == "per_tensor" else "blockwise"
    order_tag = "swapab" if operand_order == "swap_ab" else "nonswapab"
    return (
        f"{REF_DATE}_multirank_{scale_tag}_{order_tag}_"
        f"TileM{tile[0]}_TileN{tile[1]}.csv"
    )


def _emit_row(
    args,
    *,
    scale_mode: str,
    operand_order: str,
    tile,
    tokens: int,
    world_size: int,
    result: PointResult,
    header_done: bool,
) -> None:
    fc1, fc2, total = _flops_per_rank(tokens, args.top_k, args.hidden, args.intermediate)
    ref_csv = _ref_csv_name(scale_mode, operand_order, tile)
    if not header_done:
        print(CSV_HEADER, flush=True)
    if result.status != "pass":
        print(
            f"BENCH_CSV,sm90_pull_fp8,{scale_mode},{operand_order},{tile[0]},{tile[1]},128,"
            f"{tokens},{args.top_k},{world_size},{args.num_experts},"
            f"{args.num_experts // world_size},{args.hidden},{args.intermediate},"
            f"{2 * args.intermediate},{args.warmup},{args.iters},{result.status},"
            + ",".join(["nan"] * 8)
            + f",{fc1},{fc2},{total},nan,nan,nan,{ref_csv}",
            flush=True,
        )
        if result.error:
            print(f"# SKIP detail: {result.error}", flush=True)
        return

    e2e_min, e2e_max, e2e_mean = (
        min(result.e2e_us),
        max(result.e2e_us),
        fmean(result.e2e_us),
    )
    c_min, c_max, c_mean = (
        min(result.compute_us),
        max(result.compute_us),
        fmean(result.compute_us),
    )
    e2e_med = fmean(result.e2e_median_us)
    c_med = fmean(result.compute_median_us)
    # Critical-path conventions: TFLOPS over the SLOWEST rank (the drop's
    # critical_tflops_per_rank = total_flops / max_mega_us), tok/s over the
    # slowest rank's e2e.
    tflops_c = _tflops(total, c_max)
    tflops_e2e = _tflops(total, e2e_max)
    tok_s = tokens * world_size / (e2e_max * 1e-6)
    print(
        f"BENCH_CSV,sm90_pull_fp8,{scale_mode},{operand_order},{tile[0]},{tile[1]},128,"
        f"{tokens},{args.top_k},{world_size},{args.num_experts},"
        f"{args.num_experts // world_size},{args.hidden},{args.intermediate},"
        f"{2 * args.intermediate},{args.warmup},{args.iters},pass,"
        f"{e2e_min:.2f},{e2e_max:.2f},{e2e_mean:.2f},{e2e_med:.2f},"
        f"{c_min:.2f},{c_max:.2f},{c_mean:.2f},{c_med:.2f},"
        f"{fc1},{fc2},{total},{tflops_c:.2f},{tflops_e2e:.2f},{tok_s:.1f},{ref_csv}",
        flush=True,
    )


def main() -> int:
    args = _parse_args()

    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.core.runtime import sm90_pull_fp8_runtime_requirements

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    if args.num_experts % world_size != 0:
        raise SystemExit(
            f"--num-experts ({args.num_experts}) must be divisible by the "
            f"torchrun world size ({world_size})"
        )
    if world_size != 4 and rank == 0:
        print(
            f"# note: world_size={world_size}; the drop reference CSVs are "
            "EP4 (4xH100) — numbers are only directly comparable at 4 ranks.",
            flush=True,
        )

    tokens_list = [int(t) for t in args.tokens.split(",") if t]
    scale_modes = (
        ("per_tensor", "blockwise")
        if args.scale_mode == "both"
        else (args.scale_mode,)
    )
    orders = (
        ("non_swap_ab", "swap_ab")
        if args.operand_order == "both"
        else (args.operand_order,)
    )
    tile_override = None
    if args.mma_tiler is not None:
        m, n = (int(v) for v in args.mma_tiler.split(","))
        tile_override = (m, n)

    # One NVSHMEM bootstrap for the whole sweep (layers run with
    # auto_bootstrap=False against this shared runtime).
    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, sm90_pull_fp8_runtime_requirements(bootstrap)
    )

    header_done = False
    try:
        for scale_mode in scale_modes:
            for operand_order in orders:
                tile = tile_override or DEFAULT_TILE[operand_order]
                for tokens in tokens_list:
                    if rank == 0:
                        print(
                            f"# [sweep] {scale_mode} {operand_order} "
                            f"TileM{tile[0]}N{tile[1]} tokens_per_rank={tokens}",
                            flush=True,
                        )
                    result = _run_point(args, scale_mode, operand_order, tile, tokens)
                    if rank == 0:
                        _emit_row(
                            args,
                            scale_mode=scale_mode,
                            operand_order=operand_order,
                            tile=tile,
                            tokens=tokens,
                            world_size=world_size,
                            result=result,
                            header_done=header_done,
                        )
                        header_done = True
    finally:
        finalize_moe_ep_runtime(runtime)
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
