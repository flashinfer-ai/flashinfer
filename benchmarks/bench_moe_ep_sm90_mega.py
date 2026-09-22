"""SM90 (Hopper) pull-style FP8/MXFP4 mega-MoE token-sweep benchmark.

Reproduces the kernel drop's Hopper P03 multirank token sweep
(``moe_hopper_fp8/run_token_sweep_benchmark.py``) through the FlashInfer
``MoEEpLayer`` mega path, so results are directly comparable with the
drop's reference CSVs.  By default each point uses the drop's token-bucket
heuristic launch config, the drop's block-permutation balanced routing and
perf data recipe, and a short pre-series cooldown — see TUNING.md for the
methodology.  Fixed-layout runs
(``--both-orders`` / ``--swap-ab`` / ``--no-swap-ab``) map to the drop's
``20260720_multirank_{pertensor|blockwise}_{nonswapab|swapab}_TileM{M}_TileN{N}.csv``
reference files.

Geometry defaults (the drop's DSV4 P03 case; all are CLI flags):
tokens/rank sweep 8..32768 (powers of two), topk=6, 384 total experts
(EP4 -> 96 local), hidden=7168, intermediate=3072 (FI post-SwiGLU convention;
the drop's ``INTERMEDIATE_GATEUP=6144`` is 2x), gate_up_clamp=10.0,
kind=fp8_e4m3, 1xacc, load_balance_mode=atomic_counter and
token-back=reuse_dispatch_warps (both the drop's P03 perf-run settings),
warmup=3, iters=20, tile K=128.

Default tiles per layout (== the shim's per-layout defaults):
  * non_swap_ab: M64 N128  -> compare against ``..._nonswapab_TileM64_TileN128.csv``
  * swap_ab:     M256 N32  -> compare against ``..._swapab_TileM256_TileN32.csv``

One synchronized wall-time cold/JIT measurement and two warm CUDA-event
timed series are recorded per point:
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

Launch with the active environment's Python (one process per GPU,
4-rank EP; srun-safe and non-interactive):

    python -m torch.distributed.run --standalone --nproc_per_node=4 benchmarks/bench_moe_ep_sm90_mega.py

The Humming MXFP4 path is selected explicitly and by default uses a
fully-specified MXFP4-only tactic (no FP8 heuristic/cache fallback).
``--mxfp4-tactic-source cache_or_heuristic`` instead passes ``knobs=None``
through the production fused backend. FP8 and MXFP4 use direct launches for
both timed series; CUDA-event samples retain the historical timing boundaries.

    python -m torch.distributed.run --standalone --nproc_per_node=4 benchmarks/bench_moe_ep_sm90_mega.py --backend sm90_fp8_mxfp4_bf16_pull_cutedsl --tokens 64

For MXFP4, this benchmark constructs deterministic canonical packed E2M1
payloads plus raw K32 E8M0 scale bytes in PrequantizedMoEWeights and runs the
production Humming preprocessor. Every point reports the first synchronized
call (including compile/JIT) separately from warm e2e and bare compute
latency. The same command supports 1, 2, 4, and 8 ranks by changing
--nproc_per_node.

Rank 0 prints one ``BENCH_CSV`` row per (scale_mode, layout, tokens) point
(header once), each carrying the matching drop reference CSV filename.  A
point that OOMs prints a SKIP row and the sweep continues.  Between points
the layer/session and symmetric-heap buffers are destroyed before the next
allocation (the 32768-token workspace needs the heap to itself: the combine
plane alone is ~2.7 GB).
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from statistics import fmean, median
from typing import Sequence

_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_here)
# Direct script launch normally puts only benchmarks/ at sys.path[0]. Pin this
# checkout ahead of any installed FlashInfer while retaining the original
# benchmarks-shadow removal.
sys.path[:] = [_repo_root] + [
    p for p in sys.path if os.path.abspath(p or os.getcwd()) not in (_here, _repo_root)
]

# Drop parity (run_perf_test.sh): multirank Hopper needs NVLS off unless the
# environment has a working NCCL/NVSHMEM NVLS setup. setdefault so users can
# override.
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
os.environ.setdefault("NVSHMEM_DISABLE_NVLS", "1")

from flashinfer.moe_ep.sm90_routing import (
    generate_sm90_published_exact_balanced_routes_numpy,
    generate_sm90_routing_numpy,
    normalize_sm90_routing_profile,
    sm90_routing_audit_payload,
    sm90_routing_profile_from_benchmark_mode,
)

DEFAULT_TOKENS = tuple(1 << p for p in range(3, 16))  # 8 .. 32768
FP8_BACKEND = "sm90_fp8_fp8_bf16_pull_cutedsl"
MXFP4_BACKEND = "sm90_fp8_mxfp4_bf16_pull_cutedsl"
SUPPORTED_BACKENDS = (FP8_BACKEND, MXFP4_BACKEND)
ROUTING_SEED = 1234
ACTIVATION_AND_TOPK_WEIGHT_SEED_BASE = 42
MXFP4_WEIGHT_SEED_BASE = 0x4D584650
FP8_WEIGHT_SEED_BASE = 13

# Phase-A known-correct baseline. These fields are passed explicitly to the
# MXFP4 config, which bypasses both the generic FP8 heuristic and every knob
# cache. A CLI tile/token-back override remains fixed for that run and is
# printed verbatim in each result row.
MXFP4_DEFAULT_TILE = (128, 32)
MXFP4_TILE_K = 128
MXFP4_CLUSTER = (1, 1, 1)
MXFP4_PINGPONG = False
MXFP4_TOKEN_BACK = "epi_warps"

# Raw E8M0 bytes encode powers of two as 2**(e-127). A compact, finite span
# gives realistic small weights and stays well inside Humming's range-11
# contract; the payload itself still samples every legal E2M1 nibble code.
MXFP4_E8M0_MIN = 118
MXFP4_E8M0_MAX_EXCLUSIVE = 124
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

CSV_FIELDS = (
    "kernel,scale_mode,operand_order,tile_m,tile_n,tile_k,"
    "tokens_per_rank,topk,world_size,total_experts,local_experts,hidden,"
    "intermediate_downproj,intermediate_gateup,warmup,iters,status,"
    "e2e_min_us,e2e_max_us,e2e_mean_us,e2e_median_us,"
    "compute_min_us,compute_max_us,compute_mean_us,compute_median_us,"
    "fc1_flops_per_rank,fc2_flops_per_rank,total_flops_per_rank,"
    "critical_tflops_compute,critical_tflops_e2e,tok_s_e2e,ref_csv"
)
BENCH_EXT_CSV_FIELDS = (
    "tactic,cold_first_call_min_us,cold_first_call_max_us,cold_first_call_mean_us"
)
# Keep CSV_FIELDS and its historical meanings/order stable: existing FP8
# parsers consume that prefix verbatim. Removed split/Graph fields are not emitted.
FORMAL_TUNING_CSV_FIELDS = (
    "compute_max_rank_median_us,"
    "fused_pingpong,fused_cga_m,fused_cga_n,fused_cga_k,"
    "fused_group_hint,fused_num_sched_stages,fused_load_balance_mode,"
    "fused_token_back_mode"
)
# Actual compute-workspace FP8 tactic identity. Keep this append-only: the
# historical fields and table-derived HEUR_CSV_FIELDS retain their meaning
# even when knobs=None resolves a persistent-cache entry instead of the table.
FP8_RUNTIME_CSV_FIELDS = (
    "fp8_tactic_mode,fp8_swap_ab,fp8_pingpong,"
    "fp8_tile_m,fp8_tile_n,fp8_tile_k,"
    "fp8_cga_m,fp8_cga_n,fp8_cga_k,"
    "fp8_accum_mode,fp8_group_hint,fp8_num_sched_stages,fp8_flag_batch,"
    "fp8_epi_flag_batch_fc1,fp8_epi_flag_batch_fc2,"
    "fp8_load_balance_mode,fp8_token_back_mode,fp8_in_kernel_fc2_reduce"
)
# Canonical actual-runtime identity shared by ordinary FP8 and MXFP4.
# Human-readable group/stage columns make runtime defaults auditable;
# the SHA-256 is over the complete tactic below.
RUNTIME_TACTIC_CSV_FIELDS = (
    "runtime_tactic_sha256,runtime_group_hint,runtime_num_sched_stages"
)
# Global input-routing identity. Keep this block unchanged; later extensions
# are appended after it so every existing field keeps its position and meaning.
ROUTING_CSV_FIELDS = "routing_mode,routing_profile,routing_seed,route_ids_sha256"
# Label direct measurements explicitly so historical Graph samples stay distinct.
COMPUTE_LAUNCH_CSV_FIELDS = "compute_launch_mode"
# The latest common fused execution axes.  Requested/effective columns are
# append-only because the compiled kernel may self-gate store offload or force
# early publication/folding.  The effective values also participate in the
# canonical candidate identity below.
FUSED_EXECUTION_RUNTIME_CSV_FIELDS = (
    "runtime_dedup_dispatch,runtime_grouped_token_back,runtime_combine_format,"
    "runtime_active_dispatch_warps,"
    "requested_fc1_store_offload,effective_fc1_store_offload,"
    "requested_fc1_early_done_publish,effective_fc1_early_done_publish,"
    "requested_fold_producer_warps,effective_fold_producer_warps"
)
_FUSED_EXECUTION_TACTIC_FIELDS = frozenset(
    {
        "dedup_dispatch",
        "grouped_token_back",
        "combine_format",
        "active_dispatch_warps",
        "fc1_store_offload",
        "fc1_early_done_publish",
        "fold_producer_warps",
    }
)
FP8_RUNTIME_TACTIC_FIELDS = (
    frozenset(
        {
            "swap_ab",
            "pingpong",
            "mma_tiler_mnk",
            "cluster_shape_mnk",
            "fp8_accum_mode",
            "load_balance_mode",
            "token_back_mode",
            "group_hint",
            "tail_split_pairs",
            "num_sched_stages",
            "flag_batch",
            "epi_flag_batch",
            "in_kernel_fc2_reduce",
            "compact_pull_buffer",
            "generate_c",
        }
    )
    | _FUSED_EXECUTION_TACTIC_FIELDS
)
MXFP4_FUSED_RUNTIME_TACTIC_FIELDS = (
    frozenset(
        {
            "swap_ab",
            "pingpong",
            "mma_tiler_mnk",
            "cluster_shape_mnk",
            "fp8_accum_mode",
            "load_balance_mode",
            "token_back_mode",
            "group_hint",
            "tail_split_pairs",
            "num_sched_stages",
            "in_kernel_fc2_reduce",
            "fc2_tail_n8",
            "fc1_ready_mode",
        }
    )
    | _FUSED_EXECUTION_TACTIC_FIELDS
)
CSV_HEADER = (
    "BENCH_CSV,"
    + CSV_FIELDS
    + ","
    + BENCH_EXT_CSV_FIELDS
    + ","
    + FORMAL_TUNING_CSV_FIELDS
    + ","
    + FP8_RUNTIME_CSV_FIELDS
    + ","
    + RUNTIME_TACTIC_CSV_FIELDS
    + ","
    + ROUTING_CSV_FIELDS
    + ","
    + COMPUTE_LAUNCH_CSV_FIELDS
    + ","
    + FUSED_EXECUTION_RUNTIME_CSV_FIELDS
)

# Resolved launch-config columns appended to --output-csv rows (blank for
# fixed-layout runs; filled from the shim's token-bucket table under
# --heuristic so the file records what each point actually launched).
HEUR_CSV_FIELDS = (
    "heur_swap_ab,heur_pingpong,heur_tile_m,heur_tile_n,heur_tile_k,"
    "heur_cga_m,heur_cga_n,heur_accum_mode,heur_token_back,heur_token_bucket,"
    "heur_group_hint,heur_tail_split"
)


def _heuristic_cols(
    backend: str, scale_mode: str, operand_order: str, tokens: int
) -> list[str]:
    """The launch config the shim resolves for this point (heuristic mode)."""
    if backend != FP8_BACKEND or operand_order != "heuristic":
        return [""] * 12
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
        bootstrap_paths,
    )

    bootstrap_paths()
    from moe_hopper_fp8.heuristic_config import select_heuristic_config

    sel = select_heuristic_config(scale_mode, tokens)
    c = sel.config
    return [
        str(int(c.swap_ab)),
        str(int(c.pingpong)),
        str(c.mma_tiler_mnk[0]),
        str(c.mma_tiler_mnk[1]),
        str(c.mma_tiler_mnk[2]),
        str(c.cluster_shape_mnk[0]),
        str(c.cluster_shape_mnk[1]),
        c.accum_mode,
        c.token_back_mode,
        str(sel.token_bucket),
        "" if c.group_hint is None else str(c.group_hint),
        str(int(c.tail_split_pairs)),
    ]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--backend",
        choices=SUPPORTED_BACKENDS,
        default=FP8_BACKEND,
        help="production backend identity; MXFP4 is explicit and never "
        "falls back to the ordinary FP8 backend",
    )
    p.add_argument(
        "--tokens",
        type=str,
        default=",".join(str(t) for t in DEFAULT_TOKENS),
        help="comma-separated tokens-per-rank sweep",
    )
    p.add_argument(
        "--scale-mode",
        choices=["per_tensor", "blockwise", "both", "mxfp4_hybrid"],
        default="both",
        help="scale ABI(s) to sweep. The historical FP8 default is both; "
        "an omitted flag resolves to mxfp4_hybrid on the MXFP4 backend",
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
    order.add_argument(
        "--heuristic",
        dest="operand_order",
        action="store_const",
        const="heuristic",
        help="leave swap_ab/pingpong/mma_tiler/cluster unset so the shim "
        "resolves the drop's token-bucket heuristic table per point "
        "(moe_hopper_fp8/heuristic_config.py).  This is the default.",
    )
    order.add_argument(
        "--both-orders",
        dest="operand_order",
        action="store_const",
        const="both",
        help="sweep both fixed layouts (non-swap then swap-AB) instead of "
        "the heuristic selection",
    )
    p.set_defaults(operand_order="heuristic")
    p.add_argument(
        "--mma-tiler",
        type=str,
        default=None,
        metavar="M,N[,K]",
        help="override the MMA tile; omitted K defaults to 128. "
        "This benchmark uses K=128 for FP8 and supports K=128/256 for MXFP4. "
        "Default: the shim's per-layout default "
        "(non-swap 64,128 / swap-AB 256,32).",
    )
    p.add_argument(
        "--fp8-knobs-json",
        type=str,
        default=None,
        metavar="JSON_OBJECT",
        help="ordinary-FP8-only explicit tuner tactic. Accepts the dicts "
        "returned by hopper_fp8_candidates(); tuple knobs use JSON arrays. "
        "This bypasses cache/heuristic lookup and conflicts with legacy "
        "layout, --mma-tiler, and --token-back flags.",
    )
    p.add_argument(
        "--group-hint",
        type=int,
        default=None,
        help="fused scheduler group hint.",
    )
    p.add_argument(
        "--num-sched-stages",
        type=int,
        default=None,
        help="fused scheduler pipeline stage count.",
    )
    p.add_argument(
        "--mxfp4-tactic-source",
        choices=["explicit", "cache_or_heuristic"],
        default="explicit",
        help="MXFP4 tactic selector. Default 'explicit' preserves the legacy "
        "CLI-controlled fused tactic. 'cache_or_heuristic' passes "
        "knobs=None with no manual geometry so the dedicated persistent "
        "fused cache is consulted before the manifest heuristic.",
    )
    p.add_argument(
        "--cga",
        type=str,
        default=None,
        metavar="M,N",
        help="cluster shape (M,N; K fixed at 1) for explicit MXFP4 or "
        "FP8 --swap-ab/--no-swap-ab layouts. Default: 1,1.",
    )
    p.add_argument("--top-k", type=int, default=6)
    # TOTAL experts across all EP ranks (DSV4-Pro: 384), fixed regardless of
    # world size -- each rank owns num_experts // world_size local experts
    # (4 ranks -> 96/rank, 8 ranks -> 48/rank).  Do NOT scale this per rank.
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
        "--fp8-accum-mode",
        choices=["1xacc", "2xacc"],
        default="1xacc",
    )
    p.add_argument(
        "--load-balance-mode",
        choices=["static", "atomic_counter"],
        default="atomic_counter",
        help="atomic_counter matches the drop's perf-run setting",
    )
    p.add_argument(
        "--mxfp4-fc2-tail-n8",
        action="store_true",
        help="fused MXFP4 only: use N8 math for eligible FC2 token tails",
    )
    p.add_argument(
        "--mxfp4-fc1-ready-mode",
        choices=("tile", "k256"),
        default="tile",
        help="fused MXFP4 FC1 completion protocol (supported geometries only)",
    )
    p.add_argument(
        "--token-back",
        choices=[
            "heuristic",
            "epi_warps",
            "reuse_dispatch_warps",
            "standalone_warps",
        ],
        default="heuristic",
        help="fc2 token-back path. 'heuristic' (default) follows the "
        "per-token-bucket table (epi_warps small/mid buckets, "
        "reuse_dispatch_warps at the GEMM-bound tail); the explicit modes "
        "pin one path for A/B runs (reuse_dispatch_warps matches the "
        "drop's P03 perf-run setting).",
    )
    p.add_argument(
        "--dedup-dispatch",
        action="store_true",
        help="send each token once per destination rank on dispatch "
        "(duplicate top-k routes copy the carrier's pool row locally)",
    )
    p.add_argument(
        "--grouped-token-back",
        action="store_true",
        help="combine dedup: pre-reduce each (src_rank, src_token) group in "
        "fp32 on the expert rank and return one row per contributing rank "
        "(forces token-back reuse_dispatch_warps)",
    )
    p.add_argument(
        "--combine-format",
        choices=["bf16", "32e4m3xe8m0", "32e5m2xe8m0"],
        default="bf16",
        help="combine wire format; the quantized fp8 wires halve the return "
        "bytes and require --grouped-token-back",
    )
    p.add_argument(
        "--fc1-store-offload",
        dest="fc1_store_offload",
        action="store_true",
        default=True,
        help="empty-warp FC1 store offload (default on; self-gating)",
    )
    p.add_argument(
        "--no-fc1-store-offload",
        dest="fc1_store_offload",
        action="store_false",
    )
    p.add_argument(
        "--fc1-early-pub",
        dest="fc1_early_done_publish",
        action="store_true",
        default=False,
        help="early fc1_done publication (measured neutral-to-negative "
        "outside the offload's domain; kept as a tuner axis)",
    )
    p.add_argument(
        "--no-fc1-early-pub",
        dest="fc1_early_done_publish",
        action="store_false",
    )
    p.add_argument(
        "--fold-producer-warps",
        dest="fold_producer_warps",
        action="store_true",
        default=True,
        help="fold TMA-A/TMA-B/sched into the idle dispatch-WG slots and "
        "drop the producer warpgroup (active_dispatch_warps=1 only; "
        "forces early fc1_done publish, no store offload)",
    )
    p.add_argument(
        "--no-fold-producer-warps",
        dest="fold_producer_warps",
        action="store_false",
    )
    p.add_argument(
        "--compact-pull-buffer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="size the dispatch pull buffer by the active dispatch warps only "
        "(saves 3 x hidden bytes of SMEM per CTA for AB stages); "
        "--no-compact-pull-buffer restores the 4-slot buffer",
    )
    p.add_argument(
        "--generate-c",
        action="store_true",
        default=False,
        help="training forward: also write the raw pre-SwiGLU fc1 gate+up "
        "tensor (generate_c=True on every point, both layouts); default off",
    )
    p.add_argument(
        "--epi-mode",
        choices=["auto", "basic", "pingpong", "cooperative"],
        default="auto",
        help="heuristic-order only: force every bucket's epilogue mode, "
        "deriving the tile from the bucket's heuristic entry. basic = ONE "
        "epilogue WG owning a per-WG-size task tile (non-swap N128 / swap "
        "M128); pingpong = TWO epilogue WGs, each owning its own per-WG-size "
        "tile, alternating; cooperative = TWO epilogue WGs splitting one "
        "doubled tile (N256 / M256, no pingpong). Cluster "
        "shape / accum / token-back stay the bucket's; buckets already in "
        "the requested mode run unchanged (that no-op is the A/B sanity gate).",
    )
    p.add_argument(
        "--swap-token-tile",
        type=int,
        default=None,
        help="heuristic-order only: on swap-AB buckets replace the token tile "
        "(mma N) with this value, keeping the bucket's M / pingpong / cluster "
        "shape / token-back (e.g. 8 to probe the experimental N=8 tile).",
    )
    p.add_argument(
        "--pingpong",
        choices=["auto", "on", "off"],
        default="auto",
        help="force task-tile ping-pong scheduling on/off instead of the "
        "heuristic table's per-bucket choice (auto)",
    )
    p.add_argument(
        "--active-dispatch-warps",
        type=int,
        choices=[1, 2, 4],
        default=1,
        help="dispatch warps doing token-comm work; the rest stay idle "
        "(physical layout stays 4)",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument(
        "--routing-mode",
        choices=["block_permutation", "published_exact_balanced"],
        default="block_permutation",
        help="balanced-routing generator. The default preserves the drop's "
        "padded block-permutation workload; published_exact_balanced uses "
        "the exact-balanced routing from the published Hopper comparison.",
    )
    p.add_argument(
        "--no-sparse-data",
        dest="use_sparse_data",
        action="store_false",
        help="use dense quantized-randn fp8 payloads (realistic model "
        "data) instead of the default drop-harness perf recipe (weights: "
        "positive-only random E4M3 bytes; activations: uniform random "
        "finite E4M3 bytes). MXFP4 always keeps its canonical "
        "payload/scale relation.",
    )
    p.set_defaults(use_sparse_data=True)
    p.add_argument(
        "--output-csv",
        type=str,
        default="auto",
        metavar="PATH",
        help="also write the BENCH_CSV rows to this file (rank 0 only), "
        "with the resolved heuristic launch-config columns appended "
        "(blank in fixed-layout modes).  Default 'auto' writes to "
        "benchmark_data/<date>/<date>_<time>_mega_sm90_<order>_<scale>.csv "
        "under the SM90 kernel tree (directories created as needed); "
        "pass 'none' to disable.",
    )
    p.add_argument(
        "--cooldown-s",
        type=float,
        default=5.0,
        help="idle the GPUs this many seconds before each timed series so "
        "clocks recover from power capping, mirroring the near-idle "
        "process-startup window the drop's process-per-case sweep gets "
        "before each timed burst.  Pass 0 to disable.",
    )
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = p.parse_args(raw_argv)
    # Resolve abbreviations to the same applicability checks as full names.
    args._specified_options = frozenset(
        (
            p._option_string_actions.get(value.split("=", 1)[0])
            or p._get_option_tuples(value)[0][0]
        ).option_strings[0]
        for value in raw_argv
        if value.startswith("--") and value != "--"
    )
    return args


_FP8_EXPLICIT_KNOBS = frozenset(
    {
        "swap_ab",
        "pingpong",
        "mma_tiler_mnk",
        "cluster_shape_mnk",
        "fp8_accum_mode",
        "group_hint",
        "tail_split_pairs",
        "num_sched_stages",
        "flag_batch",
        "epi_flag_batch",
        "in_kernel_fc2_reduce",
        "token_back_mode",
        "load_balance_mode",
        "dedup_dispatch",
        "grouped_token_back",
        "combine_format",
        "active_dispatch_warps",
        "fc1_store_offload",
        "fc1_early_done_publish",
        "fold_producer_warps",
    }
)
_FP8_REQUIRED_EXPLICIT_KNOBS = frozenset(
    {
        "swap_ab",
        "pingpong",
        "mma_tiler_mnk",
        "cluster_shape_mnk",
        "fp8_accum_mode",
        "token_back_mode",
    }
)


_NEUTRAL_FUSED_GEOMETRY_OPTIONS = frozenset(
    {
        "--group-hint",
        "--num-sched-stages",
        "--pingpong",
    }
)
_FUSED_EXECUTION_KNOB_OPTIONS = frozenset(
    {
        "--dedup-dispatch",
        "--grouped-token-back",
        "--combine-format",
        "--active-dispatch-warps",
        "--fc1-store-offload",
        "--no-fc1-store-offload",
        "--fc1-early-pub",
        "--no-fc1-early-pub",
        "--fold-producer-warps",
        "--no-fold-producer-warps",
    }
)
_MXFP4_STRATEGY_OPTIONS = frozenset({"--mxfp4-fc2-tail-n8", "--mxfp4-fc1-ready-mode"})


def _parse_fp8_knobs_json(value: str | None) -> dict[str, object] | None:
    """Parse one strict tuner tactic without cache/heuristic fallback."""
    if value is None:
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--fp8-knobs-json is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("--fp8-knobs-json must decode to a JSON object")
    unknown = sorted(set(payload) - _FP8_EXPLICIT_KNOBS)
    if unknown:
        raise ValueError(
            "--fp8-knobs-json has unsupported knob(s): " + ", ".join(unknown)
        )
    missing = sorted(_FP8_REQUIRED_EXPLICIT_KNOBS - set(payload))
    if missing:
        raise ValueError("--fp8-knobs-json must fully specify: " + ", ".join(missing))
    knobs: dict[str, object] = dict(payload)
    for name, length in (
        ("mma_tiler_mnk", 3),
        ("cluster_shape_mnk", 3),
        ("epi_flag_batch", 2),
    ):
        if name not in knobs:
            continue
        raw = knobs[name]
        if (
            not isinstance(raw, list)
            or len(raw) != length
            or any(not isinstance(v, int) or isinstance(v, bool) for v in raw)
        ):
            raise ValueError(
                f"--fp8-knobs-json {name} must be a length-{length} integer array"
            )
        knobs[name] = tuple(raw)
    for name in (
        "swap_ab",
        "pingpong",
        "in_kernel_fc2_reduce",
        "dedup_dispatch",
        "grouped_token_back",
        "fc1_store_offload",
        "fc1_early_done_publish",
        "fold_producer_warps",
        "tail_split_pairs",
    ):
        if name in knobs and not isinstance(knobs[name], bool):
            raise ValueError(f"--fp8-knobs-json {name} must be boolean")
    if knobs["fp8_accum_mode"] not in ("1xacc", "2xacc"):
        raise ValueError("--fp8-knobs-json fp8_accum_mode must be 1xacc or 2xacc")
    if knobs["token_back_mode"] not in (
        "epi_warps",
        "standalone_warps",
        "reuse_dispatch_warps",
    ):
        raise ValueError("--fp8-knobs-json has an invalid token_back_mode")
    if "load_balance_mode" in knobs and knobs["load_balance_mode"] not in (
        "static",
        "atomic_counter",
    ):
        raise ValueError("--fp8-knobs-json has an invalid load_balance_mode")
    if "combine_format" in knobs and knobs["combine_format"] not in (
        "bf16",
        "32e4m3xe8m0",
        "32e5m2xe8m0",
    ):
        raise ValueError("--fp8-knobs-json has an invalid combine_format")
    active_dispatch_warps = knobs.get("active_dispatch_warps")
    if active_dispatch_warps is not None and active_dispatch_warps not in (1, 2, 4):
        raise ValueError("--fp8-knobs-json active_dispatch_warps must be 1, 2, or 4")
    for name in ("group_hint", "num_sched_stages"):
        item = knobs.get(name)
        if item is not None and (
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
        ):
            raise ValueError(f"--fp8-knobs-json {name} must be null or positive")
    flag_batch = knobs.get("flag_batch")
    if flag_batch is not None and (
        not isinstance(flag_batch, int)
        or isinstance(flag_batch, bool)
        or not 1 <= flag_batch <= 32
    ):
        raise ValueError("--fp8-knobs-json flag_batch must be in [1, 32]")
    epi = knobs.get("epi_flag_batch")
    if epi is not None and any(not 1 <= v <= 32 for v in epi):
        raise ValueError("--fp8-knobs-json epi_flag_batch values must be in [1, 32]")
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import is_valid

    if not is_valid(knobs):
        raise ValueError("--fp8-knobs-json is not a valid Hopper FP8 tactic")
    return knobs


def _resolve_sweep(
    args: argparse.Namespace, world_size: int
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, int] | None]:
    """Resolve backend-specific defaults and reject cross-format fallback."""

    strategy_options = args._specified_options.intersection(_MXFP4_STRATEGY_OPTIONS)
    if strategy_options and args.backend != MXFP4_BACKEND:
        raise ValueError("MXFP4 strategy flags require the fused MXFP4 backend")

    if args.backend == MXFP4_BACKEND:
        fp8_only = args._specified_options.intersection(
            {
                "--swap-token-tile",
                "--generate-c",
                "--compact-pull-buffer",
                "--no-compact-pull-buffer",
            }
        )
        if fp8_only:
            raise ValueError("FP8-only options: " + ", ".join(sorted(fp8_only)))
        if args.fp8_knobs_json is not None:
            raise ValueError(f"--fp8-knobs-json requires --backend {FP8_BACKEND}")
        if world_size not in (1, 2, 4, 8):
            raise ValueError(
                "the MXFP4 benchmark supports exactly 1, 2, 4, or 8 ranks; "
                f"got world_size={world_size}"
            )
        if args.kind != "fp8_e4m3":
            raise ValueError("MXFP4 requires --kind fp8_e4m3")
        if args.fp8_accum_mode != "1xacc":
            raise ValueError("MXFP4 requires --fp8-accum-mode 1xacc")
        if args.grouped_token_back:
            raise ValueError("MXFP4 currently requires grouped_token_back=False")
        if args.combine_format != "bf16":
            raise ValueError("MXFP4 currently requires --combine-format bf16")
        scale_mode = (
            "mxfp4_hybrid"
            if "--scale-mode" not in args._specified_options
            else args.scale_mode
        )
        if scale_mode != "mxfp4_hybrid":
            raise ValueError(
                "the MXFP4 backend requires --scale-mode mxfp4_hybrid; "
                "ordinary FP8 scale modes are not fallback candidates"
            )
        layout_options = {
            "--swap-ab",
            "--no-swap-ab",
            "--heuristic",
            "--both-orders",
        }
        operand_order = (
            "swap_ab"
            if not args._specified_options.intersection(layout_options)
            else args.operand_order
        )
        if operand_order != "swap_ab":
            raise ValueError(
                "the MXFP4 backend requires --swap-ab; native/both/heuristic "
                "operand order is not a fallback candidate"
            )
        if "--token-back" in args._specified_options and args.token_back == "heuristic":
            raise ValueError(
                "MXFP4 benchmark tactics must be fixed; choose an explicit "
                "--token-back mode or omit it for epi_warps"
            )
        cache_mode = args.mxfp4_tactic_source == "cache_or_heuristic"
        if cache_mode:
            common_tactic_options = (
                {
                    "--load-balance-mode",
                    "--mma-tiler",
                    "--cga",
                    "--token-back",
                }
                | _FUSED_EXECUTION_KNOB_OPTIONS
                | _MXFP4_STRATEGY_OPTIONS
            )
            fused_tactic_options = _NEUTRAL_FUSED_GEOMETRY_OPTIONS
            conflicts = sorted(
                args._specified_options.intersection(
                    common_tactic_options | fused_tactic_options
                )
            )
            if conflicts:
                raise ValueError(
                    "--mxfp4-tactic-source cache_or_heuristic conflicts with "
                    + ", ".join(conflicts)
                )
        tile = (
            _parse_mma_tile(args.mma_tiler)[:2]
            if args.mma_tiler is not None
            else MXFP4_DEFAULT_TILE
        )
        if not cache_mode:
            _mxfp4_fused_tactic(args, tile)
        return ("mxfp4_hybrid",), ("swap_ab",), tile

    if "--mxfp4-tactic-source" in args._specified_options:
        raise ValueError(f"--mxfp4-tactic-source requires --backend {MXFP4_BACKEND}")
    mxfp4_only_options = sorted(
        args._specified_options.intersection(
            _NEUTRAL_FUSED_GEOMETRY_OPTIONS - {"--pingpong"}
        )
    )
    if mxfp4_only_options:
        raise ValueError(
            "full-MNK fused tactic flags currently require "
            f"--backend {MXFP4_BACKEND}: " + ", ".join(mxfp4_only_options)
        )
    if args.scale_mode == "mxfp4_hybrid":
        raise ValueError(
            "mxfp4_hybrid is accepted only by "
            f"--backend {MXFP4_BACKEND}; no cross-format fallback"
        )
    scale_mode = args.scale_mode
    scale_modes = ("per_tensor", "blockwise") if scale_mode == "both" else (scale_mode,)
    fp8_knobs = _parse_fp8_knobs_json(args.fp8_knobs_json)
    if fp8_knobs is not None:
        conflicts = []
        if args._specified_options.intersection(
            {"--swap-ab", "--no-swap-ab", "--heuristic", "--both-orders"}
        ):
            conflicts.append("layout/--heuristic")
        if args.mma_tiler is not None:
            conflicts.append("--mma-tiler")
        if "--token-back" in args._specified_options:
            conflicts.append("--token-back")
        conflicts.extend(
            sorted(args._specified_options.intersection({"--cga", "--swap-token-tile"}))
        )
        if conflicts:
            raise ValueError(
                "--fp8-knobs-json is mutually exclusive with " + ", ".join(conflicts)
            )
        mma = fp8_knobs["mma_tiler_mnk"]
        assert isinstance(mma, tuple)
        order = "swap_ab" if bool(fp8_knobs["swap_ab"]) else "non_swap_ab"
        return scale_modes, (order,), (int(mma[0]), int(mma[1]))
    operand_order = args.operand_order
    orders = ("non_swap_ab", "swap_ab") if operand_order == "both" else (operand_order,)
    mma = _parse_mma_tile(args.mma_tiler) if args.mma_tiler is not None else None
    if mma is not None and mma[2] != 128:
        raise ValueError("FP8 benchmark --mma-tiler currently supports only K=128")
    return scale_modes, orders, None if mma is None else mma[:2]


def _parse_mma_tile(value: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(v) for v in value.split(","))
    except ValueError as exc:
        raise ValueError("--mma-tiler must be M,N[,K] with integer fields") from exc
    if len(values) not in (2, 3) or any(v <= 0 for v in values):
        raise ValueError("--mma-tiler must be two or three positive integers M,N[,K]")
    return (*values, 128) if len(values) == 2 else values


def _runtime_positive_triplet(value: object, field: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(f"runtime tactic {field} must be a list/tuple triple")
    if len(value) != 3 or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in value
    ):
        raise RuntimeError(
            f"runtime tactic {field} must contain three positive integers"
        )
    return tuple(value)


def _resolved_load_balance_mode(args: argparse.Namespace) -> str:
    return str(args.load_balance_mode)


def _resolved_token_back(args: argparse.Namespace) -> str | None:
    if args.backend == MXFP4_BACKEND:
        if "--token-back" not in args._specified_options:
            return MXFP4_TOKEN_BACK
        return str(args.token_back)
    return None if args.token_back == "heuristic" else str(args.token_back)


def _fused_execution_knobs_from_args(
    args: argparse.Namespace,
) -> dict[str, object]:
    """Shared FP8/MXFP4 fused execution knobs and their public defaults."""

    return {
        "dedup_dispatch": bool(args.dedup_dispatch),
        "grouped_token_back": bool(args.grouped_token_back),
        "combine_format": str(args.combine_format),
        "active_dispatch_warps": int(args.active_dispatch_warps),
        "fc1_store_offload": bool(args.fc1_store_offload),
        "fc1_early_done_publish": bool(args.fc1_early_done_publish),
        "fold_producer_warps": bool(args.fold_producer_warps),
    }


def _fp8_effective_explicit_knobs(args: argparse.Namespace) -> dict[str, object] | None:
    knobs = _parse_fp8_knobs_json(args.fp8_knobs_json)
    if knobs is None:
        return None
    return {
        **knobs,
        "group_hint": knobs.get("group_hint"),
        "num_sched_stages": knobs.get("num_sched_stages"),
        "flag_batch": knobs.get("flag_batch", 1),
        "epi_flag_batch": knobs.get("epi_flag_batch", (2, 4)),
        "load_balance_mode": knobs.get(
            "load_balance_mode", _resolved_load_balance_mode(args)
        ),
        "in_kernel_fc2_reduce": knobs.get("in_kernel_fc2_reduce", False),
        "dedup_dispatch": knobs.get("dedup_dispatch", False),
        "grouped_token_back": knobs.get("grouped_token_back", False),
        "combine_format": knobs.get("combine_format", "bf16"),
        "active_dispatch_warps": knobs.get("active_dispatch_warps", 1),
        "fc1_store_offload": knobs.get("fc1_store_offload", True),
        "fc1_early_done_publish": knobs.get("fc1_early_done_publish", False),
        "fold_producer_warps": knobs.get("fold_producer_warps", True),
    }


def _mxfp4_fused_tactic(
    args: argparse.Namespace,
    legacy_tile: tuple[int, int],
) -> dict[str, object]:
    """Return one fully identified, MXFP4-only fused tactic.

    --mma-tiler accepts M,N or M,N,K; an omitted K defaults to 128.
    """

    mma = (
        _parse_mma_tile(args.mma_tiler)
        if args.mma_tiler is not None
        else (legacy_tile[0], legacy_tile[1], MXFP4_TILE_K)
    )
    cluster = MXFP4_CLUSTER
    if args.cga is not None:
        cm, cn = (int(v) for v in args.cga.split(","))
        cluster = (cm, cn, 1)
    pingpong = MXFP4_PINGPONG if args.pingpong == "auto" else args.pingpong == "on"
    m, n, k = mma
    if m not in (128, 256) or n not in (16, 32, 64, 128):
        raise ValueError(
            "MXFP4 fused MMA tile requires M in (128,256) and "
            f"N in (16,32,64,128), got {mma!r}"
        )
    if k not in (128, 256):
        raise ValueError(f"MXFP4 fused MMA tile K must be 128 or 256, got {k}")
    for name, logical_k in (
        ("hidden", args.hidden),
        ("intermediate", args.intermediate),
    ):
        if logical_k % k:
            raise ValueError(
                f"MXFP4 fused {name} ({logical_k}) must be divisible by tile K={k}"
            )
    if pingpong and m != 128:
        raise ValueError("MXFP4 fused ping-pong requires MMA tile M=128")
    if cluster not in (
        (1, 1, 1),
        (2, 1, 1),
        (1, 2, 1),
        (2, 2, 1),
    ):
        raise ValueError(f"unsupported MXFP4 fused cluster shape {cluster!r}")
    for option, value in (
        ("--group-hint", args.group_hint),
        ("--num-sched-stages", args.num_sched_stages),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{option} must be a positive integer")

    tactic: dict[str, object] = {
        "swap_ab": True,
        "pingpong": pingpong,
        "mma_tiler_mnk": mma,
        "cluster_shape_mnk": cluster,
        "fp8_accum_mode": "1xacc",
        "load_balance_mode": _resolved_load_balance_mode(args),
        "token_back_mode": _resolved_token_back(args),
        "in_kernel_fc2_reduce": False,
        "group_hint": args.group_hint,
        "num_sched_stages": args.num_sched_stages,
        **_fused_execution_knobs_from_args(args),
        "fc2_tail_n8": args.mxfp4_fc2_tail_n8,
        "fc1_ready_mode": args.mxfp4_fc1_ready_mode,
        "tail_split_pairs": False,
    }
    return tactic


def _tactic_label(
    args: argparse.Namespace,
    *,
    operand_order: str,
    tile: tuple[int, int] | tuple[str, str],
) -> str:
    if args.backend == MXFP4_BACKEND:
        if args.mxfp4_tactic_source == "cache_or_heuristic":
            return "mxfp4_fused_cache_or_heuristic"
        token_back = _resolved_token_back(args)
        tactic = _mxfp4_fused_tactic(args, (int(tile[0]), int(tile[1])))
        mma = tactic["mma_tiler_mnk"]
        cluster = tactic["cluster_shape_mnk"]
        assert isinstance(mma, tuple) and isinstance(cluster, tuple)
        group = "auto" if tactic["group_hint"] is None else tactic["group_hint"]
        stages = (
            "auto" if tactic["num_sched_stages"] is None else tactic["num_sched_stages"]
        )
        return (
            f"swapab_m{mma[0]}n{mma[1]}k{mma[2]}_"
            f"cga{cluster[0]}x{cluster[1]}x{cluster[2]}_"
            f"pp{int(bool(tactic['pingpong']))}_gh{group}_s{stages}_"
            f"{tactic['load_balance_mode']}_{token_back}"
        )
    fp8_knobs = _fp8_effective_explicit_knobs(args)
    if fp8_knobs is not None:
        mma = fp8_knobs["mma_tiler_mnk"]
        cluster = fp8_knobs["cluster_shape_mnk"]
        epi = fp8_knobs["epi_flag_batch"]
        assert (
            isinstance(mma, tuple)
            and isinstance(cluster, tuple)
            and isinstance(epi, tuple)
        )
        layout = "swapab" if bool(fp8_knobs["swap_ab"]) else "nonswap"
        group = "auto" if fp8_knobs["group_hint"] is None else fp8_knobs["group_hint"]
        stages = (
            "auto"
            if fp8_knobs["num_sched_stages"] is None
            else fp8_knobs["num_sched_stages"]
        )
        return (
            f"fp8_{layout}_m{mma[0]}n{mma[1]}k{mma[2]}_"
            f"cga{cluster[0]}x{cluster[1]}x{cluster[2]}_pp{int(bool(fp8_knobs['pingpong']))}_"
            f"acc{fp8_knobs['fp8_accum_mode']}_gh{group}_ns{stages}_"
            f"fb{fp8_knobs['flag_batch']}_efb{epi[0]}x{epi[1]}_"
            f"{fp8_knobs['load_balance_mode']}_{fp8_knobs['token_back_mode']}_"
            f"ikr{int(bool(fp8_knobs['in_kernel_fc2_reduce']))}"
        )
    if operand_order == "heuristic":
        return "fp8_token_bucket_heuristic"
    token_back = _resolved_token_back(args) or "heuristic"
    return (
        f"{operand_order}_m{tile[0]}n{tile[1]}k128_"
        f"{_resolved_load_balance_mode(args)}_{token_back}"
    )


def _assert_backend_identity(backend, requested: str) -> str:
    actual = backend.kernel_name()
    if actual != requested:
        raise RuntimeError(
            f"requested backend {requested!r}, registry created {actual!r}; "
            "benchmark fallback is forbidden"
        )
    return actual


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
    cold_us: list[float]  # first synchronized call, including compile/JIT
    e2e_us: list[float]  # cross-rank per-rank mean e2e us (len == world)
    e2e_median_us: list[float]
    compute_us: list[float]
    compute_median_us: list[float]
    runtime_metadata: list[dict[str, object]] | None = None
    error: str = ""


def _published_exact_balanced_routes(
    *, world_size: int, tokens: int, topk: int, total_experts: int, seed: int
):
    """Published Hopper exact-balanced routes, including ragged token cases."""
    return generate_sm90_published_exact_balanced_routes_numpy(
        world_size=world_size,
        tokens=tokens,
        topk=topk,
        total_experts=total_experts,
        seed=seed,
    )


def _routing_audit_payload(
    routes,
    *,
    mode: str,
    seed: int,
    num_experts: int,
    world_size: int,
) -> dict[str, object]:
    return sm90_routing_audit_payload(
        routes,
        routing_profile=sm90_routing_profile_from_benchmark_mode(mode),
        seed=seed,
        total_experts=num_experts,
        world_size=world_size,
    )


def _balanced_routing(
    num_tokens: int,
    topk: int,
    num_experts: int,
    rank: int,
    world_size: int,
    device,
    seed: int = ROUTING_SEED,
    *,
    mode: str = "block_permutation",
):
    """Generate a deterministic balanced-routing workload.

    block_permutation preserves the benchmark's historical padded-tail
    workload. published_exact_balanced reproduces the exact expert-count
    balance used by the published Hopper comparison.
    """
    import torch

    routing_profile = sm90_routing_profile_from_benchmark_mode(mode)
    all_ids = generate_sm90_routing_numpy(
        routing_profile=routing_profile,
        world_size=world_size,
        tokens=num_tokens,
        topk=topk,
        total_experts=num_experts,
        seed=seed,
    )

    if rank == 0:
        audit = _routing_audit_payload(
            all_ids,
            mode=mode,
            seed=seed,
            num_experts=num_experts,
            world_size=world_size,
        )
        print(
            "ROUTING_AUDIT," + json.dumps(audit, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
    return torch.from_numpy(all_ids[rank].astype("int64")).to(device)


def _make_point_inputs(args, tokens: int, rank: int, world_size: int, device):
    import torch

    g = torch.Generator(device="cuda").manual_seed(
        ACTIVATION_AND_TOPK_WEIGHT_SEED_BASE + rank
    )
    hidden_states = torch.randn(
        tokens, args.hidden, dtype=torch.bfloat16, device=device, generator=g
    )
    topk_ids = _balanced_routing(
        tokens,
        args.top_k,
        args.num_experts,
        rank,
        world_size,
        device,
        mode=args.routing_mode,
    )
    topk_weights = torch.softmax(
        torch.randn(tokens, args.top_k, device=device, generator=g), dim=-1
    )
    return hidden_states, topk_ids, topk_weights.to(torch.float32)


def _raw_mxfp4_shapes(
    *, local_experts: int, hidden: int, intermediate: int
) -> dict[str, tuple[int, int, int]]:
    if local_experts <= 0:
        raise ValueError("local_experts must be positive")
    if hidden <= 0 or hidden % 128:
        raise ValueError("MXFP4 hidden must be a positive multiple of 128")
    if intermediate <= 0 or intermediate % 128:
        raise ValueError("MXFP4 intermediate must be a positive multiple of 128")
    return {
        "w13": (local_experts, 2 * intermediate, hidden // 2),
        "w13_scale": (local_experts, 2 * intermediate, hidden // 32),
        "w2": (local_experts, hidden, intermediate // 2),
        "w2_scale": (local_experts, hidden, intermediate // 32),
    }


def _make_raw_mxfp4_weights(args, local_experts: int, rank: int, device):
    """Deterministic canonical packed E2M1 payload + raw K32 E8M0 scales."""

    import torch

    from flashinfer.moe_ep.weights import PrequantizedMoEWeights

    shapes = _raw_mxfp4_shapes(
        local_experts=local_experts,
        hidden=args.hidden,
        intermediate=args.intermediate,
    )
    generator = torch.Generator(device=device).manual_seed(
        MXFP4_WEIGHT_SEED_BASE + rank
    )

    def payload(name: str) -> torch.Tensor:
        # Every uint8 is exactly two canonical E2M1 nibbles.
        return torch.randint(
            0,
            256,
            shapes[name],
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )

    def exponent(name: str) -> torch.Tensor:
        return torch.randint(
            MXFP4_E8M0_MIN,
            MXFP4_E8M0_MAX_EXCLUSIVE,
            shapes[name],
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )

    return PrequantizedMoEWeights(
        w13=payload("w13"),
        w2=payload("w2"),
        w13_scale=exponent("w13_scale"),
        w2_scale=exponent("w2_scale"),
    )


def _make_transformed_weights(
    args,
    backend: str,
    scale_mode: str,
    local_experts: int,
    rank: int,
    device,
):
    """Canonical input pack -> production kernel-ready weight tuples."""
    import torch

    if backend == MXFP4_BACKEND:
        from flashinfer.moe_ep import preprocess_sm90_pull_mxfp4_mega_weights

        raw = _make_raw_mxfp4_weights(args, local_experts, rank, device)
        transformed = preprocess_sm90_pull_mxfp4_mega_weights(
            raw,
            intermediate_size=args.intermediate,
            hidden_size=args.hidden,
        )
        del raw
        return transformed

    from flashinfer.moe_ep import preprocess_sm90_pull_fp8_mega_weights
    from flashinfer.moe_ep.weights import MoEWeightPack

    g = torch.Generator(device="cuda").manual_seed(FP8_WEIGHT_SEED_BASE + rank)
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


def _pingpong_tile_ok(c) -> bool:
    m, n, _ = c.mma_tiler_mnk
    return (n == 128) if not c.swap_ab else (m == 128)


def _megakernel_config(args, scale_mode: str, operand_order: str, tile, tokens=None):
    if args.backend == MXFP4_BACKEND:
        from flashinfer.moe_ep import (
            Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
        )

        if scale_mode != "mxfp4_hybrid" or operand_order != "swap_ab":
            raise RuntimeError("resolved MXFP4 benchmark contract is inconsistent")
        cache_mode = args.mxfp4_tactic_source == "cache_or_heuristic"
        return Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=args.intermediate,
            top_k=args.top_k,
            kind="fp8_e4m3",
            fp8_scale_mode="mxfp4_hybrid",
            fp8_accum_mode="1xacc",
            knobs=None if cache_mode else _mxfp4_fused_tactic(args, tile),
            # Explicit fused tactics already carry this axis; cache/heuristic
            # mode must leave it omitted so the selected tactic owns it.
            load_balance_mode=None,
            gate_up_clamp=args.gate_up_clamp,
            enable_in_kernel_fc2_reduce=False,
            routing_profile=sm90_routing_profile_from_benchmark_mode(args.routing_mode),
        )

    from flashinfer.moe_ep import Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig

    fp8_knobs = _parse_fp8_knobs_json(args.fp8_knobs_json)
    pingpong = None if args.pingpong == "auto" else args.pingpong == "on"
    cluster_shape_mnk = None
    accum_override = None
    token_back_override = None
    if fp8_knobs is not None:
        swap_ab = None
        mma_tiler_mnk = None
    elif operand_order == "heuristic":
        # All geometry knobs None -> the shim resolves the drop's token-bucket
        # heuristic per point (keyed on scale mode and max tokens per rank).
        swap_ab = None
        mma_tiler_mnk = None
        epi_mode = getattr(args, "epi_mode", "auto")
        swap_token_tile = getattr(args, "swap_token_tile", None)
        if (
            pingpong is not None or epi_mode != "auto" or swap_token_tile is not None
        ) and tokens is not None:
            # A pingpong override alone would flip the shim into its
            # manual-geometry branch (default tiles).  Resolve the bucket's
            # heuristic config here and pass the full geometry with only
            # pingpong flipped, so the comparison keeps the bucket's tile.
            from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
                bootstrap_paths,
            )

            bootstrap_paths()
            from moe_hopper_fp8.heuristic_config import select_heuristic_config

            c = select_heuristic_config(scale_mode, tokens).config
            swap_ab = c.swap_ab
            mma_tiler_mnk = tuple(c.mma_tiler_mnk)
            # Explicit geometry flips the shim into manual mode, which fills
            # every UNSET knob from the drop driver's manual defaults -- not
            # the bucket.  Forward the bucket's cluster shape, accum mode and
            # token-back too, so the only difference really is pingpong.
            cluster_shape_mnk = tuple(c.cluster_shape_mnk)
            accum_override = c.accum_mode
            token_back_override = c.token_back_mode
            if epi_mode != "auto":
                m, n, k = mma_tiler_mnk
                # Epilogue modes differ in warpgroup count AND task-tile size:
                #   basic       1 WG,  per-WG tile (N128 non-swap / M128 swap)
                #   pingpong    2 WGs, per-WG tile each, alternating tiles
                #   cooperative 2 WGs, one doubled tile (N256 / M256) split
                # The tile dim that encodes this is N for non-swap, M for swap.
                per_wg_tile, doubled_tile = 128, 256
                if epi_mode == "cooperative":
                    mma_tiler_mnk = (
                        (doubled_tile, n, k) if c.swap_ab else (m, doubled_tile, k)
                    )
                    pingpong = False
                else:  # basic and pingpong share the per-WG tile size
                    mma_tiler_mnk = (
                        (per_wg_tile, n, k) if c.swap_ab else (m, per_wg_tile, k)
                    )
                    pingpong = epi_mode == "pingpong"
            if swap_token_tile is not None and c.swap_ab:
                m, _n, k = mma_tiler_mnk
                mma_tiler_mnk = (m, swap_token_tile, k)
                if pingpong is None:
                    pingpong = c.pingpong
            if (
                pingpong
                and not c.pingpong
                and not _pingpong_tile_ok(
                    type(
                        "T", (), {"mma_tiler_mnk": mma_tiler_mnk, "swap_ab": c.swap_ab}
                    )
                )
            ):
                pingpong = c.pingpong  # bucket tile can't run ping-pong
    else:
        swap_ab = operand_order == "swap_ab"
        mma_tiler_mnk = (tile[0], tile[1], 128)
        if getattr(args, "cga", None):
            cm, cn = (int(v) for v in args.cga.split(","))
            cluster_shape_mnk = (cm, cn, 1)
    return Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=args.intermediate,
        top_k=args.top_k,
        kind=args.kind,
        fp8_scale_mode=scale_mode,
        fp8_accum_mode=(
            args.fp8_accum_mode if args.fp8_accum_mode is not None else accum_override
        ),
        swap_ab=swap_ab,
        mma_tiler_mnk=mma_tiler_mnk,
        cluster_shape_mnk=cluster_shape_mnk,
        load_balance_mode=args.load_balance_mode,
        gate_up_clamp=args.gate_up_clamp,
        enable_in_kernel_fc2_reduce=False,
        token_back_mode=(
            "reuse_dispatch_warps"
            if args.grouped_token_back
            else (
                token_back_override
                if args.token_back == "heuristic"
                else args.token_back
            )
        ),
        pingpong=pingpong,
        dedup_dispatch=args.dedup_dispatch,
        grouped_token_back=args.grouped_token_back,
        combine_format=args.combine_format,
        active_dispatch_warps=args.active_dispatch_warps,
        compact_pull_buffer=args.compact_pull_buffer,
        fc1_store_offload=args.fc1_store_offload,
        fc1_early_done_publish=args.fc1_early_done_publish,
        fold_producer_warps=args.fold_producer_warps,
        knobs=fp8_knobs,
        generate_c=args.generate_c,
        fc1_activation_dequant_scale=FC1_ACT_SCALE,
        fc2_activation_dequant_scale=FC2_ACT_SCALE,
    )


def _canonical_runtime_tactic_sha256(
    implementation: str, tactic: dict[str, object]
) -> str:
    payload = {"implementation": implementation, "tactic": tactic}
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _runtime_tactic_envelope(
    implementation: str, tactic: dict[str, object]
) -> dict[str, object]:
    expected_by_implementation = {
        "fp8_per_tensor": FP8_RUNTIME_TACTIC_FIELDS,
        "fp8_blockwise": FP8_RUNTIME_TACTIC_FIELDS,
        "mxfp4_fused": MXFP4_FUSED_RUNTIME_TACTIC_FIELDS,
    }
    try:
        expected = expected_by_implementation[implementation]
    except KeyError as exc:
        raise RuntimeError(
            f"unsupported runtime tactic implementation {implementation!r}"
        ) from exc
    normalized = json.loads(json.dumps(tactic))
    if set(normalized) != expected:
        raise RuntimeError(
            f"{implementation} runtime tactic fields differ: "
            f"missing={sorted(expected - set(normalized))}, "
            f"extra={sorted(set(normalized) - expected)}"
        )
    return {
        "runtime_implementation": implementation,
        "runtime_tactic": normalized,
        "runtime_tactic_sha256": _canonical_runtime_tactic_sha256(
            implementation, normalized
        ),
    }


def _runtime_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be a positive integer, got {value!r}")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{label} must be a positive integer, got {value!r}"
        ) from exc
    if resolved <= 0 or value != resolved:
        raise RuntimeError(f"{label} must be a positive integer, got {value!r}")
    return resolved


def _kernel_schedule_values(kernel: object, label: str) -> tuple[int, int]:
    for name in ("group_hint", "num_sched_stages"):
        if not hasattr(kernel, name):
            raise RuntimeError(f"{label} compiled kernel lacks {name}")
    return (
        _runtime_positive_int(kernel.group_hint, f"{label} group_hint"),
        _runtime_positive_int(
            kernel.num_sched_stages,
            f"{label} num_sched_stages",
        ),
    )


def _compiled_fused_kernel(frontend: object, label: str) -> object:
    mega = getattr(frontend, "_mega", None)
    kernel = getattr(mega, "kernel", None)
    if kernel is None:
        raise RuntimeError(f"{label} frontend lacks an actual compiled kernel")
    return kernel


def _compiled_fused_schedule(frontend: object, label: str) -> tuple[int, int]:
    return _kernel_schedule_values(_compiled_fused_kernel(frontend, label), label)


def _compiled_tail_split_pairs(frontend: object, config: object, label: str) -> bool:
    kernel = _compiled_fused_kernel(frontend, label)
    if not hasattr(kernel, "tail_split_pairs"):
        raise RuntimeError(f"{label} compiled kernel lacks tail_split_pairs")
    actual = bool(kernel.tail_split_pairs)
    if actual != config.tail_split_pairs:
        raise RuntimeError(
            f"{label} compiled tail_split_pairs={actual} "
            f"!= requested {config.tail_split_pairs}"
        )
    return actual


def _compiled_fused_execution_knobs(
    frontend: object,
    config: object,
    label: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Read requested config and effective codegen values separately."""

    kernel = _compiled_fused_kernel(frontend, label)
    token_comm = getattr(kernel, "token_comm", None)
    if token_comm is None or not hasattr(token_comm, "active_dispatch_warps"):
        raise RuntimeError(
            f"{label} compiled kernel lacks token_comm.active_dispatch_warps"
        )
    required_kernel_fields = (
        "dedup_dispatch",
        "grouped_token_back",
        "combine_format",
        "fc1_store_offload",
        "fc1_early_done_publish",
        "fold_producer_warps",
    )
    missing = [name for name in required_kernel_fields if not hasattr(kernel, name)]
    if missing:
        raise RuntimeError(
            f"{label} compiled kernel lacks execution field(s): " + ", ".join(missing)
        )
    combine = kernel.combine_format
    combine_name = str(getattr(combine, "name", combine))
    requested = {
        "dedup_dispatch": bool(config.dedup_dispatch),
        "grouped_token_back": bool(config.grouped_token_back),
        "combine_format": str(config.combine_format),
        "active_dispatch_warps": int(config.active_dispatch_warps),
        "fc1_store_offload": bool(config.fc1_store_offload),
        "fc1_early_done_publish": bool(config.fc1_early_done_publish),
        "fold_producer_warps": bool(config.fold_producer_warps),
    }
    effective = {
        "dedup_dispatch": bool(kernel.dedup_dispatch),
        "grouped_token_back": bool(kernel.grouped_token_back),
        "combine_format": combine_name,
        "active_dispatch_warps": int(token_comm.active_dispatch_warps),
        "fc1_store_offload": bool(kernel.fc1_store_offload),
        "fc1_early_done_publish": bool(kernel.fc1_early_done_publish),
        "fold_producer_warps": bool(kernel.fold_producer_warps),
    }
    return requested, effective


def _verify_requested_schedule(
    config: object, group_hint: int, num_sched_stages: int, label: str
) -> None:
    for name, actual in (
        ("group_hint", group_hint),
        ("num_sched_stages", num_sched_stages),
    ):
        requested = getattr(config, name, None)
        if requested is not None and int(requested) != actual:
            raise RuntimeError(
                f"{label} compiled {name}={actual} != requested {requested}"
            )


def _fp8_runtime_metadata(
    args: argparse.Namespace, workspace: object
) -> dict[str, object]:
    """Return the actual compiled compute-workspace tactic."""
    if args.backend != FP8_BACKEND:
        return {}
    frontend = getattr(workspace, "_frontend", None)
    config = getattr(frontend, "config", None)
    if config is None:
        raise RuntimeError("FP8 benchmark workspace lacks resolved frontend config")
    group_hint, num_sched_stages = _compiled_fused_schedule(frontend, "FP8")
    _verify_requested_schedule(config, group_hint, num_sched_stages, "FP8")
    requested_execution, effective_execution = _compiled_fused_execution_knobs(
        frontend, config, "FP8"
    )
    scale_mode = str(config.fp8_scale_mode)
    try:
        implementation = {
            "per_tensor": "fp8_per_tensor",
            "blockwise": "fp8_blockwise",
        }[scale_mode]
    except KeyError as exc:
        raise RuntimeError(
            f"FP8 runtime has unsupported scale mode {scale_mode!r}"
        ) from exc
    mode = (
        "explicit_knobs"
        if args.fp8_knobs_json is not None
        else (
            "cache_or_heuristic"
            if args.operand_order in (None, "heuristic")
            else "manual_geometry"
        )
    )
    tactic: dict[str, object] = {
        "swap_ab": bool(config.swap_ab),
        "pingpong": bool(config.pingpong),
        "mma_tiler_mnk": tuple(int(v) for v in config.mma_tiler_mnk),
        "cluster_shape_mnk": tuple(int(v) for v in config.cluster_shape_mnk),
        "fp8_accum_mode": str(config.fp8_accum_mode),
        "group_hint": group_hint,
        "tail_split_pairs": _compiled_tail_split_pairs(frontend, config, "FP8"),
        "num_sched_stages": num_sched_stages,
        "flag_batch": int(config.flag_batch),
        "epi_flag_batch": tuple(int(v) for v in config.epi_flag_batch),
        "load_balance_mode": str(config.load_balance_mode),
        "token_back_mode": str(config.resolved_token_back_mode),
        "in_kernel_fc2_reduce": bool(config.in_kernel_fc2_reduce),
        "compact_pull_buffer": bool(config.compact_pull_buffer),
        "generate_c": bool(config.generate_c),
        **effective_execution,
    }
    return {
        "tactic_mode": mode,
        "execution_knobs_requested": requested_execution,
        "execution_knobs_effective": effective_execution,
        **tactic,
        **_runtime_tactic_envelope(implementation, tactic),
    }


def _verified_mxfp4_runtime_routing_profile(
    args: argparse.Namespace, runtime_config: object, label: str
) -> str:
    expected = sm90_routing_profile_from_benchmark_mode(args.routing_mode)
    try:
        actual = normalize_sm90_routing_profile(runtime_config.routing_profile)
    except (AttributeError, ValueError) as exc:
        raise RuntimeError(
            f"{label} lacks a valid canonical routing_profile identity"
        ) from exc
    if actual != expected:
        raise RuntimeError(
            f"{label} routing_profile {actual!r} != requested {expected!r}"
        )
    return actual


def _mxfp4_fused_runtime_metadata(
    args: argparse.Namespace, workspace: object
) -> dict[str, object]:
    """Return the actual compiled Humming fused tactic."""
    if args.backend != MXFP4_BACKEND:
        return {}
    frontend = getattr(workspace, "_frontend", None)
    config = getattr(frontend, "config", None)
    if config is None:
        raise RuntimeError(
            "MXFP4 fused benchmark workspace lacks resolved frontend config"
        )
    routing_profile = _verified_mxfp4_runtime_routing_profile(
        args, config, "MXFP4 fused runtime"
    )
    if not hasattr(frontend, "requested_tactic") or not hasattr(
        frontend, "effective_tactic"
    ):
        raise RuntimeError(
            "MXFP4 fused frontend lacks canonical requested/effective tactic APIs"
        )
    requested_tactic = frontend.requested_tactic()
    tactic = frontend.effective_tactic()
    if tactic["tail_split_pairs"] != _compiled_tail_split_pairs(
        frontend, config, "MXFP4 fused"
    ):
        raise RuntimeError(
            "MXFP4 effective tactic differs from compiled tail_split_pairs"
        )
    _verify_requested_schedule(
        config,
        int(tactic["group_hint"]),
        int(tactic["num_sched_stages"]),
        "MXFP4 fused",
    )
    requested_execution = {
        name: requested_tactic[name] for name in _FUSED_EXECUTION_TACTIC_FIELDS
    }
    effective_execution = {
        name: tactic[name] for name in _FUSED_EXECUTION_TACTIC_FIELDS
    }
    return {
        "routing_profile": routing_profile,
        "execution_knobs_requested": requested_execution,
        "execution_knobs_effective": effective_execution,
        **tactic,
        **_runtime_tactic_envelope("mxfp4_fused", tactic),
    }


def _time_first_call(call) -> float:
    """Synchronized wall time in us, intentionally including compile/JIT."""

    import torch
    import torch.distributed as dist

    dist.barrier()
    torch.cuda.synchronize()
    start_ns = time.perf_counter_ns()
    call()
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter_ns() - start_ns) / 1e3
    dist.barrier()
    return elapsed_us


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


def _cooldown(seconds: float) -> None:
    """Idle the GPUs so clocks recover before the next timed series.

    Drain all work first, then host-sleep with the device idle; the trailing
    barrier re-aligns ranks so no rank starts its timed series against peers
    still sleeping.
    """
    import time

    import torch
    import torch.distributed as dist

    if seconds <= 0:
        return
    torch.cuda.synchronize()
    dist.barrier()
    time.sleep(seconds)
    dist.barrier()


def _is_oom(exc: BaseException) -> bool:
    import torch

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "oom" in msg or "nvshmem_malloc" in msg


def _run_point(
    args, scale_mode: str, operand_order: str, tile, tokens: int
) -> PointResult:
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
    compute_call = None
    result: PointResult | None = None
    error = ""
    try:
        kcfg = _megakernel_config(args, scale_mode, operand_order, tile, tokens=tokens)
        transformed = _make_transformed_weights(
            args,
            args.backend,
            scale_mode,
            local_experts,
            rank,
            device,
        )
        if args.use_sparse_data and args.backend == FP8_BACKEND:
            # Drop perf recipe for weights: positive-only random E4M3 bytes.
            for tw in (transformed[0][0], transformed[1][0]):
                tw.view(torch.uint8).random_(0, 127)
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
        # Resolve the registry before timing and prove it returned the exact
        # requested implementation. This makes a missing MXFP4 registration a
        # hard failure rather than a run of the ordinary FP8 backend.
        bench_backend = create_mega_kernel(kcfg)
        _assert_backend_identity(bench_backend, args.backend)

        # Weights are preprocessed once above and shared by both series
        # (transformed_weights path: the layer never touches the canonical pack).
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

        # --- series 1: first compile/JIT call, then warm FI e2e. ---
        _cooldown(args.cooldown_s)
        cold = _time_first_call(lambda: layer.forward(t))
        e2e = _time_calls(
            lambda: layer.forward(t), warmup=args.warmup, iters=args.iters
        )

        # --- series 2: compute-only via the documented backend plugin API.
        # prepare_workspace may share the layer's pooled buffer.
        # Stage once, then time bare
        # compute(output=None) launches (zero-copy output) — the closest FI
        # analogue of the drop's mega+topk timed region.
        bench_backend.bind_ep_bootstrap(bootstrap)
        bench_workspace = bench_backend.prepare_workspace(bootstrap, fleet_params)
        bench_backend.stage_inputs(t, bench_workspace, quantize_input=True)
        if args.use_sparse_data and args.backend == FP8_BACKEND:
            # Drop perf recipe for activations: replace the staged fp8
            # payload with uniform random finite E4M3 bytes (127=nan skipped).
            xb = bench_workspace.x.view(torch.uint8)
            idx = torch.randint(0, 254, xb.shape, device=xb.device, dtype=torch.int16)
            xb.copy_(torch.where(idx < 127, idx, idx + 1).to(torch.uint8))
        compute_call = lambda: bench_backend.compute(
            bench_workspace, transformed, output=None
        )
        _cooldown(args.cooldown_s)
        compute = _time_calls(
            compute_call,
            warmup=args.warmup,
            iters=args.iters,
        )
        torch.cuda.synchronize()
        if args.backend == FP8_BACKEND:
            runtime_metadata = _fp8_runtime_metadata(args, bench_workspace)
        else:
            runtime_metadata = _mxfp4_fused_runtime_metadata(args, bench_workspace)
        my_stats = (
            "pass",
            cold,
            fmean(e2e),
            median(e2e),
            fmean(compute),
            median(compute),
            runtime_metadata,
            "",
        )
    except Exception as exc:  # noqa: BLE001 - sweep must survive one bad point
        status = "skip_oom" if _is_oom(exc) else "failed"
        error = f"{type(exc).__name__}: {exc}"
        my_stats = (
            status,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            {},
            error,
        )
    finally:
        # Free THIS point's session before the next allocation (the 32k-token
        # workspace needs the symmetric heap to itself). Backend release first
        # (drops the pool refcount), then the layer's (last release frees).
        compute_call = None
        gc.collect()
        if bench_backend is not None and bench_workspace is not None:
            with contextlib.suppress(Exception):
                bench_backend.destroy(bench_workspace)
        if layer is not None:
            with contextlib.suppress(Exception):
                layer.destroy()
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
            cold_us=[s[1] for s in all_stats],
            e2e_us=[s[2] for s in all_stats],
            e2e_median_us=[s[3] for s in all_stats],
            compute_us=[s[4] for s in all_stats],
            compute_median_us=[s[5] for s in all_stats],
            runtime_metadata=[s[6] for s in all_stats],
        )
    else:
        worst = "skip_oom" if "skip_oom" in statuses else "failed"
        errors = "; ".join(f"rank{i}:{s[7]}" for i, s in enumerate(all_stats) if s[7])
        result = PointResult(
            status=worst,
            cold_us=[],
            e2e_us=[],
            e2e_median_us=[],
            compute_us=[],
            compute_median_us=[],
            error=errors,
        )
    return result


def _ref_csv_name(backend: str, scale_mode: str, operand_order: str, tile) -> str:
    if backend == MXFP4_BACKEND:
        return "not_applicable(mxfp4)"
    if operand_order == "heuristic":
        # Per-point geometry follows the token bucket; no single drop CSV.
        return "heuristic(no-single-ref)"
    scale_tag = "pertensor" if scale_mode == "per_tensor" else "blockwise"
    order_tag = "swapab" if operand_order == "swap_ab" else "nonswapab"
    return (
        f"{REF_DATE}_multirank_{scale_tag}_{order_tag}_"
        f"TileM{tile[0]}_TileN{tile[1]}.csv"
    )


def _formal_tuning_cols(
    args: argparse.Namespace,
    tile,
    result: PointResult,
) -> list[str]:
    """Append-only score and complete MXFP4 tactic identity columns."""

    score = (
        f"{max(result.compute_median_us):.6f}"
        if result.status == "pass" and result.compute_median_us
        else "nan"
    )
    fused = [""] * 8
    resolved_tactic: dict[str, object] | None = None
    if args.mxfp4_tactic_source == "cache_or_heuristic" and result.status == "pass":
        _, resolved_tactic, _ = _all_rank_runtime_tactic(
            result, len(result.compute_median_us)
        )
    if args.backend == MXFP4_BACKEND:
        tactic = resolved_tactic
        if tactic is None and args.mxfp4_tactic_source != "cache_or_heuristic":
            tactic = _mxfp4_fused_tactic(
                args,
                (int(tile[0]), int(tile[1])),
            )
        if tactic is not None:
            cluster = _runtime_positive_triplet(
                tactic["cluster_shape_mnk"],
                "cluster_shape_mnk",
            )
            group = tactic["group_hint"]
            stages = tactic["num_sched_stages"]
            fused = [
                str(int(bool(tactic["pingpong"]))),
                str(cluster[0]),
                str(cluster[1]),
                str(cluster[2]),
                "" if group is None else str(group),
                "" if stages is None else str(stages),
                str(tactic["load_balance_mode"]),
                str(tactic["token_back_mode"]),
            ]
    cols = [score, *fused]
    expected = len(FORMAL_TUNING_CSV_FIELDS.split(","))
    if len(cols) != expected:
        raise RuntimeError(
            f"formal tuning CSV schema mismatch: {len(cols)} values != {expected}"
        )
    return cols


def _fp8_runtime_cols(args: argparse.Namespace, result: PointResult) -> list[str]:
    """Append-only actual FP8 compute-workspace tactic identity."""
    count = len(FP8_RUNTIME_CSV_FIELDS.split(","))
    if args.backend != FP8_BACKEND or result.status != "pass":
        return [""] * count
    metadata = result.runtime_metadata
    if (
        metadata is None
        or not metadata
        or len(metadata) != len(result.compute_median_us)
    ):
        raise RuntimeError("FP8 result lacks all-rank resolved tactic metadata")
    names = (
        "tactic_mode",
        "swap_ab",
        "pingpong",
        "mma_tiler_mnk",
        "cluster_shape_mnk",
        "fp8_accum_mode",
        "group_hint",
        "num_sched_stages",
        "flag_batch",
        "epi_flag_batch",
        "load_balance_mode",
        "token_back_mode",
        "in_kernel_fc2_reduce",
    )
    try:
        resolved = [{name: record[name] for name in names} for record in metadata]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("FP8 result has malformed resolved tactic metadata") from exc
    if any(record != resolved[0] for record in resolved[1:]):
        raise RuntimeError("FP8 ranks disagree on the resolved tactic identity")
    actual = resolved[0]
    mma, cluster, epi = (
        tuple(actual["mma_tiler_mnk"]),
        tuple(actual["cluster_shape_mnk"]),
        tuple(actual["epi_flag_batch"]),
    )
    if len(mma) != 3 or len(cluster) != 3 or len(epi) != 2:
        raise RuntimeError("FP8 resolved tactic has malformed tuple fields")
    cols = [
        str(actual["tactic_mode"]),
        str(int(bool(actual["swap_ab"]))),
        str(int(bool(actual["pingpong"]))),
        *(str(v) for v in mma),
        *(str(v) for v in cluster),
        str(actual["fp8_accum_mode"]),
        "auto" if actual["group_hint"] is None else str(actual["group_hint"]),
        (
            "auto"
            if actual["num_sched_stages"] is None
            else str(actual["num_sched_stages"])
        ),
        str(actual["flag_batch"]),
        *(str(v) for v in epi),
        str(actual["load_balance_mode"]),
        str(actual["token_back_mode"]),
        str(int(bool(actual["in_kernel_fc2_reduce"]))),
    ]
    if len(cols) != count:
        raise RuntimeError(f"FP8 runtime CSV schema mismatch: {len(cols)} != {count}")
    return cols


def _expected_runtime_implementation(args: argparse.Namespace, scale_mode: str) -> str:
    if args.backend == FP8_BACKEND:
        try:
            return {
                "per_tensor": "fp8_per_tensor",
                "blockwise": "fp8_blockwise",
            }[scale_mode]
        except KeyError as exc:
            raise RuntimeError(
                f"unsupported FP8 runtime scale mode {scale_mode!r}"
            ) from exc
    return "mxfp4_fused"


def _all_rank_runtime_tactic(
    result: PointResult, world_size: int
) -> tuple[str, dict[str, object], str]:
    metadata = result.runtime_metadata
    if metadata is None or len(metadata) != world_size:
        raise RuntimeError("result lacks all-rank runtime tactic metadata")
    identities: list[tuple[str, dict[str, object], str]] = []
    try:
        for record in metadata:
            implementation = str(record["runtime_implementation"])
            tactic = json.loads(json.dumps(record["runtime_tactic"]))
            digest = str(record["runtime_tactic_sha256"])
            expected = _runtime_tactic_envelope(implementation, tactic)
            if digest != expected["runtime_tactic_sha256"]:
                raise RuntimeError(
                    f"{implementation} runtime tactic SHA-256 does not match "
                    "its canonical tactic"
                )
            identities.append((implementation, tactic, digest))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("result has malformed runtime tactic metadata") from exc
    if any(identity != identities[0] for identity in identities[1:]):
        raise RuntimeError("ranks disagree on canonical runtime tactic identity")
    return identities[0]


def _runtime_tactic_cols(
    args: argparse.Namespace,
    scale_mode: str,
    result: PointResult,
    world_size: int,
) -> list[str]:
    count = len(RUNTIME_TACTIC_CSV_FIELDS.split(","))
    if result.status != "pass":
        return [""] * count
    implementation, tactic, digest = _all_rank_runtime_tactic(result, world_size)
    expected = _expected_runtime_implementation(args, scale_mode)
    if implementation != expected:
        raise RuntimeError(
            f"runtime implementation {implementation!r} != expected {expected!r}"
        )
    cols = [digest, str(tactic["group_hint"]), str(tactic["num_sched_stages"])]
    if len(cols) != count:
        raise RuntimeError(
            f"runtime tactic CSV schema mismatch: {len(cols)} != {count}"
        )
    return cols


def _routing_csv_cols(
    args: argparse.Namespace,
    tokens: int,
    world_size: int,
    result: PointResult,
) -> list[str]:
    """Append the global input-route identity and verify MXFP4 runtime profile."""

    profile = sm90_routing_profile_from_benchmark_mode(args.routing_mode)
    if args.backend == MXFP4_BACKEND and result.status == "pass":
        metadata = result.runtime_metadata
        if metadata is None or len(metadata) != world_size:
            raise RuntimeError("MXFP4 result lacks all-rank routing metadata")
        try:
            runtime_profiles = [
                normalize_sm90_routing_profile(record["routing_profile"])
                for record in metadata
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("MXFP4 result has malformed routing metadata") from exc
        if any(value != profile for value in runtime_profiles):
            raise RuntimeError(
                "MXFP4 ranks disagree with the requested routing_profile"
            )

    seed = ROUTING_SEED
    routes = generate_sm90_routing_numpy(
        routing_profile=profile,
        world_size=world_size,
        tokens=tokens,
        topk=args.top_k,
        total_experts=args.num_experts,
        seed=seed,
    )
    audit = sm90_routing_audit_payload(
        routes,
        routing_profile=profile,
        seed=seed,
        total_experts=args.num_experts,
        world_size=world_size,
    )
    cols = [
        args.routing_mode,
        profile,
        str(seed),
        str(audit["route_ids_sha256"]),
    ]
    expected = len(ROUTING_CSV_FIELDS.split(","))
    if len(cols) != expected:
        raise RuntimeError(
            f"routing CSV schema mismatch: {len(cols)} values != {expected}"
        )
    return cols


def _fused_execution_runtime_cols(
    args: argparse.Namespace,
    result: PointResult,
    world_size: int,
) -> list[str]:
    """Append requested/effective latest fused fields from the actual kernel."""

    count = len(FUSED_EXECUTION_RUNTIME_CSV_FIELDS.split(","))
    if result.status != "pass" or args.backend not in (FP8_BACKEND, MXFP4_BACKEND):
        return [""] * count
    metadata = result.runtime_metadata
    if metadata is None or len(metadata) != world_size:
        raise RuntimeError("fused result lacks all-rank execution metadata")
    try:
        identities = [
            (
                json.loads(json.dumps(record["execution_knobs_requested"])),
                json.loads(json.dumps(record["execution_knobs_effective"])),
            )
            for record in metadata
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("fused result has malformed execution metadata") from exc
    if any(identity != identities[0] for identity in identities[1:]):
        raise RuntimeError("ranks disagree on fused execution knob identity")
    requested, effective = identities[0]
    if set(requested) != _FUSED_EXECUTION_TACTIC_FIELDS:
        raise RuntimeError("requested fused execution metadata fields differ")
    if set(effective) != _FUSED_EXECUTION_TACTIC_FIELDS:
        raise RuntimeError("effective fused execution metadata fields differ")
    cols = [
        str(int(bool(effective["dedup_dispatch"]))),
        str(int(bool(effective["grouped_token_back"]))),
        str(effective["combine_format"]),
        str(effective["active_dispatch_warps"]),
        str(int(bool(requested["fc1_store_offload"]))),
        str(int(bool(effective["fc1_store_offload"]))),
        str(int(bool(requested["fc1_early_done_publish"]))),
        str(int(bool(effective["fc1_early_done_publish"]))),
        str(int(bool(requested["fold_producer_warps"]))),
        str(int(bool(effective["fold_producer_warps"]))),
    ]
    if len(cols) != count:
        raise RuntimeError(
            f"fused execution CSV schema mismatch: {len(cols)} != {count}"
        )
    return cols


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
    csv_file=None,
) -> None:
    fc1, fc2, total = _flops_per_rank(
        tokens, args.top_k, args.hidden, args.intermediate
    )
    ref_csv = _ref_csv_name(args.backend, scale_mode, operand_order, tile)
    tactic = _tactic_label(args, operand_order=operand_order, tile=tile)
    if "," in tactic:
        raise RuntimeError(f"tactic label must be CSV-safe, got {tactic!r}")
    if not header_done:
        print(CSV_HEADER, flush=True)
        if csv_file is not None:
            csv_file.write(
                f"{CSV_FIELDS},{HEUR_CSV_FIELDS},{BENCH_EXT_CSV_FIELDS},"
                f"{FORMAL_TUNING_CSV_FIELDS},{FP8_RUNTIME_CSV_FIELDS},"
                f"{RUNTIME_TACTIC_CSV_FIELDS},{ROUTING_CSV_FIELDS},"
                f"{COMPUTE_LAUNCH_CSV_FIELDS},"
                f"{FUSED_EXECUTION_RUNTIME_CSV_FIELDS}\n"
            )

    reported_tile = tile
    tile_k = 128
    if (
        args.backend == MXFP4_BACKEND
        and args.mxfp4_tactic_source == "cache_or_heuristic"
        and result.status != "pass"
    ):
        reported_tile = ("", "")
        tile_k = ""
    if args.backend == FP8_BACKEND and args.fp8_knobs_json is not None:
        fp8_knobs = _parse_fp8_knobs_json(args.fp8_knobs_json)
        assert fp8_knobs is not None
        fp8_mma = fp8_knobs["mma_tiler_mnk"]
        assert isinstance(fp8_mma, tuple)
        tile_k = int(fp8_mma[2])
    if args.backend == MXFP4_BACKEND:
        fused_tactic = None
        if args.mxfp4_tactic_source == "cache_or_heuristic":
            if result.status == "pass":
                implementation, fused_tactic, _ = _all_rank_runtime_tactic(
                    result,
                    world_size,
                )
                if implementation != "mxfp4_fused":
                    raise RuntimeError("fused cache mode resolved a non-fused tactic")
        else:
            fused_tactic = _mxfp4_fused_tactic(
                args,
                (int(tile[0]), int(tile[1])),
            )
        if fused_tactic is not None:
            fused_mma = _runtime_positive_triplet(
                fused_tactic["mma_tiler_mnk"],
                "mma_tiler_mnk",
            )
            reported_tile = (fused_mma[0], fused_mma[1])
            tile_k = fused_mma[2]
    prefix = (
        f"{args.backend},{scale_mode},{operand_order},"
        f"{reported_tile[0]},{reported_tile[1]},{tile_k},"
        f"{tokens},{args.top_k},{world_size},{args.num_experts},"
        f"{args.num_experts // world_size},{args.hidden},{args.intermediate},"
        f"{2 * args.intermediate},{args.warmup},{args.iters}"
    )
    if result.status != "pass":
        row = (
            f"{prefix},{result.status},"
            + ",".join(["nan"] * 8)
            + f",{fc1},{fc2},{total},nan,nan,nan,{ref_csv}"
        )
    else:
        cold_min, cold_max, cold_mean = (
            min(result.cold_us),
            max(result.cold_us),
            fmean(result.cold_us),
        )
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
        # critical_tflops_per_rank = total_flops / max_mega_us), tok/s over
        # the slowest rank's e2e.
        tflops_c = _tflops(total, c_max)
        tflops_e2e = _tflops(total, e2e_max)
        tok_s = tokens * world_size / (e2e_max * 1e-6)
        row = (
            f"{prefix},pass,"
            f"{e2e_min:.2f},{e2e_max:.2f},{e2e_mean:.2f},{e2e_med:.2f},"
            f"{c_min:.2f},{c_max:.2f},{c_mean:.2f},{c_med:.2f},"
            f"{fc1},{fc2},{total},{tflops_c:.2f},{tflops_e2e:.2f},"
            f"{tok_s:.1f},{ref_csv}"
        )

    cold_cols = (
        [
            f"{cold_min:.2f}",
            f"{cold_max:.2f}",
            f"{cold_mean:.2f}",
        ]
        if result.status == "pass"
        else ["nan"] * 3
    )
    bench_ext_cols = [
        tactic,
        *cold_cols,
    ]
    expected_ext = len(BENCH_EXT_CSV_FIELDS.split(","))
    if len(bench_ext_cols) != expected_ext:
        raise RuntimeError(
            f"benchmark extension CSV schema mismatch: "
            f"{len(bench_ext_cols)} values != {expected_ext}"
        )
    bench_ext_row = ",".join(str(value) for value in bench_ext_cols)
    formal_tuning_row = ",".join(_formal_tuning_cols(args, tile, result))
    fp8_runtime_row = ",".join(_fp8_runtime_cols(args, result))
    runtime_tactic_row = ",".join(
        _runtime_tactic_cols(args, scale_mode, result, world_size)
    )
    routing_row = ",".join(_routing_csv_cols(args, tokens, world_size, result))
    compute_launch_row = "direct"
    fused_execution_row = ",".join(
        _fused_execution_runtime_cols(args, result, world_size)
    )
    print(
        f"BENCH_CSV,{row},{bench_ext_row},"
        f"{formal_tuning_row},{fp8_runtime_row},{runtime_tactic_row},"
        f"{routing_row},{compute_launch_row},{fused_execution_row}",
        flush=True,
    )
    if result.status != "pass" and result.error:
        print(f"# SKIP detail: {result.error}", flush=True)
    if csv_file is not None:
        heur = _heuristic_cols(args.backend, scale_mode, operand_order, tokens)
        csv_file.write(
            row
            + ","
            + ",".join(heur)
            + ","
            + bench_ext_row
            + ","
            + formal_tuning_row
            + ","
            + fp8_runtime_row
            + ","
            + runtime_tactic_row
            + ","
            + routing_row
            + ","
            + compute_launch_row
            + ","
            + fused_execution_row
            + "\n"
        )
        csv_file.flush()


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
    if args.backend == FP8_BACKEND and world_size != 4 and rank == 0:
        print(
            f"# note: world_size={world_size}; the drop reference CSVs are "
            "EP4 — numbers are only directly comparable at 4 ranks.",
            flush=True,
        )

    tokens_list = [int(t) for t in args.tokens.split(",") if t]
    if not tokens_list or any(t <= 0 for t in tokens_list):
        raise SystemExit("--tokens must contain at least one positive integer")
    try:
        scale_modes, orders, tile_override = _resolve_sweep(args, world_size)
    except ValueError as exc:
        dist.destroy_process_group()
        raise SystemExit(str(exc)) from exc
    if rank == 0:
        print(
            f"# backend: {args.backend} execution_mode=fused launch=direct",
            flush=True,
        )
        print(
            "# timing: cold_first_call=sync_wall_including_compile_jit; "
            "e2e/compute=warm_cuda_event",
            flush=True,
        )

    # One NVSHMEM bootstrap for the whole sweep (layers run with
    # auto_bootstrap=False against this shared runtime).
    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, sm90_pull_fp8_runtime_requirements(bootstrap)
    )

    header_done = False
    csv_file = None
    if rank == 0 and args.output_csv and args.output_csv.lower() != "none":
        csv_path = args.output_csv
        if csv_path == "auto":
            import datetime as _dt

            now = _dt.datetime.now()
            if args.backend == FP8_BACKEND:
                # Preserve the original FP8 auto-output naming contract.
                output_tag = (
                    f"{args.operand_order or 'heuristic'}_{args.scale_mode or 'both'}"
                )
            else:
                output_tag = (
                    f"{args.backend}_fused_{'-'.join(orders)}_{'-'.join(scale_modes)}"
                )
            csv_path = os.path.join(
                os.path.dirname(_here),
                "flashinfer",
                "moe_ep",
                "kernel_src",
                "sm90",
                "pull_style_cutedsl_megakernel",
                "benchmark_data",
                now.strftime("%Y%m%d"),
                f"{now.strftime('%Y%m%d_%H%M%S')}_mega_sm90_{output_tag}.csv",
            )
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        # Closed in the ``finally`` below (the handle outlives this block).
        csv_file = open(csv_path, "w")  # noqa: SIM115
        print(f"# output_csv: {csv_path}", flush=True)
    failed_any = False
    try:
        for scale_mode in scale_modes:
            for operand_order in orders:
                if operand_order == "heuristic":
                    tile = ("auto", "auto")
                else:
                    tile = tile_override or DEFAULT_TILE[operand_order]
                tactic = _tactic_label(args, operand_order=operand_order, tile=tile)
                for tokens in tokens_list:
                    if rank == 0:
                        print(
                            f"# [sweep] backend={args.backend} tactic={tactic} "
                            f"scale={scale_mode} tokens_per_rank={tokens}",
                            flush=True,
                        )
                    result = _run_point(args, scale_mode, operand_order, tile, tokens)
                    if args.backend == MXFP4_BACKEND and result.status != "pass":
                        failed_any = True
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
                            csv_file=csv_file,
                        )
                        header_done = True
    finally:
        if csv_file is not None:
            csv_file.close()
        finalize_moe_ep_runtime(runtime)
        dist.barrier()
        dist.destroy_process_group()
    return 1 if failed_any else 0


if __name__ == "__main__":
    sys.exit(main())
