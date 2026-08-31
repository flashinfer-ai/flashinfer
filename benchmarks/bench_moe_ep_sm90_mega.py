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
through the production backend.  --execution-mode chooses the Phase-A fused
path or the Phase-B Green-Context split path:

    python -m torch.distributed.run --standalone --nproc_per_node=4 benchmarks/bench_moe_ep_sm90_mega.py --backend sm90_fp8_mxfp4_bf16_pull_cutedsl --tokens 64

    python -m torch.distributed.run --standalone --nproc_per_node=4 benchmarks/bench_moe_ep_sm90_mega.py --backend sm90_fp8_mxfp4_bf16_pull_cutedsl --execution-mode split --tokens 64

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

# Phase-A known-correct baseline. These fields are passed explicitly to the
# MXFP4 config, which bypasses both the generic FP8 heuristic and every knob
# cache. A CLI tile/token-back override remains fixed for that run and is
# printed verbatim in each result row.
MXFP4_DEFAULT_TILE = (128, 32)
MXFP4_TILE_K = 128
MXFP4_CLUSTER = (1, 1, 1)
MXFP4_PINGPONG = False
MXFP4_TOKEN_BACK = "epi_warps"

# Phase-B known-correct H200 baseline. CUDA returns an 80-SM aligned primary
# Green partition plus the 52-SM remainder on a 132-SM H200. K1 and K2 use
# independent tactics and compile limits; the benchmark verifies the actual
# driver partition exposed by the captured session rather than only printing
# these requested values.
MXFP4_SPLIT_K1_TILE = (256, 64, 128)
MXFP4_SPLIT_K2_TILE = (128, 64, 128)
MXFP4_SPLIT_K1_CLUSTER = (1, 1, 1)
MXFP4_SPLIT_K2_CLUSTER = (1, 1, 1)
MXFP4_SPLIT_K1_SMS = 80
MXFP4_SPLIT_K2_SMS = 52
MXFP4_SPLIT_SCHED_STAGES = 2

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
    "execution_mode,tactic,k1_tactic,k2_tactic,graph_variant,"
    "counter_banks,k1_sm_count,k2_sm_count,"
    "cold_first_call_min_us,cold_first_call_max_us,cold_first_call_mean_us"
)
SPLIT_RUNTIME_CSV_FIELDS = (
    "k1_max_active_clusters,k2_max_active_clusters,handoff_token_n,"
    "rank_session_generations"
)
# Appended-only extension fields. Keep CSV_FIELDS and its historical
# meanings/order stable: existing FP8 parsers consume that prefix verbatim.
FORMAL_TUNING_CSV_FIELDS = (
    "compute_max_rank_median_us,"
    "fused_pingpong,fused_cga_m,fused_cga_n,fused_cga_k,"
    "fused_group_hint,fused_num_sched_stages,fused_load_balance_mode,"
    "fused_token_back_mode,"
    "split_k1_tile_m,split_k1_tile_n,split_k1_tile_k,"
    "split_k2_tile_m,split_k2_tile_n,split_k2_tile_k,"
    "split_k1_cga_m,split_k1_cga_n,split_k1_cga_k,"
    "split_k2_cga_m,split_k2_cga_n,split_k2_cga_k,"
    "split_k1_group_hint,split_k2_group_hint,"
    "split_k1_num_sched_stages,split_k2_num_sched_stages,split_enable_iket"
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
# Canonical actual-runtime identity shared by ordinary FP8, MXFP4 fused, and
# MXFP4 split. Human-readable group/stage columns make the two runtime-default
# aliases directly auditable; the SHA-256 is over the complete tactic below.
RUNTIME_TACTIC_CSV_FIELDS = (
    "runtime_tactic_sha256,runtime_group_hint,runtime_num_sched_stages,"
    "runtime_k1_group_hint,runtime_k2_group_hint,"
    "runtime_k1_num_sched_stages,runtime_k2_num_sched_stages"
)
# Global input-routing identity. This extension remains at the absolute tail
# of both stdout and file CSV schemas so every historical field keeps its
# position and meaning.
ROUTING_CSV_FIELDS = "routing_mode,routing_profile,routing_seed,route_ids_sha256"
FP8_RUNTIME_TACTIC_FIELDS = frozenset(
    {
        "swap_ab",
        "pingpong",
        "mma_tiler_mnk",
        "cluster_shape_mnk",
        "fp8_accum_mode",
        "load_balance_mode",
        "token_back_mode",
        "group_hint",
        "num_sched_stages",
        "flag_batch",
        "epi_flag_batch",
        "in_kernel_fc2_reduce",
    }
)
MXFP4_FUSED_RUNTIME_TACTIC_FIELDS = frozenset(
    {
        "swap_ab",
        "pingpong",
        "mma_tiler_mnk",
        "cluster_shape_mnk",
        "fp8_accum_mode",
        "load_balance_mode",
        "token_back_mode",
        "group_hint",
        "num_sched_stages",
        "in_kernel_fc2_reduce",
    }
)
MXFP4_SPLIT_RUNTIME_TACTIC_FIELDS = frozenset(
    {
        "k1_mma_tiler_mnk",
        "k2_mma_tiler_mnk",
        "k1_cluster_shape_mnk",
        "k2_cluster_shape_mnk",
        "k1_group_hint",
        "k2_group_hint",
        "k1_num_sched_stages",
        "k2_num_sched_stages",
        "k1_sm_count",
        "k2_sm_count",
        "counter_epoch_banks",
        "graph_variant",
        "enable_iket",
    }
)
CSV_HEADER = (
    "BENCH_CSV,"
    + CSV_FIELDS
    + ","
    + BENCH_EXT_CSV_FIELDS
    + ","
    + SPLIT_RUNTIME_CSV_FIELDS
    + ","
    + FORMAL_TUNING_CSV_FIELDS
    + ","
    + FP8_RUNTIME_CSV_FIELDS
    + ","
    + RUNTIME_TACTIC_CSV_FIELDS
    + ","
    + ROUTING_CSV_FIELDS
)

# Resolved launch-config columns appended to --output-csv rows (blank for
# fixed-layout runs; filled from the shim's token-bucket table under
# --heuristic so the file records what each point actually launched).
HEUR_CSV_FIELDS = (
    "heur_swap_ab,heur_pingpong,heur_tile_m,heur_tile_n,heur_tile_k,"
    "heur_cga_m,heur_cga_n,heur_accum_mode,heur_token_back,heur_token_bucket"
)


def _heuristic_cols(
    backend: str, scale_mode: str, operand_order: str, tokens: int
) -> list[str]:
    """The launch config the shim resolves for this point (heuristic mode)."""
    if backend != FP8_BACKEND or operand_order != "heuristic":
        return [""] * 10
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
        "--execution-mode",
        choices=["fused", "split"],
        default="fused",
        help="MXFP4 execution strategy. 'split' is the concurrent Green "
        "Context K0 -> {K1 || K2} -> K3 path and never falls back to fused.",
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
        default=None,
        help="scale ABI(s) to sweep. Default: both for ordinary FP8, "
        "mxfp4_hybrid for the MXFP4 backend",
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
    p.set_defaults(operand_order=None)
    p.add_argument(
        "--mma-tiler",
        type=str,
        default=None,
        metavar="M,N",
        help="override the mma tile (M,N; K fixed at 128). Default: the "
        "shim's per-layout default (non-swap 64,128 / swap-AB 256,32).",
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
        "--mxfp4-mma-tiler",
        type=str,
        default=None,
        metavar="M,N,K",
        help="MXFP4 fused-only full MMA tile. Supports K=128 or K=256; "
        "the legacy --mma-tiler M,N remains a K128-compatible alias.",
    )
    p.add_argument(
        "--mxfp4-cluster",
        type=str,
        default=None,
        metavar="M,N,K",
        help="MXFP4 fused-only cluster shape (default: 1,1,1).",
    )
    p.add_argument(
        "--mxfp4-group-hint",
        type=int,
        default=None,
        help="MXFP4 fused-only scheduler group hint.",
    )
    p.add_argument(
        "--mxfp4-num-sched-stages",
        type=int,
        default=None,
        help="MXFP4 fused-only scheduler pipeline stage count.",
    )
    p.add_argument(
        "--mxfp4-pingpong",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable/disable MXFP4 fused ping-pong scheduling (default: disabled).",
    )
    p.add_argument(
        "--mxfp4-tactic-source",
        choices=["explicit", "cache_or_heuristic"],
        default="explicit",
        help="MXFP4 tactic selector. Default 'explicit' preserves the legacy "
        "CLI-controlled fused/split tactic. 'cache_or_heuristic' passes "
        "knobs=None with no manual geometry so the dedicated persistent "
        "fused/split cache is consulted before the manifest heuristic.",
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
    p.add_argument("--fp8-accum-mode", choices=["1xacc", "2xacc"], default="1xacc")
    p.add_argument(
        "--load-balance-mode",
        choices=["static", "atomic_counter"],
        default=None,
        help="scheduler mode. Default: atomic_counter for fused/FP8, static "
        "for concurrent split K1/K2.",
    )
    p.add_argument(
        "--split-k1-mma-tiler",
        default=",".join(str(v) for v in MXFP4_SPLIT_K1_TILE),
        metavar="M,N,K",
    )
    p.add_argument(
        "--split-k2-mma-tiler",
        default=",".join(str(v) for v in MXFP4_SPLIT_K2_TILE),
        metavar="M,N,K",
    )
    p.add_argument(
        "--split-k1-cluster",
        default=",".join(str(v) for v in MXFP4_SPLIT_K1_CLUSTER),
        metavar="M,N,K",
    )
    p.add_argument(
        "--split-k2-cluster",
        default=",".join(str(v) for v in MXFP4_SPLIT_K2_CLUSTER),
        metavar="M,N,K",
    )
    p.add_argument("--split-k1-group-hint", type=int, default=None)
    p.add_argument("--split-k2-group-hint", type=int, default=None)
    p.add_argument(
        "--split-k1-num-sched-stages",
        type=int,
        default=MXFP4_SPLIT_SCHED_STAGES,
    )
    p.add_argument(
        "--split-k2-num-sched-stages",
        type=int,
        default=MXFP4_SPLIT_SCHED_STAGES,
    )
    p.add_argument("--split-k1-sm-count", type=int, default=MXFP4_SPLIT_K1_SMS)
    p.add_argument("--split-k2-sm-count", type=int, default=MXFP4_SPLIT_K2_SMS)
    p.add_argument("--split-counter-banks", type=int, choices=[1, 2], default=1)
    p.add_argument(
        "--split-graph-variant",
        choices=["cold_k0", "steady_k3_reset"],
        default="steady_k3_reset",
    )
    p.add_argument("--split-enable-iket", action="store_true")
    p.add_argument(
        "--token-back",
        choices=[
            "heuristic",
            "epi_warps",
            "reuse_dispatch_warps",
            "standalone_warps",
        ],
        default=None,
        help="fc2 token-back path. 'heuristic' (default) follows the "
        "per-token-bucket table (epi_warps small/mid buckets, "
        "reuse_dispatch_warps at the GEMM-bound tail); the explicit modes "
        "pin one path for A/B runs (reuse_dispatch_warps matches the "
        "drop's P03 perf-run setting).",
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
    # Argparse defaults intentionally preserve the historical explicit split
    # tactic. Retain which option spellings the caller actually supplied so
    # cache mode can reject (rather than silently ignore) legacy tactic flags.
    args._specified_options = frozenset(
        value.split("=", 1)[0] for value in raw_argv if value.startswith("--")
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
        "num_sched_stages",
        "flag_batch",
        "epi_flag_batch",
        "in_kernel_fc2_reduce",
        "token_back_mode",
        "load_balance_mode",
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
    for name in ("swap_ab", "pingpong", "in_kernel_fc2_reduce"):
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

    if args.execution_mode == "split" and args.backend != MXFP4_BACKEND:
        raise ValueError(
            "--execution-mode split is accepted only by the MXFP4 backend; "
            "ordinary FP8 cannot masquerade as split"
        )

    if args.backend == MXFP4_BACKEND:
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
        scale_mode = args.scale_mode or "mxfp4_hybrid"
        if scale_mode != "mxfp4_hybrid":
            raise ValueError(
                "the MXFP4 backend requires --scale-mode mxfp4_hybrid; "
                "ordinary FP8 scale modes are not fallback candidates"
            )
        operand_order = args.operand_order or "swap_ab"
        if operand_order != "swap_ab":
            raise ValueError(
                "the MXFP4 backend requires --swap-ab; native/both/heuristic "
                "operand order is not a fallback candidate"
            )
        if args.token_back == "heuristic":
            raise ValueError(
                "MXFP4 benchmark tactics must be fixed; choose an explicit "
                "--token-back mode or omit it for epi_warps"
            )
        cache_mode = args.mxfp4_tactic_source == "cache_or_heuristic"
        if cache_mode:
            common_tactic_options = {
                "--load-balance-mode",
                "--mma-tiler",
                "--token-back",
            }
            fused_tactic_options = {
                "--mxfp4-cluster",
                "--mxfp4-group-hint",
                "--mxfp4-mma-tiler",
                "--mxfp4-num-sched-stages",
                "--mxfp4-pingpong",
                "--no-mxfp4-pingpong",
            }
            split_tactic_options = {
                "--split-k1-mma-tiler",
                "--split-k2-mma-tiler",
                "--split-k1-cluster",
                "--split-k2-cluster",
                "--split-k1-group-hint",
                "--split-k2-group-hint",
                "--split-k1-num-sched-stages",
                "--split-k2-num-sched-stages",
                "--split-k1-sm-count",
                "--split-k2-sm-count",
                "--split-counter-banks",
                "--split-graph-variant",
                "--split-enable-iket",
            }
            mode_options = (
                split_tactic_options
                if args.execution_mode == "split"
                else fused_tactic_options
            )
            conflicts = sorted(
                args._specified_options.intersection(
                    common_tactic_options | mode_options
                )
            )
            if conflicts:
                raise ValueError(
                    "--mxfp4-tactic-source cache_or_heuristic conflicts with "
                    + ", ".join(conflicts)
                )
        if args.execution_mode == "fused":
            split_conflicts = sorted(
                args._specified_options.intersection(
                    {
                        "--split-k1-mma-tiler",
                        "--split-k2-mma-tiler",
                        "--split-k1-cluster",
                        "--split-k2-cluster",
                        "--split-k1-group-hint",
                        "--split-k2-group-hint",
                        "--split-k1-num-sched-stages",
                        "--split-k2-num-sched-stages",
                        "--split-k1-sm-count",
                        "--split-k2-sm-count",
                        "--split-counter-banks",
                        "--split-graph-variant",
                        "--split-enable-iket",
                    }
                )
            )
            if split_conflicts:
                raise ValueError(
                    "split tactic flags require --execution-mode split: "
                    + ", ".join(split_conflicts)
                )
        fused_overrides = (
            args.mxfp4_mma_tiler,
            args.mxfp4_cluster,
            args.mxfp4_group_hint,
            args.mxfp4_num_sched_stages,
            args.mxfp4_pingpong,
        )
        if args.execution_mode == "split" and any(
            value is not None for value in fused_overrides
        ):
            raise ValueError(
                "MXFP4 fused tactic flags (--mxfp4-*) cannot be used with "
                "--execution-mode split"
            )
        if args.execution_mode == "split" and args.mma_tiler is not None:
            raise ValueError(
                "legacy --mma-tiler is fused-only; use "
                "--split-k1-mma-tiler and --split-k2-mma-tiler"
            )
        if args.mma_tiler is not None and args.mxfp4_mma_tiler is not None:
            raise ValueError(
                "pass either legacy --mma-tiler M,N or "
                "--mxfp4-mma-tiler M,N,K, not both"
            )
        tile = MXFP4_DEFAULT_TILE
        if args.mma_tiler is not None:
            tile = _parse_mma_tile(args.mma_tiler)
        elif args.mxfp4_mma_tiler is not None:
            tile = _parse_positive_triplet(args.mxfp4_mma_tiler, "--mxfp4-mma-tiler")[
                :2
            ]
        if args.execution_mode == "split":
            if _resolved_token_back(args) != "epi_warps":
                raise ValueError(
                    "split requires direct epilogue token-back (epi_warps)"
                )
            if _resolved_load_balance_mode(args) != "static":
                raise ValueError(
                    "split requires --load-balance-mode static because K1/K2 "
                    "cannot share an atomic scheduler counter"
                )
            for name in (
                "split_k1_mma_tiler",
                "split_k2_mma_tiler",
                "split_k1_cluster",
                "split_k2_cluster",
            ):
                _parse_positive_triplet(
                    getattr(args, name), f"--{name.replace('_', '-')}"
                )
            if args.split_k1_sm_count <= 0 or args.split_k2_sm_count <= 0:
                raise ValueError("split K1/K2 SM counts must both be positive")
        elif not cache_mode:
            _mxfp4_fused_tactic(args, tile)
        return ("mxfp4_hybrid",), ("swap_ab",), tile

    if "--mxfp4-tactic-source" in args._specified_options:
        raise ValueError(f"--mxfp4-tactic-source requires --backend {MXFP4_BACKEND}")
    split_conflicts = sorted(
        option for option in args._specified_options if option.startswith("--split-")
    )
    if split_conflicts:
        raise ValueError(
            f"split tactic flags require --backend {MXFP4_BACKEND}: "
            + ", ".join(split_conflicts)
        )
    if any(
        value is not None
        for value in (
            args.mxfp4_mma_tiler,
            args.mxfp4_cluster,
            args.mxfp4_group_hint,
            args.mxfp4_num_sched_stages,
            args.mxfp4_pingpong,
        )
    ):
        raise ValueError(
            f"--mxfp4-* fused tactic flags require --backend {MXFP4_BACKEND}"
        )
    if args.scale_mode == "mxfp4_hybrid":
        raise ValueError(
            "mxfp4_hybrid is accepted only by "
            f"--backend {MXFP4_BACKEND}; no cross-format fallback"
        )
    scale_mode = args.scale_mode or "both"
    scale_modes = ("per_tensor", "blockwise") if scale_mode == "both" else (scale_mode,)
    fp8_knobs = _parse_fp8_knobs_json(args.fp8_knobs_json)
    if fp8_knobs is not None:
        conflicts = []
        if args.operand_order is not None:
            conflicts.append("layout/--heuristic")
        if args.mma_tiler is not None:
            conflicts.append("--mma-tiler")
        if args.token_back is not None:
            conflicts.append("--token-back")
        if conflicts:
            raise ValueError(
                "--fp8-knobs-json is mutually exclusive with " + ", ".join(conflicts)
            )
        mma = fp8_knobs["mma_tiler_mnk"]
        assert isinstance(mma, tuple)
        order = "swap_ab" if bool(fp8_knobs["swap_ab"]) else "non_swap_ab"
        return scale_modes, (order,), (int(mma[0]), int(mma[1]))
    operand_order = args.operand_order or "heuristic"
    orders = ("non_swap_ab", "swap_ab") if operand_order == "both" else (operand_order,)
    tile = _parse_mma_tile(args.mma_tiler) if args.mma_tiler is not None else None
    return scale_modes, orders, tile


def _parse_mma_tile(value: str) -> tuple[int, int]:
    try:
        values = tuple(int(v) for v in value.split(","))
    except ValueError as exc:
        raise ValueError("--mma-tiler must be M,N with integer fields") from exc
    if len(values) != 2 or any(v <= 0 for v in values):
        raise ValueError("--mma-tiler must be two positive integers M,N")
    return values


def _parse_positive_triplet(value: str, option: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(v) for v in value.split(","))
    except ValueError as exc:
        raise ValueError(f"{option} must be M,N,K with integer fields") from exc
    if len(values) != 3 or any(v <= 0 for v in values):
        raise ValueError(f"{option} must be three positive integers M,N,K")
    return values


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
    if args.load_balance_mode is not None:
        return args.load_balance_mode
    return "static" if args.execution_mode == "split" else "atomic_counter"


def _resolved_token_back(args: argparse.Namespace) -> str | None:
    if args.backend == MXFP4_BACKEND:
        return args.token_back or MXFP4_TOKEN_BACK
    return None if args.token_back in (None, "heuristic") else args.token_back


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
    }


def _mxfp4_fused_tactic(
    args: argparse.Namespace,
    legacy_tile: tuple[int, int],
) -> dict[str, object]:
    """Return one fully identified, MXFP4-only fused tactic.

    --mma-tiler M,N remains the historical K128 alias. Formal discovery uses
    --mxfp4-mma-tiler M,N,K so K256 is preserved in config and output records.
    """

    if args.execution_mode != "fused":
        raise ValueError("MXFP4 fused tactic resolution requires fused mode")
    mma = (
        _parse_positive_triplet(args.mxfp4_mma_tiler, "--mxfp4-mma-tiler")
        if args.mxfp4_mma_tiler is not None
        else (legacy_tile[0], legacy_tile[1], MXFP4_TILE_K)
    )
    cluster = (
        _parse_positive_triplet(args.mxfp4_cluster, "--mxfp4-cluster")
        if args.mxfp4_cluster is not None
        else MXFP4_CLUSTER
    )
    pingpong = (
        MXFP4_PINGPONG if args.mxfp4_pingpong is None else bool(args.mxfp4_pingpong)
    )
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
        ("--mxfp4-group-hint", args.mxfp4_group_hint),
        ("--mxfp4-num-sched-stages", args.mxfp4_num_sched_stages),
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
        "group_hint": args.mxfp4_group_hint,
        "num_sched_stages": args.mxfp4_num_sched_stages,
    }
    return tactic


def _split_tactic_labels_from_tactic(
    tactic: dict[str, object],
) -> tuple[str, str]:
    """Render the actual K1/K2 schedule identity without CSV delimiters."""

    k1 = _runtime_positive_triplet(tactic["k1_mma_tiler_mnk"], "k1_mma_tiler_mnk")
    k2 = _runtime_positive_triplet(tactic["k2_mma_tiler_mnk"], "k2_mma_tiler_mnk")
    k1_cluster = _runtime_positive_triplet(
        tactic["k1_cluster_shape_mnk"], "k1_cluster_shape_mnk"
    )
    k2_cluster = _runtime_positive_triplet(
        tactic["k2_cluster_shape_mnk"], "k2_cluster_shape_mnk"
    )
    k1_group = "auto" if tactic["k1_group_hint"] is None else tactic["k1_group_hint"]
    k2_group = "auto" if tactic["k2_group_hint"] is None else tactic["k2_group_hint"]
    k1_stages = (
        "auto"
        if tactic["k1_num_sched_stages"] is None
        else tactic["k1_num_sched_stages"]
    )
    k2_stages = (
        "auto"
        if tactic["k2_num_sched_stages"] is None
        else tactic["k2_num_sched_stages"]
    )
    k1_label = (
        f"k1_m{k1[0]}n{k1[1]}k{k1[2]}_sm{tactic['k1_sm_count']}_"
        f"s{k1_stages}_"
        f"cga{k1_cluster[0]}x{k1_cluster[1]}x{k1_cluster[2]}_gh{k1_group}"
    )
    k2_label = (
        f"k2_m{k2[0]}n{k2[1]}k{k2[2]}_sm{tactic['k2_sm_count']}_"
        f"s{k2_stages}_"
        f"cga{k2_cluster[0]}x{k2_cluster[1]}x{k2_cluster[2]}_gh{k2_group}"
    )
    return k1_label, k2_label


def _split_tactic_labels(args: argparse.Namespace) -> tuple[str, str]:
    if args.execution_mode != "split":
        return "", ""
    if args.mxfp4_tactic_source == "cache_or_heuristic":
        return "", ""
    tactic: dict[str, object] = {
        "k1_mma_tiler_mnk": _parse_positive_triplet(
            args.split_k1_mma_tiler, "--split-k1-mma-tiler"
        ),
        "k2_mma_tiler_mnk": _parse_positive_triplet(
            args.split_k2_mma_tiler, "--split-k2-mma-tiler"
        ),
        "k1_cluster_shape_mnk": _parse_positive_triplet(
            args.split_k1_cluster, "--split-k1-cluster"
        ),
        "k2_cluster_shape_mnk": _parse_positive_triplet(
            args.split_k2_cluster, "--split-k2-cluster"
        ),
        "k1_group_hint": args.split_k1_group_hint,
        "k2_group_hint": args.split_k2_group_hint,
        "k1_num_sched_stages": args.split_k1_num_sched_stages,
        "k2_num_sched_stages": args.split_k2_num_sched_stages,
        "k1_sm_count": args.split_k1_sm_count,
        "k2_sm_count": args.split_k2_sm_count,
    }
    return _split_tactic_labels_from_tactic(tactic)


def _tactic_label(
    args: argparse.Namespace,
    *,
    operand_order: str,
    tile: tuple[int, int] | tuple[str, str],
) -> str:
    if args.backend == MXFP4_BACKEND:
        if args.mxfp4_tactic_source == "cache_or_heuristic":
            return f"mxfp4_{args.execution_mode}_cache_or_heuristic"
        token_back = _resolved_token_back(args)
        if args.execution_mode == "split":
            k1, k2 = _split_tactic_labels(args)
            return (
                f"green_split_{k1}_{k2}_banks{args.split_counter_banks}_"
                f"{args.split_graph_variant}_iket{int(args.split_enable_iket)}"
            )
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
    seed: int = 1234,
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

    g = torch.Generator(device="cuda").manual_seed(42 + rank)
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
    generator = torch.Generator(device=device).manual_seed(0x4D584650 + rank)

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
    args, backend: str, scale_mode: str, local_experts: int, rank: int, device
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
    if args.backend == MXFP4_BACKEND:
        from flashinfer.moe_ep import (
            Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
        )

        if scale_mode != "mxfp4_hybrid" or operand_order != "swap_ab":
            raise RuntimeError("resolved MXFP4 benchmark contract is inconsistent")
        cache_mode = args.mxfp4_tactic_source == "cache_or_heuristic"
        split_cache_mode = cache_mode and args.execution_mode == "split"
        if args.execution_mode == "fused":
            knobs = None if cache_mode else _mxfp4_fused_tactic(args, tile)
            swap_ab = None
            pingpong = None
            mma_tiler_mnk = None
            cluster_shape_mnk = None
            token_back_mode = None
        else:
            # Split has a separate K1/K2 tactic/session identity and must never
            # consume a fused knob or cache entry.
            knobs = None
            swap_ab = None
            pingpong = None
            mma_tiler_mnk = None
            cluster_shape_mnk = None
            token_back_mode = None
        return Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=args.intermediate,
            top_k=args.top_k,
            kind="fp8_e4m3",
            fp8_scale_mode="mxfp4_hybrid",
            fp8_accum_mode="1xacc",
            knobs=knobs,
            swap_ab=swap_ab,
            pingpong=pingpong,
            mma_tiler_mnk=mma_tiler_mnk,
            cluster_shape_mnk=cluster_shape_mnk,
            load_balance_mode=_resolved_load_balance_mode(args),
            gate_up_clamp=args.gate_up_clamp,
            in_kernel_fc2_reduce=False,
            token_back_mode=token_back_mode,
            execution_mode=args.execution_mode,
            split_k1_mma_tiler_mnk=(
                (128, 32, 128)
                if split_cache_mode
                else _parse_positive_triplet(
                    args.split_k1_mma_tiler, "--split-k1-mma-tiler"
                )
            ),
            split_k2_mma_tiler_mnk=(
                (128, 32, 128)
                if split_cache_mode
                else _parse_positive_triplet(
                    args.split_k2_mma_tiler, "--split-k2-mma-tiler"
                )
            ),
            split_k1_cluster_shape_mnk=(
                (1, 1, 1)
                if split_cache_mode
                else _parse_positive_triplet(
                    args.split_k1_cluster, "--split-k1-cluster"
                )
            ),
            split_k2_cluster_shape_mnk=(
                (1, 1, 1)
                if split_cache_mode
                else _parse_positive_triplet(
                    args.split_k2_cluster, "--split-k2-cluster"
                )
            ),
            split_k1_group_hint=None if split_cache_mode else args.split_k1_group_hint,
            split_k2_group_hint=None if split_cache_mode else args.split_k2_group_hint,
            split_k1_num_sched_stages=(
                None if split_cache_mode else args.split_k1_num_sched_stages
            ),
            split_k2_num_sched_stages=(
                None if split_cache_mode else args.split_k2_num_sched_stages
            ),
            split_k1_sm_count=None if split_cache_mode else args.split_k1_sm_count,
            split_k2_sm_count=None if split_cache_mode else args.split_k2_sm_count,
            split_counter_epoch_banks=(
                1 if split_cache_mode else args.split_counter_banks
            ),
            split_graph_variant=(
                "steady_k3_reset" if split_cache_mode else args.split_graph_variant
            ),
            split_enable_iket=False if split_cache_mode else args.split_enable_iket,
            routing_profile=sm90_routing_profile_from_benchmark_mode(args.routing_mode),
        )

    from flashinfer.moe_ep import Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig

    fp8_knobs = _parse_fp8_knobs_json(args.fp8_knobs_json)
    if fp8_knobs is not None:
        swap_ab = None
        mma_tiler_mnk = None
        token_back_mode = None
    elif operand_order == "heuristic":
        # All geometry knobs None -> the shim resolves the drop's token-bucket
        # heuristic per point (keyed on scale mode and max tokens per rank).
        swap_ab = None
        mma_tiler_mnk = None
        token_back_mode = _resolved_token_back(args)
    else:
        swap_ab = operand_order == "swap_ab"
        mma_tiler_mnk = (tile[0], tile[1], 128)
        token_back_mode = _resolved_token_back(args)
    return Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=args.intermediate,
        top_k=args.top_k,
        kind=args.kind,
        fp8_scale_mode=scale_mode,
        fp8_accum_mode=args.fp8_accum_mode,
        knobs=fp8_knobs,
        swap_ab=swap_ab,
        mma_tiler_mnk=mma_tiler_mnk,
        load_balance_mode=_resolved_load_balance_mode(args),
        gate_up_clamp=args.gate_up_clamp,
        in_kernel_fc2_reduce=False,
        token_back_mode=token_back_mode,
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
        "mxfp4_split": MXFP4_SPLIT_RUNTIME_TACTIC_FIELDS,
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


def _compiled_fused_schedule(frontend: object, label: str) -> tuple[int, int]:
    mega = getattr(frontend, "_mega", None)
    kernel = getattr(mega, "kernel", None)
    if kernel is None:
        raise RuntimeError(f"{label} frontend lacks an actual compiled kernel")
    return _kernel_schedule_values(kernel, label)


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
        "num_sched_stages": num_sched_stages,
        "flag_batch": int(config.flag_batch),
        "epi_flag_batch": tuple(int(v) for v in config.epi_flag_batch),
        "load_balance_mode": str(config.load_balance_mode),
        "token_back_mode": str(config.resolved_token_back_mode),
        "in_kernel_fc2_reduce": bool(config.in_kernel_fc2_reduce),
    }
    return {
        "tactic_mode": mode,
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
    if args.backend != MXFP4_BACKEND or args.execution_mode != "fused":
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
    group_hint, num_sched_stages = _compiled_fused_schedule(frontend, "MXFP4 fused")
    _verify_requested_schedule(config, group_hint, num_sched_stages, "MXFP4 fused")
    tactic: dict[str, object] = {
        "swap_ab": bool(config.swap_ab),
        "pingpong": bool(config.pingpong),
        "mma_tiler_mnk": tuple(int(v) for v in config.mma_tiler_mnk),
        "cluster_shape_mnk": tuple(int(v) for v in config.cluster_shape_mnk),
        "fp8_accum_mode": str(config.fp8_accum_mode),
        "load_balance_mode": str(config.load_balance_mode),
        "token_back_mode": str(config.resolved_token_back_mode),
        "group_hint": group_hint,
        "num_sched_stages": num_sched_stages,
        "in_kernel_fc2_reduce": bool(config.in_kernel_fc2_reduce),
    }
    return {
        "routing_profile": routing_profile,
        **tactic,
        **_runtime_tactic_envelope("mxfp4_fused", tactic),
    }


def _expected_split_runtime_geometry(
    args: argparse.Namespace,
) -> tuple[tuple[int, int], tuple[int, int], int]:
    k1_tiler = _parse_positive_triplet(args.split_k1_mma_tiler, "--split-k1-mma-tiler")
    k2_tiler = _parse_positive_triplet(args.split_k2_mma_tiler, "--split-k2-mma-tiler")
    k1_cluster = _parse_positive_triplet(args.split_k1_cluster, "--split-k1-cluster")
    k2_cluster = _parse_positive_triplet(args.split_k2_cluster, "--split-k2-cluster")
    partition = (args.split_k1_sm_count, args.split_k2_sm_count)
    max_active_clusters = (
        partition[0] // (k1_cluster[0] * k1_cluster[1]),
        partition[1] // (k2_cluster[0] * k2_cluster[1]),
    )
    return partition, max_active_clusters, max(k1_tiler[1], k2_tiler[1])


def _expected_split_session_config(args: argparse.Namespace) -> dict[str, object]:
    """The complete CLI-controlled K1/K2 compile/session identity."""

    return {
        "k1_mma_tiler_mnk": _parse_positive_triplet(
            args.split_k1_mma_tiler, "--split-k1-mma-tiler"
        ),
        "k2_mma_tiler_mnk": _parse_positive_triplet(
            args.split_k2_mma_tiler, "--split-k2-mma-tiler"
        ),
        "k1_cluster_shape_mnk": _parse_positive_triplet(
            args.split_k1_cluster, "--split-k1-cluster"
        ),
        "k2_cluster_shape_mnk": _parse_positive_triplet(
            args.split_k2_cluster, "--split-k2-cluster"
        ),
        "k1_group_hint": args.split_k1_group_hint,
        "k2_group_hint": args.split_k2_group_hint,
        "k1_num_sched_stages": args.split_k1_num_sched_stages,
        "k2_num_sched_stages": args.split_k2_num_sched_stages,
        "k1_sm_count": args.split_k1_sm_count,
        "k2_sm_count": args.split_k2_sm_count,
        "counter_epoch_banks": args.split_counter_banks,
        "graph_variant": args.split_graph_variant,
        "enable_iket": bool(args.split_enable_iket),
    }


def _kernel_triplet(kernel: object, name: str, label: str) -> tuple[int, int, int]:
    try:
        value = tuple(int(v) for v in getattr(kernel, name))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} compiled kernel lacks valid {name}") from exc
    if len(value) != 3 or any(v <= 0 for v in value):
        raise RuntimeError(f"{label} compiled kernel has malformed {name}={value!r}")
    return value


def _split_kernel_cluster_shape_mnk(kernel: object, label: str) -> tuple[int, int, int]:
    """Recover the split kernel's compiled cluster from its vendor MN field."""

    try:
        cluster_mn = tuple(int(v) for v in kernel.cluster_shape_mn)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{label} compiled kernel lacks valid cluster_shape_mn"
        ) from exc
    if len(cluster_mn) != 2 or any(v <= 0 for v in cluster_mn):
        raise RuntimeError(
            f"{label} compiled kernel has malformed cluster_shape_mn={cluster_mn!r}"
        )
    return (*cluster_mn, 1)


def _split_pair_runtime_tactic(
    session_config: object, graph_variant: str, pair: object, label: str
) -> dict[str, object]:
    plan = getattr(pair, "plan", None)
    workspace = getattr(pair, "workspace", None)
    if plan is None or workspace is None:
        raise RuntimeError(f"{label} lacks split plan/workspace compile identity")
    role_values: dict[str, dict[str, object]] = {}
    for role, kernel_name in (("k1", "k1_kernel"), ("k2", "k2_kernel")):
        kernel = getattr(pair, kernel_name, None)
        if kernel is None:
            raise RuntimeError(f"{label} lacks {role.upper()} compiled kernel")
        mma = _kernel_triplet(kernel, "mma_tiler_mnk", f"{label} {role.upper()}")
        cluster = _split_kernel_cluster_shape_mnk(kernel, f"{label} {role.upper()}")
        group_hint, num_sched_stages = _kernel_schedule_values(
            kernel, f"{label} {role.upper()}"
        )
        sm_count = _runtime_positive_int(
            getattr(plan, f"{role}_sm_count", None),
            f"{label} {role.upper()} sm_count",
        )
        expected = {
            "mma": tuple(getattr(session_config, f"{role}_mma_tiler_mnk")),
            "cluster": tuple(getattr(session_config, f"{role}_cluster_shape_mnk")),
            "sm_count": int(getattr(session_config, f"{role}_sm_count")),
        }
        actual = {"mma": mma, "cluster": cluster, "sm_count": sm_count}
        if actual != expected:
            raise RuntimeError(
                f"{label} {role.upper()} compiled geometry {actual!r} "
                f"!= session {expected!r}"
            )
        requested_group = getattr(session_config, f"{role}_group_hint")
        expected_group = (
            int(requested_group)
            if requested_group is not None
            else 3 * sm_count // (cluster[0] * cluster[1])
        )
        requested_stages = getattr(session_config, f"{role}_num_sched_stages")
        expected_stages = int(requested_stages) if requested_stages is not None else 2
        if group_hint != expected_group or num_sched_stages != expected_stages:
            raise RuntimeError(
                f"{label} {role.upper()} compiled schedule "
                f"(group_hint={group_hint}, num_sched_stages={num_sched_stages}) "
                f"!= resolved session ({expected_group}, {expected_stages})"
            )
        role_values[role] = {
            "mma": mma,
            "cluster": cluster,
            "group_hint": group_hint,
            "num_sched_stages": num_sched_stages,
            "sm_count": sm_count,
        }

    counter_banks = _runtime_positive_int(
        getattr(workspace, "counter_epoch_banks", None),
        f"{label} counter_epoch_banks",
    )
    if counter_banks != int(session_config.counter_epoch_banks):
        raise RuntimeError(
            f"{label} workspace counter_epoch_banks={counter_banks} "
            f"!= session {session_config.counter_epoch_banks}"
        )
    k1 = role_values["k1"]
    k2 = role_values["k2"]
    return {
        "k1_mma_tiler_mnk": k1["mma"],
        "k2_mma_tiler_mnk": k2["mma"],
        "k1_cluster_shape_mnk": k1["cluster"],
        "k2_cluster_shape_mnk": k2["cluster"],
        "k1_group_hint": k1["group_hint"],
        "k2_group_hint": k2["group_hint"],
        "k1_num_sched_stages": k1["num_sched_stages"],
        "k2_num_sched_stages": k2["num_sched_stages"],
        "k1_sm_count": k1["sm_count"],
        "k2_sm_count": k2["sm_count"],
        "counter_epoch_banks": counter_banks,
        "graph_variant": graph_variant,
        "enable_iket": bool(session_config.enable_iket),
    }


def _split_session_metadata(
    args: argparse.Namespace, workspace: object
) -> dict[str, object]:
    if args.execution_mode != "split":
        return {}
    session = getattr(workspace, "_session", None)
    if session is None:
        raise RuntimeError(
            "split benchmark workspace has no Green session; fused fallback "
            "or an incomplete backend is forbidden"
        )
    if not bool(getattr(session, "captured", False)):
        raise RuntimeError("split benchmark session did not capture a CUDA graph")
    if bool(getattr(session, "poisoned", False)):
        raise RuntimeError("split benchmark session is poisoned")
    if bool(getattr(session, "destroyed", False)):
        raise RuntimeError("split benchmark session was destroyed before timing")
    session_config = getattr(session, "config", None)
    if session_config is None or not hasattr(session_config, "handoff_token_n"):
        raise RuntimeError("split benchmark session lacks handoff_token_n identity")
    routing_profile = _verified_mxfp4_runtime_routing_profile(
        args, session_config, "MXFP4 split runtime"
    )
    cache_mode = args.mxfp4_tactic_source == "cache_or_heuristic"
    expected_session_config = {} if cache_mode else _expected_split_session_config(args)
    if cache_mode:
        expected_graph = str(session_config.graph_variant)
        expected_partition = (
            int(session_config.k1_sm_count),
            int(session_config.k2_sm_count),
        )
        expected_clusters = tuple(
            expected_partition[i]
            // (
                int(getattr(session_config, f"{role}_cluster_shape_mnk")[0])
                * int(getattr(session_config, f"{role}_cluster_shape_mnk")[1])
            )
            for i, role in enumerate(("k1", "k2"))
        )
        expected_handoff = max(
            int(session_config.k1_mma_tiler_mnk[1]),
            int(session_config.k2_mma_tiler_mnk[1]),
        )
    else:
        expected_graph = args.split_graph_variant
        expected_partition, expected_clusters, expected_handoff = (
            _expected_split_runtime_geometry(args)
        )
    graph_variant = str(getattr(session, "graph_variant", ""))
    if graph_variant != expected_graph:
        raise RuntimeError(
            f"split graph variant {graph_variant!r} != expected {expected_graph!r}"
        )
    green_sm_counts = tuple(int(v) for v in session.green_sm_counts)
    if green_sm_counts != expected_partition:
        raise RuntimeError(
            "driver Green SM partition "
            f"{green_sm_counts} != requested {expected_partition}"
        )
    max_active_clusters = tuple(int(v) for v in session.max_active_clusters)
    if max_active_clusters != expected_clusters:
        raise RuntimeError(
            "split max_active_clusters "
            f"{max_active_clusters} != expected {expected_clusters}"
        )
    for name, expected in expected_session_config.items():
        if not hasattr(session_config, name):
            raise RuntimeError(f"split benchmark session lacks {name} compile identity")
        actual = getattr(session_config, name)
        if isinstance(expected, tuple):
            actual = tuple(actual)
        elif isinstance(expected, bool):
            actual = bool(actual)
        if actual != expected:
            raise RuntimeError(
                f"split session {name}={actual!r} != requested {expected!r}"
            )
    handoff_token_n = int(session_config.handoff_token_n)
    if handoff_token_n != expected_handoff:
        raise RuntimeError(
            f"split handoff_token_n {handoff_token_n} != expected {expected_handoff}"
        )
    counter_banks = int(session_config.counter_epoch_banks)
    pairs = tuple(getattr(session, "_pairs", ()))
    if len(pairs) != counter_banks:
        raise RuntimeError(
            f"split session has {len(pairs)} compiled bank pairs, "
            f"expected {counter_banks}"
        )
    try:
        pair_banks = tuple(int(pair.counter_epoch_bank) for pair in pairs)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("split bank pairs lack counter bank identity") from exc
    if pair_banks != tuple(range(counter_banks)):
        raise RuntimeError(
            f"split bank pair identities {pair_banks} != {tuple(range(counter_banks))}"
        )
    pair_tactics = [
        _split_pair_runtime_tactic(
            session_config, graph_variant, pair, f"split bank pair {bank}"
        )
        for bank, pair in zip(pair_banks, pairs, strict=True)
    ]
    if any(tactic != pair_tactics[0] for tactic in pair_tactics[1:]):
        raise RuntimeError("split compiled bank pairs disagree on tactic identity")
    runtime_identity = _runtime_tactic_envelope("mxfp4_split", pair_tactics[0])
    generation = int(session.generation)
    if generation <= 0:
        raise RuntimeError("split session generation must be positive")
    return {
        "routing_profile": routing_profile,
        "generation": generation,
        "graph_variant": graph_variant,
        "green_sm_counts": green_sm_counts,
        "max_active_clusters": max_active_clusters,
        "handoff_token_n": handoff_token_n,
        "counter_banks": counter_banks,
        **runtime_identity,
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
    result: PointResult | None = None
    error = ""
    try:
        kcfg = _megakernel_config(args, scale_mode, operand_order, tile)
        transformed = _make_transformed_weights(
            args, args.backend, scale_mode, local_experts, rank, device
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
        # Fused prepare_workspace may share the layer's pooled buffer. Split
        # deliberately disables pooling and creates an independent fixed-pointer
        # Green session; its first warmup compiles/captures before timed replays.
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
        _cooldown(args.cooldown_s)
        compute = _time_calls(
            lambda: bench_backend.compute(bench_workspace, transformed, output=None),
            warmup=args.warmup,
            iters=args.iters,
        )
        torch.cuda.synchronize()
        if args.backend == FP8_BACKEND:
            runtime_metadata = _fp8_runtime_metadata(args, bench_workspace)
        elif args.execution_mode == "fused":
            runtime_metadata = _mxfp4_fused_runtime_metadata(args, bench_workspace)
        else:
            runtime_metadata = _split_session_metadata(args, bench_workspace)

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
    split = [""] * 17
    resolved_tactic: dict[str, object] | None = None
    if args.mxfp4_tactic_source == "cache_or_heuristic" and result.status == "pass":
        _, resolved_tactic, _ = _all_rank_runtime_tactic(
            result, len(result.compute_median_us)
        )
    if args.backend == MXFP4_BACKEND and args.execution_mode == "fused":
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
    elif args.backend == MXFP4_BACKEND and args.execution_mode == "split":
        if resolved_tactic is not None:
            k1 = _runtime_positive_triplet(
                resolved_tactic["k1_mma_tiler_mnk"],
                "k1_mma_tiler_mnk",
            )
            k2 = _runtime_positive_triplet(
                resolved_tactic["k2_mma_tiler_mnk"],
                "k2_mma_tiler_mnk",
            )
            k1_cluster = _runtime_positive_triplet(
                resolved_tactic["k1_cluster_shape_mnk"],
                "k1_cluster_shape_mnk",
            )
            k2_cluster = _runtime_positive_triplet(
                resolved_tactic["k2_cluster_shape_mnk"],
                "k2_cluster_shape_mnk",
            )
            k1_group = resolved_tactic["k1_group_hint"]
            k2_group = resolved_tactic["k2_group_hint"]
            k1_stages = resolved_tactic["k1_num_sched_stages"]
            k2_stages = resolved_tactic["k2_num_sched_stages"]
            enable_iket = bool(resolved_tactic["enable_iket"])
        elif args.mxfp4_tactic_source != "cache_or_heuristic":
            k1 = _parse_positive_triplet(
                args.split_k1_mma_tiler, "--split-k1-mma-tiler"
            )
            k2 = _parse_positive_triplet(
                args.split_k2_mma_tiler, "--split-k2-mma-tiler"
            )
            k1_cluster = _parse_positive_triplet(
                args.split_k1_cluster, "--split-k1-cluster"
            )
            k2_cluster = _parse_positive_triplet(
                args.split_k2_cluster, "--split-k2-cluster"
            )
            k1_group = args.split_k1_group_hint
            k2_group = args.split_k2_group_hint
            k1_stages = args.split_k1_num_sched_stages
            k2_stages = args.split_k2_num_sched_stages
            enable_iket = bool(args.split_enable_iket)
        if (
            resolved_tactic is not None
            or args.mxfp4_tactic_source != "cache_or_heuristic"
        ):
            split = [
                *(str(v) for v in k1),
                *(str(v) for v in k2),
                *(str(v) for v in k1_cluster),
                *(str(v) for v in k2_cluster),
                ("" if k1_group is None else str(k1_group)),
                ("" if k2_group is None else str(k2_group)),
                "" if k1_stages is None else str(k1_stages),
                "" if k2_stages is None else str(k2_stages),
                str(int(enable_iket)),
            ]
    cols = [score, *fused, *split]
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
    return "mxfp4_split" if args.execution_mode == "split" else "mxfp4_fused"


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
    cols = [digest, "", "", "", "", "", ""]
    if implementation == "mxfp4_split":
        cols[3:] = [
            str(tactic["k1_group_hint"]),
            str(tactic["k2_group_hint"]),
            str(tactic["k1_num_sched_stages"]),
            str(tactic["k2_num_sched_stages"]),
        ]
    else:
        cols[1:3] = [
            str(tactic["group_hint"]),
            str(tactic["num_sched_stages"]),
        ]
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

    seed = 1234
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
    k1_tactic, k2_tactic = _split_tactic_labels(args)

    graph_variant = ""
    counter_banks: str | int = ""
    k1_sm_count: str | int = ""
    k2_sm_count: str | int = ""
    k1_max_active_clusters: str | int = ""
    k2_max_active_clusters: str | int = ""
    handoff_token_n: str | int = ""
    rank_session_generations = ""
    if args.execution_mode == "split":
        if args.mxfp4_tactic_source != "cache_or_heuristic":
            graph_variant = args.split_graph_variant
            counter_banks = args.split_counter_banks
            k1_sm_count = args.split_k1_sm_count
            k2_sm_count = args.split_k2_sm_count
        if result.status == "pass":
            metadata = result.runtime_metadata
            if metadata is None or len(metadata) != world_size:
                raise RuntimeError("split result lacks all-rank session metadata")
            try:
                actual_partitions = {
                    tuple(int(v) for v in record["green_sm_counts"])
                    for record in metadata
                }
                actual_clusters = {
                    tuple(int(v) for v in record["max_active_clusters"])
                    for record in metadata
                }
                actual_handoffs = {
                    int(record["handoff_token_n"]) for record in metadata
                }
                generations = [int(record["generation"]) for record in metadata]
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "split result has malformed all-rank session metadata"
                ) from exc
            if args.mxfp4_tactic_source == "cache_or_heuristic":
                implementation, resolved_tactic, _ = _all_rank_runtime_tactic(
                    result, world_size
                )
                if implementation != "mxfp4_split":
                    raise RuntimeError("split cache mode resolved a non-split tactic")
                expected_partition = (
                    int(resolved_tactic["k1_sm_count"]),
                    int(resolved_tactic["k2_sm_count"]),
                )
                expected_clusters = tuple(
                    expected_partition[i]
                    // (
                        int(resolved_tactic[f"{role}_cluster_shape_mnk"][0])
                        * int(resolved_tactic[f"{role}_cluster_shape_mnk"][1])
                    )
                    for i, role in enumerate(("k1", "k2"))
                )
                expected_handoff = max(
                    int(resolved_tactic["k1_mma_tiler_mnk"][1]),
                    int(resolved_tactic["k2_mma_tiler_mnk"][1]),
                )
                graph_variant = str(resolved_tactic["graph_variant"])
                counter_banks = int(resolved_tactic["counter_epoch_banks"])
                k1_sm_count, k2_sm_count = expected_partition
                k1_tactic, k2_tactic = _split_tactic_labels_from_tactic(resolved_tactic)
            else:
                expected_partition, expected_clusters, expected_handoff = (
                    _expected_split_runtime_geometry(args)
                )
            if actual_partitions != {expected_partition}:
                raise RuntimeError(
                    "split ranks disagree on the Green partition: "
                    f"{sorted(actual_partitions)}"
                )
            if actual_clusters != {expected_clusters}:
                raise RuntimeError(
                    "split ranks disagree on max_active_clusters: "
                    f"{sorted(actual_clusters)}"
                )
            if actual_handoffs != {expected_handoff}:
                raise RuntimeError(
                    "split ranks disagree on handoff_token_n: "
                    f"{sorted(actual_handoffs)}"
                )
            if any(generation <= 0 for generation in generations):
                raise RuntimeError("split session generations must be positive")
            k1_max_active_clusters, k2_max_active_clusters = next(iter(actual_clusters))
            handoff_token_n = next(iter(actual_handoffs))
            rank_session_generations = "|".join(
                f"r{rank}:g{generation}" for rank, generation in enumerate(generations)
            )
    labels = (tactic, k1_tactic, k2_tactic)
    if any("," in label for label in labels):
        raise RuntimeError(f"tactic labels must be CSV-safe, got {labels!r}")

    if not header_done:
        print(CSV_HEADER, flush=True)
        if csv_file is not None:
            csv_file.write(
                f"{CSV_FIELDS},{HEUR_CSV_FIELDS},{BENCH_EXT_CSV_FIELDS},"
                f"{SPLIT_RUNTIME_CSV_FIELDS},"
                f"{FORMAL_TUNING_CSV_FIELDS},{FP8_RUNTIME_CSV_FIELDS},"
                f"{RUNTIME_TACTIC_CSV_FIELDS},{ROUTING_CSV_FIELDS}\n"
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
    if args.backend == MXFP4_BACKEND and args.execution_mode == "fused":
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
        args.execution_mode,
        tactic,
        k1_tactic,
        k2_tactic,
        graph_variant,
        counter_banks,
        k1_sm_count,
        k2_sm_count,
        *cold_cols,
    ]
    expected_ext = len(BENCH_EXT_CSV_FIELDS.split(","))
    if len(bench_ext_cols) != expected_ext:
        raise RuntimeError(
            f"benchmark extension CSV schema mismatch: "
            f"{len(bench_ext_cols)} values != {expected_ext}"
        )
    bench_ext_row = ",".join(str(value) for value in bench_ext_cols)
    split_runtime_cols = [
        str(k1_max_active_clusters),
        str(k2_max_active_clusters),
        str(handoff_token_n),
        rank_session_generations,
    ]
    split_runtime_row = ",".join(split_runtime_cols)
    formal_tuning_row = ",".join(_formal_tuning_cols(args, tile, result))
    fp8_runtime_row = ",".join(_fp8_runtime_cols(args, result))
    runtime_tactic_row = ",".join(
        _runtime_tactic_cols(args, scale_mode, result, world_size)
    )
    routing_row = ",".join(_routing_csv_cols(args, tokens, world_size, result))
    print(
        f"BENCH_CSV,{row},{bench_ext_row},{split_runtime_row},"
        f"{formal_tuning_row},{fp8_runtime_row},{runtime_tactic_row},"
        f"{routing_row}",
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
            + split_runtime_row
            + ","
            + formal_tuning_row
            + ","
            + fp8_runtime_row
            + ","
            + runtime_tactic_row
            + ","
            + routing_row
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
            f"# backend: {args.backend} execution_mode={args.execution_mode}",
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
                    f"{args.backend}_{args.execution_mode}_"
                    f"{'-'.join(orders)}_{'-'.join(scale_modes)}"
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
        csv_file = open(csv_path, "w")  # noqa: SIM115 - closed in sweep finally
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
