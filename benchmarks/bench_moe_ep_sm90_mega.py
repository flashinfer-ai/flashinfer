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

Two warm CUDA-event timed series are recorded per point:
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
production Humming preprocessor. The same command supports 1, 2, 4, and 8
ranks by changing --nproc_per_node. To replay a tuned winner, pass its complete
knob dictionary through --mxfp4-knobs-json (or --fp8-knobs-json for FP8).
The runtime_tactic CSV field records the effective configuration after compile.

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
import csv
import gc
import json
import os
import sys
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
    generate_sm90_routing_numpy,
    sm90_routing_profile_from_benchmark_mode,
)

DEFAULT_TOKENS = tuple(1 << p for p in range(3, 16))  # 8 .. 32768
FP8_BACKEND = "sm90_fp8_fp8_bf16_pull_cutedsl"
MXFP4_BACKEND = "sm90_fp8_mxfp4_bf16_pull_cutedsl"
SUPPORTED_BACKENDS = (FP8_BACKEND, MXFP4_BACKEND)
ROUTING_SEED = 1234
ACTIVATION_AND_TOPK_WEIGHT_SEED_BASE = 42
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
# Preserve the historical CSV prefix. The complete effective tactic is emitted
# once as JSON and can be passed directly to the matching --*-knobs-json option.
BENCH_EXT_CSV_FIELDS = (
    "compute_max_rank_median_us,routing_mode,routing_profile,routing_seed,"
    "compute_launch_mode,runtime_tactic"
)
CSV_HEADER = "BENCH_CSV," + CSV_FIELDS + "," + BENCH_EXT_CSV_FIELDS

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
        "--mxfp4-knobs-json",
        default=None,
        metavar="JSON",
        help="replay a complete MXFP4 tactic, including optional strategies; "
        "bypasses cache/heuristic selection and conflicts with manual tactic flags.",
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
    args.knobs = None
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
        "compact_pull_buffer",
        "generate_c",
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


def _parse_knobs_json(value: str, backend: str) -> dict[str, object]:
    """Parse one strict tuner tactic without cache/heuristic fallback."""
    option = "--mxfp4-knobs-json" if backend == MXFP4_BACKEND else "--fp8-knobs-json"
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option} is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{option} must decode to a JSON object")
    if backend == MXFP4_BACKEND:
        from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.mxfp4_optimization import (
            normalize_mxfp4_optimization_tactic,
        )

        return normalize_mxfp4_optimization_tactic(payload)
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
            or any(type(v) is not int or v <= 0 for v in raw)
        ):
            raise ValueError(
                f"--fp8-knobs-json {name} must be a length-{length} positive integer array"
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
        "compact_pull_buffer",
        "generate_c",
    ):
        if name in knobs and not isinstance(knobs[name], bool):
            raise ValueError(f"--fp8-knobs-json {name} must be boolean")
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
    if isinstance(knobs.get("active_dispatch_warps"), bool):
        raise ValueError("--fp8-knobs-json active_dispatch_warps must be an integer")
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


_LAYOUT_OPTIONS = {"--swap-ab", "--no-swap-ab", "--heuristic", "--both-orders"}
_MANUAL_TACTIC_OPTIONS = (
    _LAYOUT_OPTIONS
    | _NEUTRAL_FUSED_GEOMETRY_OPTIONS
    | _FUSED_EXECUTION_KNOB_OPTIONS
    | _MXFP4_STRATEGY_OPTIONS
    | {
        "--load-balance-mode",
        "--mma-tiler",
        "--cga",
        "--token-back",
        "--fp8-accum-mode",
        "--epi-mode",
        "--swap-token-tile",
        "--compact-pull-buffer",
        "--no-compact-pull-buffer",
        "--generate-c",
    }
)


def _resolve_sweep(
    args: argparse.Namespace, world_size: int
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, int] | None]:
    """Resolve backend-specific defaults and reject cross-format fallback."""

    json_value = (
        args.mxfp4_knobs_json if args.backend == MXFP4_BACKEND else args.fp8_knobs_json
    )
    if (args.backend == MXFP4_BACKEND and args.fp8_knobs_json is not None) or (
        args.backend == FP8_BACKEND and args.mxfp4_knobs_json is not None
    ):
        raise ValueError("--*-knobs-json must match the selected backend")
    if json_value is not None:
        conflicts = args._specified_options.intersection(_MANUAL_TACTIC_OPTIONS)
        if conflicts:
            raise ValueError(
                "JSON tactic is mutually exclusive with " + ", ".join(sorted(conflicts))
            )
        if args.mxfp4_tactic_source == "cache_or_heuristic":
            raise ValueError("JSON tactic conflicts with cache_or_heuristic selection")
        args.knobs = _parse_knobs_json(json_value, args.backend)

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
        operand_order = (
            "swap_ab"
            if not args._specified_options.intersection(_LAYOUT_OPTIONS)
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
            conflicts = sorted(
                args._specified_options.intersection(
                    _MANUAL_TACTIC_OPTIONS - _LAYOUT_OPTIONS
                )
            )
            if conflicts:
                raise ValueError(
                    "--mxfp4-tactic-source cache_or_heuristic conflicts with "
                    + ", ".join(conflicts)
                )
            tile = None
        else:
            from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.mxfp4_optimization import (
                resolve_mxfp4_tactic_optimizations,
            )

            if args.knobs is None:
                args.knobs = _mxfp4_fused_tactic(args)
            resolve_mxfp4_tactic_optimizations(
                args.knobs,
                hidden=args.hidden,
                intermediate=args.intermediate,
                num_experts=args.num_experts,
                world_size=world_size,
            )
            tile = args.knobs["mma_tiler_mnk"][:2]
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
    if args.knobs is not None:
        mma = args.knobs["mma_tiler_mnk"]
        order = "swap_ab" if args.knobs["swap_ab"] else "non_swap_ab"
        return scale_modes, (order,), mma[:2]
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


def _mxfp4_fused_tactic(args: argparse.Namespace) -> dict[str, object]:
    """Build a manual MXFP4 tactic; the library validates it in _resolve_sweep."""
    mma = (
        _parse_mma_tile(args.mma_tiler)
        if args.mma_tiler
        else (*MXFP4_DEFAULT_TILE, MXFP4_TILE_K)
    )
    cm, cn = (int(v) for v in args.cga.split(",")) if args.cga else MXFP4_CLUSTER[:2]
    return {
        "swap_ab": True,
        "pingpong": MXFP4_PINGPONG
        if args.pingpong == "auto"
        else args.pingpong == "on",
        "mma_tiler_mnk": mma,
        "cluster_shape_mnk": (cm, cn, 1),
        "fp8_accum_mode": "1xacc",
        "load_balance_mode": args.load_balance_mode,
        "token_back_mode": _resolved_token_back(args),
        "in_kernel_fc2_reduce": False,
        "group_hint": args.group_hint,
        "num_sched_stages": args.num_sched_stages,
        **_fused_execution_knobs_from_args(args),
        "fc2_tail_n8": args.mxfp4_fc2_tail_n8,
        "fc1_ready_mode": args.mxfp4_fc1_ready_mode,
        "tail_split_pairs": False,
    }


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
    e2e_us: list[float]  # cross-rank per-rank mean e2e us (len == world)
    e2e_median_us: list[float]
    compute_us: list[float]
    compute_median_us: list[float]
    runtime_tactic: dict[str, object] | None = None
    error: str = ""


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


def _make_raw_mxfp4_weights(args, local_experts: int, rank: int, device):
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl.tuner import (
        create_tuning_weights,
    )

    return create_tuning_weights(
        local_experts=local_experts,
        hidden=args.hidden,
        intermediate=args.intermediate,
        rank=rank,
        device=device,
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
        return Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=args.intermediate,
            top_k=args.top_k,
            kind="fp8_e4m3",
            fp8_scale_mode="mxfp4_hybrid",
            fp8_accum_mode="1xacc",
            knobs=args.knobs,
            # Explicit fused tactics already carry this axis; cache/heuristic
            # mode must leave it omitted so the selected tactic owns it.
            load_balance_mode=None,
            gate_up_clamp=args.gate_up_clamp,
            enable_in_kernel_fc2_reduce=False,
            routing_profile=sm90_routing_profile_from_benchmark_mode(args.routing_mode),
        )

    from flashinfer.moe_ep import Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig

    fp8_knobs = args.knobs
    pingpong = None if args.pingpong == "auto" else args.pingpong == "on"
    cluster_shape_mnk = None
    accum_override = None
    token_back_override = None
    grouped_token_back = args.grouped_token_back
    combine_format = args.combine_format
    if fp8_knobs is not None:
        swap_ab = None
        mma_tiler_mnk = None
        grouped_token_back = fp8_knobs.get("grouped_token_back", grouped_token_back)
        combine_format = fp8_knobs.get("combine_format", combine_format)
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
            if grouped_token_back
            else (
                token_back_override
                if args.token_back == "heuristic"
                else args.token_back
            )
        ),
        pingpong=pingpong,
        dedup_dispatch=args.dedup_dispatch,
        grouped_token_back=grouped_token_back,
        combine_format=combine_format,
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


def _runtime_tactic(args: argparse.Namespace, workspace) -> dict[str, object]:
    """Read the effective compute tactic after the kernel has been compiled."""
    frontend = workspace._frontend
    if args.backend == MXFP4_BACKEND:
        return frontend.effective_tactic()

    # FP8 has no effective_tactic API. Geometry comes from the resolved config;
    # schedule and warp-layout fields must come from the compiled kernel since
    # it can choose defaults and normalize requested flags.
    config = frontend.config
    kernel = frontend._mega.kernel
    return {
        "swap_ab": config.swap_ab,
        "pingpong": config.pingpong,
        "mma_tiler_mnk": tuple(config.mma_tiler_mnk),
        "cluster_shape_mnk": tuple(config.cluster_shape_mnk),
        "fp8_accum_mode": config.fp8_accum_mode,
        "group_hint": int(kernel.group_hint),
        "tail_split_pairs": bool(kernel.tail_split_pairs),
        "num_sched_stages": int(kernel.num_sched_stages),
        "flag_batch": config.flag_batch,
        "epi_flag_batch": tuple(config.epi_flag_batch),
        "load_balance_mode": config.load_balance_mode,
        "token_back_mode": config.resolved_token_back_mode,
        "in_kernel_fc2_reduce": config.in_kernel_fc2_reduce,
        "compact_pull_buffer": config.compact_pull_buffer,
        "generate_c": config.generate_c,
        "dedup_dispatch": bool(kernel.dedup_dispatch),
        "grouped_token_back": bool(kernel.grouped_token_back),
        "combine_format": str(kernel.combine_format),
        "active_dispatch_warps": int(kernel.token_comm.active_dispatch_warps),
        "fc1_store_offload": bool(kernel.fc1_store_offload),
        "fc1_early_done_publish": bool(kernel.fc1_early_done_publish),
        "fold_producer_warps": bool(kernel.fold_producer_warps),
    }


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

        # --- series 1: warm FI e2e. ---
        _cooldown(args.cooldown_s)
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
        my_stats = (
            "pass",
            fmean(e2e),
            median(e2e),
            fmean(compute),
            median(compute),
            _runtime_tactic(args, bench_workspace),
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

    return _summarize_ranks(all_stats)


def _summarize_ranks(all_stats: list) -> PointResult:
    """Check status and the complete tactic once, on every rank."""
    statuses = [s[0] for s in all_stats]
    if all(status == "pass" for status in statuses):
        tactic = all_stats[0][5]
        if all(s[5] == tactic for s in all_stats):
            return PointResult(
                status="pass",
                e2e_us=[s[1] for s in all_stats],
                e2e_median_us=[s[2] for s in all_stats],
                compute_us=[s[3] for s in all_stats],
                compute_median_us=[s[4] for s in all_stats],
                runtime_tactic=tactic,
            )
        error = "effective runtime tactic differs across ranks"
        status = "failed"
    else:
        status = "skip_oom" if "skip_oom" in statuses else "failed"
        error = "; ".join(f"rank{i}:{s[6]}" for i, s in enumerate(all_stats) if s[6])
    return PointResult(status, [], [], [], [], error=error)


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
    if not header_done:
        print(CSV_HEADER, flush=True)
        if csv_file is not None:
            csv_file.write(f"{CSV_FIELDS},{HEUR_CSV_FIELDS},{BENCH_EXT_CSV_FIELDS}\n")

    reported_tile = tile
    tile_k = 128
    if args.backend == MXFP4_BACKEND or args.knobs is not None:
        tactic = result.runtime_tactic or args.knobs
        mma = tactic["mma_tiler_mnk"] if tactic else ("", "", "")
        reported_tile, tile_k = mma[:2], mma[2]
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

    extra = [
        f"{max(result.compute_median_us):.6f}" if result.status == "pass" else "nan",
        args.routing_mode,
        sm90_routing_profile_from_benchmark_mode(args.routing_mode),
        ROUTING_SEED,
        "direct",
        json.dumps(result.runtime_tactic, sort_keys=True, separators=(",", ":"))
        if result.runtime_tactic is not None
        else "",
    ]
    csv.writer(sys.stdout, lineterminator="\n").writerow(
        ["BENCH_CSV", *row.split(","), *extra]
    )
    sys.stdout.flush()
    if result.status != "pass" and result.error:
        print(f"# SKIP detail: {result.error}", flush=True)
    if csv_file is not None:
        heur = _heuristic_cols(args.backend, scale_mode, operand_order, tokens)
        csv.writer(csv_file, lineterminator="\n").writerow(
            [*row.split(","), *heur, *extra]
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
            "# timing: e2e/compute=warm_cuda_event",
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
                for tokens in tokens_list:
                    if rank == 0:
                        print(
                            f"# [sweep] backend={args.backend} order={operand_order} "
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
