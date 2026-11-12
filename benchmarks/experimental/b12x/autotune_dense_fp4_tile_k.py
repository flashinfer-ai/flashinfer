#!/usr/bin/env python3
"""Offline graph-replay autotuner for complete dense NVFP4 plans.

The serving path never times kernels. This tool compiles explicit tile-MN, BK,
load-path, and operand-storage candidates, requires exact output equality
against the trusted b12x BK128 plan, times already-captured graphs in balanced
order, and writes raw samples for a reviewed planner policy.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import statistics
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import cutlass
import torch

from benchmarks.experimental.b12x.dense_autotune_common import (
    bench_events,
    capture_graph_replay,
    make_nvfp4_operand,
)
from benchmarks.experimental.b12x.common import make_l2_flush_fn, resolve_l2_flush_bytes
from b12x._lib.dense_gemm import _select_default_dense_gemm_plan, dense_gemm
from b12x._lib.dense_gemm import DenseGemmKernel
from b12x._lib.utils import get_num_sm


@dataclass(frozen=True)
class Shape:
    name: str
    n: int
    k: int
    note: str


@dataclass
class CandidateRun:
    candidate: Candidate
    replay: object
    samples_ms: list[float]


@dataclass(frozen=True)
class Candidate:
    tile_mn: tuple[int, int]
    tile_k: int
    load_path: str
    swap_ab: bool
    target_occupancy: int | None

    def label(self) -> str:
        storage = "swap" if self.swap_ab else "normal"
        return (
            f"{self.tile_mn[0]}x{self.tile_mn[1]}/BK{self.tile_k}/"
            f"{self.load_path}/{storage}/"
            f"occ{self.target_occupancy if self.target_occupancy is not None else 'auto'}"
        )


QWEN38_SHAPES = (
    Shape("qwen38_27b_gate_or_up", 17408, 5120, "dense SwiGLU gate or up"),
    Shape("qwen38_27b_down", 5120, 17408, "dense SwiGLU down"),
)

QWEN38_TP_SHAPES = tuple(
    shape
    for tp in (1, 2, 4, 8)
    for shape in (
        Shape(
            f"qwen38_27b_gate_or_up_tp{tp}",
            17408 // tp,
            5120,
            f"column-parallel dense SwiGLU gate or up, TP={tp}",
        ),
        Shape(
            f"qwen38_27b_down_tp{tp}",
            5120,
            17408 // tp,
            f"row-parallel dense SwiGLU down, TP={tp}",
        ),
    )
)

COMMON_SHAPES = (
    Shape("nemotron_shared_down_full", 4096, 5376, "shared expert down"),
    Shape("nemotron_shared_down_n2048", 2048, 5376, "output shard proxy"),
    Shape("nemotron_shared_down_n1024", 1024, 5376, "output shard proxy"),
    Shape("nemotron_shared_down_n512", 512, 5376, "output shard proxy"),
    Shape("qwen35_moe_fc1_fused", 2048, 4096, "Qwen MoE fused gate/up"),
    Shape("qwen35_moe_fc2_down", 4096, 1024, "Qwen MoE down"),
    Shape("nemotron_backbone_relu2_fc1", 2688, 1024, "Nemotron relu2 up"),
    Shape("nemotron_backbone_relu2_fc2", 1024, 2688, "Nemotron relu2 down"),
    Shape("nano35_relu2_fc1", 1856, 2688, "Nano relu2 up"),
    Shape("nano35_relu2_fc2_unaligned_k", 2688, 1856, "unaligned-K down"),
    Shape("dsv4f_silu_fc1_fused", 2048, 6144, "DSV4 fused gate/up"),
    Shape("dsv4f_silu_fc2_down", 6144, 1024, "DSV4 down"),
    Shape("deepseek_qkv_a_down", 1536, 4096, "DeepSeek qkv_a down"),
    Shape("deepseek_q_b_up", 16384, 1024, "DeepSeek q_b up"),
    Shape("deepseek_wo_a", 1024, 4096, "DeepSeek WO-A"),
    Shape("deepseek_wo_b", 4096, 4096, "DeepSeek WO-B"),
)


def _parse_int_list(raw: str) -> list[int]:
    values = [int(item) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    return values


def _parse_tile_list(raw: str) -> list[tuple[int, int]]:
    try:
        values = [
            tuple(int(axis) for axis in item.lower().split("x"))
            for item in raw.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tiles must use MxN syntax") from exc
    if not values or any(len(tile) != 2 or min(tile) <= 0 for tile in values):
        raise argparse.ArgumentTypeError("tiles must use positive MxN syntax")
    return [(tile[0], tile[1]) for tile in values]


def _parse_load_paths(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values or any(item not in ("tma", "cpasync") for item in values):
        raise argparse.ArgumentTypeError("load paths must be tma and/or cpasync")
    return values


def _parse_bool_list(raw: str) -> list[bool]:
    aliases = {"0": False, "false": False, "1": True, "true": True}
    try:
        values = [
            aliases[item.strip().lower()] for item in raw.split(",") if item.strip()
        ]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "swap values must be 0/1 or false/true"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("swap list must not be empty")
    return values


def _plan_candidates(
    plan: object,
    *,
    tile_k_list: list[int],
    joint: bool,
    tile_mn_list: list[tuple[int, int]] | None,
    load_path_list: list[str] | None,
    swap_ab_list: list[bool] | None,
    target_occupancy_list: list[int] | None = None,
) -> list[Candidate]:
    if joint:
        tile_storage = (
            ((64, 32), True),
            ((64, 64), False),
            ((64, 128), False),
            ((128, 64), False),
            ((128, 128), False),
        )
        # TMA is the serving load path and dominated cp.async throughout the
        # Qwen TP sweep. Keep cp.async available through --load-path-list for
        # focused diagnostics without multiplying every production-plan run.
        load_paths = ("tma",)
    else:
        tiles = tile_mn_list or [plan.mma_tiler_mn]
        swaps = swap_ab_list if swap_ab_list is not None else [plan.swap_ab]
        tile_storage = tuple((tile, swap) for tile in tiles for swap in swaps)
        load_paths = tuple(load_path_list or [plan.load_path])
    return [
        Candidate(tile, tile_k, load_path, swap, target_occupancy)
        for tile, swap in tile_storage
        for tile_k in tile_k_list
        for load_path in load_paths
        for target_occupancy in (target_occupancy_list or [None])
        if not (tile == (128, 128) and tile_k == 512)
    ]


def _shapes(args: argparse.Namespace) -> list[Shape]:
    if args.n is not None:
        return [Shape(args.name, args.n, args.k, "explicit CLI shape")]
    if args.n_list is not None:
        return [
            Shape(f"grid_n{n}_k{k}", n, k, "CLI Cartesian grid")
            for n in args.n_list
            for k in args.k_list
        ]
    if args.shape_set == "qwen38":
        return list(QWEN38_SHAPES)
    if args.shape_set == "qwen38-tp":
        return list(QWEN38_TP_SHAPES)
    if args.shape_set == "common":
        return list(COMMON_SHAPES)
    return list(QWEN38_SHAPES + COMMON_SHAPES)


def _baseline(
    a_packed: torch.Tensor,
    a_sf: torch.Tensor,
    b_packed: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    *,
    m: int,
    n: int,
    k: int,
    plan: object,
) -> torch.Tensor:
    out = torch.empty((m, n, 1), device="cuda", dtype=torch.bfloat16)
    dense_gemm(
        (a_packed, a_sf),
        (b_packed, b_sf),
        out=out,
        alpha=alpha,
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype="bfloat16",
        sf_vec_size=16,
        mma_tiler_mn=plan.mma_tiler_mn,
        load_path=plan.load_path,
        swap_ab=plan.swap_ab,
        _tile_k_override=128 if k % 128 == 0 else None,
    )
    torch.cuda.synchronize()
    return out[:, :, 0].clone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape-set",
        choices=("qwen38", "qwen38-tp", "common", "all"),
        default="all",
    )
    parser.add_argument("--name", default="custom")
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--n-list", type=_parse_int_list)
    parser.add_argument("--k-list", type=_parse_int_list)
    parser.add_argument(
        "--m-list", type=_parse_int_list, default=_parse_int_list("1,2,4,6,8")
    )
    parser.add_argument(
        "--tile-k-list", type=_parse_int_list, default=_parse_int_list("128,256,512")
    )
    parser.add_argument(
        "--joint",
        action="store_true",
        help=(
            "search the production TMA FP4 tile/storage set; otherwise only "
            "explicitly supplied axes or the default plan are used"
        ),
    )
    parser.add_argument("--tile-mn-list", type=_parse_tile_list)
    parser.add_argument("--load-path-list", type=_parse_load_paths)
    parser.add_argument("--swap-ab-list", type=_parse_bool_list)
    parser.add_argument("--target-occupancy-list", type=_parse_int_list)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("results.dense.fp4_tile_k.tsv"),
    )
    parser.add_argument("--no-l2-flush", action="store_true")
    args = parser.parse_args()

    if (args.n is None) != (args.k is None):
        parser.error("--n and --k must be supplied together")
    if (args.n_list is None) != (args.k_list is None):
        parser.error("--n-list and --k-list must be supplied together")
    if args.n is not None and args.n_list is not None:
        parser.error("explicit --n/--k and --n-list/--k-list are mutually exclusive")
    if args.n is not None and (args.n <= 0 or args.k <= 0):
        parser.error("--n and --k must be positive")
    if args.n_list is not None and (
        any(n <= 0 for n in args.n_list) or any(k <= 0 for k in args.k_list)
    ):
        parser.error("--n-list and --k-list values must be positive")
    if args.warmup < 0 or args.iters <= 0 or args.repeats <= 0:
        parser.error("warmup must be nonnegative; iters and repeats must be positive")
    if any(tile_k not in (128, 256, 512) for tile_k in args.tile_k_list):
        parser.error("--tile-k-list is restricted to 128,256,512")
    if args.target_occupancy_list is not None and any(
        occupancy not in (1, 2, 3, 4) for occupancy in args.target_occupancy_list
    ):
        parser.error("--target-occupancy-list is restricted to 1,2,3,4")
    if args.joint and any(
        value is not None
        for value in (
            args.tile_mn_list,
            args.load_path_list,
            args.swap_ab_list,
            args.target_occupancy_list,
        )
    ):
        parser.error(
            "--joint is mutually exclusive with explicit tile/load/swap candidate lists"
        )

    torch.empty(1, device="cuda")
    sm_count = get_num_sm(torch.device("cuda"))
    l2_flush = make_l2_flush_fn(enabled=not args.no_l2_flush, bytes_hint=0)
    l2_bytes = resolve_l2_flush_bytes(0) if l2_flush is not None else 0
    shape_cases = _shapes(args)

    metadata = (
        f"gpu={torch.cuda.get_device_name()} sm_count={sm_count} "
        f"arch={os.getenv('CUTE_DSL_ARCH', '')} cutlass={cutlass.__version__} "
        f"warmup={args.warmup} iters={args.iters} repeats={args.repeats} "
        f"l2_flush_bytes={l2_bytes}"
    )
    print(metadata)
    print(
        f"shapes={len(shape_cases)} M={args.m_list} BK={args.tile_k_list} "
        f"joint={args.joint}"
    )

    columns = (
        "shape\tm\tn\tk\ttile_m\ttile_n\ttile_k\tload_path\tswap_ab\t"
        "target_occupancy\t"
        "status\tmedian_us\tmin_us\tmax_us\tcos\tmax_abs\tsamples_us\n"
    )
    with args.output.open("w", encoding="utf-8") as output:
        output.write(f"# {metadata}\n")
        output.write(columns)

        for shape in shape_cases:
            print(f"\n### {shape.name}: Mx{shape.n}x{shape.k} ({shape.note})")
            torch.manual_seed(42 + shape.n + shape.k)
            b_packed, b_sf, b_gs = make_nvfp4_operand(shape.n, shape.k)

            for m in args.m_list:
                torch.manual_seed(42 + m + shape.n + shape.k)
                a_packed, a_sf, a_gs = make_nvfp4_operand(m, shape.k)
                alpha = (1.0 / (a_gs[0] * b_gs[0])).view(1)
                plan = _select_default_dense_gemm_plan(
                    m,
                    shape.n,
                    shape.k,
                    sm_count,
                    is_mxfp8=False,
                )
                candidates = _plan_candidates(
                    plan,
                    tile_k_list=args.tile_k_list,
                    joint=args.joint,
                    tile_mn_list=args.tile_mn_list,
                    load_path_list=args.load_path_list,
                    swap_ab_list=args.swap_ab_list,
                    target_occupancy_list=args.target_occupancy_list,
                )
                if not any(shape.k % candidate.tile_k == 0 for candidate in candidates):
                    for candidate in candidates:
                        output.write(
                            f"{shape.name}\t{m}\t{shape.n}\t{shape.k}\t"
                            f"{candidate.tile_mn[0]}\t{candidate.tile_mn[1]}\t"
                            f"{candidate.tile_k}\t{candidate.load_path}\t"
                            f"{int(candidate.swap_ab)}\t"
                            f"{candidate.target_occupancy or 'auto'}\t"
                            "not_divisible\t\t\t\t\t\t\n"
                        )
                    print(f"  M={m:<3} no requested tile K divides K={shape.k}")
                    continue
                baseline = _baseline(
                    a_packed,
                    a_sf,
                    b_packed,
                    b_sf,
                    alpha,
                    m=m,
                    n=shape.n,
                    k=shape.k,
                    plan=plan,
                )
                prepared: list[CandidateRun] = []
                for candidate in candidates:
                    if shape.k % candidate.tile_k:
                        output.write(
                            f"{shape.name}\t{m}\t{shape.n}\t{shape.k}\t"
                            f"{candidate.tile_mn[0]}\t{candidate.tile_mn[1]}\t"
                            f"{candidate.tile_k}\t{candidate.load_path}\t"
                            f"{int(candidate.swap_ab)}\t"
                            f"{candidate.target_occupancy or 'auto'}\t"
                            "not_divisible\t\t\t\t\t\t\n"
                        )
                        continue
                    if not DenseGemmKernel.can_implement(
                        cutlass.Float4E2M1FN,
                        cutlass.Float8E4M3FN,
                        16,
                        cutlass.BFloat16,
                        candidate.tile_mn,
                        (1, 1),
                        shape.n,
                        shape.k,
                        1,
                        "k",
                        "k",
                        "n",
                        load_path=candidate.load_path,
                        swap_ab=candidate.swap_ab,
                    ):
                        output.write(
                            f"{shape.name}\t{m}\t{shape.n}\t{shape.k}\t"
                            f"{candidate.tile_mn[0]}\t{candidate.tile_mn[1]}\t"
                            f"{candidate.tile_k}\t{candidate.load_path}\t"
                            f"{int(candidate.swap_ab)}\t"
                            f"{candidate.target_occupancy or 'auto'}\t"
                            "unsupported\t\t\t\t\t\t\n"
                        )
                        continue
                    out = torch.empty(
                        (m, shape.n, 1), device="cuda", dtype=torch.bfloat16
                    )

                    def launch(
                        out: torch.Tensor = out,
                        candidate: Candidate = candidate,
                    ) -> None:
                        dense_gemm(
                            (a_packed, a_sf),
                            (b_packed, b_sf),
                            out=out,
                            alpha=alpha,
                            ab_dtype="float4_e2m1fn",
                            sf_dtype="float8_e4m3fn",
                            c_dtype="bfloat16",
                            sf_vec_size=16,
                            mma_tiler_mn=candidate.tile_mn,
                            load_path=candidate.load_path,
                            swap_ab=candidate.swap_ab,
                            _tile_k_override=candidate.tile_k,
                            _target_occupancy_override=candidate.target_occupancy,
                        )

                    try:
                        replay = capture_graph_replay(launch)
                        replay()
                        torch.cuda.synchronize()
                    except Exception as exc:
                        message = str(exc).splitlines()[0].replace("\t", " ")[:160]
                        print(
                            f"  M={m:<3} {candidate.label()} compile/launch failed: "
                            f"{message}"
                        )
                        output.write(
                            f"{shape.name}\t{m}\t{shape.n}\t{shape.k}\t"
                            f"{candidate.tile_mn[0]}\t{candidate.tile_mn[1]}\t"
                            f"{candidate.tile_k}\t{candidate.load_path}\t"
                            f"{int(candidate.swap_ab)}\t"
                            f"{candidate.target_occupancy or 'auto'}\t"
                            f"launch_failed:{message}\t\t\t\t\t\t\n"
                        )
                        continue

                    candidate_out = out[:, :, 0]
                    max_abs = (
                        (candidate_out.float() - baseline.float()).abs().max().item()
                    )
                    if not torch.equal(candidate_out, baseline):
                        raise RuntimeError(
                            f"correctness failure for {shape.name} M={m} "
                            f"plan={candidate.label()}: "
                            f"max_abs={max_abs}"
                        )
                    prepared.append(CandidateRun(candidate, replay, []))

                for repeat in range(args.repeats):
                    ordered = prepared if repeat % 2 == 0 else list(reversed(prepared))
                    for candidate in ordered:
                        candidate.samples_ms.extend(
                            bench_events(
                                candidate.replay,
                                warmup=args.warmup,
                                iters=args.iters,
                                l2_flush=l2_flush,
                            )
                        )

                scored: list[tuple[float, CandidateRun]] = []
                for candidate in prepared:
                    samples_us = [sample * 1000 for sample in candidate.samples_ms]
                    median_us = statistics.median(samples_us)
                    scored.append((median_us, candidate))
                    sample_text = ",".join(f"{sample:.3f}" for sample in samples_us)
                    output.write(
                        f"{shape.name}\t{m}\t{shape.n}\t{shape.k}\t"
                        f"{candidate.candidate.tile_mn[0]}\t"
                        f"{candidate.candidate.tile_mn[1]}\t"
                        f"{candidate.candidate.tile_k}\t"
                        f"{candidate.candidate.load_path}\t"
                        f"{int(candidate.candidate.swap_ab)}\t"
                        f"{candidate.candidate.target_occupancy or 'auto'}\t"
                        f"pass\t{median_us:.3f}\t"
                        f"{min(samples_us):.3f}\t{max(samples_us):.3f}\t1.0\t0.0\t"
                        f"{sample_text}\n"
                    )
                    print(
                        f"  M={m:<3} {candidate.candidate.label():<31} "
                        f"{median_us:8.2f} us exact"
                    )
                if scored:
                    winner_us, winner = min(scored, key=lambda item: item[0])
                    baseline_us = next(
                        (
                            value
                            for value, item in scored
                            if item.candidate.tile_k == 128
                            and item.candidate.tile_mn == plan.mma_tiler_mn
                            and item.candidate.load_path == plan.load_path
                            and item.candidate.swap_ab == plan.swap_ab
                        ),
                        None,
                    )
                    speedup = ""
                    if baseline_us is not None:
                        speedup = f" ({baseline_us / winner_us:.3f}x vs default/BK128)"
                    print(
                        f"  -> M={m:<3} winner {winner.candidate.label()} "
                        f"{winner_us:.2f} us{speedup}"
                    )

    print(f"\nraw_results={args.output}")


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
