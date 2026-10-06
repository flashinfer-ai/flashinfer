#!/usr/bin/env python3
"""Offline graph-replay autotuner for complete dense MXFP8 plans.

Serving never times kernels. This tool searches explicit tile-MN, staged-K,
and decode split-K choices, checks every candidate against a trusted b12x
direct/BK128 result, and writes balanced raw timing samples. Reference backends
are intentionally absent from calibration; they belong only in the final
acceptance benchmark.
"""

from __future__ import annotations

import argparse
import gc
import os
import pathlib
import statistics
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import cutlass
import torch

from benchmarks.experimental.b12x.common import make_l2_flush_fn, resolve_l2_flush_bytes
from benchmarks.experimental.b12x.dense_autotune_common import (
    bench_events,
    capture_graph_replay,
    cosine_similarity,
    make_mxfp8_operand,
)
from b12x._lib.dense_gemm import (
    DenseGemmKernel,
    _select_default_dense_gemm_plan,
    dense_gemm,
)
from b12x._lib.utils import get_num_sm


BASELINE_COSINE_THRESHOLD = 0.9999
PRODUCTION_TILES = (
    (16, 64),
    (16, 128),
    (32, 64),
    (32, 128),
    (64, 64),
    (64, 128),
    (128, 64),
    (128, 128),
)


@dataclass(frozen=True)
class Shape:
    name: str
    n: int
    k: int
    note: str


@dataclass(frozen=True)
class Candidate:
    tile_mn: tuple[int, int]
    tile_k: int
    split_k_slices: int
    large_m_unroll: bool | None

    def label(self) -> str:
        unroll = "auto" if self.large_m_unroll is None else int(self.large_m_unroll)
        return (
            f"{self.tile_mn[0]}x{self.tile_mn[1]}/BK{self.tile_k}/"
            f"split{self.split_k_slices}/unroll{unroll}"
        )


@dataclass
class CandidateRun:
    candidate: Candidate
    cos: float
    max_abs: float
    samples_ms: list[float]


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

Q_B_TP_SHAPES = tuple(
    Shape(
        f"deepseek_q_b_tp{tp}",
        32768 // tp,
        1024,
        f"column-parallel q_b projection, TP={tp}",
    )
    for tp in (1, 2, 4, 8)
)

COMMON_SHAPES = (
    Shape("deepseek_wo_b", 4096, 4096, "WO-B projection"),
    Shape("deepseek_wo_a", 1024, 4096, "WO-A projection"),
    Shape("deepseek_qkv_a_down", 1536, 4096, "qkv_a down projection"),
    Shape("nemotron_shared_down", 4096, 5376, "shared expert down"),
    Shape("glm5_dense_down", 6144, 1536, "GLM dense down"),
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


def _parse_unroll_list(raw: str) -> list[bool | None]:
    aliases: dict[str, bool | None] = {
        "auto": None,
        "0": False,
        "false": False,
        "1": True,
        "true": True,
    }
    try:
        values = [aliases[item.strip().lower()] for item in raw.split(",") if item.strip()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "large-M unroll values must be auto, 0/false, or 1/true"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("large-M unroll list must not be empty")
    return values


def _shapes(args: argparse.Namespace) -> list[Shape]:
    if args.n is not None:
        return [Shape(args.name, args.n, args.k, "explicit CLI shape")]
    if args.n_list is not None:
        return [
            Shape(f"grid_n{n}_k{k}", n, k, "CLI Cartesian grid")
            for n in args.n_list
            for k in args.k_list
        ]
    if args.shape_set == "qwen38-tp":
        return list(QWEN38_TP_SHAPES)
    if args.shape_set == "q-b-tp":
        return list(Q_B_TP_SHAPES)
    if args.shape_set == "common":
        return list(COMMON_SHAPES)
    return list(QWEN38_TP_SHAPES + Q_B_TP_SHAPES + COMMON_SHAPES)


def _plan_candidates(
    *,
    m: int,
    tile_mn_list: list[tuple[int, int]],
    tile_k_list: list[int],
    split_k_list: list[int],
    large_m_unroll_list: list[bool | None],
) -> list[Candidate]:
    slices = split_k_list if m <= 8 else [1]
    return [
        Candidate(tile, tile_k, split_k, large_m_unroll)
        for tile in tile_mn_list
        for tile_k in tile_k_list
        for split_k in slices
        for large_m_unroll in large_m_unroll_list
        if tile_k != 64 or tile[0] == 128
    ]


def _baseline(
    a_quantized: torch.Tensor,
    a_scale_mma: torch.Tensor,
    b_quantized: torch.Tensor,
    b_scale_mma: torch.Tensor,
    *,
    m: int,
    n: int,
    k: int,
    plan: object,
) -> torch.Tensor:
    out = torch.empty((m, n, 1), device="cuda", dtype=torch.bfloat16)
    dense_gemm(
        (a_quantized.view(m, k, 1), a_scale_mma),
        (b_quantized.view(n, k, 1), b_scale_mma),
        out=out,
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
        expected_m=m,
        mma_tiler_mn=plan.mma_tiler_mn,
        load_path="tma",
        swap_ab=False,
        _tile_k_override=128,
        _split_k_slices_override=1,
    )
    torch.cuda.synchronize()
    return out[:, :, 0].clone()


def _check_candidate(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    *,
    shape: Shape,
    m: int,
    plan: Candidate,
) -> tuple[float, float]:
    if not bool(torch.isfinite(candidate).all().item()):
        raise RuntimeError(f"non-finite output for {shape.name} M={m} {plan.label()}")
    if not bool(torch.count_nonzero(candidate).item()):
        raise RuntimeError(f"zero output for {shape.name} M={m} {plan.label()}")
    max_abs = (candidate.float() - baseline.float()).abs().max().item()
    if plan.split_k_slices == 1 and not torch.equal(candidate, baseline):
        raise RuntimeError(
            f"direct-plan correctness failure for {shape.name} M={m} "
            f"{plan.label()}: max_abs={max_abs}"
        )
    cos = cosine_similarity(candidate, baseline)
    if cos < BASELINE_COSINE_THRESHOLD:
        raise RuntimeError(
            f"correctness failure for {shape.name} M={m} {plan.label()}: "
            f"max_abs={max_abs}, cos={cos}"
        )
    return cos, max_abs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape-set",
        choices=("qwen38-tp", "q-b-tp", "common", "all"),
        default="qwen38-tp",
    )
    parser.add_argument("--name", default="custom")
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--n-list", type=_parse_int_list)
    parser.add_argument("--k-list", type=_parse_int_list)
    parser.add_argument(
        "--m-list",
        type=_parse_int_list,
        default=_parse_int_list("1,6,64,128,2048,4096"),
    )
    parser.add_argument(
        "--tile-mn-list",
        type=_parse_tile_list,
        default=list(PRODUCTION_TILES),
    )
    parser.add_argument(
        "--tile-k-list", type=_parse_int_list, default=_parse_int_list("64,128")
    )
    parser.add_argument(
        "--split-k-list", type=_parse_int_list, default=_parse_int_list("1,2,4")
    )
    parser.add_argument(
        "--large-m-unroll-list",
        type=_parse_unroll_list,
        default=_parse_unroll_list("auto"),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("results.dense.mxfp8_plan.tsv"),
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
    if any(m <= 0 for m in args.m_list):
        parser.error("--m-list values must be positive")
    if any(tile_k not in (64, 128) for tile_k in args.tile_k_list):
        parser.error("--tile-k-list is restricted to 64,128")
    if any(slices not in (1, 2, 4) for slices in args.split_k_list):
        parser.error("--split-k-list is restricted to 1,2,4")
    if args.warmup < 0 or args.iters <= 0 or args.repeats <= 0:
        parser.error("warmup must be nonnegative; iters and repeats must be positive")

    torch.empty(1, device="cuda")
    sm_count = get_num_sm(torch.device("cuda"))
    l2_flush = make_l2_flush_fn(enabled=not args.no_l2_flush, bytes_hint=0)
    l2_bytes = resolve_l2_flush_bytes(0) if l2_flush is not None else 0
    shape_cases = _shapes(args)
    metadata = (
        f"gpu={torch.cuda.get_device_name()} sm_count={sm_count} "
        f"arch={os.getenv('CUTE_DSL_ARCH', '')} cutlass={cutlass.__version__} "
        f"warmup={args.warmup} iters={args.iters} repeats={args.repeats} "
        f"l2_flush_bytes={l2_bytes} calibration=b12x-only graph_replay=1"
    )
    print(metadata)
    print(
        f"shapes={len(shape_cases)} M={args.m_list} tiles={args.tile_mn_list} "
        f"BK={args.tile_k_list} splitK={args.split_k_list} "
        f"largeMUnroll={args.large_m_unroll_list}"
    )

    columns = (
        "shape\tm\tn\tk\ttile_m\ttile_n\ttile_k\twork_tiles\tsm_count\t"
        "split_k_slices\tlarge_m_unroll\tstatus\tmedian_us\tmin_us\tmax_us\tcos\tmax_abs\t"
        "samples_us\n"
    )
    with args.output.open("w", encoding="utf-8") as output:
        output.write(f"# {metadata}\n")
        output.write(columns)
        for shape in shape_cases:
            print(f"\n### {shape.name}: Mx{shape.n}x{shape.k} ({shape.note})")
            torch.manual_seed(42 + shape.n + shape.k)
            b_q, b_scale_mma = make_mxfp8_operand(shape.n, shape.k)
            for m in args.m_list:
                torch.manual_seed(42 + m + shape.n + shape.k)
                a_q, a_scale_mma = make_mxfp8_operand(m, shape.k)
                default_plan = _select_default_dense_gemm_plan(
                    m,
                    shape.n,
                    shape.k,
                    sm_count,
                    is_mxfp8=True,
                    expected_m=m,
                )
                baseline = _baseline(
                    a_q,
                    a_scale_mma,
                    b_q,
                    b_scale_mma,
                    m=m,
                    n=shape.n,
                    k=shape.k,
                    plan=default_plan,
                )
                candidates = _plan_candidates(
                    m=m,
                    tile_mn_list=args.tile_mn_list,
                    tile_k_list=args.tile_k_list,
                    split_k_list=args.split_k_list,
                    large_m_unroll_list=args.large_m_unroll_list,
                )
                runnable: list[Candidate] = []
                for candidate in candidates:
                    work_tiles = (
                        ((m + candidate.tile_mn[0] - 1) // candidate.tile_mn[0])
                        * ((shape.n + candidate.tile_mn[1] - 1) // candidate.tile_mn[1])
                    )
                    prefix = (
                        f"{shape.name}\t{m}\t{shape.n}\t{shape.k}\t"
                        f"{candidate.tile_mn[0]}\t{candidate.tile_mn[1]}\t"
                        f"{candidate.tile_k}\t{work_tiles}\t{sm_count}\t"
                        f"{candidate.split_k_slices}\t{candidate.large_m_unroll}\t"
                    )
                    if shape.k % (candidate.tile_k * candidate.split_k_slices):
                        output.write(prefix + "not_divisible\t\t\t\t\t\t\n")
                        continue
                    if candidate.split_k_slices > 1 and (
                        m > 8 or m > candidate.tile_mn[0]
                    ):
                        output.write(prefix + "unsupported_split\t\t\t\t\t\t\n")
                        continue
                    if not DenseGemmKernel.can_implement(
                        cutlass.Float8E4M3FN,
                        cutlass.Float8E8M0FNU,
                        32,
                        cutlass.BFloat16,
                        candidate.tile_mn,
                        (1, 1),
                        shape.n,
                        shape.k,
                        1,
                        "k",
                        "k",
                        "n",
                        load_path="tma",
                        swap_ab=False,
                    ):
                        output.write(prefix + "unsupported\t\t\t\t\t\t\n")
                        continue
                    runnable.append(candidate)

                # Retaining every candidate graph at once can exhaust or corrupt
                # driver graph resources on large-M shapes even though every
                # candidate replays correctly in isolation. Capture one stable-
                # allocation graph at a time, measure it, and destroy it before
                # moving on. Alternating candidate order between repeats retains
                # balanced timing without accumulating graph executables.
                measured: dict[Candidate, CandidateRun] = {}
                failed: set[Candidate] = set()
                for repeat in range(args.repeats):
                    ordered = runnable if repeat % 2 == 0 else list(reversed(runnable))
                    for candidate in ordered:
                        if candidate in failed:
                            continue
                        out = torch.empty(
                            (m, shape.n, 1), device="cuda", dtype=torch.bfloat16
                        )

                        def launch(
                            out: torch.Tensor = out,
                            candidate: Candidate = candidate,
                        ) -> None:
                            dense_gemm(
                                (a_q.view(m, shape.k, 1), a_scale_mma),
                                (b_q.view(shape.n, shape.k, 1), b_scale_mma),
                                out=out,
                                ab_dtype="float8_e4m3fn",
                                sf_dtype="float8_e8m0fnu",
                                c_dtype="bfloat16",
                                sf_vec_size=32,
                                expected_m=m,
                                mma_tiler_mn=candidate.tile_mn,
                                load_path="tma",
                                swap_ab=False,
                                _tile_k_override=candidate.tile_k,
                                _split_k_slices_override=candidate.split_k_slices,
                                _large_m_unroll_override=candidate.large_m_unroll,
                            )

                        replay = None
                        try:
                            replay = capture_graph_replay(launch)
                            replay()
                            torch.cuda.synchronize()
                            cos, max_abs = _check_candidate(
                                out[:, :, 0],
                                baseline,
                                shape=shape,
                                m=m,
                                plan=candidate,
                            )
                            samples = bench_events(
                                replay,
                                warmup=args.warmup,
                                iters=args.iters,
                                l2_flush=l2_flush,
                            )
                        except Exception as exc:
                            message = str(exc).splitlines()[0].replace("\t", " ")[:160]
                            print(f"  M={m:<4} {candidate.label()} failed: {message}")
                            tile_m, tile_n = candidate.tile_mn
                            work_tiles = ((m + tile_m - 1) // tile_m) * (
                                (shape.n + tile_n - 1) // tile_n
                            )
                            output.write(
                                f"{shape.name}\t{m}\t{shape.n}\t{shape.k}\t"
                                f"{tile_m}\t{tile_n}\t{candidate.tile_k}\t"
                                f"{work_tiles}\t{sm_count}\t"
                                f"{candidate.split_k_slices}\t"
                                f"{candidate.large_m_unroll}\tfailed:{message}\t"
                                "\t\t\t\t\t\n"
                            )
                            failed.add(candidate)
                            measured.pop(candidate, None)
                        else:
                            run = measured.setdefault(
                                candidate,
                                CandidateRun(candidate, cos, max_abs, []),
                            )
                            run.samples_ms.extend(samples)
                            run.cos = min(run.cos, cos)
                            run.max_abs = max(run.max_abs, max_abs)
                        finally:
                            del replay
                            del launch
                            del out
                            gc.collect()
                            torch.cuda.empty_cache()

                scored: list[tuple[float, CandidateRun]] = []
                for candidate in measured.values():
                    samples_us = [sample * 1000 for sample in candidate.samples_ms]
                    median_us = statistics.median(samples_us)
                    scored.append((median_us, candidate))
                    tile_m, tile_n = candidate.candidate.tile_mn
                    work_tiles = ((m + tile_m - 1) // tile_m) * (
                        (shape.n + tile_n - 1) // tile_n
                    )
                    sample_text = ",".join(f"{sample:.3f}" for sample in samples_us)
                    output.write(
                        f"{shape.name}\t{m}\t{shape.n}\t{shape.k}\t{tile_m}\t{tile_n}\t"
                        f"{candidate.candidate.tile_k}\t{work_tiles}\t{sm_count}\t"
                        f"{candidate.candidate.split_k_slices}\t"
                        f"{candidate.candidate.large_m_unroll}\tpass\t{median_us:.3f}\t"
                        f"{min(samples_us):.3f}\t{max(samples_us):.3f}\t"
                        f"{candidate.cos:.10f}\t{candidate.max_abs:.8f}\t{sample_text}\n"
                    )
                    print(
                        f"  M={m:<4} {candidate.candidate.label():<25} "
                        f"{median_us:8.2f} us cos={candidate.cos:.8f}"
                    )
                if scored:
                    winner_us, winner = min(scored, key=lambda item: item[0])
                    default_us = next(
                        (
                            value
                            for value, item in scored
                            if item.candidate.tile_mn == default_plan.mma_tiler_mn
                            and item.candidate.tile_k == 128
                            and item.candidate.split_k_slices == 1
                            and item.candidate.large_m_unroll is None
                        ),
                        None,
                    )
                    speedup = (
                        f" ({default_us / winner_us:.3f}x vs default/direct/BK128)"
                        if default_us is not None
                        else ""
                    )
                    print(
                        f"  -> M={m:<4} winner {winner.candidate.label()} "
                        f"{winner_us:.2f} us{speedup}"
                    )
    print(f"\nraw_results={args.output}")


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
