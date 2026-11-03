#!/usr/bin/env python3
"""Offline graph-replay autotuner for dense MXFP8 split-K slices.

Serving never times kernels. This tool compiles explicit 1/2/4-slice plans,
checks every replay against the trusted b12x direct plan, times captured graphs
in balanced order, and writes raw samples from which a reviewed SM/grid rule
can be derived.
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
    cosine_similarity,
    make_mxfp8_operand,
)
from benchmarks.experimental.b12x.common import make_l2_flush_fn, resolve_l2_flush_bytes
from b12x._lib.dense_gemm import (
    _select_default_dense_gemm_plan,
    _select_mxfp8_tile_k,
    dense_gemm,
)
from b12x._lib.utils import get_num_sm


BASELINE_COSINE_THRESHOLD = 0.9999


@dataclass(frozen=True)
class Shape:
    name: str
    n: int
    k: int
    note: str


@dataclass
class CandidateRun:
    slices: int
    replay: object
    out: torch.Tensor
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

SPLIT_K_SHAPES = (
    Shape("deepseek_wo_b", 4096, 4096, "production WO-B split-K shape"),
    *QWEN38_TP_SHAPES,
)

COMMON_SHAPES = (
    Shape("nemotron_shared_down_full", 4096, 5376, "shared expert down"),
    Shape("nemotron_shared_down_n2048", 2048, 5376, "output shard proxy"),
    Shape("nemotron_shared_down_n1024", 1024, 5376, "output shard proxy"),
    Shape("nemotron_shared_down_n512", 512, 5376, "output shard proxy"),
    Shape("deepseek_qkv_a_down", 1536, 4096, "DeepSeek qkv_a down"),
    Shape("deepseek_q_b_up", 16384, 1024, "DeepSeek q_b up"),
    Shape("deepseek_wo_a", 1024, 4096, "DeepSeek WO-A"),
    Shape("deepseek_wo_b", 4096, 4096, "DeepSeek WO-B"),
    Shape("glm5_dense_down", 6144, 1536, "GLM dense down"),
    Shape("wo_b_like_n7168_k512", 7168, 512, "short-K wide-N proxy"),
)


def _parse_int_list(raw: str) -> list[int]:
    values = [int(item) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
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
    if args.shape_set == "common":
        return list(COMMON_SHAPES)
    if args.shape_set == "all":
        return list(SPLIT_K_SHAPES + COMMON_SHAPES)
    return list(SPLIT_K_SHAPES)


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
        load_path=plan.load_path,
        swap_ab=plan.swap_ab,
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
    slices: int,
) -> tuple[float, float]:
    if not bool(torch.isfinite(candidate).all().item()):
        raise RuntimeError(f"non-finite output for {shape.name} M={m} split_k={slices}")
    if not bool(torch.count_nonzero(candidate).item()):
        raise RuntimeError(f"zero output for {shape.name} M={m} split_k={slices}")
    max_abs = (candidate.float() - baseline.float()).abs().max().item()
    cos = cosine_similarity(candidate, baseline)
    if cos < BASELINE_COSINE_THRESHOLD:
        raise RuntimeError(
            f"correctness failure for {shape.name} M={m} split_k={slices}: "
            f"max_abs={max_abs}, cos={cos}"
        )
    return cos, max_abs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape-set",
        choices=("split", "qwen38-tp", "common", "all"),
        default="split",
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
        "--split-k-list", type=_parse_int_list, default=_parse_int_list("1,2,4")
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("results.dense.mxfp8_split_k.tsv"),
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
    if any(m < 1 or m > 8 for m in args.m_list):
        parser.error("--m-list is restricted to the split-K decode range 1..8")
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
        f"l2_flush_bytes={l2_bytes} atomic_bf16=1"
    )
    print(metadata)
    print(f"shapes={len(shape_cases)} M={args.m_list} splitK={args.split_k_list}")

    columns = (
        "shape\tm\tn\tk\ttile_m\ttile_n\ttile_k\twork_tiles\tsm_count\t"
        "split_k_slices\tstatus\tmedian_us\tmin_us\tmax_us\tcos\tmax_abs\t"
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
                plan = _select_default_dense_gemm_plan(
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
                    plan=plan,
                )
                tile_k = _select_mxfp8_tile_k(m, shape.n, shape.k, m, sm_count)
                work_tiles = (
                    (m + plan.mma_tiler_mn[0] - 1) // plan.mma_tiler_mn[0]
                ) * ((shape.n + plan.mma_tiler_mn[1] - 1) // plan.mma_tiler_mn[1])
                prepared: list[CandidateRun] = []

                for slices in args.split_k_list:
                    if shape.k % (tile_k * slices):
                        output.write(
                            f"{shape.name}\t{m}\t{shape.n}\t{shape.k}\t"
                            f"{plan.mma_tiler_mn[0]}\t{plan.mma_tiler_mn[1]}\t{tile_k}\t"
                            f"{work_tiles}\t{sm_count}\t{slices}\tnot_divisible\t\t\t\t\t\t\n"
                        )
                        continue
                    out = torch.empty(
                        (m, shape.n, 1), device="cuda", dtype=torch.bfloat16
                    )

                    def launch(
                        out: torch.Tensor = out,
                        slices: int = slices,
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
                            mma_tiler_mn=plan.mma_tiler_mn,
                            load_path=plan.load_path,
                            swap_ab=plan.swap_ab,
                            _split_k_slices_override=slices,
                        )

                    try:
                        replay = capture_graph_replay(launch)
                        replay()
                        torch.cuda.synchronize()
                        cos, max_abs = _check_candidate(
                            out[:, :, 0],
                            baseline,
                            shape=shape,
                            m=m,
                            slices=slices,
                        )
                    except Exception as exc:
                        message = str(exc).splitlines()[0].replace("\t", " ")[:160]
                        print(
                            f"  M={m:<2} splitK={slices} compile/check failed: {message}"
                        )
                        output.write(
                            f"{shape.name}\t{m}\t{shape.n}\t{shape.k}\t"
                            f"{plan.mma_tiler_mn[0]}\t{plan.mma_tiler_mn[1]}\t{tile_k}\t"
                            f"{work_tiles}\t{sm_count}\t{slices}\tfailed:{message}\t\t\t\t\t\t\n"
                        )
                        continue
                    prepared.append(CandidateRun(slices, replay, out, cos, max_abs, []))

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
                        f"{plan.mma_tiler_mn[0]}\t{plan.mma_tiler_mn[1]}\t{tile_k}\t"
                        f"{work_tiles}\t{sm_count}\t{candidate.slices}\tpass\t"
                        f"{median_us:.3f}\t{min(samples_us):.3f}\t{max(samples_us):.3f}\t"
                        f"{candidate.cos:.10f}\t{candidate.max_abs:.8f}\t{sample_text}\n"
                    )
                    print(
                        f"  M={m:<2} splitK={candidate.slices} {median_us:8.2f} us "
                        f"cos={candidate.cos:.8f}"
                    )
                if scored:
                    winner_us, winner = min(scored, key=lambda item: item[0])
                    direct = next(
                        (value for value, item in scored if item.slices == 1), None
                    )
                    speedup = (
                        f" ({direct / winner_us:.3f}x vs direct)" if direct else ""
                    )
                    print(
                        f"  -> M={m:<2} winner splitK={winner.slices} "
                        f"{winner_us:.2f} us{speedup}"
                    )

    print(f"\nraw_results={args.output}")


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
