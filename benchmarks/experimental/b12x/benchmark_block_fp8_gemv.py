#!/usr/bin/env python3
"""Small-row serialized block-FP8: GEMV regime vs the dense GEMM plan.

Times fixed ``expected_m`` plans (the decode shape vLLM declares per CUDA graph
size) through ``blockscaled.mm_block_fp8``, once with the GEMV regime and once
with it disabled, using the preparation engine's GPU sampler.  ``--cold`` flushes L2 before
every call, which is what a decode step sees for weights larger than L2.

    python benchmarks/experimental/b12x/benchmark_block_fp8_gemv.py [--cold] [--rows 1,2,4,8]
        [--shapes 3712x4096,1024x7168]
"""

from __future__ import annotations

import argparse
import contextlib

import torch

from b12x.gemm import blockscaled
from b12x.gemm.blockscaled import _block_fp8_gemv
from b12x.preparation import PreparationSession, PreparedCall
from b12x.testing.benchmark import measure_calls

DEFAULT_SHAPES = [
    (n, k)
    for k in (1536, 4096, 7168)
    for n in (256, 512, 1024, 2048, 3712, 4096, 6144, 8192, 16384)
]


@contextlib.contextmanager
def _gemv_enabled(enabled: bool):
    original = _block_fp8_gemv.supports
    if not enabled:
        _block_fp8_gemv.supports = lambda *args: False
    try:
        yield
    finally:
        _block_fp8_gemv.supports = original


def _prepare(lhs, lhs_scale, rhs, rhs_scale, rows):
    query = blockscaled.FixedBlockscaledQuery(
        recipe="block_fp8",
        call_kind="serialized",
        max_rows=rows,
        in_features=rhs.shape[1],
        padded_in_features=rhs.shape[1],
        out_features=rhs.shape[0],
        input_dtype="float8_e4m3fn",
        output_dtype="bfloat16",
        expected_m=rows,
    )
    plan = blockscaled.plan(query)

    def call(state):
        return PreparedCall(
            run=lambda: state.run_serialized(
                lhs,
                lhs_scale,
                rhs,
                rhs_scale,
                None,
                ab_dtype="float8_e4m3fn",
                sf_dtype="float32",
                c_dtype="bfloat16",
                sf_vec_size=128,
                block_fp8=True,
                stream=None,
            )
        )

    session = PreparationSession(device=lhs.device, autotune=False, compile_workers=4)
    session.__enter__()
    session.prepare((plan.request(name="bench", prepare_call=call),))
    return session, plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1,2,4,8")
    parser.add_argument("--shapes", default=None, help="NxK list, e.g. 3712x4096")
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--iters", type=int, default=25)
    args = parser.parse_args()
    shapes = (
        [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
        if args.shapes
        else DEFAULT_SHAPES
    )
    rows_list = [int(r) for r in args.rows.split(",")]
    device = torch.device("cuda")
    flush = (
        torch.empty(320 << 20, dtype=torch.uint8, device=device) if args.cold else None
    )
    print(
        f"{'N':>6} {'K':>6} {'M':>2} {'dense_us':>9} {'gemv_us':>8} {'speedup':>7}  eligible"
    )
    for n, k in shapes:
        gen = torch.Generator(device=device).manual_seed(n * 131 + k)
        rhs = torch.randn((n, k), generator=gen, device=device).to(torch.float8_e4m3fn)
        rhs_scale = torch.rand(
            ((n + 127) // 128, k // 128), generator=gen, device=device
        )
        for rows in rows_list:
            lhs = torch.randn((rows, k), generator=gen, device=device).to(
                torch.float8_e4m3fn
            )
            lhs_scale = torch.rand((rows, k // 128), generator=gen, device=device)
            lhs_ref = lhs.double() * lhs_scale.double().repeat_interleave(128, dim=1)
            rhs_ref = rhs.double() * rhs_scale.double().repeat_interleave(128, dim=0)[
                :n
            ].repeat_interleave(128, dim=1)
            reference = lhs_ref @ rhs_ref.T
            ulp = reference.abs().clamp_min(1e-30) * 2.0**-7
            calls = {}
            with contextlib.ExitStack() as stack:
                for label, enabled in (("dense", False), ("gemv", True)):
                    with _gemv_enabled(enabled):
                        original_max = _block_fp8_gemv.MAX_OUT_FEATURES
                        _block_fp8_gemv.MAX_OUT_FEATURES = 1 << 30
                        try:
                            session, plan = _prepare(
                                lhs, lhs_scale, rhs, rhs_scale, rows
                            )
                        finally:
                            _block_fp8_gemv.MAX_OUT_FEATURES = original_max
                        stack.callback(session.close)

                    def fn(plan=plan):
                        return blockscaled.mm_block_fp8(
                            lhs, lhs_scale, rhs, rhs_scale, plan=plan
                        )

                    output = fn()
                    error = (output.double() - reference).abs()
                    assert torch.isfinite(output).all()
                    if enabled:
                        assert bool(
                            (error <= ulp + 1e-6 * reference.abs().max()).all()
                        ), label
                    else:
                        # Dense split-K may round partial sums to BF16.
                        assert float(error.norm() / reference.norm()) < 4e-3, label
                    calls[label] = PreparedCall(
                        run=fn, produce=lambda: None, owners=(plan, session)
                    )
                measured = measure_calls(
                    calls,
                    samples=args.iters,
                    eviction=(lambda: None) if flush is None else flush.zero_,
                )
                times = measured.latencies_us
            eligible = _block_fp8_gemv.supports(rows, n, k)
            print(
                f"{n:6d} {k:6d} {rows:2d} {times['dense']:9.2f} {times['gemv']:8.2f} "
                f"{times['dense'] / times['gemv']:7.2f}  {'yes' if eligible else 'no'}",
                flush=True,
            )


if __name__ == "__main__":
    main()
