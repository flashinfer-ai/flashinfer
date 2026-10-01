# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Tune dense PrimsTS GEMM buckets, then time the winning tactic.

A runtime M uses the tactic of the smallest tuned bucket that is at least
that M. The default buckets run from 1024 to 65536 and round up. Below
1024 uses 1024; above 65536 uses 65536. ``--tuning-buckets`` replaces that
list. The same override stays
active while the winner is looked up; dropping it would look up a different
mapping and can miss the tuned tactic.

``--all-tactics`` also times every other tactic the search tried. A tactic
that fails to launch is reported as failed and does not stop the rest.

``--epilogue qkv_qknorm_rope`` runs the fused QK-norm and RoPE epilogue.
That kernel accepts only head_dim 128, so N must be divisible by 384.

Run from the repository root: python -m benchmarks.bench_prims_ts_gemm
"""

import argparse
import contextlib
import statistics

import torch

from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.gemm import (
    fp4_linear,
    fp4_linear_swiglu,
    fp4_qkv_qknorm_rope,
    fp8_linear,
    fp8_linear_swiglu,
    fp8_qkv_qknorm_rope,
)
from flashinfer.prims_ts.gemm.runner import (
    GemmIdentity,
    PrimsTsGemmRunner,
    dense_gemm_op_name,
    tuning_config_for,
)
from flashinfer.prims_ts.gemm.support import nvfp4_128x4_numel
from flashinfer.prims_ts.gemm.tactics import legal_tactics
from flashinfer.testing import bench_gpu_time
from flashinfer.utils import get_compute_capability

# The fused QKV kernel accepts only head_dim 128 and is_neox=False.
_QKV_HEAD_DIM = 128


def _format_tactic(tactic) -> str:
    if tactic == -1:
        return "fallback"
    (
        cluster_m,
        cluster_n,
        tile_n,
        tile_k,
        overlap,
        stages,
        mma_k,
        use_clc,
        warps,
        use_tma_store,
    ) = tactic
    return (
        f"cluster=({cluster_m},{cluster_n},1) tile_n={tile_n} tile_k={tile_k} "
        f"overlap={overlap} stages={stages} mma_k={mma_k} clc={use_clc} "
        f"epi_warps={warps} tma_store={int(use_tma_store)}"
    )


def _rope(m: int, q_norm, k_norm, num_heads: int):
    half = _QKV_HEAD_DIM // 2
    angles = torch.randn((m, half), device="cuda")
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()
    positions = torch.arange(m, device="cuda", dtype=torch.int64)
    return q_norm, k_norm, cos_sin, positions, num_heads


def _call(dtype: str, epilogue: str, a, weight, a_scale, weight_scale, out, rope=None):
    if epilogue == "qkv_qknorm_rope":
        q_norm, k_norm, cos_sin, positions, num_heads = rope
        common = dict(
            num_q_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=_QKV_HEAD_DIM,
            out=out,
        )
        if dtype == "fp8":
            return fp8_qkv_qknorm_rope(
                a,
                weight,
                a_scale,
                weight_scale,
                q_norm,
                k_norm,
                cos_sin,
                positions,
                **common,
            )
        return fp4_qkv_qknorm_rope(
            a,
            a_scale,
            1.0,
            weight,
            weight_scale,
            1.0,
            q_norm,
            k_norm,
            cos_sin,
            positions,
            **common,
        )
    if dtype == "fp8":
        fn = fp8_linear_swiglu if epilogue == "swiglu" else fp8_linear
        return fn(a, weight, a_scale, weight_scale, out=out)
    fn = fp4_linear_swiglu if epilogue == "swiglu" else fp4_linear
    return fn(a, a_scale, 1.0, weight, weight_scale, 1.0, out=out)


def _selected_tactic(
    op_name: str,
    dtype: str,
    epilogue: str,
    a,
    weight,
    a_scale,
    weight_scale,
    out,
    rope=None,
):
    """Return the tactic ``choose_one`` selects for these tensors.

    The caller must already be inside the ``autotune`` context used for
    profiling, so lookup uses the same buckets and rounding.
    """
    recorded = []
    tuner = AutoTuner.get()
    choose_one = tuner.choose_one

    def _choose_one(custom_op, runners, tuning_config, inputs, **kwargs):
        runner, tactic = choose_one(custom_op, runners, tuning_config, inputs, **kwargs)
        if custom_op == op_name:
            recorded.append(tactic)
        return runner, tactic

    tuner.choose_one = _choose_one
    try:
        _call(dtype, epilogue, a, weight, a_scale, weight_scale, out, rope)
    finally:
        tuner.choose_one = choose_one
    return recorded[-1] if recorded else -1


def _bucket_for(epilogue: str, m: int) -> int:
    config = tuning_config_for(epilogue)
    mapper = AutoTuner.get().get_effective_map_to_tuning_buckets(config)
    return mapper(m)


def _clear_cuda_error() -> None:
    # A failed launch leaves a sticky CUDA error. synchronize surfaces it;
    # cudaGetLastError clears it so the next tactic can run.
    with contextlib.suppress(Exception):
        torch.cuda.synchronize()
    with contextlib.suppress(Exception):
        torch.cuda.cudart().cudaGetLastError()


def _runner_inputs(dtype: str, a, weight, out, a_scale, weight_scale, rope=None):
    # Order matches PrimsTsGemmRunner.forward: a, weight, output, bias,
    # fp8 scales, q/k norm, cos_sin, positions, nvfp4 sfa/sfb, then the
    # unused fused outputs.
    q_norm = k_norm = cos_sin = positions = None
    if rope is not None:
        q_norm, k_norm, cos_sin, positions, _num_heads = rope
    if dtype == "fp8":
        return [
            a,
            weight,
            out,
            None,
            a_scale,
            weight_scale,
            q_norm,
            k_norm,
            cos_sin,
            positions,
            None,
            None,
            None,
            None,
            None,
            None,
        ]
    return [
        a,
        weight,
        out,
        None,
        None,
        None,
        q_norm,
        k_norm,
        cos_sin,
        positions,
        a_scale,
        weight_scale,
        None,
        None,
        None,
        None,
    ]


def _time_tactic(runner, inputs, tactic, dry_run_iters: int, repeat_iters: int):
    def run():
        runner.forward(inputs, tactic=tactic, scale=1.0)

    try:
        samples = bench_gpu_time(
            run,
            enable_cupti=True,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
        )
    except Exception:
        _clear_cuda_error()
        return None
    return statistics.median(samples)


def _print_tactic(median_ms, m: int, n: int, k: int, tactic, tuned: bool) -> None:
    mark = "tuned" if tuned else "     "
    if median_ms is None:
        print(f"  {'failed':>14}  {mark}  {_format_tactic(tactic)}", flush=True)
        return
    tflops = 2 * m * n * k / median_ms / 1e9
    print(
        f"  {median_ms * 1000:10.3f} us  {tflops:8.2f} TFLOP/s  {mark}  {_format_tactic(tactic)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("fp8", "fp4"), default="fp4")
    parser.add_argument(
        "--epilogue",
        choices=("linear", "swiglu", "qkv_qknorm_rope"),
        default="linear",
    )
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=[8192],
        help="One or more M values. Each uses the tactic of the next bucket at or above it.",
    )
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=7680)
    parser.add_argument(
        "--tuning-buckets",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit M buckets to tune. Runtime M rounds up to the next bucket. "
            "Omit to use the default 1024..65536 buckets, which also round up."
        ),
    )
    parser.add_argument("--dry-run-iters", type=int, default=10)
    parser.add_argument("--repeat-iters", type=int, default=50)
    parser.add_argument(
        "--all-tactics",
        action="store_true",
        help="Time every searched tactic and mark the winner. Default prints only the winner.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA GPU")
    if any(m <= 0 for m in args.m):
        raise ValueError("M must be positive")
    if args.tuning_buckets is not None and any(
        bucket <= 0 for bucket in args.tuning_buckets
    ):
        raise ValueError("tuning buckets must be positive")
    if args.epilogue == "swiglu" and args.n % 2:
        raise ValueError("SwiGLU requires even N")
    num_heads = None
    if args.epilogue == "qkv_qknorm_rope":
        group = 3 * _QKV_HEAD_DIM
        if args.n % group:
            raise ValueError(f"QKV N must be divisible by {group}, got {args.n}")
        num_heads = args.n // group
    if args.dtype == "fp4" and args.k % 256:
        raise ValueError("NVFP4 requires K divisible by 256")
    if args.dtype == "fp8" and args.k % 128:
        raise ValueError("FP8 requires K divisible by 128")

    major, minor = get_compute_capability(torch.device("cuda"))
    logical_n = args.n // 2 if args.epilogue == "swiglu" else args.n
    buckets = None if args.tuning_buckets is None else tuple(args.tuning_buckets)
    context = {} if buckets is None else {"tuning_buckets": buckets, "round_up": True}
    if args.dtype == "fp8":
        weight = (torch.randn((args.n, args.k), device="cuda") / 5).to(
            torch.float8_e4m3fn
        )
        weight_scale = torch.ones(args.n, device="cuda", dtype=torch.float32)
        operand = "fp8_e4m3"
    else:
        weight = torch.randint(
            0, 256, (args.n, args.k // 2), device="cuda", dtype=torch.uint8
        )
        weight_scale = torch.randint(
            0,
            128,
            (nvfp4_128x4_numel(args.n, args.k),),
            device="cuda",
            dtype=torch.uint8,
        )
        operand = "nvfp4_e2m1"

    q_norm = k_norm = None
    if num_heads is not None:
        q_norm = torch.rand((_QKV_HEAD_DIM,), device="cuda", dtype=torch.bfloat16)
        k_norm = torch.rand((_QKV_HEAD_DIM,), device="cuda", dtype=torch.bfloat16)

    def problem(m: int):
        out = torch.empty((m, logical_n), device="cuda", dtype=torch.bfloat16)
        if args.dtype == "fp8":
            a = (torch.randn((m, args.k), device="cuda") / 5).to(torch.float8_e4m3fn)
            a_scale = torch.ones(m, device="cuda", dtype=torch.float32)
        else:
            a = torch.randint(
                0, 256, (m, args.k // 2), device="cuda", dtype=torch.uint8
            )
            a_scale = torch.randint(
                0,
                128,
                (nvfp4_128x4_numel(m, args.k),),
                device="cuda",
                dtype=torch.uint8,
            )
        rope = None
        if num_heads is not None:
            rope = _rope(m, q_norm, k_norm, num_heads)
        return a, a_scale, out, rope

    op_name = dense_gemm_op_name(operand, args.epilogue)
    bucket_desc = (
        "1024..65536 (round up)"
        if buckets is None
        else ",".join(str(bucket) for bucket in sorted(set(buckets))) + " (round up)"
    )
    heads = "" if num_heads is None else f" heads={num_heads}"
    print(
        f"GPU={torch.cuda.get_device_name()} SM{major}{minor} "
        f"dtype={args.dtype} epilogue={args.epilogue} "
        f"N={args.n} K={args.k}{heads} buckets={bucket_desc}",
        flush=True,
    )

    tune_m = max(args.m)
    a, a_scale, out, rope = problem(tune_m)
    with autotune(True, **context):
        _call(args.dtype, args.epilogue, a, weight, a_scale, weight_scale, out, rope)

    runner = None
    tactics = None
    if args.all_tactics:
        arch = major * 10 + minor
        head_dim = _QKV_HEAD_DIM if num_heads is not None else None
        is_neox = False if num_heads is not None else None
        runner = PrimsTsGemmRunner(
            op_name,
            GemmIdentity(
                arch, operand, "bf16", args.epilogue, False, head_dim, is_neox, False
            ),
        )
        tactics = legal_tactics(arch, operand, "bf16", args.epilogue)

    with autotune(False, **context):
        for m in args.m:
            a, a_scale, out, rope = problem(m)
            selected = _selected_tactic(
                op_name,
                args.dtype,
                args.epilogue,
                a,
                weight,
                a_scale,
                weight_scale,
                out,
                rope,
            )
            bucket = _bucket_for(args.epilogue, m)
            if not args.all_tactics:

                def run(a=a, a_scale=a_scale, out=out, rope=rope):
                    _call(
                        args.dtype,
                        args.epilogue,
                        a,
                        weight,
                        a_scale,
                        weight_scale,
                        out,
                        rope,
                    )

                samples = bench_gpu_time(
                    run,
                    enable_cupti=True,
                    dry_run_iters=args.dry_run_iters,
                    repeat_iters=args.repeat_iters,
                )
                median_ms = statistics.median(samples)
                tflops = 2 * m * args.n * args.k / median_ms / 1e9
                print(
                    f"M={m} bucket={bucket} tuned={_format_tactic(selected)} "
                    f"median_us={median_ms * 1000:.3f} TFLOP/s={tflops:.2f}",
                    flush=True,
                )
                continue

            lookup = " lookup=fallback" if selected == -1 else ""
            print(f"M={m} bucket={bucket}{lookup}", flush=True)
            inputs = _runner_inputs(
                args.dtype, a, weight, out, a_scale, weight_scale, rope
            )
            timed = [
                (
                    _time_tactic(
                        runner, inputs, tactic, args.dry_run_iters, args.repeat_iters
                    ),
                    tactic,
                )
                for tactic in tactics
            ]
            timed.sort(
                key=lambda item: (
                    item[0] is None,
                    item[0] if item[0] is not None else 0,
                )
            )
            for median_ms, tactic in timed:
                _print_tactic(median_ms, m, args.n, args.k, tactic, tactic == selected)


if __name__ == "__main__":
    main()
