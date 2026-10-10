# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Unified-runner coverage for dense Prims-TS FP8 and NVFP4 GEMMs."""

from collections import defaultdict
import statistics

import torch
import torch.nn.functional as F

from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.gemm import (
    fp4_linear,
    fp4_linear_swiglu,
    fp4_qkv_qknorm_rope,
    fp8_linear,
    fp8_linear_swiglu,
    fp8_qkv_qknorm_rope,
)
from flashinfer.prims_ts.gemm import prepare_fp4_linear, prepare_fp4_linear_swiglu
from flashinfer.prims_ts.gemm.runner import dense_gemm_op_name
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import get_compute_capability

from .flashinfer_benchmark_utils import get_device, print_perf_metrics

_HEAD_DIM = 128
_FP4_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


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


def _linear_scale_to_128x4(scales: torch.Tensor) -> torch.Tensor:
    rows, cols = scales.shape
    rows_pad = (rows + 127) // 128 * 128
    cols_pad = (cols + 3) // 4 * 4
    result = torch.zeros(rows_pad * cols_pad, device=scales.device, dtype=torch.uint8)
    source = scales.view(torch.uint8)
    row = torch.arange(rows, device=scales.device, dtype=torch.int64)[:, None]
    col = torch.arange(cols, device=scales.device, dtype=torch.int64)[None, :]
    index = (
        col % 4
        + (col // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + (row // 128) * (128 * cols_pad)
    )
    result[index.reshape(-1)] = source.reshape(-1)
    return result


def _unpack_nvfp4(packed: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(_FP4_LUT, device=packed.device, dtype=torch.float32)

    def decode(nibbles):
        values = lut[(nibbles & 7).long()]
        return torch.where((nibbles & 8) != 0, -values, values)

    return torch.stack((decode(packed & 15), decode((packed >> 4) & 15)), -1).reshape(
        packed.shape[0], packed.shape[1] * 2
    )


def _dequantize_nvfp4(packed, logical_scales):
    return _unpack_nvfp4(packed) * logical_scales.float().repeat_interleave(16, dim=1)


def _dequantize_nvfp4_output(packed, scales_128x4, encode_scale):
    m, packed_n = packed.shape
    scale_cols = packed_n * 2 // 16
    padded_cols = (scale_cols + 3) // 4 * 4
    rows = torch.arange(m, device=packed.device, dtype=torch.int64)[:, None]
    cols = torch.arange(scale_cols, device=packed.device, dtype=torch.int64)[None, :]
    index = (
        cols % 4
        + (cols // 4) * 512
        + (rows % 32) * 16
        + ((rows % 128) // 32) * 4
        + (rows // 128) * (128 * padded_cols)
    )
    scale = scales_128x4[index].view(torch.float8_e4m3fn).float()
    return (
        _unpack_nvfp4(packed)
        * scale.repeat_interleave(16, dim=1)
        / encode_scale.float()
    )


def _rope_inputs(m, num_heads, device):
    q_norm = torch.rand(_HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_norm = torch.rand(_HEAD_DIM, device=device, dtype=torch.bfloat16)
    angles = torch.randn((m, _HEAD_DIM // 2), device=device)
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()
    positions = torch.arange(m, device=device, dtype=torch.int64)
    return q_norm, k_norm, cos_sin, positions, num_heads


def _qkv_reference(base, rope):
    q_norm, k_norm, cos_sin, _positions, num_heads = rope
    m = base.shape[0]
    shaped = base.bfloat16().float().view(m, 3, num_heads, _HEAD_DIM)
    cos = cos_sin[:, None, : _HEAD_DIM // 2]
    sin = cos_sin[:, None, _HEAD_DIM // 2 :]
    for part, norm_weight in ((0, q_norm), (1, k_norm)):
        value = shaped[:, part]
        value = (
            value
            * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
            * norm_weight.float()
        )
        pairs = value.view(m, num_heads, _HEAD_DIM // 2, 2)
        shaped[:, part] = torch.stack(
            (
                pairs[..., 0] * cos - pairs[..., 1] * sin,
                pairs[..., 1] * cos + pairs[..., 0] * sin,
            ),
            -1,
        ).flatten(-2)
    return shaped.reshape_as(base)


def _reference(dtype, epilogue, a, weight, a_scale, weight_scale, rope):
    if dtype == "fp8":
        base = (
            F.linear(a.float(), weight.float())
            * a_scale[:, None]
            * weight_scale[None, :]
        )
    else:
        base = F.linear(
            _dequantize_nvfp4(a, a_scale),
            _dequantize_nvfp4(weight, weight_scale),
        )
    if epilogue == "swiglu":
        return base[:, 0::2] * F.silu(base[:, 1::2])
    if epilogue == "qkv_qknorm_rope":
        return _qkv_reference(base, rope)
    return base


def _one_shot_call(dtype, epilogue, tensors, out):
    a, weight, a_scale, weight_scale, rope = tensors
    if epilogue == "qkv_qknorm_rope":
        q_norm, k_norm, cos_sin, positions, num_heads = rope
        common = dict(
            num_q_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=_HEAD_DIM,
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
        function = fp8_linear_swiglu if epilogue == "swiglu" else fp8_linear
        return function(a, weight, a_scale, weight_scale, out=out)
    function = fp4_linear_swiglu if epilogue == "swiglu" else fp4_linear
    return function(a, a_scale, 1.0, weight, weight_scale, 1.0, out=out)


def _selected_tactic(op_name, run):
    recorded = []
    tuner = AutoTuner.get()
    choose_one = tuner.choose_one

    def capture(custom_op, runners, tuning_config, inputs, **kwargs):
        runner, tactic = choose_one(custom_op, runners, tuning_config, inputs, **kwargs)
        if custom_op == op_name:
            recorded.append(tactic)
        return runner, tactic

    tuner.choose_one = capture
    try:
        run()
    finally:
        tuner.choose_one = choose_one
    return recorded[-1] if recorded else -1


def _make_problem(args, device):
    logical_n = args.n // 2 if args.epilogue == "swiglu" else args.n
    rope = None
    if args.epilogue == "qkv_qknorm_rope":
        rope = _rope_inputs(args.m, args.n // (3 * _HEAD_DIM), device)
    if args.dtype == "fp8":
        a = (torch.randn((args.m, args.k), device=device) / 5).to(torch.float8_e4m3fn)
        weight = (torch.randn((args.n, args.k), device=device) / 5).to(
            torch.float8_e4m3fn
        )
        a_scale = torch.ones(args.m, device=device, dtype=torch.float32)
        weight_scale = torch.ones(args.n, device=device, dtype=torch.float32)
    else:
        a = torch.randint(
            0, 256, (args.m, args.k // 2), device=device, dtype=torch.uint8
        )
        weight = torch.randint(
            0, 256, (args.n, args.k // 2), device=device, dtype=torch.uint8
        )
        a_scale = (torch.rand((args.m, args.k // 16), device=device) / 4).to(
            torch.float8_e4m3fn
        )
        weight_scale = (torch.rand((args.n, args.k // 16), device=device) / 4).to(
            torch.float8_e4m3fn
        )
    return (a, weight, a_scale, weight_scale, rope), logical_n


def _validate_case(args, arch):
    if arch not in (100, 103, 107):
        raise RuntimeError(
            f"Prims-TS GEMM requires SM100, SM103, or SM107, got SM{arch}"
        )
    if args.m <= 0 or args.n <= 0 or args.k <= 0:
        raise ValueError("M, N, and K must be positive")
    if args.dtype == "fp8" and args.k % 128:
        raise ValueError("FP8 requires K divisible by 128")
    if args.dtype == "fp4" and args.k % 256:
        raise ValueError("NVFP4 requires K divisible by 256")
    if args.epilogue == "swiglu" and args.n % 2:
        raise ValueError("SwiGLU requires even N")
    if args.epilogue == "qkv_qknorm_rope" and args.n % (3 * _HEAD_DIM):
        raise ValueError("QKV N must be divisible by 384")
    if args.mode == "prepared" and args.dtype != "fp4":
        raise ValueError("prepared mode is available only for FP4")
    if args.mode == "one_shot" and args.tuning_bucket is None:
        raise ValueError("one_shot mode requires --tuning_bucket")
    if args.mode == "prepared" and args.epilogue == "qkv_qknorm_rope":
        raise ValueError("this benchmark adapter supports prepared linear and SwiGLU")
    if args.output_format == "nvfp4" and not (
        args.dtype == "fp4" and args.mode == "prepared" and args.epilogue == "swiglu"
    ):
        raise ValueError("NVFP4 output requires prepared FP4 SwiGLU")
    if (
        args.output_format == "bf16"
        and args.mode == "prepared"
        and args.mma_k == 96
        and arch != 103
    ):
        raise ValueError("mma_k=96 requires SM103")
    if args.output_format == "nvfp4" and args.mma_k == 96 and arch != 103:
        raise ValueError("mma_k=96 requires SM103")


def _assert_output(args, actual, expected, encode_scale):
    if args.output_format == "nvfp4":
        payload, scales = actual
        decoded = _dequantize_nvfp4_output(payload, scales, encode_scale)
        denominator = (
            expected.reshape(args.m, -1, 16).abs().amax(-1, keepdim=True).clamp(min=1)
        )
        error = (
            (decoded - expected).reshape(args.m, -1, 16).abs() / denominator
        ).amax()
        if error.item() >= 0.2:
            raise AssertionError(
                f"packed NVFP4 block-relative error is {error.item():.4f}"
            )
        return
    if args.dtype == "fp8":
        atol, rtol = 0.3, 3e-2
    elif args.epilogue == "linear":
        atol, rtol = 0.5, 4e-2
    else:
        atol, rtol = 1.0, 8e-2
    torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=rtol)


def _check_output(args, actual, expected, encode_scale):
    try:
        _assert_output(args, actual, expected, encode_scale)
    except AssertionError as exc:
        if not args.allow_output_mismatch:
            raise
        print(f"[ERROR] prims-ts output mismatch: {exc}")
        return False
    return True


def run_prims_ts_gemm_test(args):
    device = get_device(args)
    major, minor = get_compute_capability(device)
    arch = major * 10 + minor
    _validate_case(args, arch)
    if args.generate_repro_command:
        print(f"[INFO] To reproduce this test case, run: {args.repro_command}")

    tensors, logical_n = _make_problem(args, device)
    a, weight, a_scale, weight_scale, rope = tensors
    call_tensors = tensors
    if args.dtype == "fp4":
        call_tensors = (
            a,
            weight,
            _linear_scale_to_128x4(a_scale),
            _linear_scale_to_128x4(weight_scale),
            rope,
        )
    expected = None
    if args.refcheck:
        expected = _reference(
            args.dtype, args.epilogue, a, weight, a_scale, weight_scale, rope
        )

    encode_scale = None
    if args.output_format == "nvfp4":
        encode_scale = (
            (448.0 * 6.0 / expected.abs().amax()).reshape(1)
            if expected is not None
            else torch.ones(1, device=device, dtype=torch.float32)
        )
        out = torch.empty((args.m, logical_n // 2), device=device, dtype=torch.uint8)
    else:
        out = torch.empty((args.m, logical_n), device=device, dtype=torch.bfloat16)

    refcheck_passed = False
    if args.mode == "one_shot":
        context = {"tuning_buckets": (args.tuning_bucket,), "round_up": True}

        def run():
            return _one_shot_call(args.dtype, args.epilogue, call_tensors, out)

        with autotune(True, **context):
            run()
        operand = "fp8_e4m3" if args.dtype == "fp8" else "nvfp4_e2m1"
        op_name = dense_gemm_op_name(operand, args.epilogue)
        with autotune(False, **context):
            tactic = _format_tactic(_selected_tactic(op_name, run))
            actual = run()
            if expected is not None:
                refcheck_passed = _check_output(args, actual, expected, encode_scale)
            samples = bench_gpu_time(
                run,
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                sleep_after_run=True,
                enable_cupti=args.use_cupti,
                use_cuda_graph=False,
            )
    else:
        weight_sf = _linear_scale_to_128x4(weight_scale)
        prepare = (
            prepare_fp4_linear_swiglu
            if args.epilogue == "swiglu"
            else prepare_fp4_linear
        )
        prepared = prepare(
            weight,
            weight_sf,
            1.0,
            1.0,
            out_dtype=torch.uint8 if args.output_format == "nvfp4" else torch.bfloat16,
            output_quant_scale=encode_scale,
            mma_k=args.mma_k,
        )
        activation_sf = _linear_scale_to_128x4(a_scale)

        def run():
            return prepared(a, activation_sf, out=out)

        actual = run()
        tactic = repr(prepared.config)
        if expected is not None:
            refcheck_passed = _check_output(args, actual, expected, encode_scale)
        samples = bench_gpu_time(
            run,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=True,
            enable_cupti=args.use_cupti,
            use_cuda_graph=False,
        )

    median_time = statistics.median(samples)
    std_time = statistics.pstdev(samples)
    tflops = 2 * args.m * args.n * args.k / (1e9 * median_time)
    print_perf_metrics("prims-ts", median_time, std_time, tflops, 0.0)
    print(f"[INFO] GPU={torch.cuda.get_device_name()} SM{arch} tactic={tactic}")

    result = defaultdict(str)
    result.update(
        routine=args.routine,
        median_time=median_time,
        std_time=std_time,
        tflops=tflops,
        backend="prims-ts",
        resolved_backend="prims-ts",
        m=args.m,
        n=args.n,
        k=args.k,
        dtype=args.dtype,
        epilogue=args.epilogue,
        mode=args.mode,
        output_format=args.output_format,
        tuning_bucket=args.tuning_bucket,
        mma_k=args.mma_k,
        gpu_sm=arch,
        tactic=tactic,
        refcheck_passed=refcheck_passed,
        case_tag=args.case_tag,
    )
    return [result]
