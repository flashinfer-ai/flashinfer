"""Benchmark the unified GEMM API (``GemmLayer``) per backend and as a cross-backend shootout.

One row per candidate backend (a single-candidate ``GemmLayer``) plus one row
for the layer built over every usable candidate (``backend="unified"``, with
the backend it selected in ``winner_backend``). Every row also carries the
median host (CPU) time per ``layer(act)`` call, since the dispatch overhead
of the layer, not its kernels, is what the unified API adds.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F

import flashinfer
from flashinfer.autotuner import autotune
from flashinfer.gemm import (
    BackendUnsupported,
    CublasGemmConfig,
    CudnnGemmConfig,
    CuteDslGemmConfig,
    CutlassGemmConfig,
    EpilogueConfig,
    ExecutionConfig,
    GemmActivationPack,
    GemmConfig,
    GemmLayer,
    GemmProblem,
    GemmWeightPack,
    TrtllmGemmConfig,
    prepare_activations,
    prepare_weight_pack,
    supported_backends,
)
from flashinfer.gemm.api import BackendOptions, QuantConfig, QuantFormat
from flashinfer.quantization.fp4_quantization import e2m1_and_ufp8sf_scale_to_float
from flashinfer.testing.utils import bench_gpu_time

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    filter_backends_by_compute_capability,
    get_device,
    print_perf_metrics,
)

# Backend key (the legacy ``backend=`` string) -> unified backend config.
# Default `auto` order (DEFAULT_GEMM_BACKENDS first, then the opt-in backends).
_BACKEND_CONFIGS = {
    "cutlass": CutlassGemmConfig,
    "cublas": CublasGemmConfig,
    "cudnn": CudnnGemmConfig,
    "cute-dsl": CuteDslGemmConfig,
    "trtllm": TrtllmGemmConfig,
}
_QUANT_PAIRS = {
    "nvfp4": (QuantFormat.NVFP4, QuantFormat.NVFP4),
    "fp8": (QuantFormat.FP8PerTensor, QuantFormat.FP8PerTensor),
}
_OUTPUT_FORMATS = {"bfloat16": QuantFormat.BF16, "float16": QuantFormat.FP16}
# Bytes per element of the quantized operands (NVFP4 packs two per byte).
_OPERAND_BYTES = {"nvfp4": 0.5, "fp8": 1.0}
# Host timing: back-to-back calls per block, blocks whose per-call median is reported.
_HOST_CALLS_PER_BLOCK = 100
_HOST_BLOCKS = 5
# Exceptions a backend raises when it cannot run this problem here (not bugs).
_UNSUPPORTED_ERRORS = (
    BackendUnsupported,
    NotImplementedError,
    RuntimeError,
    ValueError,
)


def parse_unified_gemm_args(line, parser: argparse.ArgumentParser):
    """Add arguments for the unified GEMM comparison routine."""
    parser.add_argument(
        "--m", type=int, required=True, help="Number of rows of the activation."
    )
    parser.add_argument(
        "--n", type=int, required=True, help="Number of output features (weight rows)."
    )
    parser.add_argument(
        "--k", type=int, required=True, help="Number of input features (reduction)."
    )
    parser.add_argument(
        "--quant",
        type=str,
        required=True,
        choices=tuple(_QUANT_PAIRS),
        help="Quantization pair: nvfp4 (NVFP4 x NVFP4) or fp8 (per-tensor FP8 x FP8).",
    )
    parser.add_argument(
        "--out_dtype",
        type=str,
        required=False,
        default="bfloat16",
        choices=tuple(_OUTPUT_FORMATS),
        help="Data type of the output.",
    )
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=None,
        choices=tuple(_BACKEND_CONFIGS),
        help=(
            "Candidate backends. Default: every backend supported_backends() "
            "reports usable for --quant on this GPU."
        ),
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        default=False,
        help="Add an [N] bias after the GEMM (EpilogueConfig(bias=True)).",
    )
    parser.add_argument(
        "--autotune",
        action="store_true",
        default=False,
        help=(
            "Warm every layer up under autotune(True): tactics are profiled and the "
            "all-candidates layer measures each backend before picking its winner."
        ),
    )
    args = parser.parse_args(line)
    if args.backends is None:
        args.backends = list(_BACKEND_CONFIGS)
    args.backends = list(dict.fromkeys(args.backends))
    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


def _quant_config(args) -> QuantConfig:
    weight_format, activation_format = _QUANT_PAIRS[args.quant]
    return QuantConfig(
        weight=weight_format,
        activation=activation_format,
        output=_OUTPUT_FORMATS[args.out_dtype],
    )


def _gemm_config(args, quant: QuantConfig, candidates: tuple) -> GemmConfig:
    return GemmConfig(
        quant=quant,
        epilogue=EpilogueConfig(bias=args.bias),
        backend=BackendOptions(candidates),
        execution=ExecutionConfig(enable_pdl=args.enable_pdl, tune_max_m=args.m),
    )


def _usable_backends(args, quant: QuantConfig, device: torch.device) -> list[str]:
    """Backends whose config-level and shape-level checks pass here, with reasons for the rest."""
    usable = []
    problem = GemmProblem(n=args.n, k=args.k, m=args.m)
    for key in args.backends:
        config = _gemm_config(args, quant, (_BACKEND_CONFIGS[key](),))
        (report,) = supported_backends(config, problem, device)
        if report.ok:
            usable.append(key)
        else:
            print(f"[INFO] {key} does not support this configuration: {report.reason}")
    return usable


def _dequant_reference(
    act: GemmActivationPack, pack: GemmWeightPack, quant_name: str
) -> torch.Tensor:
    """fp32 ``A @ W^T`` (plus bias) from the quantized operands the kernels consume."""
    if quant_name == "fp8":
        view = pack.view_for_recipe("fp8_per_tensor")
        a = act.a.float() * act.a_scale.float()
        w = view["b"].float() * view["b_scale"].float()
    else:
        # Prefer the row-major view; only trtllm's shuffled copy exists when
        # trtllm is the sole candidate.
        view = pack.view_for_recipe("nvfp4_128x4") or pack.view_for_recipe(
            "nvfp4_128x4_shuffled"
        )
        a = e2m1_and_ufp8sf_scale_to_float(
            act.a, act.a_scale.view(torch.uint8), act.a_global_scale, 16, 1, True
        ).float()
        w = e2m1_and_ufp8sf_scale_to_float(
            view["b"],
            view["b_descale"].view(torch.uint8),
            view["b_global_scale"],
            16,
            1,
            True,
        ).float()
    ref = a @ w.T
    if pack.bias is not None:
        ref = ref + pack.bias.float()
    return ref


def _build_layer(config: GemmConfig, w: torch.Tensor, bias, device):
    """Prepare the weight pack once for ``config`` and bind it to a new layer."""
    pack = prepare_weight_pack(w, config, bias=bias)
    layer = GemmLayer(config, device=device)
    layer.prepare(pack)
    return layer, pack


def _warmup(args, layer: GemmLayer, act: GemmActivationPack, out: torch.Tensor):
    """Select (and under --autotune, profile) once; the timed calls then hit the cache."""
    cache_path = args.autotune_cache
    if args.autotune:
        warmup_iters = args.dry_run_iters if args.dry_run_iters > 0 else 10
        with autotune(True, cache=cache_path):
            for _ in range(warmup_iters):
                layer(act, out=out)
    elif cache_path:
        with autotune(False, cache=cache_path):
            layer(act, out=out)
    else:
        layer(act, out=out)
    torch.cuda.synchronize()


def _gpu_time_ms(args, layer: GemmLayer, act, out) -> tuple[float, float]:
    times = bench_gpu_time(
        fn=layer,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        sleep_after_run=True,
        enable_cupti=args.use_cupti,
        use_cuda_graph=not args.no_cuda_graph,
        cold_l2_cache=True,
        input_args=(act,),
        input_kwargs={"out": out},
    )
    return float(np.median(times)), float(np.std(times))


def _host_us_per_call(layer: GemmLayer, act: GemmActivationPack) -> float:
    """Median host time per ``layer(act)`` over blocks of back-to-back calls.

    The clock runs over the enqueues only: no synchronization inside a block,
    one after it, so GPU completion time never counts as host cost. The plain
    call (output allocated by the layer) is timed because that is what a
    framework pays per call.
    """
    for _ in range(10):
        layer(act)
    torch.cuda.synchronize()
    samples = []
    for _ in range(_HOST_BLOCKS):
        t0 = time.perf_counter()
        for _ in range(_HOST_CALLS_PER_BLOCK):
            layer(act)
        samples.append((time.perf_counter() - t0) / _HOST_CALLS_PER_BLOCK * 1e6)
        torch.cuda.synchronize()
    return float(np.median(samples))


def _refcheck(args, label: str, out: torch.Tensor, reference) -> bool | str:
    if reference is None:
        return ""
    cos_sim = F.cosine_similarity(reference.reshape(-1), out.float().reshape(-1), dim=0)
    passed = bool(cos_sim > 0.99) or bool(
        torch.allclose(out.float(), reference, rtol=5e-2, atol=5e-1)
    )
    if not passed:
        print(
            f"[ERROR] {label} output mismatch with the dequantized reference "
            f"(cos_sim={float(cos_sim):.5f})"
        )
        if not args.allow_output_mismatch:
            raise AssertionError(
                f"[ERROR] Backend {label} output mismatch with cos_sim={float(cos_sim)}"
            )
    return passed


def _measure(args, label: str, layer: GemmLayer, act, out, reference) -> dict:
    """Warm up, time the GPU and the host, refcheck, print, and build one result row."""
    _warmup(args, layer, act, out)
    layer(act, out=out)
    torch.cuda.synchronize()
    refcheck_passed = _refcheck(args, label, out.detach().clone(), reference)
    median_time, std_time = _gpu_time_ms(args, layer, act, out)
    host_us = _host_us_per_call(layer, act)

    m, n, k = args.m, args.n, args.k
    res_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    operand_bytes = _OPERAND_BYTES[args.quant]
    problem_flops = 2 * m * n * k
    problem_bytes = (
        m * k * operand_bytes + n * k * operand_bytes + m * n * res_dtype.itemsize
    )
    tflops = problem_flops / (10**9 * median_time)  # in TFLOPs/sec
    tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec
    print_perf_metrics(label, median_time, std_time, tflops, tb_per_sec)
    print(
        f"[PERF] {label.ljust(max(15, len(label)))}:: host {host_us:.1f} us/call "
        f"(median of {_HOST_BLOCKS} x {_HOST_CALLS_PER_BLOCK} calls); "
        f"winner {layer.winner_backend}"
    )

    cur_res = defaultdict(str)
    cur_res.update(
        routine=args.routine,
        median_time=median_time,
        std_time=std_time,
        tflops=tflops,
        tb_per_sec=tb_per_sec,
        backend=label,
        resolved_backend=layer.winner_backend,
        winner_backend=layer.winner_backend,
        host_us_per_call=host_us,
        m=m,
        n=n,
        k=k,
        out_dtype=res_dtype,
        quant_dtype=args.quant,
        bias=args.bias,
        autotune=args.autotune,
        refcheck_passed=refcheck_passed,
        enable_pdl=args.enable_pdl,
        case_tag=args.case_tag,
    )
    return cur_res


def run_unified_gemm_test(args):
    """Benchmark each usable unified GEMM backend, then the all-candidates layer."""
    if args.verbose >= 1:
        print("[INFO] Running unified_gemm")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    if args.generate_repro_command:
        print(f"[INFO] To reproduce this test case, run: {args.repro_command}")

    device = get_device(args)
    res = []
    if not filter_backends_by_compute_capability(["unified"], args.routine, device):
        print("[ERROR] No backends to test. Exiting.")
        return res

    quant = _quant_config(args)
    usable = _usable_backends(args, quant, device)
    if not usable:
        print("[ERROR] No backends passed validation. Exiting.")
        return res

    x = torch.randn(args.m, args.k, device=device, dtype=torch.bfloat16)
    w = torch.randn(args.n, args.k, device=device, dtype=torch.bfloat16)
    bias = (
        torch.randn(args.n, device=device, dtype=torch.bfloat16) if args.bias else None
    )
    act = prepare_activations(x, quant=quant)
    out = torch.empty(
        args.m, args.n, device=device, dtype=dtype_str_to_torch_dtype(args.out_dtype)
    )
    if args.verbose >= 2:
        print(f"[VVERBOSE] {act = }")
        print(f"[VVERBOSE] {usable = }")

    # One layer per candidate; a backend that passes the config checks can
    # still decline the real tensors (legacy requirement checks) or fail to
    # build, so each is probed like mm_fp4 probes its backends.
    layers = {}
    for key in usable:
        config = _gemm_config(args, quant, (_BACKEND_CONFIGS[key](),))
        try:
            layer, _ = _build_layer(config, w, bias, device)
            layer(act, out=out)
            torch.cuda.synchronize()
        except _UNSUPPORTED_ERRORS as error:
            print(
                f"[INFO] {key} does not support this configuration: "
                f"{type(error).__name__}: {error}"
            )
            continue
        layers[key] = layer
    if not layers:
        print("[ERROR] No backends passed runtime validation. Exiting.")
        return res

    # The all-candidates layer: what a framework builds from one config.
    unified_config = _gemm_config(
        args, quant, tuple(_BACKEND_CONFIGS[key]() for key in layers)
    )
    unified_layer, unified_pack = _build_layer(unified_config, w, bias, device)
    reference = (
        _dequant_reference(act, unified_pack, args.quant) if args.refcheck else None
    )

    for key, layer in layers.items():
        res.append(_measure(args, key, layer, act, out, reference))
    res.append(_measure(args, "unified", unified_layer, act, out, reference))
    return res
