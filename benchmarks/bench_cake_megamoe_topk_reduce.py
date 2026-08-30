"""CUPTI benchmark for the frozen MegaMoE workspace TopK reducer.

The default baseline is an eager PyTorch implementation that performs the six
BF16-to-FP32 additions in strict K=0..5 order and converts the result to BF16.
It is a semantic reference made of several GPU kernels, not the MegaMoE CuTeDSL
kernel and not a claimed historical FlashInfer performance baseline.

``--baseline vendored-cutedsl`` selects the exact vendored implementation at
``flashinfer/moe_ep/kernel_src/cutedsl_megamoe/src/moe_nvfp4_swapab/topk_reduce.py``.
It preserves the issue denominator's fixed C=4096 workspace for every live-token
shape; preparation, zero-fill, and compilation remain outside timing. That module
is not a stable public API. Source drift, import failure, or compilation failure
is fatal; the benchmark never substitutes the PyTorch path while labelling it
CuTeDSL.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from flashinfer.jit.cake_megamoe_topk_reduce import (
    run_cake_megamoe_topk_reduce,
)
from flashinfer.testing import bench_gpu_time
from flashinfer.utils import is_sm100a_supported


_SHAPES = (
    (1, 256),
    (8, 256),
    (64, 256),
    (128, 256),
    (256, 256),
    (4096, 4096),
)
_HIDDEN_SIZE = 4096
_TOP_K = 6
_BASELINE_CAPACITY = 4096
_ATOL = 1e-2
_RTOL = 1e-2
_CUTEDSL_SOURCE = (
    "flashinfer/moe_ep/kernel_src/cutedsl_megamoe/src/moe_nvfp4_swapab/topk_reduce.py"
)
_CUTEDSL_SOURCE_SHA256 = (
    "d7d1fc2361c30dcd8a37269edda27e75953dadf396014e47a3c3cbd7a2551184"
)


def _require_exact_sm100a() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability != (10, 0):
        raise RuntimeError(
            f"frozen reducer requires exact SM100a, found capability {capability}"
        )
    if not is_sm100a_supported(torch.device("cuda")):
        raise RuntimeError("frozen reducer requires SM100a with CUDA 12.8+")


def _require_cupti() -> None:
    """Prevent bench_gpu_time from silently falling back to CUDA events."""

    try:
        from cupti import cupti as _cupti  # noqa: F401

        version = importlib.metadata.version("cupti-python")
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise RuntimeError(
            "CUPTI timing is mandatory; install cupti-python>=13.0.0"
        ) from error
    try:
        major = int(version.split(".", maxsplit=1)[0])
    except ValueError as error:
        raise RuntimeError(f"invalid cupti-python version {version!r}") from error
    if major < 13:
        raise RuntimeError(
            f"CUPTI timing requires cupti-python>=13.0.0, found {version}"
        )


def _ordered_reference(partials: torch.Tensor, num_tokens: int) -> torch.Tensor:
    acc = partials[:num_tokens, 0].float()
    for topk_idx in range(1, _TOP_K):
        acc = acc + partials[:num_tokens, topk_idx].float()
    return acc.to(torch.bfloat16)


def _make_ordered_pytorch_runner(
    num_tokens: int,
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    # Keep all scratch outside the timed region. Each invocation is six
    # stream-ordered BF16->FP32 accumulations followed by one BF16 store.
    acc = torch.empty((num_tokens, _HIDDEN_SIZE), dtype=torch.float32, device="cuda")

    def run(partials: torch.Tensor, out: torch.Tensor) -> None:
        acc.copy_(partials[:num_tokens, 0])
        for topk_idx in range(1, _TOP_K):
            acc.add_(partials[:num_tokens, topk_idx])
        out[:num_tokens].copy_(acc)

    return run


def _make_vendored_cutedsl_runner(
    partials: torch.Tensor,
) -> tuple[Callable[[torch.Tensor, torch.Tensor], None], torch.Tensor]:
    """Prepare the pinned fixed-C=4096 CuTeDSL denominator."""

    try:
        import cuda.bindings.driver as cuda_driver
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch

        from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim._paths import (
            bootstrap_paths,
        )

        bootstrap_paths()
        from moe_nvfp4_swapab.topk_reduce import TopkReduce
        from src.token_comm import CombineFormat
    except Exception as error:
        raise RuntimeError(
            f"vendored CuTeDSL baseline {_CUTEDSL_SOURCE} is unavailable; "
            "no fallback was used"
        ) from error

    source_path = inspect.getsourcefile(TopkReduce)
    if source_path is None:
        raise RuntimeError(f"cannot locate vendored CuTeDSL source {_CUTEDSL_SOURCE}")
    actual_source_sha256 = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    if actual_source_sha256 != _CUTEDSL_SOURCE_SHA256:
        raise RuntimeError(
            f"vendored CuTeDSL baseline {_CUTEDSL_SOURCE} changed: expected "
            f"{_CUTEDSL_SOURCE_SHA256}, got {actual_source_sha256}"
        )

    # The production denominator owns a fixed full-capacity workspace. Keeping
    # this physical shape is essential: slicing to num_tokens asks CuTeDSL to
    # compile a smaller shape-specialized kernel and is not the issue baseline.
    baseline_partials = torch.zeros(
        (_BASELINE_CAPACITY, _TOP_K, _HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=partials.device,
    )
    baseline_partials[: partials.shape[0]].copy_(partials)
    baseline_out = torch.empty(
        (_BASELINE_CAPACITY, _HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=partials.device,
    )
    partials_cute = cutlass_torch.from_dlpack(baseline_partials, assumed_align=16)
    out_cute = cutlass_torch.from_dlpack(baseline_out, assumed_align=16)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    kernel = TopkReduce(
        _HIDDEN_SIZE,
        _TOP_K,
        CombineFormat(
            act_dtype=cutlass.BFloat16,
            scale_dtype=None,
            scale_block=None,
        ),
        sm_arch="sm_100a",
    )
    compile_kwargs = {
        "combine_quant": partials_cute,
        "combine_sf": None,
        "reduced_output": out_cute,
        "topk_score": None,
        "stream": stream,
    }
    try:
        compiled = cute.compile(kernel, **compile_kwargs)
    except Exception as error:
        raise RuntimeError(
            f"vendored CuTeDSL baseline {_CUTEDSL_SOURCE} failed to compile; "
            "no fallback was used"
        ) from error

    def run(_partials: torch.Tensor, _out: torch.Tensor) -> None:
        compiled(
            combine_quant=partials_cute,
            combine_sf=None,
            reduced_output=out_cute,
            topk_score=None,
            stream=cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream),
        )

    return run, baseline_out


def _median_cupti_ms(
    runner: Callable[[torch.Tensor, torch.Tensor], None],
    partials: torch.Tensor,
    out: torch.Tensor,
    dry_run_iters: int,
    repeat_iters: int,
) -> float:
    measurements = bench_gpu_time(
        runner,
        input_args=(partials, out),
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
        enable_cupti=True,
        cold_l2_cache=True,
    )
    return float(statistics.median(measurements))


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _run_shape(
    num_tokens: int,
    capacity: int,
    baseline: str,
    dry_run_iters: int,
    repeat_iters: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    partials = torch.randn(
        capacity,
        _TOP_K,
        _HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).contiguous()
    out = torch.empty((capacity, _HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")
    expected = _ordered_reference(partials, num_tokens)

    def native_runner(native_partials: torch.Tensor, native_out: torch.Tensor) -> None:
        run_cake_megamoe_topk_reduce(native_partials, native_out, num_tokens)

    native_runner(partials, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out[:num_tokens], expected, atol=_ATOL, rtol=_RTOL)
    native_ms = _median_cupti_ms(
        native_runner, partials, out, dry_run_iters, repeat_iters
    )

    if baseline == "ordered-pytorch":
        baseline_label = "ordered_pytorch_fp32_k0_to_k5"
        baseline_runner = _make_ordered_pytorch_runner(num_tokens)
        baseline_validation_out = out
        baseline_source = "PyTorch eager: six ordered FP32 accumulations + BF16 store"
    else:
        baseline_label = "vendored_cutedsl_topk_reduce_fixed_c4096"
        baseline_runner, baseline_validation_out = _make_vendored_cutedsl_runner(
            partials
        )
        baseline_source = _CUTEDSL_SOURCE

    baseline_runner(partials, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        baseline_validation_out[:num_tokens], expected, atol=_ATOL, rtol=_RTOL
    )
    baseline_ms = _median_cupti_ms(
        baseline_runner, partials, out, dry_run_iters, repeat_iters
    )
    return {
        "num_tokens": num_tokens,
        "capacity": capacity,
        "native_reducer_median_ms": native_ms,
        "baseline_label": baseline_label,
        "baseline_source": baseline_source,
        "baseline_source_sha256": (
            _CUTEDSL_SOURCE_SHA256 if baseline == "vendored-cutedsl" else None
        ),
        "baseline_capacity": (
            _BASELINE_CAPACITY if baseline == "vendored-cutedsl" else num_tokens
        ),
        "baseline_median_ms": baseline_ms,
        "native_speedup_vs_baseline": baseline_ms / native_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        choices=("ordered-pytorch", "vendored-cutedsl"),
        default="ordered-pytorch",
        help=(
            "comparison path; vendored-cutedsl is a fixed non-public module path "
            "and fails closed if unavailable"
        ),
    )
    parser.add_argument("--dry-run-iters", type=int, default=20)
    parser.add_argument("--repeat-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4727)
    parser.add_argument(
        "--json", action="store_true", help="also print one JSON document"
    )
    args = parser.parse_args()
    if args.dry_run_iters <= 0 or args.repeat_iters <= 0:
        parser.error("iteration counts must be positive")

    _require_exact_sm100a()
    _require_cupti()

    results = []
    for shape_idx, (num_tokens, capacity) in enumerate(_SHAPES):
        result = _run_shape(
            num_tokens,
            capacity,
            args.baseline,
            args.dry_run_iters,
            args.repeat_iters,
            args.seed + shape_idx,
        )
        results.append(result)
        print(
            f"T={num_tokens:4d} C={capacity:4d} "
            f"native_reducer={result['native_reducer_median_ms']:.6f} ms "
            f"baseline[{result['baseline_label']}]="
            f"{result['baseline_median_ms']:.6f} ms "
            f"speedup={result['native_speedup_vs_baseline']:.6f}x"
        )

    native_geomean_ms = _geomean(
        [result["native_reducer_median_ms"] for result in results]
    )
    baseline_geomean_ms = _geomean([result["baseline_median_ms"] for result in results])
    speedup_geomean = _geomean(
        [result["native_speedup_vs_baseline"] for result in results]
    )
    summary = {
        "architecture": "sm_100a",
        "dtype": "bfloat16",
        "atol": _ATOL,
        "rtol": _RTOL,
        "timing": "CUPTI activity, cold L2",
        "native_reducer_geomean_ms": native_geomean_ms,
        "baseline_geomean_ms": baseline_geomean_ms,
        "native_speedup_geomean_vs_baseline": speedup_geomean,
        "results": results,
    }
    print(
        f"geomean native_reducer={native_geomean_ms:.6f} ms "
        f"baseline[{results[0]['baseline_label']}]="
        f"{baseline_geomean_ms:.6f} ms "
        f"speedup={speedup_geomean:.6f}x"
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
