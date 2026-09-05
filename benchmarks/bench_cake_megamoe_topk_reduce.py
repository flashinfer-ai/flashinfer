"""CUPTI benchmark for the frozen MegaMoE workspace TopK reducer.

``--baseline ordered-pytorch`` provides an eager semantic reference that
performs the six BF16-to-FP32 additions in strict K=0..5 order and converts the
result to BF16. It is made of several GPU kernels, not the MegaMoE CuTeDSL
kernel and not a claimed historical FlashInfer performance baseline.

``--baseline vendored-cutedsl-matched`` (the default) selects the exact vendored
implementation at
``flashinfer/moe_ep/kernel_src/cutedsl_megamoe/src/moe_nvfp4_swapab/topk_reduce.py``.
Both kernels process the same live ``T x 6 x 4096`` elements in this primary
comparison. ``--baseline vendored-cutedsl-fixed-capacity`` separately measures
the serving scenario where the old reducer processes a prefill-sized ``C=4096``
workspace for a smaller live batch. That fixed-capacity result measures avoided
workspace work, not an apples-to-apples kernel speedup. Preparation, zero-fill,
and compilation remain outside timing. The vendored module is not a stable
public API. Source drift, import failure, or compilation failure is fatal; the
benchmark never substitutes the PyTorch path while labelling it CuTeDSL.
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
_GRID_CTAS_PER_TOKEN = 4
_LEGACY_BASELINE_CAPACITY = 4096
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


def _matched_cutedsl_views(
    partials: torch.Tensor,
    out: torch.Tensor,
    comparison_plan: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the exact tensor extent declared by a matched-work plan."""

    assert comparison_plan["comparison_kind"] == "matched_work_kernel"
    tensor_extent = comparison_plan["baseline_tensor_extent"]
    assert isinstance(tensor_extent, int)
    return partials[:tensor_extent], out[:tensor_extent]


def _make_vendored_cutedsl_runner(
    partials: torch.Tensor,
    out: torch.Tensor,
    num_tokens: int,
    comparison_plan: dict[str, Any],
) -> tuple[Callable[[torch.Tensor, torch.Tensor], None], torch.Tensor]:
    """Prepare the pinned CuTeDSL reducer with explicit physical work."""

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

    if comparison_plan["comparison_kind"] == "matched_work_kernel":
        # Primary kernel comparison: CuTeDSL sees exactly the same live rows
        # that the native reducer launches for. The surrounding native
        # workspace may have more capacity, but neither timed kernel touches it.
        assert comparison_plan["baseline_tensor_extent"] == num_tokens
        baseline_partials, baseline_out = _matched_cutedsl_views(
            partials,
            out,
            comparison_plan,
        )
    else:
        fixed_capacity = comparison_plan["baseline_tensor_extent"]
        assert isinstance(fixed_capacity, int)
        if fixed_capacity < partials.shape[0]:
            raise ValueError("fixed baseline capacity cannot truncate input")
        # Serving scenario only: reproduce the legacy prefill-sized physical
        # workspace. The CuTeDSL grid is shape-derived and therefore processes
        # all fixed_capacity rows even when only num_tokens rows are live.
        baseline_partials = torch.zeros(
            (fixed_capacity, _TOP_K, _HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=partials.device,
        )
        baseline_partials[: partials.shape[0]].copy_(partials)
        baseline_out = torch.empty(
            (fixed_capacity, _HIDDEN_SIZE),
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


def _comparison_plan(num_tokens: int, baseline: str) -> dict[str, Any]:
    """Define timed work once for both runner construction and JSON output."""

    native_io_bytes = num_tokens * (_TOP_K + 1) * _HIDDEN_SIZE * 2
    if baseline == "vendored-cutedsl-fixed-capacity":
        baseline_work_tokens = _LEGACY_BASELINE_CAPACITY
        comparison_kind = "legacy_fixed_capacity_serving_scenario"
        same_token_extent = num_tokens == baseline_work_tokens
        same_reduction_work: bool | None = same_token_extent
        baseline_tensor_extent: int | None = baseline_work_tokens
    elif baseline == "vendored-cutedsl-matched":
        baseline_work_tokens = num_tokens
        comparison_kind = "matched_work_kernel"
        same_token_extent = True
        same_reduction_work = True
        baseline_tensor_extent = num_tokens
    else:
        baseline_work_tokens = None
        comparison_kind = "matched_live_tokens_semantic_reference"
        same_token_extent = True
        same_reduction_work = None
        baseline_tensor_extent = None
    return {
        "comparison_kind": comparison_kind,
        "native_work_tokens": num_tokens,
        "baseline_work_tokens": baseline_work_tokens,
        "baseline_tensor_extent": baseline_tensor_extent,
        "same_token_extent": same_token_extent,
        "same_reduction_work": same_reduction_work,
        "native_grid_ctas": _GRID_CTAS_PER_TOKEN * num_tokens,
        "baseline_grid_ctas": (
            _GRID_CTAS_PER_TOKEN * baseline_work_tokens
            if baseline_work_tokens is not None
            else None
        ),
        "native_io_bytes": native_io_bytes,
        "baseline_io_bytes": (
            baseline_work_tokens * (_TOP_K + 1) * _HIDDEN_SIZE * 2
            if baseline_work_tokens is not None
            else None
        ),
    }


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
    comparison_plan = _comparison_plan(num_tokens, baseline)

    if baseline == "ordered-pytorch":
        baseline_label = "ordered_pytorch_fp32_k0_to_k5"
        baseline_runner = _make_ordered_pytorch_runner(num_tokens)
        baseline_validation_out = out
        baseline_source = "PyTorch eager: six ordered FP32 accumulations + BF16 store"
    elif baseline == "vendored-cutedsl-matched":
        baseline_label = "vendored_cutedsl_topk_reduce_matched_live_t"
        baseline_runner, baseline_validation_out = _make_vendored_cutedsl_runner(
            partials,
            out,
            num_tokens,
            comparison_plan,
        )
        baseline_source = _CUTEDSL_SOURCE
    else:
        baseline_label = "vendored_cutedsl_topk_reduce_fixed_c4096"
        baseline_runner, baseline_validation_out = _make_vendored_cutedsl_runner(
            partials,
            out,
            num_tokens,
            comparison_plan,
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
    if baseline == "vendored-cutedsl-matched":
        assert comparison_plan["same_reduction_work"]
    return {
        "num_tokens": num_tokens,
        "capacity": capacity,
        "native_reducer_median_ms": native_ms,
        "baseline_label": baseline_label,
        "baseline_source": baseline_source,
        "baseline_source_sha256": (
            _CUTEDSL_SOURCE_SHA256 if baseline.startswith("vendored-cutedsl") else None
        ),
        "baseline_median_ms": baseline_ms,
        "baseline_over_native_latency_ratio": baseline_ms / native_ms,
        **comparison_plan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        choices=(
            "vendored-cutedsl-matched",
            "vendored-cutedsl-fixed-capacity",
            "ordered-pytorch",
        ),
        default="vendored-cutedsl-matched",
        help=(
            "comparison path; vendored CuTeDSL modes use a fixed non-public "
            "module path and fail closed if unavailable"
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
        baseline_work = result["baseline_work_tokens"]
        baseline_work_label = (
            str(baseline_work) if baseline_work is not None else "multi-kernel"
        )
        print(
            f"T={num_tokens:4d} C={capacity:4d} "
            "work_tokens(native:baseline)="
            f"{result['native_work_tokens']}:"
            f"{baseline_work_label} tokens "
            f"native_reducer={result['native_reducer_median_ms']:.6f} ms "
            f"baseline[{result['baseline_label']}]="
            f"{result['baseline_median_ms']:.6f} ms "
            "baseline/native latency ratio="
            f"{result['baseline_over_native_latency_ratio']:.6f}x"
        )

    native_geomean_ms = _geomean(
        [result["native_reducer_median_ms"] for result in results]
    )
    baseline_geomean_ms = _geomean([result["baseline_median_ms"] for result in results])
    latency_ratio_geomean = _geomean(
        [result["baseline_over_native_latency_ratio"] for result in results]
    )
    summary = {
        "architecture": "sm_100a",
        "dtype": "bfloat16",
        "comparison_kind": results[0]["comparison_kind"],
        "atol": _ATOL,
        "rtol": _RTOL,
        "timing": "CUPTI activity, cold L2",
        "native_reducer_geomean_ms": native_geomean_ms,
        "baseline_geomean_ms": baseline_geomean_ms,
        "baseline_over_native_latency_ratio_geomean": latency_ratio_geomean,
        "results": results,
    }
    print(
        f"geomean native_reducer={native_geomean_ms:.6f} ms "
        f"baseline[{results[0]['baseline_label']}]="
        f"{baseline_geomean_ms:.6f} ms "
        f"baseline/native latency ratio={latency_ratio_geomean:.6f}x"
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
