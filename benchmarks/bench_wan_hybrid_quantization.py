"""Measure reusable exact-shape Wan value quantization with cold-L2 CUPTI."""

import json
import statistics
from importlib.metadata import version as distribution_version

import torch

from flashinfer.testing import bench_gpu_time
from flashinfer.wan_hybrid import (
    WanHybridAttentionWorkspace,
    _quantize_wan_hybrid_value,
)


_SHAPE = (1, 4800, 40, 128)
_WARMUP_RUNS = 2
_MEASURE_RUNS = 20


def main() -> None:
    try:
        from cupti import cupti as _cupti  # noqa: F401
    except ModuleNotFoundError as error:
        raise RuntimeError("cupti-python is required for this benchmark") from error
    cupti_version = distribution_version("cupti-python")
    if int(cupti_version.split(".", maxsplit=1)[0]) < 13:
        raise RuntimeError(
            f"cupti-python>=13 is required, found {cupti_version}"
        )

    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device)
    generator.manual_seed(4254)
    value = torch.randn(
        _SHAPE,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    workspace = WanHybridAttentionWorkspace(device)

    def quantize() -> None:
        _quantize_wan_hybrid_value(value, workspace)

    quantize()
    torch.cuda.synchronize(device)
    allocated_before = torch.cuda.memory_allocated(device)
    for _ in range(10):
        quantize()
    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device)

    samples = [
        float(sample)
        for sample in bench_gpu_time(
            fn=quantize,
            dry_run_iters=_WARMUP_RUNS,
            repeat_iters=_MEASURE_RUNS,
            enable_cupti=True,
            use_cuda_graph=False,
            cold_l2_cache=True,
        )
    ]
    properties = torch.cuda.get_device_properties(device)
    report = {
        "shape": list(_SHAPE),
        "layout": "NHD",
        "dtype": "bfloat16",
        "device": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "cupti_python_version": cupti_version,
        "cold_l2": True,
        "warmup_runs": _WARMUP_RUNS,
        "measure_runs": _MEASURE_RUNS,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
        "allocation_stable": allocated_after == allocated_before,
        "memory_allocated_before": allocated_before,
        "memory_allocated_after": allocated_after,
        "packed_storage_bytes_per_level": 13_107_200,
        "scale_plane_bytes": 819_200,
        "scale_plane_count": 4,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["allocation_stable"]:
        raise RuntimeError("reused quantization allocated persistent CUDA storage")


if __name__ == "__main__":
    main()
