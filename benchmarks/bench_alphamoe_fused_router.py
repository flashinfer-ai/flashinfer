"""CUPTI benchmark for the standalone AlphaMoE fused router (SM100/SM103)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flashinfer.fused_moe import (
    allocate_alphamoe_route_plan,
    alphamoe_fused_router,
)
from flashinfer.jit.cpp_ext import is_cuda_version_at_least
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import is_sm100a_supported


DEFAULT_SHAPES = (
    (1, 512, 2, 16, True),
    (8, 257, 9, 8, True),
    (32, 512, 8, 16, False),
    (128, 512, 8, 16, False),
)


def benchmark_shape(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    block_m: int,
    has_shared_expert: bool,
) -> dict[str, object]:
    logits = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.float32)
    plan = allocate_alphamoe_route_plan(
        logits,
        top_k=top_k,
        block_m=block_m,
        has_shared_expert=has_shared_expert,
    )

    def run() -> None:
        alphamoe_fused_router(
            logits,
            top_k=top_k,
            block_m=block_m,
            has_shared_expert=has_shared_expert,
            plan=plan,
            skip_check=True,
        )

    # Build/load and validate the launch once outside the measured boundary.
    run()
    torch.cuda.synchronize()
    measurements = bench_gpu_time(
        run,
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        enable_cupti=True,
        cold_l2_cache=True,
    )
    return {
        "num_tokens": num_tokens,
        "num_experts": num_experts,
        "top_k": top_k,
        "block_m": block_m,
        "has_shared_expert": has_shared_expert,
        "median_ms": float(np.median(measurements)),
        "p20_ms": float(np.percentile(measurements, 20)),
        "p80_ms": float(np.percentile(measurements, 80)),
        "samples": len(measurements),
        "timing": "CUPTI GPU activity with cold-L2 flushing",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", type=Path, help="Optional path for machine-readable results."
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", torch.cuda.current_device())
    capability = torch.cuda.get_device_capability(device)
    if not (
        capability in {(10, 0), (10, 3)}
        and is_sm100a_supported(device)
        and is_cuda_version_at_least("12.9" if capability == (10, 3) else "12.8")
    ):
        raise RuntimeError(
            "AlphaMoE fused router requires supported SM100a or SM103a hardware/toolkit; "
            f"got compute capability {capability}"
        )

    results = [benchmark_shape(*shape) for shape in DEFAULT_SHAPES]
    payload = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
