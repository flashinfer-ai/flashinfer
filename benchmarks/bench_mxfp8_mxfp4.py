# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU benchmark for the weighted MXFP8 x MXFP4 MegaMoE output.

Run each arm in a fresh process on the same GPU, in A/B/A order:
baseline, candidate, baseline.
Uses pre-staged inputs, default tactics, cold-L2 CUPTI graph timing. Saved
outputs permit exact before/after checks independently of timing.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.integration import (
    create_dummy_mxfp8_mxfp4_inputs,
    mxfp8_mxfp4_mega_moe,
)
from flashinfer.testing import bench_gpu_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[32, 128, 512])
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--intermediate", type=int, required=True)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-outputs", type=Path)
    args = parser.parse_args()
    # Refuse an event fallback: stable workspace closures cannot rotate inputs.
    from cupti import cupti  # noqa: F401

    assert int(importlib.metadata.version("cupti-python").split(".")[0]) >= 13
    assert torch.cuda.get_device_capability() == (10, 0)
    device = torch.cuda.get_device_properties(0)
    metadata = dict(
        gpu=device.name,
        gpu_uuid=str(device.uuid),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cutlass_dsl=importlib.metadata.version("nvidia-cutlass-dsl"),
        cupti=importlib.metadata.version("cupti-python"),
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        dirty=bool(
            subprocess.check_output(["git", "diff", "--name-only"], text=True).strip()
        ),
        hidden=args.hidden,
        intermediate=args.intermediate,
        experts=args.experts,
        topk=args.top_k,
        seed=args.seed,
        world_size=1,
        timing="CUPTI, CUDA graph, cold L2, pre-staged inputs",
    )
    results, saved = [], {}
    for tokens in args.tokens:
        _, fc1, fc2, workspace = create_dummy_mxfp8_mxfp4_inputs(
            0,
            1,
            args.experts,
            tokens,
            tokens,
            args.top_k,
            args.hidden,
            args.intermediate,
            seed=args.seed,
            capacity_tactics=[{}],
        )
        try:

            def run():
                return mxfp8_mxfp4_mega_moe(
                    None, fc1, fc2, workspace, num_tokens=tokens
                )

            actual = run()
            torch.cuda.synchronize()
            assert torch.isfinite(actual).all()
            saved[tokens] = dict(
                output=actual.cpu(), route_terms=workspace.route_terms.cpu()
            )
            times = (
                np.asarray(
                    bench_gpu_time(
                        run,
                        dry_run_iters=50,
                        repeat_iters=args.repeat,
                        enable_cupti=True,
                        use_cuda_graph=True,
                        cold_l2_cache=True,
                    )
                )
                * 1000
            )
            row = dict(
                metadata,
                tokens=tokens,
                tactic=workspace.tactic,
                p50_us=float(np.median(times)),
                p90_us=float(np.percentile(times, 90)),
                samples_us=times.tolist(),
            )
            results.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "samples_us"}),
                flush=True,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
        finally:
            workspace.destroy()
    if args.save_outputs:
        args.save_outputs.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved, args.save_outputs)


if __name__ == "__main__":
    main()
