"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Benchmark the experimental Cake Kimi-K3 AttnRes kernels (SM100 / SM103).

Rows: the Kimi-K3 evaluation grid (token counts ``M`` in {1, 2, 4, ..., 16384}
x block counts ``K`` in {0, 1, 4, 8}, plus every ``K`` in 0..8 at ``M`` = 1 and
4096) in the steady mode (residual add, no snapshot write, fused output norm).
Arms: ``cake`` (PDL off), ``cake_pdl`` (PDL on), ``torch`` (the eager FP32
reference; informational) and, when importable, ``vllm`` (the vLLM native
``attn_res`` op of the same model; ``vllm.models.kimi_k3.nvidia.ops.attn_res``).

Timing: ``flashinfer.testing.bench_gpu_time`` (CUPTI activity tracing, cold L2
between iterations, per-iteration GPU span; medians over ``--iters``).  The
operator mutates ``prefix`` in place; the drift across iterations changes values
only, not the work, so the loop does not restore state.  ``--accuracy`` reports
the maximum absolute output error of every arm against the FP32 reference.

Usage::

    python benchmarks/bench_cake_kimi_k3_attn_res.py [--rows m1_k4 m4096_k8 ...]
        [--arms cake,cake_pdl,torch,vllm] [--iters 100] [--accuracy] [--json out.json]
"""

import argparse
import json
import sys

import torch

from flashinfer.experimental.cake_kimi_k3_attn_res import cake_backend as cb
from flashinfer.kimi_k3_attn_res import prepare_kimi_k3_attn_res
from flashinfer.testing import bench_gpu_time

TOKEN_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
PRIMARY_K = (0, 1, 4, 8)
H = cb.HIDDEN_SIZE


def rows():
    cells = [(M, K) for M in TOKEN_COUNTS for K in PRIMARY_K]
    cells += [(M, K) for M in (1, 4096) for K in range(9) if K not in PRIMARY_K]
    return {f"m{M}_k{K}": (M, K) for M, K in cells}


def make_inputs(M, K, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    prefix = torch.empty((M, H), device="cuda", dtype=torch.bfloat16).uniform_(
        -1, 1, generator=g
    )
    delta = torch.empty((M, H), device="cuda", dtype=torch.bfloat16).uniform_(
        -0.015625, 0.015625, generator=g
    )
    blocks = torch.empty(
        (M, cb.MAX_BLOCKS, H), device="cuda", dtype=torch.bfloat16
    ).uniform_(-1, 1, generator=g)
    norm_weight = torch.empty(H, device="cuda", dtype=torch.bfloat16).normal_(
        1.0, 0.05, generator=g
    )
    qk_weight = torch.empty(H, device="cuda", dtype=torch.bfloat16).normal_(
        0.0, 0.02, generator=g
    )
    output_norm_weight = torch.empty(H, device="cuda", dtype=torch.bfloat16).normal_(
        1.0, 0.05, generator=g
    )
    out = torch.empty((M, H), device="cuda", dtype=torch.bfloat16)
    return dict(
        prefix=prefix,
        delta=delta,
        blocks=blocks,
        norm_weight=norm_weight,
        qk_weight=qk_weight,
        output_norm_weight=output_norm_weight,
        out=out,
    )


def vllm_op():
    try:
        from vllm.models.kimi_k3.nvidia.ops.attn_res import attn_res  # type: ignore
    except Exception:  # noqa: BLE001 - optional arm
        return None
    return attn_res


def build_arms(names, inputs, K):
    arms = {}
    if "cake" in names:
        arms["cake"] = prepare_kimi_k3_attn_res(**inputs, num_blocks=K).launch
    if "cake_pdl" in names:
        arms["cake_pdl"] = prepare_kimi_k3_attn_res(
            **inputs, num_blocks=K, enable_pdl=True
        ).launch
    if "torch" in names:
        arms["torch"] = lambda: cb.reference_kimi_k3_attn_res(**inputs, num_blocks=K)
    if "vllm" in names:
        op = vllm_op()
        if op is not None:
            # TODO(verify-on-node): confirm the vLLM Python wrapper signature at the pinned revision.
            arms["vllm"] = lambda: op(
                inputs["prefix"],
                inputs["delta"],
                inputs["blocks"],
                inputs["norm_weight"],
                inputs["qk_weight"],
                inputs["output_norm_weight"],
                inputs["out"],
                K,
                -1,
                1e-5,
                1e-5,
            )
    return arms


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", nargs="*", default=None)
    parser.add_argument("--arms", default="cake,cake_pdl,torch,vllm")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--json", default=None)
    parser.add_argument("--seed", type=int, default=31000)
    args = parser.parse_args(argv)
    device = torch.device("cuda")
    names = [a for a in args.arms.split(",") if a]
    table = rows()
    selected = args.rows or list(table)
    results = []
    for label in selected:
        M, K = table[label]
        if not cb.generated_program_available(device, M, K):
            print(f"{label}: no generated program registered; skipped", file=sys.stderr)
            continue
        inputs = make_inputs(M, K, args.seed)
        pristine = {k: v.clone() for k, v in inputs.items()}
        arms = build_arms(names, inputs, K)
        row = dict(
            label=label,
            M=M,
            K=K,
            plan=cb.plan_route(
                cb.arch_for(device), cb.sm_count_for(device), M, K, False
            )._asdict(),
        )
        for name, fn in arms.items():
            for k, v in pristine.items():
                inputs[k].copy_(v)
            times = bench_gpu_time(
                fn, dry_run_iters=5, repeat_iters=args.iters, l2_flush=True
            )
            row[f"{name}_ms"] = float(sorted(times)[len(times) // 2])
            if args.accuracy:
                for k, v in pristine.items():
                    inputs[k].copy_(v)
                fn()
                got = inputs["out"].float().clone()
                ref_inputs = {k: v.clone() for k, v in pristine.items()}
                cb.reference_kimi_k3_attn_res(**ref_inputs, num_blocks=K)
                row[f"{name}_max_abs_err"] = float(
                    (got - ref_inputs["out"].float()).abs().max()
                )
        base = row.get("vllm_ms") or row.get("torch_ms")
        if base and "cake_ms" in row:
            row["speedup_cake_vs_baseline"] = base / row["cake_ms"]
        results.append(row)
        print(json.dumps(row))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                dict(
                    device=torch.cuda.get_device_name(device),
                    capability=list(torch.cuda.get_device_capability(device)),
                    torch=torch.__version__,
                    rows=results,
                ),
                f,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
