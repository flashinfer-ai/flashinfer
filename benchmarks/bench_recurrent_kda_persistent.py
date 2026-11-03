# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Input generation and correctness metric adapted from humanfia/kda-for-kda-release
# Copyright (c) 2026 KDA Team; MIT License, see licenses/LICENSE.kda-for-kda.

"""INT21 KDA prefill: persistent CuTe DSL versus MoonshotAI/FlashKDA.

python benchmarks/bench_recurrent_kda_persistent.py \
    --flash-kda-source-dir /path/to/FlashKDA --output results.json

Install fla-core==0.5.2 and cupti-python for independent correctness and timing.
Both arms update FP32 state in place and are captured once, then measured with
cold-L2 CUPTI GPU activity spans. Allocation, compilation, scheduling, capture
and state reset are outside timing. States evolve during repeated launches;
both arms reset to the same initial values before each timing trial.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

import torch

from flashinfer import RecurrentKDAPrefillWorkspace, recurrent_kda
from flashinfer.testing import bench_gpu_time_with_cupti

D = 128
ATOL_RMS, RTOL, MAX_REL_L2 = 0.5, 0.05, 0.03
MIXED = [1300, 547, 2048, 963, 271, 3063]
WORKLOADS = {
    "h96-fixed": (96, [8192]),
    "h96-mixed_varlen": (96, MIXED),
    "h96-uniform_varlen": (96, [1024] * 8),
    "h64-fixed": (64, [8192]),
    "h64-mixed_varlen": (64, MIXED),
    "h64-uniform_varlen": (64, [1024] * 8),
}


def make_inputs(heads, seq_lens, seed, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    T = sum(seq_lens)

    def randn(shape, scale, dtype=torch.bfloat16):
        return (torch.randn(shape, generator=gen, device=device) * scale).to(dtype)

    A_log = torch.log(
        torch.empty(heads, device=device).uniform_(1.0, 16.0, generator=gen)
    )
    dt = torch.exp(
        torch.rand(heads * D, generator=gen, device=device)
        * (math.log(0.1) - math.log(0.001))
        + math.log(0.001)
    ).clamp_(min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    q, k, v, g = (randn((1, T, heads, D), 0.5) for _ in range(4))
    beta = randn((1, T, heads), 0.5)
    initial_state = randn((len(seq_lens), heads, D, D), 0.25, torch.float32)
    cu_seqlens = None
    if len(seq_lens) > 1:
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(seq_lens).cumsum(0).tolist()], device=device
        )
    return [
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        1.0 / math.sqrt(D),
        initial_state,
        cu_seqlens,
    ]


def check(got, want):
    """The task's correctness gate on one tensor; returns (ok, worst ratio, rel L2)."""
    x, y = got.float(), want.float()
    err = (x - y).abs()
    ratio = torch.minimum(
        err / (ATOL_RMS * y.pow(2).mean().sqrt()), err / (RTOL * y.abs() + 1e-30)
    )
    rel_l2 = float(torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(y))
    worst = float(torch.nan_to_num(ratio, nan=float("inf")).max())
    return (
        worst <= 1.0 and rel_l2 <= MAX_REL_L2 and bool(torch.isfinite(x).all()),
        worst,
        rel_l2,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flash-kda-source-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--workloads", nargs="+", choices=list(WORKLOADS), default=list(WORKLOADS)
    )
    args = parser.parse_args()
    if min(args.warmup, args.iters, args.trials) <= 0:
        parser.error("warmup, iters and trials must be positive")
    source = args.flash_kda_source_dir.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip():
        raise RuntimeError("FlashKDA source checkout must be clean")
    sys.path.insert(0, str(source))
    import flash_kda
    import flash_kda_C
    from cupti import cupti

    cupti.get_timestamp()  # Fail explicitly instead of reporting event timing as CUPTI.

    for module in (flash_kda, flash_kda_C):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise RuntimeError(f"{module.__name__} was not loaded from {source}")
    os.environ["FLA_FLASH_KDA"] = "0"
    os.environ["FLA_TILELANG"] = "0"
    from fla.ops.kda import chunk_kda

    report = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "flash_kda_revision": revision,
        "flash_kda_extension_sha256": hashlib.sha256(
            Path(flash_kda_C.__file__).read_bytes()
        ).hexdigest(),
        "timing": "CUPTI, cold L2, captured public API/raw FlashKDA, evolving in-place FP32 state; reset outside each trial",
        "warmup": args.warmup,
        "iters": args.iters,
        "trials": args.trials,
        "rows": [],
    }
    print(
        f"{report['gpu']} | FlashKDA {revision} | {args.trials} x {args.iters}",
        flush=True,
    )
    for name in args.workloads:
        heads, lengths = WORKLOADS[name]
        seed = list(WORKLOADS).index(name)
        q, k, v, g, beta, A_log, dt_bias, scale, initial, cu = make_inputs(
            heads, lengths, seed
        )
        reference = chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial,
            cu_seqlens=cu,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            state_v_first=True,
            safe_gate=True,
            lower_bound=-5.0,
        )
        candidate_state = initial.clone()
        baseline_state = initial.clone()
        candidate_out = torch.empty_like(v)
        baseline_out = torch.empty_like(v)
        workspace = RecurrentKDAPrefillWorkspace(device=q.device)
        baseline_workspace = torch.empty(
            flash_kda.get_workspace_size(q.shape[1], heads, len(lengths)),
            dtype=torch.uint8,
            device=q.device,
        )

        def candidate():
            return recurrent_kda(
                q,
                k,
                v,
                g,
                beta,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                initial_state=candidate_state,
                cu_seqlens=cu,
                output=candidate_out,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                beta_is_logit=True,
                lower_bound=-5.0,
                prefill_workspace=workspace,
                backend="cute-dsl-persistent",
            )

        def baseline():
            flash_kda._fwd_raw(
                q,
                k,
                v,
                g,
                beta,
                scale,
                baseline_out,
                baseline_workspace,
                A_log,
                dt_bias.view(heads, D),
                -5.0,
                initial_state=baseline_state,
                final_state=baseline_state,
                cu_seqlens=cu,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            candidate()
            baseline()
        stream.synchronize()
        verdicts = [
            check(got, want)
            for got, want in zip(
                (candidate_out, candidate_state), reference, strict=True
            )
        ]
        base_verdicts = [
            check(got, want)
            for got, want in zip((baseline_out, baseline_state), reference, strict=True)
        ]
        if not all(row[0] for row in verdicts + base_verdicts):
            raise AssertionError(
                f"{name}: candidate={verdicts}, baseline={base_verdicts}"
            )
        graphs = []
        for fn in (candidate, baseline):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                fn()
            graphs.append(graph)
        samples = [[], []]
        for trial in range(args.trials):
            # Alternate order to reduce drift; retain all trial medians.
            for arm in (0, 1) if trial % 2 == 0 else (1, 0):
                (candidate_state if arm == 0 else baseline_state).copy_(initial)
                torch.cuda.synchronize()
                times = bench_gpu_time_with_cupti(
                    graphs[arm].replay,
                    dry_run_iters=args.warmup,
                    repeat_iters=args.iters,
                    cold_l2_cache=True,
                    use_cuda_graph=False,
                )
                samples[arm].append(statistics.median(times))
        candidate_ms, baseline_ms = map(statistics.median, samples)
        row = {
            "name": name,
            "heads": heads,
            "sequence_lengths": lengths,
            "candidate_ms": candidate_ms,
            "baseline_ms": baseline_ms,
            "speedup": baseline_ms / candidate_ms,
            "candidate_trials_ms": samples[0],
            "baseline_trials_ms": samples[1],
            "candidate_correctness": verdicts,
            "baseline_correctness": base_verdicts,
        }
        report["rows"].append(row)
        print(
            f"{name:24s} FlashKDA {baseline_ms:.6f} ms | persistent {candidate_ms:.6f} ms | {row['speedup']:.3f}x | PASS",
            flush=True,
        )
    report["geomean_speedup"] = math.exp(
        statistics.mean(math.log(row["speedup"]) for row in report["rows"])
    )
    print(f"geomean: {report['geomean_speedup']:.3f}x", flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    with torch.no_grad():
        main()
