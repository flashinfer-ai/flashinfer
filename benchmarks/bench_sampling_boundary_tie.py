# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare sampling revisions using separate checkouts and JIT workspaces.

Run this script once per source checkout with an explicit PYTHONPATH and
FLASHINFER_WORKSPACE_BASE. It never switches tracked files or clears caches.
"""

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import warnings
from pathlib import Path

import torch

import flashinfer
from flashinfer.jit.sampling import gen_sampling_module
from flashinfer.testing import bench_gpu_time


def _runner(name, deterministic, top_k, top_p):
    s = flashinfer.sampling
    kwargs = dict(deterministic=deterministic, seed=42, offset=0)
    if name == "basic":
        return lambda p: s.sampling_from_probs(p, **kwargs)
    if name == "min_p":
        return lambda p: s.min_p_sampling_from_probs(p, 0.1, **kwargs)
    if name == "top_k":
        return lambda p: s.top_k_sampling_from_probs(p, top_k, **kwargs)
    if name == "top_p":
        return lambda p: s.top_p_sampling_from_probs(p, top_p, **kwargs)
    if name == "joint":
        return lambda p: s.top_k_top_p_sampling_from_probs(
            p, top_k, top_p, filter_apply_order="joint", **kwargs
        )
    if name == "top_k_first":
        return lambda p: s.top_k_top_p_sampling_from_probs(
            p, top_k, top_p, filter_apply_order="top_k_first", **kwargs
        )
    if name == "top_p_renorm":
        return lambda p: s.top_p_renorm_probs(p, top_p)
    if name == "top_k_renorm":
        return lambda p: s.top_k_renorm_probs(p, top_k)
    if name == "top_k_mask":
        return lambda p: s.top_k_mask_logits(p, top_k)
    raise ValueError(name)


def main():
    # A published CUPTI comparison must not silently switch timing backends.
    warnings.filterwarnings("error", message=".*Falling back to CUDA events.*")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--revision", default="")
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--vocab", type=int, default=128512)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 512])
    parser.add_argument("--distributions", nargs="+", default=["unique", "uniform"])
    parser.add_argument(
        "--routines", nargs="+", default=["basic", "min_p", "top_k", "top_p", "joint"]
    )
    parser.add_argument("--repeat-iters", type=int, default=40)
    parser.add_argument("--both-deterministic", action="store_true")
    parser.add_argument("--deterministic", choices=("true", "false"), default="true")
    parser.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    source = Path(flashinfer.__file__).resolve().parent.parent
    headers = [
        "include/flashinfer/sampling.cuh",
        "include/flashinfer/topk.cuh",
        "include/flashinfer/topk_common.cuh",
        "include/flashinfer/air_top_p.cuh",
        "include/flashinfer/sampling_tie_break.cuh",
        "flashinfer/sampling.py",
    ]
    metadata = {
        "label": args.label,
        "revision": args.revision,
        "source": str(source),
        "top_k": args.top_k,
        "top_p": args.top_p,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "timing": "CUPTI, cold L2, "
        + ("CUDA graph" if args.cuda_graph else "no CUDA graph"),
        "source_hashes": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in headers
            if (source / name).exists()
        },
        "nvidia_smi": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv"],
            text=True,
        ),
    }
    args.out.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    rows = []
    generator = torch.Generator(device="cuda").manual_seed(20261102)
    for distribution in args.distributions:
        for batch in args.batches:
            if distribution == "unique":
                # Adjacent logits are sufficiently separated to produce distinct
                # fp32 probabilities. Explicitly verify this instead of assuming
                # random fp32 softmax has no accidental ties.
                row = torch.linspace(-1, 1, args.vocab, device="cuda")
                row = row[
                    torch.randperm(args.vocab, device="cuda", generator=generator)
                ]
                probs = row.softmax(-1).expand(batch, -1).contiguous()
                assert row.numel() == probs[0].unique().numel()
            elif distribution == "uniform":
                probs = torch.full((batch, args.vocab), 1 / args.vocab, device="cuda")
            elif distribution == "bf16_derived":
                # Keep probabilities rounded to bf16, as in serving pipelines
                # that promote stored bf16 probabilities back to fp32. Record
                # the resulting sum drift; do not erase ties by re-softmaxing.
                logits = torch.randn(
                    (batch, args.vocab), device="cuda", generator=generator
                )
                probs = logits.softmax(-1).bfloat16().float()
            else:
                raise ValueError(distribution)
            input_hash = hashlib.sha256(probs.cpu().numpy().tobytes()).hexdigest()
            row_sum = probs.double().sum(-1)
            unique_values = probs[0].unique().numel()
            deterministic_modes = (
                [True, False]
                if args.both_deterministic
                else [args.deterministic == "true"]
            )
            for deterministic in deterministic_modes:
                for routine in args.routines:
                    run = _runner(routine, deterministic, args.top_k, args.top_p)
                    samples = run(probs)
                    torch.cuda.synchronize()
                    spec = gen_sampling_module()
                    assert not spec.is_aot, (
                        "Source comparisons must use the checkout's JIT kernel"
                    )
                    binary = spec.get_library_path()
                    metadata["sampling_binary"] = {
                        "path": str(binary),
                        "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                    }
                    if routine in ("top_k_first", "top_k_mask", "top_k_renorm"):
                        from flashinfer.jit.topk import gen_topk_module

                        topk_spec = gen_topk_module()
                        assert not topk_spec.is_aot
                        topk_binary = topk_spec.get_library_path()
                        metadata["topk_binary"] = {
                            "path": str(topk_binary),
                            "sha256": hashlib.sha256(
                                topk_binary.read_bytes()
                            ).hexdigest(),
                        }
                    args.out.with_suffix(".metadata.json").write_text(
                        json.dumps(metadata, indent=2) + "\n"
                    )
                    if args.label.startswith("fixed") and distribution == "uniform":
                        if routine in ("top_k", "joint", "top_k_first"):
                            assert (samples < args.top_k).all()
                    times = bench_gpu_time(
                        run,
                        input_args=(probs,),
                        enable_cupti=True,
                        cold_l2_cache=True,
                        use_cuda_graph=args.cuda_graph,
                        dry_run_iters=5,
                        repeat_iters=args.repeat_iters,
                    )
                    rows.append(
                        dict(
                            label=args.label,
                            routine=routine,
                            distribution=distribution,
                            batch=batch,
                            vocab=args.vocab,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            deterministic=deterministic,
                            input_sha256=input_hash,
                            row_sum_min=row_sum.min().item(),
                            row_sum_max=row_sum.max().item(),
                            first_row_unique_values=unique_values,
                            median_us=1000 * statistics.median(times),
                            min_us=1000 * min(times),
                            max_us=1000 * max(times),
                        )
                    )
                    print(json.dumps(rows[-1]), flush=True)
                    with args.out.open("w", newline="") as out:
                        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
    args.out.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
