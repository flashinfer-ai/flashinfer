# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare TRTLLM-GEN, cuDNN, and the automatic sparse MLA selection.

python benchmarks/bench_cudnn_sparse_mla.py --csv sparse_mla.csv
python benchmarks/bench_cudnn_sparse_mla.py --rows 128 --topk 128 640 2048 2051
python benchmarks/bench_cudnn_sparse_mla.py --layout compact --rows 128 4096

The fixed layout is [rows, 1, H, D]; compact is one prefill chunk [rows, H, D].
Both use identical physical token selections, BF16, H64, and GLM's scale 1/16.
Index selection and projections are excluded. All backends are warmed and
reference-checked before measuring; TRTLLM-GEN is autotuned. Timings use the
repository benchmark helper with cold L2 and CUDA graphs (CUPTI when enabled).
"""

import argparse
import csv
import functools
import importlib.metadata
import itertools
import statistics

import torch

import flashinfer
from flashinfer.mla import trtllm_prefill_with_kv_cache_mla
from flashinfer.testing import bench_gpu_time


def make_inputs(rows, dim, topk, context, layout):
    if context <= rows + topk or context % 32:
        raise ValueError("Context must be divisible by 32 and greater than rows + topk")
    torch.manual_seed(42)
    q = torch.randn(rows, 64, dim, device="cuda", dtype=torch.bfloat16) * 0.5
    kv = (
        torch.randn(context // 32, 1, 32, dim, device="cuda", dtype=torch.bfloat16)
        * 0.5
    )
    prefix = context - rows
    base = torch.randperm(prefix, device="cuda")[:topk]
    shifts = torch.randint(prefix, (rows, 1), device="cuda")
    indices = ((base + shifts) % prefix).to(torch.int32)
    # Match the serving adapter's multiple-of-four metadata contract. Masked
    # padding is the same for every backend; cuDNN's additional padding is timed.
    width = (topk + 3) // 4 * 4
    indices = torch.nn.functional.pad(indices, (0, width - topk), value=-1)
    kwargs = dict(
        query=q.view(rows, 1, 64, dim),
        kv_cache=kv,
        workspace_buffer=torch.zeros(256 << 20, dtype=torch.uint8, device="cuda"),
        qk_nope_head_dim=192 if dim == 576 else 256,
        kv_lora_rank=512,
        qk_rope_head_dim=dim - 512,
        block_tables=indices.view(rows, 1, width),
        seq_lens=torch.full((rows,), context, device="cuda", dtype=torch.int32),
        max_seq_len=context,
        sparse_mla_top_k=width,
        bmm1_scale=1 / 16,
        bmm2_scale=1.0,
        enable_pdl=False,
        out=torch.empty(rows, 1, 64, 512, dtype=torch.bfloat16, device="cuda"),
    )
    if dim == 512:
        kwargs["sparse_mla_top_k_lens"] = torch.full(
            (rows,), topk, dtype=torch.int32, device="cuda"
        )
    if layout == "compact":
        kwargs.update(
            query=q,
            block_tables=indices,
            seq_lens=kwargs["seq_lens"][:1],
            cum_seq_lens_q=torch.tensor([0, rows], dtype=torch.int32, device="cuda"),
            max_q_len=rows,
            out=kwargs["out"].view(rows, 64, 512),
        )
    chosen = sorted({0, rows // 2, rows - 1})
    expected = []
    for row in chosen:
        selected = kv.view(-1, dim)[indices[row, :topk].long()].float()
        scores = q[row].float() @ selected.T / 16
        expected.append(scores.softmax(-1) @ selected[:, :512])
    return kwargs, chosen, torch.stack(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[1, 127, 128, 1024, 4096]
    )
    parser.add_argument("--dim", type=int, nargs="+", default=[576, 512])
    parser.add_argument("--topk", type=int, nargs="+", default=[2048])
    parser.add_argument("--context", type=int, nargs="+", default=[131072])
    parser.add_argument("--layout", choices=["fixed", "compact"], default="fixed")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["trtllm-gen", "cudnn", "auto"],
        default=["trtllm-gen", "cudnn", "auto"],
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--cupti",
        action="store_true",
        help="Prefer CUPTI timing; otherwise use CUDA events",
    )
    parser.add_argument("--csv", default="sparse_mla.csv")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    common = dict(
        gpu=torch.cuda.get_device_name(),
        sm="%d%d" % torch.cuda.get_device_capability(),
        torch_version=torch.__version__,
        flashinfer_version=flashinfer.__version__,
        cudnn_frontend=importlib.metadata.version("nvidia-cudnn-frontend"),
        dtype="bfloat16",
        heads=64,
        layout=args.layout,
        cold_l2=True,
    )
    fields = list(common) + [
        "rows",
        "dim",
        "topk",
        "context",
        "backend",
        "median_us",
        "min_us",
        "max_us",
        "speedup_vs_trtllm",
        "max_abs_error",
    ]
    with open(args.csv, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for dim, context, topk, rows in itertools.product(
            args.dim, args.context, args.topk, args.rows
        ):
            kwargs, chosen, expected = make_inputs(
                rows, dim, topk, context, args.layout
            )
            # Tune the old backend outside timing, including when auto retains it.
            with flashinfer.autotune(True):
                trtllm_prefill_with_kv_cache_mla(**kwargs, backend="trtllm-gen")
            records = []
            for backend in args.backends:
                fn = functools.partial(
                    trtllm_prefill_with_kv_cache_mla, backend=backend
                )
                result = fn(**kwargs).view(rows, 64, 512)[chosen].float()
                torch.testing.assert_close(result, expected, atol=0.003, rtol=0.03)
                times = bench_gpu_time(
                    fn,
                    input_kwargs=kwargs,
                    enable_cupti=args.cupti,
                    use_cuda_graph=True,
                    cold_l2_cache=True,
                    dry_run_iters=5,
                    repeat_iters=args.iterations,
                )
                records.append(
                    dict(
                        common,
                        rows=rows,
                        dim=dim,
                        topk=topk,
                        context=context,
                        backend=backend,
                        median_us=statistics.median(times) * 1000,
                        min_us=min(times) * 1000,
                        max_us=max(times) * 1000,
                        max_abs_error=(result - expected).abs().max().item(),
                    )
                )
            baseline = next(
                (r["median_us"] for r in records if r["backend"] == "trtllm-gen"), None
            )
            for record in records:
                record["speedup_vs_trtllm"] = (
                    baseline / record["median_us"] if baseline else ""
                )
                writer.writerow(record)
                print(record, flush=True)
            output.flush()


if __name__ == "__main__":
    main()
