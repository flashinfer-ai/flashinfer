# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Interleaved direct-API A/B, against an immutable pre-consolidation worktree.

Example (run on an otherwise idle Blackwell GPU):
    python benchmarks/bench_prims_ts_decode_api.py --baseline /path/to/baseline

JSON lines go to stdout. GPU times use FlashInfer's CUDA-graph benchmark.
Host API times exclude a final synchronize; eager end-to-end times include it.
Convenience calls synchronize metadata and cannot be captured, so only their
host/eager timings are reported. Kernel sources must be identical between the
two trees: baseline API modules intentionally share the unchanged kernel tree.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import warnings

import torch

from flashinfer.attention.prims_ts import decode, mla_decode
from flashinfer.testing import bench_gpu_time


def _load_baseline(root, name):
    path = root / "flashinfer/attention/prims_ts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(
        f"flashinfer.attention.prims_ts._baseline_{name}", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _kernel_digest(root):
    digest = hashlib.sha256()
    for file in sorted((root / "flashinfer/attention/prims_ts/kernels").rglob("*.py")):
        digest.update(str(file.relative_to(root)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def _revision(root):
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def _case(kind, batch, kv_len, sq, packed, dtype, old):
    page = 32
    columns = math.ceil(kv_len / page)
    pages = batch * columns
    lengths = torch.full((batch,), kv_len, dtype=torch.int32, device="cuda")
    tables = torch.arange(pages, dtype=torch.int32, device="cuda").reshape(
        batch, columns
    )
    offsets = (
        torch.arange(batch + 1, dtype=torch.int32, device="cuda") * sq
        if packed
        else None
    )
    if kind == "qwen":
        output_dtype = torch.float16 if dtype == torch.float8_e4m3fn else dtype
        shape = (batch, 28, 128) if sq == 1 else (batch, sq, 28, 128)
        q = (torch.randn(shape, device="cuda") * 0.1).to(dtype)
        if packed:
            q = q.reshape(batch * sq, 28, 128)
        cache = tuple(
            (torch.randn((pages, 4, page, 128), device="cuda") * 0.1).to(dtype)
            for _ in range(2)
        )
        size = old.get_prims_ts_batch_decode_workspace_size(
            batch,
            28,
            4,
            128,
            page,
            kv_len,
            seq_len_q=1 if packed else sq,
            qo_indptr=offsets,
            max_seq_len_q=sq,
            q_dtype=dtype,
            kv_dtype=dtype,
            out_dtype=output_dtype,
            device=q.device,
        )
        options = dict(
            seq_len_q=1 if packed else sq,
            qo_indptr=offsets,
            max_seq_len_q=sq,
            out_dtype=output_dtype,
        )
        outputs = [torch.empty_like(q, dtype=output_dtype) for _ in range(5)]
        scratch = [
            torch.zeros(size, dtype=torch.uint8, device="cuda") for _ in range(2)
        ]
        wrapper = decode.BatchDecodePagedTSWrapper()
        wrapper.plan(
            q.device,
            batch,
            28,
            4,
            128,
            page,
            kv_len,
            max_seq_len_q=sq,
            packed_query=packed,
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=output_dtype,
        )
        calls = {
            "old_direct": lambda: old.prims_ts_batch_decode_with_kv_cache(
                q, cache, scratch[0], tables, lengths, kv_len, out=outputs[0], **options
            ),
            "new_explicit": lambda: decode.batch_decode_with_paged_kv_cache(
                q,
                cache,
                tables,
                lengths,
                workspace_buffer=scratch[1],
                max_kv_len=kv_len,
                validate=False,
                out=outputs[1],
                **options,
            ),
            "wrapper_control": lambda: wrapper.run(
                q,
                cache,
                lengths,
                tables,
                qo_indptr=offsets,
                out=outputs[2],
                validate=False,
            ),
            "old_convenience": lambda: old.batch_decode_with_paged_kv_cache(
                q, cache, tables, lengths, out=outputs[3], **options
            ),
            "new_convenience": lambda: decode.batch_decode_with_paged_kv_cache(
                q, cache, tables, lengths, out=outputs[4], **options
            ),
        }
    else:
        q = (torch.randn((batch, sq, 32, 576), device="cuda") * 0.1).to(dtype)
        if packed:
            q = q.reshape(batch * sq, 32, 576)
        cache = (torch.randn((pages, page, 576), device="cuda") * 0.1).to(dtype)
        size = old.get_prims_ts_batch_mla_decode_workspace_size(
            batch,
            32,
            512,
            64,
            page,
            kv_len,
            max_seq_len_q=sq,
            q_dtype=dtype,
            kv_dtype=dtype,
            device=q.device,
        )
        shape = (*q.shape[:-1], 512)
        outputs = [
            torch.empty(shape, dtype=torch.bfloat16, device="cuda") for _ in range(5)
        ]
        scratch = [
            torch.empty(size, dtype=torch.uint8, device="cuda") for _ in range(2)
        ]
        options = dict(
            qo_indptr=offsets, max_seq_len_q=sq, bmm1_scale=1 / math.sqrt(576)
        )
        wrapper = mla_decode.BatchMLADecodePagedTSWrapper()
        wrapper.plan(
            q.device,
            batch,
            32,
            512,
            64,
            page,
            kv_len,
            max_seq_len_q=sq,
            packed_query=packed,
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=torch.bfloat16,
        )
        calls = {
            "old_direct": lambda: old.prims_ts_batch_mla_decode_with_kv_cache(
                q,
                cache,
                scratch[0],
                512,
                64,
                tables,
                lengths,
                kv_len,
                out=outputs[0],
                **options,
            ),
            "new_explicit": lambda: mla_decode.batch_mla_decode_with_paged_kv_cache(
                q,
                cache,
                tables,
                lengths,
                workspace_buffer=scratch[1],
                max_kv_len=kv_len,
                validate=False,
                out=outputs[1],
                **options,
            ),
            "old_convenience": lambda: old.batch_mla_decode_with_paged_kv_cache(
                q, cache, tables, lengths, out=outputs[3], **options
            ),
            "new_convenience": lambda: mla_decode.batch_mla_decode_with_paged_kv_cache(
                q, cache, tables, lengths, out=outputs[4], **options
            ),
        }
        # max_seq_len_q belongs to plan, not the reusable wrapper's run().
        calls["wrapper_control"] = lambda: wrapper.run(
            q,
            cache,
            tables,
            lengths,
            qo_indptr=offsets,
            out=outputs[2],
            bmm1_scale=options["bmm1_scale"],
            validate=False,
        )
    for run in calls.values():
        run()
    torch.cuda.synchronize()
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(outputs[3], outputs[4], rtol=0, atol=0)
    return calls


def _wall_us(run, repeats):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    for _ in range(repeats):
        run()
    submitted = time.perf_counter_ns()
    torch.cuda.synchronize()
    completed = time.perf_counter_ns()
    return (submitted - start) / repeats / 1000, (completed - start) / repeats / 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Run packed FP8 short/long-KV cases instead of the core matrix.",
    )
    parser.add_argument("--model", choices=("all", "qwen", "deepseek"), default="all")
    args = parser.parse_args()
    torch.manual_seed(0)
    current = Path(decode.__file__).resolve().parents[3]
    assert _kernel_digest(args.baseline) == _kernel_digest(current), (
        "kernel trees differ"
    )
    old = {
        kind: _load_baseline(args.baseline, module)
        for kind, module in (("qwen", "decode"), ("deepseek", "mla_decode"))
    }
    print(
        json.dumps(
            dict(
                event="environment",
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                baseline=_revision(args.baseline),
                current=_revision(current),
                dirty=bool(
                    subprocess.check_output(
                        ["git", "-C", str(current), "status", "--porcelain"], text=True
                    )
                ),
                gpu_timing="CUDA graph, warm L2, 10 calls/graph",
                rounds=args.rounds,
            )
        ),
        flush=True,
    )
    matrix = [
        (kind, batch, 1536, 1, False, torch.bfloat16)
        for kind in old
        for batch in (1, 64)
    ]
    if args.extended:
        matrix = [
            (kind, 4, kv_len, 4, True, torch.float8_e4m3fn)
            for kind in old
            for kv_len in (257, 4097)
        ]
    matrix = [case for case in matrix if args.model in ("all", case[0])]
    for kind, batch, kv_len, sq, packed, dtype in matrix:
        label = dict(
            model=kind,
            batch=batch,
            kv_len=kv_len,
            sq=sq,
            packed=packed,
            dtype=str(dtype),
        )
        print(json.dumps(dict(event="start", **label)), flush=True)
        calls = _case(kind, batch, kv_len, sq, packed, dtype, old[kind])
        observations = {name: [] for name in calls}
        for round_idx in range(args.rounds):
            order = list(calls) if round_idx % 2 == 0 else list(reversed(calls))
            for name in order:
                run = calls[name]
                gpu_us = None
                if "convenience" not in name:
                    gpu_us = (
                        statistics.median(
                            bench_gpu_time(
                                run,
                                dry_run_iters=10,
                                repeat_iters=100,
                                use_cuda_graph=True,
                                cold_l2_cache=False,
                                num_iters_within_graph=10,
                            )
                        )
                        * 1000
                    )
                host_us, eager_us = _wall_us(run, 30)
                observations[name].append(
                    dict(gpu_us=gpu_us, host_us=host_us, eager_us=eager_us)
                )
        for name, samples in observations.items():
            stats = {
                key: statistics.median([sample[key] for sample in samples])
                for key in ("host_us", "eager_us")
            }
            if samples[0]["gpu_us"] is not None:
                values = [sample["gpu_us"] for sample in samples]
                stats.update(
                    gpu_us=statistics.median(values),
                    gpu_min_us=min(values),
                    gpu_max_us=max(values),
                    tokens_per_s=batch * sq * 1e6 / statistics.median(values),
                )
            print(
                json.dumps(
                    dict(event="result", **label, api=name, **stats, samples=samples)
                ),
                flush=True,
            )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
