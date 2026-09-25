# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Complete FlashInfer vs pinned vLLM MiniMax-M3 decode wrapper, without a model.

Run from a FlashInfer checkout with Triton installed::

    python benchmarks/bench_minimax_m3_sparse_decode.py \
        --reference /path/to/pinned/sparse_attn.py --output minimax_m3.json

The reference is the unmodified file at vLLM commit
866fea2b9900bf49d552c205d2eaac4716fb63ac. No vLLM install/server is needed.
CUDA graph timing includes metadata, scale preparation, native attention and
reduction, and all kernels in the complete Triton wrapper. It excludes Python
dispatch and one-time workspace allocation/JIT. Repeated input buffers are warm.
Each workspace is captured once; a timed sample replays that single-call graph
32 times. This differs from the earlier standalone wrapper's 32-call capture.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import types
from unittest.mock import patch

import torch
import triton
import triton.language as tl

from flashinfer.msa_ops import msa_sparse_decode_attention

# The benchmark shares reproducible *input generation*, not its correctness
# reference, with the tests. No CPU/GPU gathers occur in either timed callable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.msa_ops.test_minimax_m3 import (  # noqa: E402
    assert_numerics,
    make_case,
    packed_kv_cache,
)


def load_reference(path):
    path = Path(path)
    expected_sha = "5db5d5e692044e54bb57418cbb7dcf5c9cc94f789eaccf128eec6fa8df86d967"
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("reference must be the unmodified pinned vLLM sparse_attn.py")
    platform = types.ModuleType("vllm.platforms")
    platform.current_platform = types.SimpleNamespace(is_arch_support_pdl=lambda: True)
    utils = types.ModuleType("vllm.triton_utils")
    utils.triton, utils.tl = triton, tl
    spec = importlib.util.spec_from_file_location("minimax_m3_pinned_reference", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules, {"vllm.platforms": platform, "vllm.triton_utils": utils}
    ):
        spec.loader.exec_module(module)
    return module.minimax_m3_sparse_attn_decode


def capture_call(fn, stream):
    """Capture once: an MSA workspace belongs to one captured invocation."""
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    return graph


def graph_times_us(graph, inner=32, repeats=12):
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = []
    for _ in range(repeats):
        start.record()
        for _ in range(inner):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / inner)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ragged",
        action="store_true",
        help="Vary valid lengths below the context capacity (default: uniform lengths)",
    )
    parser.add_argument(
        "--no-prefix-sharing",
        action="store_true",
        help="Use disjoint physical pages (default: 50% logical prefix shared)",
    )
    parser.add_argument(
        "--strided-indices",
        action="store_true",
        help="Use the actual indexer's capacity-buffer slice layout",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--query-lens", type=int, nargs="+", default=[4])
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[8192, 100000, 200000]
    )
    parser.add_argument(
        "--num-kv-heads", type=int, nargs="+", choices=[1, 4], default=[1, 4]
    )
    args = parser.parse_args()
    reference = load_reference(args.reference)
    results = []
    for hkv in args.num_kv_heads:
        for context in args.contexts:
            for qlen in args.query_lens:
                for batch in args.batch_sizes:
                    case = make_case(
                        batch,
                        qlen,
                        hkv,
                        context,
                        shared=not args.no_prefix_sharing,
                        ragged=args.ragged,
                    )
                    if args.strided_indices:
                        total = batch * qlen
                        storage = torch.empty(
                            hkv, total * 2 + 7, 16, dtype=torch.int32, device="cuda"
                        )
                        view = storage[:, :total, :]
                        view.copy_(case["q2k_indices"])
                        case["q2k_indices"] = view
                    ref = torch.empty_like(case["out"])
                    packed = packed_kv_cache(case)

                    def candidate(case=case):
                        return msa_sparse_decode_attention(**case)

                    def baseline(case=case, ref=ref, packed=packed):
                        return reference(
                            case["q"],
                            packed,
                            case["q2k_indices"],
                            case["page_table"],
                            case["seqused_k"],
                            hkv,
                            128**-0.5,
                            ref,
                            qlen,
                            case["k_scale"],
                            case["v_scale"],
                        )

                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        candidate()
                        baseline()
                    torch.cuda.synchronize()
                    assert_numerics(case["out"], ref)
                    fi_graph = capture_call(candidate, stream)
                    tr_graph = capture_call(baseline, stream)
                    fi_a = graph_times_us(fi_graph)
                    tr_a = graph_times_us(tr_graph)
                    tr_b = graph_times_us(tr_graph)
                    fi_b = graph_times_us(fi_graph)
                    fi_us, tr_us = (
                        statistics.median(fi_a + fi_b),
                        statistics.median(tr_a + tr_b),
                    )
                    row = dict(
                        batch=batch,
                        qlen=qlen,
                        hq=hkv * 16,
                        hkv=hkv,
                        context=context,
                        ragged=args.ragged,
                        prefix_shared=not args.no_prefix_sharing,
                        seq_len_min=case["seqused_k"].min().item(),
                        seq_len_max=case["seqused_k"].max().item(),
                        topk_strides=list(case["q2k_indices"].stride()),
                        flashinfer_us=fi_us,
                        triton_us=tr_us,
                        speedup=tr_us / fi_us,
                        flashinfer_samples_us=fi_a + fi_b,
                        triton_samples_us=tr_a + tr_b,
                    )
                    results.append(row)
                    print(json.dumps(row), flush=True)
                    args.output.write_text(
                        json.dumps(
                            dict(
                                gpu=torch.cuda.get_device_name(),
                                torch=torch.__version__,
                                reference_commit="866fea2b9900bf49d552c205d2eaac4716fb63ac",
                                timing="CUDA events, single-call graphs replayed 32 times per sample, ABBA, warm repeated buffers",
                                results=results,
                            ),
                            indent=2,
                        )
                    )
                    del case, packed, ref, candidate, baseline, fi_graph, tr_graph
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
