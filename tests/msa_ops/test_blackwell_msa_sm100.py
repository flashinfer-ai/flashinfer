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

Canonical public-API correctness coverage for MSA on SM100 and SM103.
"""

import math
import warnings
from typing import Any

import pytest
import torch


BLOCK_SIZE = 128
HEAD_DIM = 128
_FP8 = "float8_e4m3fn"


def _attention_case(
    operation: str,
    *,
    q_dtype: str,
    kv_dtype: str,
    kv_layout: str,
    num_q_heads: int,
    num_kv_heads: int,
    topk: int,
    causal: bool,
    seed: int,
    q_lens: list[int] | None = None,
    kv_lens: list[int] | None = None,
    batch_size: int | None = None,
    seqlen_q: int | None = None,
    seqlen_kv: int | None = None,
    q_offset: list[int] | None = None,
    return_temperature_lse: bool = False,
    lse_temperature_scale: float = 1.0,
    selection_mode: str = "random_valid",
    force_fused: bool | None = None,
    cuda_graph: bool = False,
    use_workspace: bool = True,
) -> dict[str, Any]:
    return {
        "operation": operation,
        "q_dtype": q_dtype,
        "kv_dtype": kv_dtype,
        "kv_layout": kv_layout,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "topk": topk,
        "causal": causal,
        "seed": seed,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "batch_size": batch_size,
        "seqlen_q": seqlen_q,
        "seqlen_kv": seqlen_kv,
        "q_offset": q_offset,
        "return_softmax_lse": True,
        "return_temperature_lse": return_temperature_lse,
        "lse_temperature_scale": lse_temperature_scale,
        "selection_mode": selection_mode,
        "force_fused": force_fused,
        "cuda_graph": cuda_graph,
        "use_workspace": use_workspace,
    }


CASES = [
    pytest.param(
        {
            "operation": "sparse_topk_select",
            "num_heads": 2,
            "max_k_tiles": 64,
            "total_q": 3,
            "topk": 16,
            "num_valid_pages": 47,
            "force_begin_blocks": 2,
            "force_end_blocks": 3,
            "score_mode": "distinct",
            "seed": 11,
        },
        id="topk-small-forced-clamped",
    ),
    pytest.param(
        {
            "operation": "sparse_topk_select",
            "num_heads": 3,
            "max_k_tiles": 64,
            "total_q": 4,
            "topk": 16,
            "num_valid_pages": 47,
            "force_begin_blocks": 2,
            "force_end_blocks": 2,
            "score_mode": "tied_threshold",
            "seed": 12,
        },
        id="topk-tied-threshold-forced",
    ),
    pytest.param(
        {
            "operation": "sparse_topk_select",
            "num_heads": 8,
            "max_k_tiles": 16384,
            "total_q": 2,
            "topk": 16,
            "num_valid_pages": 16381,
            "force_begin_blocks": 0,
            "force_end_blocks": 0,
            "score_mode": "distinct",
            "seed": 13,
        },
        id="topk-large-domain",
    ),
    pytest.param(
        _attention_case(
            "sparse_prefill",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="flat_varlen",
            q_lens=[100, 37],
            kv_lens=[2048, 700],
            num_q_heads=4,
            num_kv_heads=2,
            topk=16,
            causal=False,
            seed=17,
        ),
        id="prefill-flat-bf16-ragged-noncausal",
    ),
    pytest.param(
        _attention_case(
            "sparse_prefill",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="flat_varlen",
            q_lens=[17],
            kv_lens=[1024],
            num_q_heads=16,
            num_kv_heads=1,
            topk=16,
            causal=True,
            seed=18,
            return_temperature_lse=True,
            lse_temperature_scale=0.7,
        ),
        id="prefill-flat-bf16-gqa16-tail",
    ),
    pytest.param(
        _attention_case(
            "sparse_prefill",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="flat_varlen",
            q_lens=[33, 17],
            kv_lens=[1024, 768],
            num_q_heads=16,
            num_kv_heads=1,
            topk=16,
            causal=True,
            seed=20,
        ),
        id="prefill-flat-bf16-gqa16-ragged-batch",
    ),
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="flat_varlen",
            batch_size=148,
            seqlen_q=1,
            seqlen_kv=128,
            num_q_heads=64,
            num_kv_heads=4,
            topk=16,
            causal=True,
            seed=22,
        ),
        id="decode-flat-bf16-batch148-kv128",
    ),
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="flat_varlen",
            batch_size=8,
            seqlen_q=4,
            seqlen_kv=1024,
            num_q_heads=16,
            num_kv_heads=1,
            topk=16,
            causal=True,
            seed=24,
            force_fused=True,
            use_workspace=False,
        ),
        id="decode-flat-bf16-q4-k16",
    ),
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="flat_varlen",
            batch_size=2,
            seqlen_q=4,
            kv_lens=[129, 257],
            num_q_heads=16,
            num_kv_heads=1,
            topk=16,
            causal=True,
            seed=25,
            force_fused=True,
            use_workspace=False,
        ),
        id="decode-flat-bf16-q4-k16-ragged",
    ),
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="flat_varlen",
            batch_size=8,
            seqlen_q=8,
            seqlen_kv=4096,
            num_q_heads=16,
            num_kv_heads=1,
            topk=16,
            causal=True,
            seed=26,
            force_fused=True,
            use_workspace=False,
        ),
        id="decode-flat-bf16-q8-k16",
    ),
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="flat_varlen",
            batch_size=4,
            seqlen_q=16,
            seqlen_kv=1024,
            num_q_heads=16,
            num_kv_heads=1,
            topk=16,
            causal=True,
            seed=28,
            force_fused=True,
        ),
        id="decode-flat-bf16-q16",
    ),
    pytest.param(
        _attention_case(
            "sparse_prefill",
            q_dtype="float16",
            kv_dtype="float16",
            kv_layout="flat_varlen",
            q_lens=[130, 64],
            kv_lens=[1024, 512],
            num_q_heads=2,
            num_kv_heads=2,
            topk=16,
            causal=True,
            seed=19,
            return_temperature_lse=True,
            lse_temperature_scale=0.7,
        ),
        id="prefill-flat-fp16-causal-tail",
    ),
    pytest.param(
        _attention_case(
            "sparse_prefill",
            q_dtype="bfloat16",
            kv_dtype=_FP8,
            kv_layout="paged",
            q_lens=[200],
            kv_lens=[4096],
            q_offset=[3200],
            num_q_heads=8,
            num_kv_heads=2,
            topk=16,
            causal=False,
            seed=23,
        ),
        id="prefill-paged-fp8-gqa",
    ),
    pytest.param(
        _attention_case(
            "sparse_prefill",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="flat_varlen",
            q_lens=[1],
            kv_lens=[256],
            q_offset=[0],
            num_q_heads=1,
            num_kv_heads=1,
            topk=16,
            causal=True,
            seed=29,
            return_temperature_lse=True,
            lse_temperature_scale=0.7,
            selection_mode="future_only",
        ),
        id="prefill-fully-masked-row",
    ),
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="flat_varlen",
            batch_size=8,
            seqlen_q=1,
            kv_lens=[1024, 1152, 1280, 1408, 1536, 1664, 1792, 2048],
            num_q_heads=8,
            num_kv_heads=2,
            topk=16,
            causal=True,
            seed=31,
        ),
        id="decode-flat-bf16-ragged-batch8",
    ),
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype="float16",
            kv_dtype="float16",
            kv_layout="flat_varlen",
            batch_size=4,
            seqlen_q=1,
            kv_lens=[129, 257, 385, 513],
            num_q_heads=8,
            num_kv_heads=2,
            topk=16,
            causal=True,
            seed=35,
        ),
        id="decode-flat-fp16-ragged-nonmultiple64",
    ),
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype="bfloat16",
            kv_dtype=_FP8,
            kv_layout="paged",
            batch_size=8,
            seqlen_q=1,
            kv_lens=[2048, 1920, 1792, 1664, 1536, 1408, 1280, 1152],
            num_q_heads=8,
            num_kv_heads=2,
            topk=16,
            causal=True,
            seed=37,
            force_fused=False,
        ),
        id="decode-paged-fp8",
    ),
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype="bfloat16",
            kv_dtype=_FP8,
            kv_layout="paged",
            batch_size=2,
            seqlen_q=1,
            kv_lens=[512, 384],
            num_q_heads=8,
            num_kv_heads=2,
            topk=16,
            causal=True,
            seed=38,
            force_fused=False,
            cuda_graph=True,
        ),
        id="decode-paged-fp8-graph",
    ),
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="paged",
            batch_size=4,
            seqlen_q=2,
            kv_lens=[1024, 896, 768, 640],
            num_q_heads=16,
            num_kv_heads=2,
            topk=16,
            causal=True,
            seed=41,
            force_fused=True,
        ),
        id="decode-paged-bf16-q2",
    ),
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype=_FP8,
            kv_dtype=_FP8,
            kv_layout="paged",
            batch_size=4,
            seqlen_q=2,
            kv_lens=[1024, 896, 768, 640],
            num_q_heads=16,
            num_kv_heads=2,
            topk=16,
            causal=True,
            seed=61,
            force_fused=True,
        ),
        id="decode-paged-uniform-fp8-q2",
    ),
    # the lean uniform-FP8 tiers (items = batch x seqlen_q x KV heads; the
    # page-table width is ceil(max kv / 128)): one cluster of eight per item,
    # the two-split fp32 pools, one split at each ring depth, the
    # occupancy-2 entry (compute capability 10.3; 10.0 runs the same case on
    # the one-split 11-stage program) and the two-split tier under graph capture
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype=_FP8,
            kv_dtype=_FP8,
            kv_layout="paged",
            batch_size=1,
            seqlen_q=1,
            kv_lens=[2048],
            num_q_heads=32,
            num_kv_heads=4,
            topk=16,
            causal=True,
            seed=81,
            force_fused=True,
        ),
        id="decode-paged-uniform-fp8-lean-s8-cluster",
    ),
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype=_FP8,
            kv_dtype=_FP8,
            kv_layout="paged",
            batch_size=4,
            seqlen_q=8,
            kv_lens=[1200, 1100, 900, 1024],
            num_q_heads=16,
            num_kv_heads=2,
            topk=16,
            causal=True,
            seed=83,
            force_fused=True,
        ),
        id="decode-paged-uniform-fp8-lean-s2-pools",
    ),
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype=_FP8,
            kv_dtype=_FP8,
            kv_layout="paged",
            batch_size=8,
            seqlen_q=8,
            kv_lens=[1000, 777, 1500, 2048, 640, 1024, 1900, 1111],
            num_q_heads=32,
            num_kv_heads=2,
            topk=16,
            causal=True,
            seed=85,
            force_fused=True,
        ),
        id="decode-paged-uniform-fp8-lean-s1-kv11",
    ),
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype=_FP8,
            kv_dtype=_FP8,
            kv_layout="paged",
            batch_size=4,
            seqlen_q=8,
            kv_lens=[16385, 16400, 16390, 16500],
            num_q_heads=32,
            num_kv_heads=4,
            topk=16,
            causal=True,
            seed=87,
            force_fused=True,
        ),
        id="decode-paged-uniform-fp8-lean-s1-kv8-long",
    ),
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype=_FP8,
            kv_dtype=_FP8,
            kv_layout="paged",
            batch_size=8,
            seqlen_q=8,
            kv_lens=[4096, 3000, 2048, 1500, 8192, 5000, 1024, 777],
            num_q_heads=64,
            num_kv_heads=4,
            topk=16,
            causal=True,
            seed=89,
            force_fused=True,
        ),
        id="decode-paged-uniform-fp8-lean-occ2-on-cc103",
    ),
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype=_FP8,
            kv_dtype=_FP8,
            kv_layout="paged",
            batch_size=8,
            seqlen_q=4,
            kv_lens=[1200, 640, 900, 1024, 512, 1100, 777, 1000],
            num_q_heads=16,
            num_kv_heads=2,
            topk=16,
            causal=True,
            seed=91,
            force_fused=True,
            cuda_graph=True,
        ),
        id="decode-paged-uniform-fp8-lean-s2-graph",
    ),
    # the eight-token fused program on its narrow box (32 Q heads over 4 KV
    # heads, eight tokens per request, at least 64 requests, a page table of at
    # most 16 pages = 2048 tokens): the 64-request row and a 128-request row
    # whose longest context sits on the 2048 boundary -- eager calls without a
    # workspace, the only calls that select it
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype=_FP8,
            kv_dtype=_FP8,
            kv_layout="paged",
            batch_size=64,
            seqlen_q=8,
            kv_lens=[750 + (37 * i) % 501 for i in range(64)],
            num_q_heads=32,
            num_kv_heads=4,
            topk=16,
            causal=True,
            seed=93,
            force_fused=True,
            use_workspace=False,
        ),
        id="decode-paged-uniform-fp8-fuseq8-b64",
    ),
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype=_FP8,
            kv_dtype=_FP8,
            kv_layout="paged",
            batch_size=128,
            seqlen_q=8,
            kv_lens=[752 + (31 * i) % 497 for i in range(127)] + [2048],
            num_q_heads=32,
            num_kv_heads=4,
            topk=16,
            causal=True,
            seed=95,
            force_fused=True,
            use_workspace=False,
        ),
        id="decode-paged-uniform-fp8-fuseq8-b128-kv2048",
    ),
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype="float16",
            kv_dtype="float16",
            kv_layout="paged",
            batch_size=4,
            seqlen_q=2,
            kv_lens=[1024, 896, 768, 640],
            num_q_heads=16,
            num_kv_heads=2,
            topk=16,
            causal=True,
            seed=43,
            force_fused=False,
        ),
        id="decode-paged-fp16-q2",
    ),
    pytest.param(
        _attention_case(
            "sparse_prefill",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="paged",
            q_lens=[19, 7],
            kv_lens=[257, 129],
            num_q_heads=4,
            num_kv_heads=2,
            topk=16,
            causal=True,
            seed=47,
        ),
        id="prefill-paged-bf16-ragged",
    ),
    pytest.param(
        _attention_case(
            "sparse_prefill",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="paged",
            q_lens=[17],
            kv_lens=[385],
            num_q_heads=16,
            num_kv_heads=1,
            topk=16,
            causal=True,
            seed=49,
        ),
        id="prefill-paged-bf16-gqa16",
    ),
    pytest.param(
        _attention_case(
            "sparse_prefill",
            q_dtype="float16",
            kv_dtype="float16",
            kv_layout="paged",
            q_lens=[21],
            kv_lens=[257],
            num_q_heads=4,
            num_kv_heads=2,
            topk=16,
            causal=True,
            seed=51,
        ),
        id="prefill-paged-fp16-tail",
    ),
    pytest.param(
        _attention_case(
            "sparse_prefill",
            q_dtype="bfloat16",
            kv_dtype=_FP8,
            kv_layout="flat_varlen",
            q_lens=[23],
            kv_lens=[385],
            num_q_heads=8,
            num_kv_heads=2,
            topk=16,
            causal=True,
            seed=53,
        ),
        id="prefill-flat-fp8-tail",
    ),
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="flat_varlen",
            batch_size=2,
            seqlen_q=2,
            kv_lens=[129, 257],
            num_q_heads=8,
            num_kv_heads=1,
            topk=16,
            causal=True,
            seed=55,
            force_fused=False,
        ),
        id="decode-flat-bf16-ragged-nonmultiple64",
    ),
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype="bfloat16",
            kv_dtype=_FP8,
            kv_layout="flat_varlen",
            batch_size=2,
            seqlen_q=1,
            kv_lens=[129, 257],
            num_q_heads=8,
            num_kv_heads=1,
            topk=16,
            causal=True,
            seed=57,
            force_fused=False,
        ),
        id="decode-flat-fp8-ragged",
    ),
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="paged",
            batch_size=2,
            seqlen_q=1,
            kv_lens=[257, 129],
            num_q_heads=8,
            num_kv_heads=1,
            topk=16,
            causal=True,
            seed=59,
            force_fused=True,
        ),
        id="decode-paged-bf16-m16-ragged",
    ),
    # L2 evict-first paged BF16 decode: eight requests of eight queries over
    # four KV heads (256 work items, under two waves) with a longest context
    # above 40960 tokens; the hinted program on compute capability 10.0 and
    # 10.3, the unhinted one elsewhere, both against the same reference.
    pytest.param(
        _attention_case(
            "sparse_decode",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="paged",
            batch_size=8,
            seqlen_q=8,
            kv_lens=[41088, 40960, 1024, 896, 768, 640, 512, 384],
            num_q_heads=32,
            num_kv_heads=4,
            topk=16,
            causal=True,
            seed=77,
            force_fused=True,
        ),
        id="decode-paged-bf16-q8-gqa8-l2ef-batch8",
    ),
    # Persistent K-major prefill route: paged GQA-16 above 64 pages, paged
    # GQA-8 at 64 pages (captured) and flat GQA-16 above 64 KV blocks, every
    # batch with a request below the long route's 8192-query threshold.
    pytest.param(
        _attention_case(
            "sparse_prefill",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="paged",
            q_lens=[700, 300],
            kv_lens=[8320, 8300],
            num_q_heads=16,
            num_kv_heads=1,
            topk=16,
            causal=True,
            seed=71,
        ),
        id="prefill-paged-bf16-gqa16-kmaj-persistent",
    ),
    pytest.param(
        _attention_case(
            "sparse_prefill",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="paged",
            q_lens=[900, 300],
            kv_lens=[8192, 7000],
            num_q_heads=16,
            num_kv_heads=2,
            topk=16,
            causal=True,
            seed=73,
            cuda_graph=True,
        ),
        id="prefill-paged-bf16-gqa8-kmaj-persistent-graph",
    ),
    pytest.param(
        _attention_case(
            "sparse_prefill",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="flat_varlen",
            q_lens=[1024],
            kv_lens=[8320],
            num_q_heads=16,
            num_kv_heads=1,
            topk=16,
            causal=True,
            seed=75,
        ),
        id="prefill-flat-bf16-gqa16-kmaj-persistent",
    ),
]


def _require_supported_gpu() -> torch.device:
    from flashinfer.utils import get_compute_capability, version_at_least

    if not torch.cuda.is_available():
        pytest.skip("requires an SM100 or SM103 CUDA device")
    device = torch.device("cuda")
    capability = get_compute_capability(device)
    minimum_cuda = {(10, 0): "12.8", (10, 3): "12.9"}.get(capability)
    if minimum_cuda is None:
        pytest.skip("requires compute capability 10.0 or 10.3")
    cuda_version = torch.version.cuda
    if cuda_version is None:
        pytest.skip("requires a CUDA-enabled PyTorch build")
    if not version_at_least(cuda_version, minimum_cuda):
        pytest.skip(
            f"compute capability {capability[0]}.{capability[1]} requires "
            f"CUDA {minimum_cuda} or newer"
        )
    return device


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        _FP8: torch.float8_e4m3fn,
    }[name]


def _output_dtype(q: torch.Tensor) -> torch.dtype:
    """Uniform FP8 Q/K/V decode returns BF16; every other route returns q's dtype."""
    return torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype


def _indptr(lengths: list[int], device: torch.device) -> torch.Tensor:
    result = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    result[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    return result


def _make_topk_inputs(case: dict[str, Any], device: torch.device) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(case["seed"])
    shape = (case["num_heads"], case["max_k_tiles"], case["total_q"])
    if case["score_mode"] == "distinct":
        scores = torch.randn(shape, generator=generator, device=device).float()
        scores += torch.arange(shape[1], device=device).view(1, -1, 1) * 1.0e-6
    else:
        levels = (torch.arange(shape[1], device=device) % 4).float()
        scores = levels.view(1, -1, 1).expand(shape).clone()
    scores[:, case["num_valid_pages"] :, :] = float("-inf")
    return {**case, "max_score": scores.contiguous()}


def _make_q2k(
    *,
    q_lens: list[int],
    kv_lens: list[int],
    kv_heads: int,
    topk: int,
    causal: bool,
    q_offset: torch.Tensor | None,
    selection_mode: str,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    output = torch.full((kv_heads, sum(q_lens), topk), -1, dtype=torch.int32)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q_start = 0
    for batch, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens, strict=True)):
        offset = kv_len - q_len if q_offset is None else int(q_offset[batch].item())
        all_blocks = math.ceil(kv_len / BLOCK_SIZE)
        for local_q in range(q_len):
            for kv_head in range(kv_heads):
                if selection_mode == "future_only":
                    candidates = torch.tensor([1], dtype=torch.int64)
                else:
                    visible_tokens = offset + local_q + 1 if causal else kv_len
                    visible_blocks = max(
                        0,
                        min(all_blocks, math.ceil(visible_tokens / BLOCK_SIZE)),
                    )
                    candidates = torch.randperm(visible_blocks, generator=generator)
                count = min(topk, candidates.numel())
                output[kv_head, q_start + local_q, :count] = (
                    candidates[:count].sort().values.to(torch.int32)
                )
        q_start += q_len
    return output.to(device).contiguous()


def _make_paged_cache(
    logical_k: torch.Tensor,
    logical_v: torch.Tensor,
    kv_lens: list[int],
    kv_heads: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    page_counts = [math.ceil(length / BLOCK_SIZE) for length in kv_lens]
    total_pages = sum(page_counts)
    k = torch.zeros(
        (total_pages, kv_heads, BLOCK_SIZE, HEAD_DIM),
        dtype=logical_k.dtype,
        device=device,
    )
    v = torch.zeros_like(k)
    page_table = torch.full(
        (len(kv_lens), max(page_counts)), -1, dtype=torch.int32, device=device
    )
    physical_ids = list(reversed(range(total_pages)))
    token_start = 0
    logical_page = 0
    for batch, kv_len in enumerate(kv_lens):
        for block in range(page_counts[batch]):
            physical_page = physical_ids[logical_page]
            page_table[batch, block] = physical_page
            token_count = min(BLOCK_SIZE, kv_len - block * BLOCK_SIZE)
            source = slice(
                token_start + block * BLOCK_SIZE,
                token_start + block * BLOCK_SIZE + token_count,
            )
            k[physical_page, :, :token_count] = logical_k[source].transpose(0, 1)
            v[physical_page, :, :token_count] = logical_v[source].transpose(0, 1)
            logical_page += 1
        token_start += kv_len
    return k.contiguous(), v.contiguous(), page_table.contiguous()


def _make_attention_inputs(
    case: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    if case["operation"] == "sparse_prefill":
        q_lens = case["q_lens"]
    else:
        q_lens = [case["seqlen_q"]] * case["batch_size"]
    if case["kv_lens"] is not None:
        kv_lens = case["kv_lens"]
    else:
        kv_lens = [case["seqlen_kv"]] * len(q_lens)

    generator = torch.Generator(device=device).manual_seed(case["seed"])
    q = (
        torch.randn(
            (sum(q_lens), case["num_q_heads"], HEAD_DIM),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        / 3.0
    ).to(_dtype(case["q_dtype"]))
    logical_k = (
        torch.randn(
            (sum(kv_lens), case["num_kv_heads"], HEAD_DIM),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        / 3.0
    ).to(_dtype(case["kv_dtype"]))
    logical_v = (
        torch.randn(
            (sum(kv_lens), case["num_kv_heads"], HEAD_DIM),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        / 3.0
    ).to(_dtype(case["kv_dtype"]))
    cu_q = _indptr(q_lens, device)
    cu_k = _indptr(kv_lens, device)
    q_offset = (
        None
        if case["q_offset"] is None
        else torch.tensor(case["q_offset"], dtype=torch.int32, device=device)
    )
    q2k = _make_q2k(
        q_lens=q_lens,
        kv_lens=kv_lens,
        kv_heads=case["num_kv_heads"],
        topk=case["topk"],
        causal=case["causal"],
        q_offset=q_offset,
        selection_mode=case["selection_mode"],
        seed=case["seed"] + 101,
        device=device,
    )

    page_table = None
    seqused_k = None
    if case["kv_layout"] == "paged":
        k, v, page_table = _make_paged_cache(
            logical_k,
            logical_v,
            kv_lens,
            case["num_kv_heads"],
            device,
        )
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    else:
        k, v = logical_k.contiguous(), logical_v.contiguous()

    return {
        **case,
        "q": q.contiguous(),
        "k": k,
        "v": v,
        "logical_k": logical_k.float(),
        "logical_v": logical_v.float(),
        "q2k_indices": q2k,
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
        "page_table": page_table,
        "seqused_k": seqused_k,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "q_offset": q_offset,
    }


def _reference_topk(inputs: dict[str, Any]) -> torch.Tensor:
    scores = inputs["max_score"]
    heads, _, total_q = scores.shape
    topk = inputs["topk"]
    valid = inputs["num_valid_pages"]
    forced = list(range(inputs["force_begin_blocks"]))
    forced.extend(range(valid - inputs["force_end_blocks"], valid))
    forced = sorted(set(forced))
    candidates = torch.tensor(
        [index for index in range(valid) if index not in set(forced)],
        dtype=torch.long,
        device=scores.device,
    )
    output = torch.full(
        (total_q, heads, topk), -1, dtype=torch.int32, device=scores.device
    )
    remaining = topk - len(forced)
    for query in range(total_q):
        for head in range(heads):
            chosen = list(forced)
            if remaining:
                positions = torch.topk(
                    scores[head, candidates, query], remaining, sorted=False
                ).indices
                chosen.extend(candidates[positions].tolist())
            chosen.sort()
            output[query, head] = torch.tensor(
                chosen, dtype=torch.int32, device=scores.device
            )
    return output


def _reference_attention(inputs: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    q = inputs["q"]
    q_lens = inputs["q_lens"]
    kv_lens = inputs["kv_lens"]
    batch_size = len(q_lens)
    max_q = max(q_lens)
    max_k = max(kv_lens)
    q_heads = inputs["num_q_heads"]
    kv_heads = inputs["num_kv_heads"]
    group_size = q_heads // kv_heads
    device = q.device

    q_padded = torch.zeros(
        (batch_size, max_q, q_heads, HEAD_DIM), dtype=torch.float32, device=device
    )
    k_padded = torch.zeros(
        (batch_size, max_k, kv_heads, HEAD_DIM), dtype=torch.float32, device=device
    )
    v_padded = torch.zeros_like(k_padded)
    selections = torch.full(
        (batch_size, max_q, kv_heads, inputs["topk"]),
        -1,
        dtype=torch.int32,
        device=device,
    )
    q_start = 0
    kv_start = 0
    for batch, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens, strict=True)):
        q_padded[batch, :q_len] = q[q_start : q_start + q_len].float()
        k_padded[batch, :kv_len] = inputs["logical_k"][kv_start : kv_start + kv_len]
        v_padded[batch, :kv_len] = inputs["logical_v"][kv_start : kv_start + kv_len]
        selections[batch, :q_len] = inputs["q2k_indices"][
            :, q_start : q_start + q_len
        ].permute(1, 0, 2)
        q_start += q_len
        kv_start += kv_len

    token_ids = torch.arange(max_k, device=device)
    block_ids = token_ids // BLOCK_SIZE
    allowed = (block_ids.view(1, 1, 1, max_k, 1) == selections.unsqueeze(-2)).any(-1)
    valid_q = torch.arange(max_q, device=device).view(1, max_q, 1, 1) < torch.tensor(
        q_lens, device=device
    ).view(batch_size, 1, 1, 1)
    valid_k = token_ids.view(1, 1, 1, max_k) < torch.tensor(
        kv_lens, device=device
    ).view(batch_size, 1, 1, 1)
    allowed &= valid_q & valid_k
    if inputs["causal"]:
        if inputs["q_offset"] is None:
            offsets = torch.tensor(
                [kv_len - q_len for q_len, kv_len in zip(q_lens, kv_lens, strict=True)],
                device=device,
            )
        else:
            offsets = inputs["q_offset"].to(torch.long)
        q_positions = offsets.view(batch_size, 1) + torch.arange(
            max_q, device=device
        ).view(1, max_q)
        allowed &= token_ids.view(1, 1, 1, max_k) <= q_positions.view(
            batch_size, max_q, 1, 1
        )

    out = torch.zeros_like(q_padded)
    lse = torch.full(
        (batch_size, max_q, q_heads),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    temperature_lse = torch.full_like(lse, float("-inf"))
    scale = HEAD_DIM**-0.5
    for kv_head in range(kv_heads):
        head_start = kv_head * group_size
        head_end = head_start + group_size
        logits = (
            torch.einsum(
                "bqgd,bkd->bqgk",
                q_padded[:, :, head_start:head_end],
                k_padded[:, :, kv_head],
            )
            * scale
        )
        mask = allowed[:, :, kv_head].unsqueeze(2)
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        has_tokens = mask.any(-1)
        probabilities = torch.softmax(masked_logits, dim=-1)
        probabilities = torch.where(
            has_tokens.unsqueeze(-1), probabilities, torch.zeros_like(probabilities)
        )
        out[:, :, head_start:head_end] = torch.einsum(
            "bqgk,bkd->bqgd", probabilities, v_padded[:, :, kv_head]
        )
        lse[:, :, head_start:head_end] = torch.logsumexp(masked_logits, dim=-1)
        scaled_logits = (logits * inputs["lse_temperature_scale"]).masked_fill(
            ~mask, float("-inf")
        )
        temperature_lse[:, :, head_start:head_end] = torch.logsumexp(
            scaled_logits, dim=-1
        )

    out_rows = [out[batch, :q_len] for batch, q_len in enumerate(q_lens)]
    lse_rows = [lse[batch, :q_len] for batch, q_len in enumerate(q_lens)]
    temperature_rows = [
        temperature_lse[batch, :q_len] for batch, q_len in enumerate(q_lens)
    ]
    return (
        torch.cat(out_rows).to(_output_dtype(q)),
        torch.cat(lse_rows),
        torch.cat(temperature_rows),
    )


def _assert_tied_topk(inputs: dict[str, Any], actual: torch.Tensor) -> None:
    scores = inputs["max_score"]
    valid = inputs["num_valid_pages"]
    topk = inputs["topk"]
    forced = set(range(inputs["force_begin_blocks"]))
    forced.update(range(valid - inputs["force_end_blocks"], valid))
    candidates = [index for index in range(valid) if index not in forced]
    for query in range(actual.shape[0]):
        for head in range(actual.shape[1]):
            selected = actual[query, head].tolist()
            selected_set = set(selected)
            assert selected == sorted(selected)
            assert len(selected_set) == topk
            assert all(0 <= index < valid for index in selected)
            assert forced.issubset(selected_set)
            selected_candidates = [index for index in selected if index not in forced]
            remaining = topk - len(forced)
            threshold = torch.topk(
                scores[head, candidates, query], remaining, sorted=False
            ).values.min()
            assert all(
                scores[head, index, query] >= threshold for index in selected_candidates
            )
            unselected = [index for index in candidates if index not in selected_set]
            if unselected:
                assert min(
                    scores[head, index, query] for index in selected_candidates
                ) >= max(scores[head, index, query] for index in unselected)


def _assert_lse_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    finite = torch.isfinite(expected)
    assert torch.isfinite(actual[finite]).all()
    torch.testing.assert_close(actual[finite], expected[finite], atol=0.05, rtol=0.01)


def _invoke_attention(
    inputs: dict[str, Any], workspace: Any, prefill: Any, decode: Any
) -> tuple[torch.Tensor, ...]:
    if inputs["operation"] == "sparse_prefill":
        value = prefill(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["q2k_indices"],
            inputs["cu_seqlens_q"],
            inputs["cu_seqlens_k"],
            causal=inputs["causal"],
            page_table=inputs["page_table"],
            seqused_k=inputs["seqused_k"],
            return_softmax_lse=inputs["return_softmax_lse"],
            q_offset=inputs["q_offset"],
            return_temperature_lse=inputs["return_temperature_lse"],
            lse_temperature_scale=inputs["lse_temperature_scale"],
            workspace=workspace,
        )
    else:
        value = decode(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["q2k_indices"],
            page_table=inputs["page_table"],
            seqused_k=inputs["seqused_k"],
            cu_seqlens_k=inputs["cu_seqlens_k"],
            seqlen_q=inputs["seqlen_q"],
            causal=inputs["causal"],
            return_softmax_lse=inputs["return_softmax_lse"],
            q_offset=inputs["q_offset"],
            force_fused=inputs["force_fused"],
            workspace=workspace,
        )
    if isinstance(value, tuple):
        return value
    return (value,)


@pytest.mark.parametrize("case", CASES)
def test_sm100_sm103_msa_public_api_correctness(case: dict[str, Any]) -> None:
    device = _require_supported_gpu()
    from flashinfer.msa_ops import (
        MSASparseAttentionWorkspace,
        msa_sparse_attention,
        msa_sparse_decode_attention,
        msa_topk_select,
    )

    if case["operation"] == "sparse_topk_select":
        inputs = _make_topk_inputs(case, device)
        actual = msa_topk_select(
            inputs["max_score"],
            inputs["topk"],
            num_valid_pages=inputs["num_valid_pages"],
            force_begin_blocks=inputs["force_begin_blocks"],
            force_end_blocks=inputs["force_end_blocks"],
        )
        assert actual.shape == (
            inputs["total_q"],
            inputs["num_heads"],
            inputs["topk"],
        )
        assert actual.dtype == torch.int32
        if inputs["score_mode"] == "tied_threshold":
            _assert_tied_topk(inputs, actual)
        else:
            torch.testing.assert_close(actual, _reference_topk(inputs), atol=0, rtol=0)
        return

    inputs = _make_attention_inputs(case, device)
    workspace = (
        MSASparseAttentionWorkspace(device=device) if inputs["use_workspace"] else None
    )
    run = lambda: _invoke_attention(  # noqa: E731
        inputs, workspace, msa_sparse_attention, msa_sparse_decode_attention
    )
    if inputs["cuda_graph"]:
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                run()
        capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            actual = run()
        # FP8 tensors have no in-place CUDA arithmetic: perturb in float, store back.
        inputs["q"].copy_((inputs["q"].float() * 0.5 + 0.125).to(inputs["q"].dtype))
        graph.replay()
        torch.cuda.synchronize()
    else:
        actual = run()

    expected = _reference_attention(inputs)
    assert actual[0].shape == inputs["q"].shape
    assert actual[0].dtype == _output_dtype(inputs["q"])
    output_tolerance = 0.1 if inputs["kv_dtype"] == _FP8 else 0.01
    torch.testing.assert_close(
        actual[0], expected[0], atol=output_tolerance, rtol=output_tolerance
    )
    assert actual[1].shape == expected[1].shape
    assert actual[1].dtype == torch.float32
    _assert_lse_close(actual[1], expected[1])
    if inputs["return_temperature_lse"]:
        assert actual[2].shape == expected[2].shape
        assert actual[2].dtype == torch.float32
        _assert_lse_close(actual[2], expected[2])


# The narrow-box shape under CUDA graph capture: captured with every context at
# most 2048 tokens, replayed after ``seqused_k`` and the selections grow past
# the fused program's context bound (6400-8192 tokens, and above 8192).  The
# capture keeps the lean tier, whose context bound is the page-table width x
# 128 tokens -- sized by the grown contexts, read by no call -- so the replay
# matches the reference at the grown contexts.
@pytest.mark.parametrize(
    "grown_base",
    [6500, 9000],
    ids=["replay-kv6500-7999", "replay-kv9000-10499"],
)
def test_uniform_fp8_narrow_box_graph_replays_after_the_contexts_grow(grown_base):
    device = _require_supported_gpu()
    from flashinfer.msa_ops import (
        MSASparseAttentionWorkspace,
        msa_sparse_decode_attention,
    )

    batch, seqlen_q = 64, 8
    grown_lens = [grown_base + (53 * i) % 1500 for i in range(batch)]
    captured_lens = [750 + (37 * i) % 501 for i in range(batch)]
    case = _attention_case(
        "sparse_decode",
        q_dtype=_FP8,
        kv_dtype=_FP8,
        kv_layout="paged",
        batch_size=batch,
        seqlen_q=seqlen_q,
        kv_lens=grown_lens,
        num_q_heads=32,
        num_kv_heads=4,
        topk=16,
        causal=True,
        seed=97,
        force_fused=True,
        cuda_graph=True,
    )
    # the grown state sizes the cache and the page table; the capture runs on
    # the same tensors holding the short contexts and their selections
    inputs = _make_attention_inputs(case, device)
    assert inputs["page_table"].shape[1] * BLOCK_SIZE >= max(grown_lens)
    grown_seqused = inputs["seqused_k"].clone()
    grown_q2k = inputs["q2k_indices"].clone()
    captured_q2k = _make_q2k(
        q_lens=inputs["q_lens"],
        kv_lens=captured_lens,
        kv_heads=case["num_kv_heads"],
        topk=case["topk"],
        causal=case["causal"],
        q_offset=None,
        selection_mode=case["selection_mode"],
        seed=case["seed"] + 202,
        device=device,
    )
    inputs["seqused_k"].copy_(
        torch.tensor(captured_lens, dtype=torch.int32, device=device)
    )
    inputs["q2k_indices"].copy_(captured_q2k)

    workspace = MSASparseAttentionWorkspace(device=device)
    run = lambda: _invoke_attention(  # noqa: E731
        inputs, workspace, None, msa_sparse_decode_attention
    )
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            run()
    capture_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        actual = run()

    inputs["seqused_k"].copy_(grown_seqused)
    inputs["q2k_indices"].copy_(grown_q2k)
    graph.replay()
    torch.cuda.synchronize()

    expected = _reference_attention(inputs)
    torch.testing.assert_close(actual[0], expected[0], atol=0.1, rtol=0.1)
    _assert_lse_close(actual[1], expected[1])


# The two-split lean tier without a workspace: its completion counter is cached
# per (device, stream), zeroed once when created and reset by the program after
# each item's merge, so an eager call is one kernel and repeated calls on one
# stream reuse one counter.  The shape is the hq64/hkv4 eight-token decode
# (items = batch x seqlen_q x 4), its batch chosen on the device's SM count.
def _current_device() -> torch.device:
    _require_supported_gpu()
    return torch.device("cuda", torch.cuda.current_device())


def _two_split_inputs(
    device: torch.device, *, batch: int, seqlen_q: int, kv_base: int, seed: int
) -> dict[str, Any]:
    case = _attention_case(
        "sparse_decode",
        q_dtype=_FP8,
        kv_dtype=_FP8,
        kv_layout="paged",
        batch_size=batch,
        seqlen_q=seqlen_q,
        kv_lens=[kv_base + (97 * i) % 401 for i in range(batch)],
        num_q_heads=64,
        num_kv_heads=4,
        topk=16,
        causal=True,
        seed=seed,
        force_fused=True,
        use_workspace=False,
    )
    return _make_attention_inputs(case, device)


def _lean_plan(backend: Any, inputs: dict[str, Any]) -> Any:
    device = inputs["q"].device
    max_pages = int(inputs["page_table"].shape[1])
    return backend._lean_launch_plan(
        target=backend._select_target(device),
        items=int(inputs["q"].shape[0]) * int(inputs["k"].shape[1]),
        num_sms=backend._num_sms(device),
        max_pages=max_pages,
        max_kv_len=backend._lean_kv_len_bound(max_pages),
    )


def _two_split_batch(backend: Any, device: torch.device, seqlen_q: int) -> int:
    """The smallest batch whose launch plan has two splits on this device."""
    for batch in range(1, 65):
        plan = backend._lean_launch_plan(
            target=backend._select_target(device),
            items=batch * seqlen_q * 4,
            num_sms=backend._num_sms(device),
            max_pages=10,
            max_kv_len=backend._lean_kv_len_bound(10),
        )
        if plan.splits == 2:
            return batch
    raise AssertionError(f"no batch has two splits at seqlen_q {seqlen_q}")


def _counter_storage(backend: Any, device: torch.device) -> torch.Tensor:
    """The whole cached counter of ``device``'s current stream."""
    key = (device.index, backend._stream_ptr(device), 0)
    return backend._lean_completion_counters[key]


def _assert_matches_reference(
    actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]
) -> None:
    torch.testing.assert_close(actual[0], expected[0], atol=0.1, rtol=0.1)
    _assert_lse_close(actual[1], expected[1])


def test_uniform_fp8_two_split_eager_calls_are_one_kernel_on_one_counter():
    device = _current_device()
    from torch.profiler import ProfilerActivity, profile

    from flashinfer.msa_ops import _blackwell_sm100 as backend
    from flashinfer.msa_ops import msa_sparse_decode_attention

    seqlen_q = 8
    batch = _two_split_batch(backend, device, seqlen_q)
    sets = [
        _two_split_inputs(
            device, batch=batch, seqlen_q=seqlen_q, kv_base=kv_base, seed=seed
        )
        for kv_base, seed in ((800, 131), (3706, 137))
    ]
    for inputs in sets:
        plan = _lean_plan(backend, inputs)
        assert plan.splits == 2, (batch, plan)
    expected = [_reference_attention(inputs) for inputs in sets]
    run = lambda inputs: _invoke_attention(  # noqa: E731
        inputs, None, None, msa_sparse_decode_attention
    )
    for inputs in sets:  # load the program outside the profiled window
        run(inputs)
    torch.cuda.synchronize()

    calls = 200
    outputs = []
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for call in range(calls):
            outputs.append(run(sets[call % 2]))
        torch.cuda.synchronize()
    for call, actual in enumerate(outputs):
        _assert_matches_reference(actual, expected[call % 2])
    counter = _counter_storage(backend, device)
    assert counter.numel() == backend._num_sms(device) // 2
    assert int(counter.abs().sum()) == 0, "every merge reset its counter"
    kernels = [
        event.name
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert len(kernels) == calls, sorted(set(kernels))
    assert len(set(kernels)) == 1, sorted(set(kernels))


def test_uniform_fp8_two_split_calls_on_two_streams_overlap_on_two_counters():
    device = _current_device()
    from flashinfer.msa_ops import _blackwell_sm100 as backend
    from flashinfer.msa_ops import msa_sparse_decode_attention

    seqlen_q = 8
    batch = _two_split_batch(backend, device, seqlen_q)
    sets = [
        _two_split_inputs(
            device, batch=batch, seqlen_q=seqlen_q, kv_base=kv_base, seed=seed
        )
        for kv_base, seed in ((800, 139), (7273, 149))
    ]
    for inputs in sets:
        assert _lean_plan(backend, inputs).splits == 2
    expected = [_reference_attention(inputs) for inputs in sets]
    run = lambda inputs: _invoke_attention(  # noqa: E731
        inputs, None, None, msa_sparse_decode_attention
    )
    streams = [torch.cuda.Stream(device=device) for _ in sets]
    assert streams[0].cuda_stream != streams[1].cuda_stream
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream(device))
    outputs: list[list[tuple[torch.Tensor, ...]]] = [[], []]
    # no synchronization between the calls: both streams first spin for longer
    # than two host calls take, so their launches become ready together
    for _ in range(50):
        for stream in streams:
            with torch.cuda.stream(stream):
                torch.cuda._sleep(2_000_000)
        for index, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                outputs[index].append(run(sets[index]))
    torch.cuda.synchronize()
    for index in range(2):
        for actual in outputs[index]:
            _assert_matches_reference(actual, expected[index])
    counters = []
    for stream in streams:
        with torch.cuda.stream(stream):
            counters.append(_counter_storage(backend, device))
    assert counters[0].data_ptr() != counters[1].data_ptr()
    for counter in counters:
        assert int(counter.abs().sum()) == 0


def test_uniform_fp8_two_split_counter_tail_stays_zero_across_item_counts():
    device = _current_device()
    from flashinfer.msa_ops import _blackwell_sm100 as backend
    from flashinfer.msa_ops import msa_sparse_decode_attention

    # one-token requests over four KV heads: items = 4 x batch.  64 and 32
    # items, then the fewest and the most items two splits take on this
    # device (num_sms // 4 < items <= num_sms // 2); a count that resolves
    # more splits never reads the counter
    capacity = backend._num_sms(device) // 2
    fewest = (backend._num_sms(device) // 4) // 4 * 4 + 4
    most = capacity // 4 * 4
    sequence = list(dict.fromkeys([64, 32, fewest, most]))
    run = lambda inputs: _invoke_attention(  # noqa: E731
        inputs, None, None, msa_sparse_decode_attention
    )
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    two_split = []
    with torch.cuda.stream(stream):
        for seed, items in enumerate(sequence):
            inputs = _two_split_inputs(
                device, batch=items // 4, seqlen_q=1, kv_base=900, seed=151 + seed
            )
            plan = _lean_plan(backend, inputs)
            if items in (fewest, most):
                assert plan.splits == 2, (items, plan)
            if plan.splits == 2:
                two_split.append(items)
            actual = run(inputs)
            stream.synchronize()
            _assert_matches_reference(actual, _reference_attention(inputs))
            if two_split:
                counter = _counter_storage(backend, device)
                assert counter.numel() == capacity
                assert int(counter.abs().sum()) == 0, (items, counter.tolist())
    assert len(two_split) >= 2, two_split


def test_uniform_fp8_capture_without_a_workspace_still_raises():
    device = _current_device()
    from flashinfer.msa_ops import _blackwell_sm100 as backend
    from flashinfer.msa_ops import msa_sparse_decode_attention

    seqlen_q = 8
    batch = _two_split_batch(backend, device, seqlen_q)
    inputs = _two_split_inputs(
        device, batch=batch, seqlen_q=seqlen_q, kv_base=800, seed=157
    )
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    cached = dict(backend._lean_completion_counters)
    graph = torch.cuda.CUDAGraph()
    refused = pytest.raises(RuntimeError, match="requires an explicit")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the refused capture leaves an empty graph
        with refused, torch.cuda.graph(graph, stream=stream):
            _invoke_attention(inputs, None, None, msa_sparse_decode_attention)
    torch.cuda.synchronize()
    assert backend._lean_completion_counters == cached, "capture reached no cache"


def test_uniform_fp8_two_split_on_a_second_device_uses_its_own_counter():
    first = _require_supported_gpu()
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    from flashinfer.msa_ops import _blackwell_sm100 as backend
    from flashinfer.msa_ops import msa_sparse_decode_attention
    from flashinfer.utils import get_compute_capability

    devices = [torch.device("cuda", 0), torch.device("cuda", 1)]
    if get_compute_capability(devices[1]) != get_compute_capability(first):
        pytest.skip("requires two devices of one compute capability")
    seqlen_q = 8
    run = lambda inputs: _invoke_attention(  # noqa: E731
        inputs, None, None, msa_sparse_decode_attention
    )
    counters = []
    with torch.cuda.device(0):
        # device 0 is current for every call, including those on cuda:1
        for device in reversed(devices):
            batch = _two_split_batch(backend, device, seqlen_q)
            inputs = _two_split_inputs(
                device, batch=batch, seqlen_q=seqlen_q, kv_base=800, seed=163
            )
            assert _lean_plan(backend, inputs).splits == 2
            expected = _reference_attention(inputs)
            for _ in range(20):
                actual = run(inputs)
                assert actual[0].device == device
                _assert_matches_reference(actual, expected)
            torch.cuda.synchronize(device)
            counter = _counter_storage(backend, device)
            assert counter.device == device
            assert int(counter.abs().sum()) == 0
            counters.append(counter)
            assert torch.cuda.current_device() == 0
    assert counters[0].data_ptr() != counters[1].data_ptr()
