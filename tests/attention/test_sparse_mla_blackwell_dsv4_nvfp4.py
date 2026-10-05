# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek-V4 NVFP4 sparse-MLA cache on SM100 / SM103 (B200 / GB300).

The NVFP4 cache helpers (``nvfp4_quantize_pack_sparse_mla_cache`` /
``nvfp4_quantize_append_sparse_mla_cache``) are one implementation shared with the
SM120 ``backend="sparse"`` path; these tests pin the packed bytes against the pure
torch reference on the Blackwell datacenter parts, where the cache is consumed by
``backend="cake"``.
"""

import pytest
import torch

from flashinfer.mla import (
    nvfp4_quantize_append_sparse_mla_cache,
    nvfp4_quantize_pack_sparse_mla_cache,
)
from flashinfer.utils import get_compute_capability
from tests.attention.sparse_mla_test_utils import (
    _BYTES_PER_TOKEN,
    _D_NOPE,
    _D_ROPE,
    _PACKED_NOPE_BYTES,
    _reference_rows,
    _split_cache,
)

_CACHE_OP_CCS = ((10, 0), (10, 3), (12, 0), (12, 1))


def _require_cache_op_arch() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    cc = get_compute_capability(torch.device("cuda"))
    if tuple(cc) not in _CACHE_OP_CCS:
        pytest.skip(f"NVFP4 DSv4 cache ops need SM100/SM103/SM120/SM121, got SM{cc[0]}{cc[1]}")


@pytest.mark.parametrize("page_size", [2, 32, 64, 128])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
def test_nvfp4_dsv4_cache_pack_matches_reference(page_size, kv_layout):
    _require_cache_op_arch()
    torch.manual_seed(42)
    latent_kv = torch.randn(3, page_size, _D_NOPE + _D_ROPE, dtype=torch.bfloat16, device="cuda")

    cache = nvfp4_quantize_pack_sparse_mla_cache(latent_kv, kv_layout=kv_layout)
    data, scales = _split_cache(cache)
    packed_ref, scales_ref, rope_ref = _reference_rows(latent_kv)

    assert cache.dtype == torch.uint8
    expected_shape = (
        (3, 1, page_size, _BYTES_PER_TOKEN) if kv_layout == "HND" else (3, page_size, 1, _BYTES_PER_TOKEN)
    )
    assert cache.shape == expected_shape
    torch.testing.assert_close(data[..., :_PACKED_NOPE_BYTES].reshape_as(packed_ref), packed_ref)
    torch.testing.assert_close(data[..., _PACKED_NOPE_BYTES:].reshape_as(rope_ref), rope_ref)
    torch.testing.assert_close(scales[..., :28].reshape_as(scales_ref), scales_ref)
    assert torch.count_nonzero(scales[..., 28:]) == 0


@pytest.mark.parametrize("page_size", [2, 64])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_nvfp4_dsv4_cache_append_matches_pack(page_size, index_dtype):
    _require_cache_op_arch()
    torch.manual_seed(7)
    num_pages = 3
    latent_kv = torch.randn(num_pages, page_size, _D_NOPE + _D_ROPE, dtype=torch.bfloat16, device="cuda")
    full_cache = nvfp4_quantize_pack_sparse_mla_cache(latent_kv)
    append_cache = torch.full_like(full_cache, 0xA5)
    slots = torch.arange(num_pages * page_size, dtype=index_dtype, device="cuda")
    # shuffled slot order, one padding entry (-1) and one duplicate (lowest input row wins)
    perm = torch.randperm(num_pages * page_size, device="cuda")
    rows = latent_kv.reshape(-1, _D_NOPE + _D_ROPE)[perm]
    slots = slots[perm]
    rows = torch.cat([rows, rows[:1] + 1.0])
    slots = torch.cat([slots, slots[:1]])
    rows = torch.cat([rows, rows[:1]])
    slots = torch.cat([slots, torch.full((1,), -1, dtype=index_dtype, device="cuda")])

    nvfp4_quantize_append_sparse_mla_cache(rows.contiguous(), slots.contiguous(), append_cache)
    torch.testing.assert_close(append_cache, full_cache)


# ---------------------------------------------------------------------------
# Decode through the public entry: trtllm_batch_decode_sparse_mla_dsv4(
#     kv_cache_format="nvfp4", backend="cake") on SM100 / SM103.
# ---------------------------------------------------------------------------

from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4  # noqa: E402
from flashinfer.mla.cake_dsv4 import (  # noqa: E402
    _nvfp4_plan,
    _workspace_bytes,
    cake_dsv4_workspace_layout,
    get_cake_dsv4_workspace_bytes,
)
from tests.attention.sparse_mla_test_utils import (  # noqa: E402
    _dequantize_nvfp4_cache,
    _dequantize_nvfp4_query,
    _reference_sparse_attention,
)

_CAKE_CCS = ((10, 0), (10, 3))
_HEAD_DIM = _D_NOPE + _D_ROPE
_SM_SCALE = _HEAD_DIM**-0.5
# FlashInfer NVFP4 tolerance (output 5e-2, LSE 2e-2) is the upper bound for this
# route; the kernel quantizes P to NVFP4 and V^T to NVFP4 (fp32 softmax /
# accumulation) while the reference uses the exact dequantized cache and the
# NVFP4-rounded query.
_OUT_ATOL = _OUT_RTOL = 5e-2
_LSE_ATOL = 2e-2


def _require_cake_arch() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    cc = get_compute_capability(torch.device("cuda"))
    if tuple(cc) not in _CAKE_CCS:
        pytest.skip(f"backend='cake' NVFP4 decode needs SM100/SM103, got SM{cc[0]}{cc[1]}")


def _random_indices(num_tokens, topk, pool_rows, generator, device):
    """Random selections without replacement per query token (the perf-row convention)."""
    rows = [
        torch.randperm(pool_rows, generator=generator, device=device)[:topk]
        for _ in range(num_tokens)
    ]
    return torch.stack(rows).to(torch.int32)


def _pack_pool(num_pages, page_size, generator, device, kv_layout="HND"):
    latent = (
        torch.randn(num_pages, page_size, _HEAD_DIM, generator=generator, device=device)
        .to(torch.bfloat16)
        * 0.1
    )
    cache = nvfp4_quantize_pack_sparse_mla_cache(latent, kv_layout=kv_layout)
    return cache, _dequantize_nvfp4_cache(cache).reshape(-1, _HEAD_DIM)


def _masked(indices, lens):
    if lens is None:
        return indices
    positions = torch.arange(indices.shape[1], device=indices.device)[None]
    return indices.masked_fill(positions >= lens[:, None], -1)


def _run_case(
    *,
    num_tokens,
    num_heads,
    main_pages,
    topk,
    extra_pages=0,
    extra_topk=0,
    page_size=64,
    extra_page_size=64,
    main_lens=None,
    extra_lens=None,
    sinks=False,
    kv_layout="HND",
    seed=0,
):
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(num_tokens, num_heads, _HEAD_DIM, generator=gen, device=device).to(
        torch.bfloat16
    )
    main_cache, main_rows = _pack_pool(main_pages, page_size, gen, device, kv_layout)
    main_idx = _random_indices(num_tokens, topk, main_rows.shape[0], gen, device)
    extra_cache = extra_idx = None
    kv_rows = main_rows
    ref_idx = _masked(main_idx, main_lens)
    if extra_topk:
        extra_cache, extra_rows = _pack_pool(extra_pages, extra_page_size, gen, device, kv_layout)
        extra_idx = _random_indices(num_tokens, extra_topk, extra_rows.shape[0], gen, device)
        kv_rows = torch.cat((main_rows, extra_rows))
        shifted = _masked(extra_idx, extra_lens)
        shifted = torch.where(shifted >= 0, shifted + main_rows.shape[0], shifted)
        ref_idx = torch.cat((ref_idx, shifted), dim=1)
    sink = (
        torch.randn(num_heads, generator=gen, device=device) if sinks else None
    )
    workspace = torch.empty(
        get_cake_dsv4_workspace_bytes(
            num_tokens,
            num_heads,
            topk,
            torch.bfloat16,
            kv_cache_format="nvfp4",
            extra_topk=extra_topk,
        ),
        dtype=torch.uint8,
        device=device,
    )

    def call():
        return trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=main_cache,
            workspace_buffer=workspace,
            sparse_indices=main_idx,
            sparse_topk_lens=main_lens,
            compressed_kv_cache=extra_cache,
            extra_sparse_indices=extra_idx,
            extra_sparse_topk_lens=extra_lens,
            bmm1_scale=_SM_SCALE,
            sinks=sink,
            kv_layout=kv_layout,
            kv_cache_format="nvfp4",
            backend="cake",
        )

    out = call()
    torch.cuda.synchronize()
    plan = _nvfp4_plan(
        num_query_tokens=num_tokens,
        num_heads=num_heads,
        sparse_topk=topk,
        extra_topk=extra_topk,
        sm_count=torch.cuda.get_device_properties(device).multi_processor_count,
    )
    layout = cake_dsv4_workspace_layout(num_tokens, num_heads, plan.num_splits, with_lse=True)
    lse_offset, _ = layout.lse
    lse = (
        _workspace_bytes(workspace)[lse_offset : lse_offset + num_tokens * num_heads * 4]
        .view(torch.float32)
        .view(num_tokens, num_heads)
        .clone()
    )
    ref_out, ref_lse = _reference_sparse_attention(
        _dequantize_nvfp4_query(q), kv_rows, ref_idx, _SM_SCALE, attn_sink=sink
    )
    return out, lse, ref_out, ref_lse, call, plan


@pytest.mark.parametrize(
    "num_tokens,num_heads,main_pages,topk",
    [(8, 16, 64, 128), (8, 128, 64, 128), (4, 16, 256, 512), (1, 128, 256, 512)],
)
def test_nvfp4_dsv4_cake_decode_matches_reference(num_tokens, num_heads, main_pages, topk):
    _require_cake_arch()
    out, lse, ref_out, ref_lse, _, _ = _run_case(
        num_tokens=num_tokens, num_heads=num_heads, main_pages=main_pages, topk=topk, seed=11
    )
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out, ref_out, atol=_OUT_ATOL, rtol=_OUT_RTOL)
    torch.testing.assert_close(lse, ref_lse, atol=_LSE_ATOL, rtol=0.0)


@pytest.mark.parametrize("extra_page_size", [2, 64])
def test_nvfp4_dsv4_cake_decode_dual_cache(extra_page_size):
    _require_cake_arch()
    extra_topk = 132 if extra_page_size == 2 else 512
    out, lse, ref_out, ref_lse, _, plan = _run_case(
        num_tokens=8,
        num_heads=16,
        main_pages=64,
        topk=128,
        extra_pages=256 if extra_page_size == 2 else 64,
        extra_topk=extra_topk,
        extra_page_size=extra_page_size,
        seed=23,
    )
    assert plan.extra_width == extra_topk
    torch.testing.assert_close(out, ref_out, atol=_OUT_ATOL, rtol=_OUT_RTOL)
    torch.testing.assert_close(lse, ref_lse, atol=_LSE_ATOL, rtol=0.0)


def test_nvfp4_dsv4_cake_decode_lengths_sink_and_masked_rows():
    """topk_length clamps (0 / 1 / 63 / 65 / full), -1 entries and sinks on both tables."""
    _require_cake_arch()
    device = torch.device("cuda")
    main_lens = torch.tensor([0, 1, 63, 65, 128, 128, 7, 128], dtype=torch.int32, device=device)
    extra_lens = torch.tensor([512, 0, 1, 500, 63, 65, 300, 512], dtype=torch.int32, device=device)
    out, lse, ref_out, ref_lse, _, _ = _run_case(
        num_tokens=8,
        num_heads=32,
        main_pages=64,
        topk=128,
        extra_pages=64,
        extra_topk=512,
        main_lens=main_lens,
        extra_lens=extra_lens,
        sinks=True,
        seed=5,
    )
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out, ref_out, atol=_OUT_ATOL, rtol=_OUT_RTOL)
    torch.testing.assert_close(lse, ref_lse, atol=_LSE_ATOL, rtol=0.0)


def test_nvfp4_dsv4_cake_decode_all_masked_without_sink_is_zero():
    _require_cake_arch()
    device = torch.device("cuda")
    main_lens = torch.zeros(4, dtype=torch.int32, device=device)
    out, lse, ref_out, _, _, _ = _run_case(
        num_tokens=4, num_heads=16, main_pages=64, topk=128, main_lens=main_lens, seed=7
    )
    assert torch.count_nonzero(out) == 0
    assert torch.count_nonzero(ref_out) == 0
    assert torch.isneginf(lse).all()


def test_nvfp4_dsv4_cake_decode_nhd_layout():
    _require_cake_arch()
    out, lse, ref_out, ref_lse, _, _ = _run_case(
        num_tokens=8, num_heads=16, main_pages=64, topk=128, kv_layout="NHD", seed=3
    )
    torch.testing.assert_close(out, ref_out, atol=_OUT_ATOL, rtol=_OUT_RTOL)
    torch.testing.assert_close(lse, ref_lse, atol=_LSE_ATOL, rtol=0.0)


def test_nvfp4_dsv4_cake_decode_graph_replay_is_bitwise():
    """CUDA-graph capture of the public entry; three replays reproduce the eager bytes."""
    _require_cake_arch()
    out, lse, ref_out, _, call, _ = _run_case(
        num_tokens=8, num_heads=128, main_pages=64, topk=128, extra_pages=64, extra_topk=512, seed=9
    )
    torch.testing.assert_close(out, ref_out, atol=_OUT_ATOL, rtol=_OUT_RTOL)
    eager = out.clone()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        call()  # warm the JIT modules on the capture stream
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        replayed = call()
    for _ in range(3):
        replayed.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(replayed.view(torch.int16), eager.view(torch.int16))
