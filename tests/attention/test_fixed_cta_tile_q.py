# Copyright (c) 2026 by FlashInfer team.
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

import pytest
import torch

import flashinfer


@pytest.mark.parametrize("tile", [16, 64, 128])
@pytest.mark.parametrize("cache", ["paged", "ragged"])
def test_fixed_tile_mixed_batch_and_prefix_tail(tile, cache):
    """Cross the automatic 128/64/16 thresholds without changing a request's bits."""
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("bf16 regression requires SM80 or newer")
    torch.manual_seed(42)
    device, dtype = "cuda", torch.bfloat16
    q = torch.randn(372, 16, 128, device=device, dtype=dtype)
    k = torch.randn(372, 8, 128, device=device, dtype=dtype)
    v = torch.randn_like(k)
    workspace = torch.empty(32 * 1024 * 1024, device=device, dtype=torch.uint8)

    def run(mate_lengths=(), prefix=0):
        q_lengths = [372 - prefix, *mate_lengths]
        kv_lengths = [372, *mate_lengths]
        qo_indptr = torch.tensor([0, *q_lengths], dtype=torch.int32).cumsum(0).int()
        kv_indptr = torch.tensor([0, *kv_lengths], dtype=torch.int32).cumsum(0).int()
        query = torch.cat(
            [q[prefix:]]
            + [
                torch.randn(n, 16, 128, device=device, dtype=dtype)
                for n in mate_lengths
            ]
        )
        plan_kwargs = dict(
            causal=True,
            q_data_type=dtype,
            disable_split_kv=True,
            fixed_cta_tile_q=tile,
        )
        if cache == "ragged":
            wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace, backend="fa2"
            )
            key = torch.cat(
                [k]
                + [
                    torch.randn(n, 8, 128, device=device, dtype=dtype)
                    for n in mate_lengths
                ]
            )
            value = torch.cat(
                [v]
                + [
                    torch.randn(n, 8, 128, device=device, dtype=dtype)
                    for n in mate_lengths
                ]
            )
            wrapper.plan(qo_indptr, kv_indptr, 16, 8, 128, **plan_kwargs)
            result = wrapper.run(query, key, value)
        else:
            page_size = 16
            page_counts = [(n + page_size - 1) // page_size for n in kv_lengths]
            page_indptr = (
                torch.tensor([0, *page_counts], dtype=torch.int32).cumsum(0).int()
            )
            indices = torch.arange(sum(page_counts), device=device, dtype=torch.int32)
            last_page_len = torch.tensor(
                [(n - 1) % page_size + 1 for n in kv_lengths], dtype=torch.int32
            )
            kv = torch.randn(
                sum(page_counts), 2, page_size, 8, 128, device=device, dtype=dtype
            )
            padding = k.new_zeros(page_counts[0] * page_size - 372, 8, 128)
            kv[: page_counts[0], 0].copy_(
                torch.cat([k, padding]).view(page_counts[0], page_size, 8, 128)
            )
            kv[: page_counts[0], 1].copy_(
                torch.cat([v, padding]).view(page_counts[0], page_size, 8, 128)
            )
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace, backend="fa2"
            )
            wrapper.plan(
                qo_indptr,
                page_indptr,
                indices,
                last_page_len,
                16,
                8,
                128,
                page_size,
                **plan_kwargs,
            )
            result = wrapper.run(query, kv)
        assert wrapper._plan_info[3] == tile
        return result[: q_lengths[0]]

    alone = run()
    assert torch.equal(alone, run([12] * 27))
    assert torch.equal(alone, run([1] * 63))
    assert torch.equal(alone[256:], run(prefix=256))


@pytest.mark.parametrize("tile", [16, 64, 128])
@pytest.mark.parametrize("wrapper_kind", ["prefill", "decode"])
def test_fixed_tile_overrides_uniform_cuda_graph(tile, wrapper_kind):
    batch_size, page_size = 4, 16
    indptr = torch.arange(batch_size + 1, dtype=torch.int32)
    indices = torch.arange(batch_size, device="cuda", dtype=torch.int32)
    last_page_len = torch.full((batch_size,), page_size, dtype=torch.int32)
    workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    if wrapper_kind == "prefill":
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            backend="fa2",
            use_cuda_graph=True,
            qo_indptr_buf=indptr.cuda(),
            paged_kv_indptr_buf=indptr.cuda(),
            paged_kv_indices_buf=indices,
            paged_kv_last_page_len_buf=last_page_len.cuda(),
        )
        args = (indptr, indptr, indices, last_page_len, 8, 4, 128, page_size)
        kwargs = {"uniform_q_len": 1}
    else:
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            backend="fa2",
            use_tensor_cores=True,
            use_cuda_graph=True,
            paged_kv_indptr_buffer=indptr.cuda(),
            paged_kv_indices_buffer=indices,
            paged_kv_last_page_len_buffer=last_page_len.cuda(),
        )
        args = (indptr, indices, last_page_len, 8, 4, 128, page_size)
        kwargs = {}
    kwargs.update(disable_split_kv=True, fixed_cta_tile_q=tile)
    float_size, int_size = wrapper.workspace_size(*args, **kwargs)
    wrapper.reset_workspace_buffer(
        torch.empty(float_size, device="cuda", dtype=torch.uint8),
        torch.empty(int_size, device="cuda", dtype=torch.uint8),
    )
    wrapper.plan(*args, **kwargs)
    assert wrapper._plan_info[3] == tile
    # A captured run() keeps the kernel of its planned tile.
    for other in (None, 64 if tile != 64 else 16):
        with pytest.raises(ValueError, match="frozen cudagraph shape"):
            wrapper.plan(*args, **{**kwargs, "fixed_cta_tile_q": other})
    wrapper.plan(*args, **kwargs)


@pytest.mark.parametrize("wrapper_kind", ["paged", "ragged", "decode"])
def test_fixed_tile_pins_auto_to_fa2(wrapper_kind):
    """backend="auto" must not resolve to FA3/cuDNN/CUTLASS once a tile is fixed."""
    batch_size, page_size, tile = 4, 16, 64
    indptr = torch.arange(batch_size + 1, dtype=torch.int32)
    workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    common = dict(disable_split_kv=True, fixed_cta_tile_q=tile)
    if wrapper_kind == "decode":
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, backend="auto", use_tensor_cores=True
        )
        last_page_len = torch.full((batch_size,), page_size, dtype=torch.int32)
        indices = torch.arange(batch_size, device="cuda", dtype=torch.int32)
        wrapper.plan(indptr, indices, last_page_len, 8, 4, 128, page_size, **common)
        assert wrapper.resolved_backend == "fa2"
    elif wrapper_kind == "paged":
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, backend="auto"
        )
        qo_indptr = indptr * 128
        last_page_len = torch.full((batch_size,), page_size, dtype=torch.int32)
        page_indptr = indptr * 8
        indices = torch.arange(batch_size * 8, device="cuda", dtype=torch.int32)
        wrapper.plan(
            qo_indptr,
            page_indptr,
            indices,
            last_page_len,
            8,
            4,
            128,
            page_size,
            causal=True,
            **common,
        )
        assert wrapper._backend == "fa2"
    else:
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, backend="auto"
        )
        qo_indptr = indptr * 128
        wrapper.plan(qo_indptr, qo_indptr, 8, 4, 128, causal=True, **common)
        assert wrapper._backend == "fa2"
    assert wrapper._plan_info[3] == tile


@pytest.mark.parametrize("head_dim_qk,head_dim_vo", [(448, 256), (128, 256)])
def test_fixed_tile_16_rejected_when_smem_too_small(head_dim_qk, head_dim_vo):
    """Tile 16 whose shared memory does not fit must fail at plan, not at run."""
    # Mirrors FA2CtaTileQ16FitsSmem with a 2-byte Q and KV: Q tile plus one KV
    # step, or the cross-warp merge buffers that share the same union.
    qkv = 16 * head_dim_qk * 2 + (head_dim_qk + head_dim_vo) * 16 * 4 * 2
    merge = 4 * 16 * head_dim_vo * 4 + 4 * 16 * 8
    needed = max(qkv, merge)
    optin = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    if needed <= optin:
        pytest.skip(f"tile 16 fits this GPU ({needed} <= {optin} bytes)")
    indptr = torch.tensor([0, 16], dtype=torch.int32)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        use_cuda_graph=True,
        qo_indptr_buf=indptr.cuda(),
        kv_indptr_buf=indptr.cuda(),
        backend="fa2",
    )

    def plan(tile):
        wrapper.plan(
            indptr,
            indptr,
            4,
            4,
            head_dim_qk,
            head_dim_vo=head_dim_vo,
            q_data_type=torch.float16,
            fixed_cta_tile_q=tile,
        )

    # FLASHINFER_ERROR from the native planner surfaces as RuntimeError.
    with pytest.raises(RuntimeError, match="does not fit this GPU.s shared memory"):
        plan(16)
    # The rejected plan must not freeze the CUDA-graph tile: the suggested
    # retry with 64 has to plan.
    plan(64)
    assert wrapper._plan_info[3] == 64


@pytest.mark.parametrize(
    "head_dim_qk,head_dim_vo,tile", [(512, 512, 32), (576, 512, 32), (512, 256, 16)]
)
def test_fixed_tile_large_heads(head_dim_qk, head_dim_vo, tile):
    """Exercise VO-split tile 32 and the asymmetric large-QK tile 16."""
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("16-bit FA2 large heads require SM80 or newer")
    q_len, kv_len, num_heads = 64, 64, 4
    q = torch.randn(q_len, num_heads, head_dim_qk, device="cuda", dtype=torch.float16)
    k = torch.randn(kv_len, num_heads, head_dim_qk, device="cuda", dtype=torch.float16)
    v = torch.randn(kv_len, num_heads, head_dim_vo, device="cuda", dtype=torch.float16)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8), backend="fa2"
    )
    wrapper.plan(
        torch.tensor([0, q_len], dtype=torch.int32),
        torch.tensor([0, kv_len], dtype=torch.int32),
        num_heads,
        num_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        disable_split_kv=True,
        fixed_cta_tile_q=tile,
    )
    result = wrapper.run(q, k, v)
    assert wrapper._plan_info[3] == tile
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) / head_dim_qk**0.5
    expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float())
    torch.testing.assert_close(result.float(), expected, atol=2e-3, rtol=2e-3)
