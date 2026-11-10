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


def _prefix_tail_runner(tile, cache):
    """Return run(mate_lengths, prefix) -> (out, lse) of a 372-token causal request,
    optionally batched with mates or restricted to the query tail after a prefix."""
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
            out, lse = wrapper.run(query, key, value, return_lse=True)
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
            out, lse = wrapper.run(query, kv, return_lse=True)
        assert wrapper._plan_info[3] == tile
        return out[: q_lengths[0]], lse[: q_lengths[0]]

    return run


def _assert_bit_equal(actual, expected):
    for a, e in zip(actual, expected, strict=True):
        assert torch.equal(a, e)


@pytest.mark.parametrize("tile", [16, 64, 128])
@pytest.mark.parametrize("cache", ["paged", "ragged"])
def test_fixed_tile_mixed_batch_and_prefix_tail(tile, cache):
    """Cross the automatic 128/64/16 thresholds without changing a request's bits."""
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("bf16 regression requires SM80 or newer")
    run = _prefix_tail_runner(tile, cache)
    alone = run()
    _assert_bit_equal(run([12] * 27), alone)
    _assert_bit_equal(run([1] * 63), alone)
    # With GQA group 2, prefix 256 starts the tail at packed row 512, a multiple
    # of every tile, so each tile covers the same rows as in the full request.
    out, lse = alone
    _assert_bit_equal(run(prefix=256), (out[256:], lse[256:]))


@pytest.mark.parametrize("tile", [16, 64, 128])
@pytest.mark.parametrize("cache", ["paged", "ragged"])
def test_fixed_tile_misaligned_prefix_tail(tile, cache):
    """Prefix 250 starts the tail at packed row 500, off every tile boundary.

    The FA2 online-softmax rescale preserves an unchanged running maximum
    exactly, so both output and LSE must match the full request even when the
    tail starts off a tile boundary.
    """
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("bf16 regression requires SM80 or newer")
    run = _prefix_tail_runner(tile, cache)
    out, lse = run()
    _assert_bit_equal(run(prefix=250), (out[250:], lse[250:]))


@pytest.mark.parametrize("tile", [16, 64, 128])
@pytest.mark.parametrize("wrapper_kind", ["prefill", "decode"])
def test_fixed_tile_overrides_uniform_cuda_graph(tile, wrapper_kind):
    if tile == 16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("fixed_cta_tile_q=16 requires SM80 or newer")
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
    # A CUDA-graph wrapper keeps the tile of its first successful plan.
    for other in (None, 64 if tile != 64 else 16):
        with pytest.raises(ValueError, match="frozen cudagraph shape"):
            wrapper.plan(*args, **{**kwargs, "fixed_cta_tile_q": other})
    wrapper.plan(*args, **kwargs)


@pytest.mark.parametrize("tile", [16, 64])
def test_fixed_tile_ragged_cuda_graph_replay(tile):
    """Replay one capture across ragged replans that use the reserved row capacity
    differently, and check each replay against an eager plan of the same tile."""
    if tile == 16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("fixed_cta_tile_q=16 requires SM80 or newer")
    torch.manual_seed(0)
    device, dtype = "cuda", torch.float16
    num_qo_heads, num_kv_heads, head_dim, page_size = 8, 4, 128, 16
    batch_size, max_pages, capacity = 4, 12, 256
    kv_data = torch.randn(
        batch_size * max_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    q = torch.randn(capacity, num_qo_heads, head_dim, device=device, dtype=dtype)
    # (q lengths, kv lengths) per plan. The capture fills the reserved capacity,
    # since the captured output is sized by that plan's rows; later replans move
    # rows between requests, drop to fewer rows and grow back.
    lengths = [
        ([100, 1, 90, 65], [130, 40, 190, 65]),
        ([3, 70, 17, 41], [3, 150, 60, 41]),
        ([150, 2, 33, 71], [180, 2, 100, 71]),
    ]

    def metadata(q_lens, kv_lens):
        pages = [(n + page_size - 1) // page_size for n in kv_lens]
        indices = torch.cat(
            [
                torch.arange(i * max_pages, i * max_pages + n, dtype=torch.int32)
                for i, n in enumerate(pages)
            ]
        )
        return (
            torch.tensor([0, *q_lens], dtype=torch.int32).cumsum(0).int(),
            torch.tensor([0, *pages], dtype=torch.int32).cumsum(0).int(),
            indices,
            torch.tensor([(n - 1) % page_size + 1 for n in kv_lens], dtype=torch.int32),
        )

    def make(graph):
        kwargs = {}
        if graph:
            kwargs = dict(
                use_cuda_graph=True,
                qo_indptr_buf=torch.empty(
                    batch_size + 1, dtype=torch.int32, device=device
                ),
                paged_kv_indptr_buf=torch.empty(
                    batch_size + 1, dtype=torch.int32, device=device
                ),
                paged_kv_indices_buf=torch.empty(
                    batch_size * max_pages, dtype=torch.int32, device=device
                ),
                paged_kv_last_page_len_buf=torch.empty(
                    batch_size, dtype=torch.int32, device=device
                ),
                max_total_num_rows=capacity,
            )
        workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
        return flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", backend="fa2", **kwargs
        )

    def plan(wrapper, q_lens, kv_lens, fixed_cta_tile_q=tile):
        wrapper.plan(
            *metadata(q_lens, kv_lens),
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,
            q_data_type=dtype,
            disable_split_kv=True,
            fixed_cta_tile_q=fixed_cta_tile_q,
        )

    wrapper = make(graph=True)
    plan(wrapper, *lengths[0])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        wrapper.run(q, kv_data)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = wrapper.run(q, kv_data)

    def eager_reference(q_lens, kv_lens):
        eager = make(graph=False)
        plan(eager, q_lens, kv_lens)
        rows = sum(q_lens)
        return rows, eager.run(q[:rows], kv_data)

    def check_replay(rows, expected):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out[:rows], expected, rtol=1e-3, atol=1e-3)

    previous = eager_reference(*lengths[0])
    check_replay(*previous)
    other = 64 if tile == 16 else 16
    for q_lens, kv_lens in lengths[1:]:
        with pytest.raises(ValueError, match="frozen cudagraph shape"):
            plan(wrapper, q_lens, kv_lens, fixed_cta_tile_q=other)
        # Replay before any successful replan: the rejected one must have left
        # the captured metadata of the previous valid plan untouched.
        check_replay(*previous)
        plan(wrapper, q_lens, kv_lens)
        assert wrapper._plan_info[3] == tile
        previous = eager_reference(q_lens, kv_lens)
        check_replay(*previous)


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
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("below SM80 tile 16 is rejected first; see the SM80 test")
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


@pytest.mark.parametrize("head_dim", [64, 128])
def test_fixed_tile_16_rejected_below_sm80(head_dim):
    """Below SM80 the heuristic never selects tile 16, and some head dims
    (e.g. SM75 fp16 head_dim 64) get an invalid kernel layout at launch."""
    if torch.cuda.get_device_capability()[0] >= 8:
        pytest.skip("tile 16 is supported on SM80 and newer")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8), backend="fa2"
    )
    indptr = torch.tensor([0, 16], dtype=torch.int32)

    def plan(tile):
        wrapper.plan(
            indptr,
            indptr,
            4,
            4,
            head_dim,
            q_data_type=torch.float16,
            fixed_cta_tile_q=tile,
        )

    with pytest.raises(RuntimeError, match="requires SM80 or newer"):
        plan(16)
    plan(64)
    assert wrapper._plan_info[3] == 64


@pytest.mark.parametrize(
    "head_dim_qk,head_dim_vo,tile", [(512, 512, 32), (576, 512, 32), (512, 256, 16)]
)
def test_fixed_tile_large_heads(head_dim_qk, head_dim_vo, tile):
    """Exercise VO-split tile 32 and the asymmetric large-QK tile 16."""
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("16-bit FA2 large heads require SM80 or newer")
    # Minimum FP16 launch budgets for the two asymmetric layouts: Q storage,
    # one KV step, and VO-split staging where applicable. Neither fits 99-KiB GPUs.
    asymmetric_smem = {(576, 512): 111104, (512, 256): 114688}
    needed = asymmetric_smem.get((head_dim_qk, head_dim_vo), 0)
    optin = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    if needed > optin:
        pytest.skip(f"asymmetric FA2 layout needs {needed} bytes, GPU permits {optin}")
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


@pytest.mark.parametrize("kind", ["decode", "paged", "ragged"])
def test_batch_invariant_wrappers_refuse_fixed_tile(kind):
    """batch_invariant=True plans its own query tiles: a fixed tile must raise the
    class's typed error instead of being silently ignored."""
    from flashinfer import batch_invariant as bi
    from flashinfer.attention.batch_invariant.wrappers import ATTENTION_ARCHS

    if not bi.is_supported_device(torch.device("cuda"), ATTENTION_ARCHS):
        pytest.skip("batch-invariant attention is not supported on this GPU")
    workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    indptr = torch.tensor([0, 1], dtype=torch.int32)
    indices = torch.zeros(1, device="cuda", dtype=torch.int32)
    last_page_len = torch.tensor([16], dtype=torch.int32)
    if kind == "decode":
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "NHD", batch_invariant=True
        )
        args = (indptr, indices, last_page_len, 8, 2, 128, 16)
    elif kind == "paged":
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", batch_invariant=True
        )
        args = (indptr, indptr, indices, last_page_len, 8, 2, 128, 16)
    else:
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, "NHD", batch_invariant=True
        )
        args = (indptr, torch.tensor([0, 16], dtype=torch.int32), 8, 2, 128)
    methods = [wrapper.plan]
    if kind != "ragged":
        methods.append(wrapper.workspace_size)
    for method in methods:
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="fixed_cta_tile_q"):
            method(*args, fixed_cta_tile_q=64)
