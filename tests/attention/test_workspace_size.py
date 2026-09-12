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

import pytest
import torch

import flashinfer


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="workspace sizing tests require CUDA"
)


def _paged_kv_inputs(batch_size: int, kv_len: int, page_size: int):
    pages_per_seq = (kv_len + page_size - 1) // page_size
    total_pages = batch_size * pages_per_seq
    indptr = (
        torch.arange(batch_size + 1, dtype=torch.int32, device="cuda") * pages_per_seq
    )
    indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
    last_page_len = torch.full(
        (batch_size,),
        (kv_len - 1) % page_size + 1,
        dtype=torch.int32,
        device="cuda",
    )
    return indptr, indices, last_page_len


def _byte_workspace(num_bytes: int, device: str = "cuda"):
    return torch.empty((num_bytes,), dtype=torch.uint8, device=device)


def _unaligned_byte_workspace(num_bytes: int, device: str = "cuda"):
    workspace = torch.empty((num_bytes + 1,), dtype=torch.uint8, device=device)[1:]
    assert workspace.data_ptr() % 16 != 0
    return workspace


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_batch_decode_workspace_size_plans_with_exact_buffers(use_cuda_graph):
    batch_size = 4
    kv_len = 4096
    page_size = 16
    num_qo_heads = 16
    num_kv_heads = 4
    head_dim = 128
    dtype = torch.float16

    indptr, indices, last_page_len = _paged_kv_inputs(batch_size, kv_len, page_size)
    kwargs = {}
    if use_cuda_graph:
        kwargs = {
            "use_cuda_graph": True,
            "paged_kv_indptr_buffer": torch.empty(
                batch_size + 1, dtype=torch.int32, device="cuda"
            ),
            "paged_kv_indices_buffer": torch.empty(
                len(indices), dtype=torch.int32, device="cuda"
            ),
            "paged_kv_last_page_len_buffer": torch.empty(
                batch_size, dtype=torch.int32, device="cuda"
            ),
        }
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        _byte_workspace(32 * 1024 * 1024), **kwargs
    )

    float_workspace_size, int_workspace_size = wrapper.workspace_size(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    assert float_workspace_size > 0
    assert int_workspace_size > 0

    wrapper.reset_workspace_buffer(
        _byte_workspace(float_workspace_size),
        _byte_workspace(int_workspace_size),
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    assert wrapper._plan_info is not None


def _run_batch_prefill_workspace_size_plan(
    use_cuda_graph=False,
    fixed_split_size=None,
):
    batch_size = 3
    qo_len = 64
    kv_len = 1024
    page_size = 16
    num_qo_heads = 16
    num_kv_heads = 4
    head_dim = 128

    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda") * qo_len
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = _paged_kv_inputs(
        batch_size, kv_len, page_size
    )
    kwargs = {}
    if use_cuda_graph:
        kwargs = {
            "use_cuda_graph": True,
            "qo_indptr_buf": torch.empty(
                batch_size + 1, dtype=torch.int32, device="cuda"
            ),
            "paged_kv_indptr_buf": torch.empty(
                batch_size + 1, dtype=torch.int32, device="cuda"
            ),
            "paged_kv_indices_buf": torch.empty(
                len(paged_kv_indices), dtype=torch.int32, device="cuda"
            ),
            "paged_kv_last_page_len_buf": torch.empty(
                batch_size, dtype=torch.int32, device="cuda"
            ),
        }
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        _byte_workspace(32 * 1024 * 1024), backend="fa2", **kwargs
    )

    plan_kwargs = {}
    if fixed_split_size is not None:
        plan_kwargs = {
            "fixed_split_size": fixed_split_size,
            "disable_split_kv": False,
        }

    float_workspace_size, int_workspace_size = wrapper.workspace_size(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        **plan_kwargs,
    )
    if fixed_split_size is None:
        assert float_workspace_size >= 0
    else:
        assert float_workspace_size > 0
    assert int_workspace_size > 0

    wrapper.reset_workspace_buffer(
        _byte_workspace(float_workspace_size),
        _byte_workspace(int_workspace_size),
    )
    wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        **plan_kwargs,
    )
    assert wrapper._plan_info is not None


def test_batch_prefill_workspace_size_plans_fixed_split_with_exact_buffers():
    _run_batch_prefill_workspace_size_plan(fixed_split_size=16)


def test_batch_prefill_workspace_size_plans_cuda_graph_with_exact_buffers():
    _run_batch_prefill_workspace_size_plan(use_cuda_graph=True)


def test_batch_decode_workspace_size_rejects_unaligned_workspace_buffer():
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        _byte_workspace(32 * 1024 * 1024)
    )

    with pytest.raises(
        ValueError, match="float_workspace_buffer must be 16-byte aligned"
    ):
        wrapper.reset_workspace_buffer(
            _unaligned_byte_workspace(1024),
            _byte_workspace(1024),
        )
    with pytest.raises(
        ValueError, match="int_workspace_buffer must be 16-byte aligned"
    ):
        wrapper.reset_workspace_buffer(
            _byte_workspace(1024),
            _unaligned_byte_workspace(1024),
        )


def test_batch_prefill_workspace_size_rejects_unaligned_workspace_buffer():
    with pytest.raises(
        ValueError, match="float_workspace_buffer must be 16-byte aligned"
    ):
        flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            _unaligned_byte_workspace(32 * 1024 * 1024), backend="fa2"
        )

    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        _byte_workspace(32 * 1024 * 1024), backend="fa2"
    )
    with pytest.raises(
        ValueError, match="int_workspace_buffer must be 16-byte aligned"
    ):
        wrapper.reset_workspace_buffer(
            _byte_workspace(1024),
            _unaligned_byte_workspace(1024),
        )


def _prefill_split_kv_wrapper(batch_size, kv_len, page_size, use_cuda_graph):
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = _paged_kv_inputs(
        batch_size, kv_len, page_size
    )
    kwargs = {}
    if use_cuda_graph:
        kwargs = {
            "use_cuda_graph": True,
            "qo_indptr_buf": torch.empty(
                batch_size + 1, dtype=torch.int32, device="cuda"
            ),
            "paged_kv_indptr_buf": torch.empty(
                batch_size + 1, dtype=torch.int32, device="cuda"
            ),
            "paged_kv_indices_buf": torch.empty(
                len(paged_kv_indices), dtype=torch.int32, device="cuda"
            ),
            "paged_kv_last_page_len_buf": torch.empty(
                batch_size, dtype=torch.int32, device="cuda"
            ),
        }
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        _byte_workspace(32 * 1024 * 1024), backend="fa2", **kwargs
    )
    return wrapper, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len


# Both parameter sets split deterministically on any device: under CUDA graphs
# PrefillSplitQOKVIndptr always reports split_kv, and fixed_split_size forces
# chunking without one. Leaving the eager case to the SM-count-driven binary
# search would make the test a no-op on small GPUs.
@pytest.mark.parametrize("use_cuda_graph,fixed_split_size", [(False, 64), (True, None)])
def test_batch_prefill_split_kv_workspace_counts_query_rows(
    use_cuda_graph, fixed_split_size
):
    """tmp_v and tmp_s hold one partial output per (query row, kv chunk).

    cta_tile_q counts packed (row, head) indices, so a CTA spans
    cta_tile_q / gqa_group_size query rows. Reserving padded_batch_size *
    cta_tile_q rows counts the packed indices instead and over-reserves the
    float workspace by the GQA group size.
    """
    batch_size = 3
    qo_len = 64
    kv_len = 8192
    page_size = 16
    num_qo_heads = 32
    num_kv_heads = 4
    head_dim = 128
    gqa_group_size = num_qo_heads // num_kv_heads

    wrapper, kv_indptr, kv_indices, kv_last_page_len = _prefill_split_kv_wrapper(
        batch_size, kv_len, page_size, use_cuda_graph
    )
    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda") * qo_len
    plan_args = (
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
    )

    plan_kwargs = {}
    if fixed_split_size is not None:
        plan_kwargs = {"fixed_split_size": fixed_split_size, "disable_split_kv": False}

    float_workspace_size, int_workspace_size = wrapper.workspace_size(
        *plan_args, **plan_kwargs
    )
    wrapper.reset_workspace_buffer(
        _byte_workspace(float_workspace_size),
        _byte_workspace(int_workspace_size),
    )
    wrapper.plan(*plan_args, **plan_kwargs)

    # PrefillPlanInfo field order, read positionally because plan() returns the
    # struct flattened into a vector.
    plan_info = list(wrapper._plan_info)
    padded_batch_size = plan_info[0]
    cta_tile_q = plan_info[3]
    split_kv = bool(plan_info[14])
    assert split_kv, "this configuration is meant to exercise the partial buffers"

    partial_rows = -(-(padded_batch_size * cta_tile_q) // gqa_group_size)
    tmp_v_bytes = partial_rows * num_qo_heads * head_dim * 4
    tmp_s_bytes = partial_rows * num_qo_heads * 4
    expected = -(-tmp_v_bytes // 16) * 16 + tmp_s_bytes
    assert float_workspace_size == expected


def test_batch_prefill_split_kv_exact_workspace_matches_unsplit_reference():
    """The merged output must still be right once the partial buffers are
    sized by query rows.

    The reference plans with split-KV disabled, so it writes the output
    directly and never touches tmp_v / tmp_s. The wrapper under test plans
    under CUDA graphs into exactly the reported workspace, so its partial
    buffers have no slack: a reservation short of the rows the merge reads
    would show up here.
    """
    batch_size = 4
    qo_len = 32
    kv_len = 8192
    page_size = 16
    num_qo_heads = 32
    num_kv_heads = 4
    head_dim = 128
    dtype = torch.float16

    kv_indptr, kv_indices, kv_last_page_len = _paged_kv_inputs(
        batch_size, kv_len, page_size
    )
    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda") * qo_len
    torch.manual_seed(0)
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, dtype=dtype, device="cuda"
    )
    kv_data = torch.randn(
        len(kv_indices),
        2,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
    plan_args = (
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
    )

    reference_wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        _byte_workspace(128 * 1024 * 1024), backend="fa2"
    )
    reference_wrapper.plan(*plan_args, disable_split_kv=True)
    reference = reference_wrapper.run(q, kv_data)

    wrapper, _, _, _ = _prefill_split_kv_wrapper(
        batch_size, kv_len, page_size, use_cuda_graph=True
    )
    float_workspace_size, int_workspace_size = wrapper.workspace_size(*plan_args)
    wrapper.reset_workspace_buffer(
        _byte_workspace(float_workspace_size),
        _byte_workspace(int_workspace_size),
    )
    wrapper.plan(*plan_args)
    assert bool(list(wrapper._plan_info)[14]), "expected the plan to split kv"
    out = wrapper.run(q, kv_data)

    torch.testing.assert_close(out, reference, rtol=1e-3, atol=1e-3)
