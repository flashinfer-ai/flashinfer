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

import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0")
from flashinfer.attention.prims_ts import (
    BatchSparseMLADecodePagedTSWrapper,
    SparseMLAPreparedMetadata,
    batch_sparse_mla_decode_with_paged_kv_cache,
    get_prims_ts_sparse_mla_decode_workspace_size,
)
from flashinfer.experimental.prims_ts_sparse_mla.policy import _SparseMlaTuning
from flashinfer.testing.sparse_mla import sparse_mla_reference
from flashinfer.testing.sparse_mla_metadata import prepare_sparse_mla_metadata
from tests.experimental.prims_ts_sparse_mla.sparse_mla_test_utils import prepare_fixture

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Requires SM100/SM103",
)


def test_caller_workspace_single_source_and_eager_api():
    q = torch.zeros(1, 1, 16, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.ones(2, 1, 512, device="cuda", dtype=torch.bfloat16)
    kv[1] *= 2
    indices = torch.tensor([[[0, 1, -1]]], device="cuda", dtype=torch.int32)
    args = dict(
        device=q.device, batch_size=1, num_heads=16, max_topk=3, return_lse=True
    )
    size = get_prims_ts_sparse_mla_decode_workspace_size(**args)
    workspace = torch.empty(size, device=q.device, dtype=torch.uint8)
    wrapper = BatchSparseMLADecodePagedTSWrapper(workspace)
    wrapper.plan(**args)
    metadata = prepare_sparse_mla_metadata(wrapper, q, kv, indices)
    result, lse = wrapper.run(q, kv, metadata)
    torch.testing.assert_close(result, torch.full_like(q, 1.5))
    eager, eager_lse = batch_sparse_mla_decode_with_paged_kv_cache(
        q, kv, metadata, return_lse=True
    )
    torch.testing.assert_close(result, eager, atol=0, rtol=0)
    torch.testing.assert_close(lse, eager_lse, atol=0, rtol=0)
    first_lse = lse.clone()
    shorter = prepare_sparse_mla_metadata(
        wrapper, q, kv, indices, torch.ones((1, 1), device=q.device, dtype=torch.int32)
    )
    _, next_lse = wrapper.run(q, kv, shorter)
    torch.testing.assert_close(next_lse, torch.zeros_like(next_lse), atol=0, rtol=0)
    torch.testing.assert_close(lse, first_lse, atol=0, rtol=0)
    short = BatchSparseMLADecodePagedTSWrapper(workspace[:-1])
    with pytest.raises(ValueError, match="workspace"):
        short.plan(**args)
    bad = indices.clone()
    bad[..., 0] = 2
    with pytest.raises(ValueError, match="outside pool"):
        wrapper.run(q, kv, metadata._replace(indices=bad.view(1, 3)))
    with pytest.raises(ValueError, match="positive"):
        wrapper.run(q, kv, metadata, softmax_scale=1e-100)


def test_valid_prefix_eager_contract():
    q = torch.zeros(1, 1, 16, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.ones(2, 1, 512, device=q.device, dtype=q.dtype)
    swa[1] *= 2
    compressed = swa * 10
    si = torch.tensor([[[0, 1, -1]]], device=q.device, dtype=torch.int32)
    ci = torch.tensor([[[1, -1, 0]]], device=q.device, dtype=torch.int32)
    lengths = torch.tensor([[2]], device=q.device, dtype=torch.int32)
    ci[..., 1] = 0
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper.plan(
        q.device, 1, 16, max_topk=3, max_extra_topk=3, assume_valid_prefix=True
    )
    metadata = prepare_sparse_mla_metadata(
        wrapper,
        q,
        swa,
        si,
        lengths,
        extra_kv_cache=compressed,
        extra_indices=ci,
        extra_lengths=lengths,
    )
    metadata.extra_indices[0, 1] = -1
    kwargs = dict(assume_valid_prefix=True, return_lse=True)
    with pytest.raises(ValueError, match="extra active prefix contains a hole"):
        batch_sparse_mla_decode_with_paged_kv_cache(
            q, swa, metadata, compressed, **kwargs
        )
    metadata.extra_indices[0, 1] = 0
    result, lse = batch_sparse_mla_decode_with_paged_kv_cache(
        q, swa, metadata, compressed, **kwargs
    )
    # Q=0 gives uniform attention over values 1, 2, 20, 10.
    torch.testing.assert_close(result, torch.full_like(result, 8.25))
    torch.testing.assert_close(lse, torch.full_like(lse, 4.0).log())


def test_odd_capacity_h6_independent_scales_graph_and_empty_packed():
    torch.manual_seed(83)
    q = (torch.randn(1, 6, 512, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    s = (torch.randn(5, 1, 512, device="cuda") * 0.1).to(q.dtype)
    c = (torch.randn(7, 1, 512, device="cuda") * 0.1).to(q.dtype)
    si = torch.tensor([[0, 1, -1]], device="cuda", dtype=torch.int32)
    ci = torch.tensor([[2, 3, 4]], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    cs = torch.tensor(2.0, device="cuda")
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper.plan(
        q.device,
        1,
        6,
        max_topk=3,
        max_extra_topk=3,
        packed_query=True,
        q_data_type=q.dtype,
        return_lse=True,
    )
    assert wrapper._impl._state["buffers"]["lse"][1].data_ptr() % 16 == 0
    out = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty(1, 6, device="cuda")
    kwargs = dict(
        swa_indices=si,
        compressed_indices=ci,
        qo_indptr=offsets,
        compressed_kv_scale=cs,
        out=out,
        lse=lse,
    )
    wrapper.run(**prepare_fixture(wrapper, q, s, c, **kwargs))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(**prepare_fixture(wrapper, q, s, c, validate=False, **kwargs))
    cs.fill_(3.0)
    ci[0, 0] = -1
    graph.replay()
    expected, expected_lse, bound = sparse_mla_reference(
        q, s, c, si, ci, compressed_kv_scale=3, return_fp8_error_bound=True
    )
    assert ((out.double() - expected).abs() <= bound).all()
    torch.testing.assert_close(lse.double(), expected_lse, atol=2e-4, rtol=1e-4)
    empty_offsets = torch.zeros(2, device="cuda", dtype=torch.int32)
    empty_meta = prepare_sparse_mla_metadata(
        wrapper, q[:0], s, si[:0], extra_kv_cache=c, extra_indices=ci[:0]
    )
    empty, empty_lse = wrapper.run(q[:0], s, empty_meta, c, qo_indptr=empty_offsets)
    assert empty.shape == (0, 6, 512) and empty_lse.shape == (0, 6)


def test_illegal_persistent_split_profile_is_rejected():
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper._impl._tuning = _SparseMlaTuning("swap", 16, 2, "static")
    with pytest.raises(ValueError, match="does not support split-KV"):
        wrapper.plan("cuda", 1, 16, max_topk=128)


def test_direct_metadata_without_packed_fields_graph():
    """Direct kernels accept live source lists without preparation scratch."""
    q = torch.zeros(1, 1, 16, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.ones(2, 1, 512, device=q.device, dtype=q.dtype)
    kv[1] *= 3
    meta = SparseMLAPreparedMetadata(
        torch.tensor([[0, -1, 1]], device=q.device, dtype=torch.int32),
        torch.tensor([3], device=q.device, dtype=torch.int32),
    )
    w = BatchSparseMLADecodePagedTSWrapper()
    w._impl._tuning = _SparseMlaTuning(
        family="swap",
        tile_size_q=16,
        split_kv=1,
        head_dim_ctas=1,
        fuse_epilogue=True,
        direct_inputs=True,
    )
    w.plan(q.device, 1, 16, max_topk=3, return_lse=True)
    out, lse = w.run(q, kv, meta)
    torch.testing.assert_close(out, torch.full_like(out, 2.0))
    torch.testing.assert_close(lse, torch.full_like(lse, 2.0).log())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w.run(q, kv, meta, out=out, lse=lse, validate=False)
    meta.indices[0, 2] = -1
    graph.replay()
    torch.testing.assert_close(out, torch.ones_like(out))
    torch.testing.assert_close(lse, torch.zeros_like(lse))
    meta.lengths.zero_()
    graph.replay()
    assert torch.count_nonzero(out) == 0 and torch.isneginf(lse).all()
