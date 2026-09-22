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
from tests.experimental.prims_ts_sparse_mla.sparse_mla_test_utils import prepare_fixture

pytest.importorskip("cutlass", minversion="4.7.0")
from flashinfer.attention.prims_ts import BatchSparseMLADecodePagedTSWrapper
from flashinfer.experimental.prims_ts_sparse_mla.policy import _SparseMlaTuning
from flashinfer.testing.sparse_mla import sparse_mla_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Requires SM100/SM103",
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("family,heads,tile", [("swap", 24, 16), ("keep", 96, 64)])
def test_fused_split_reduction_and_fallbacks(dtype, family, heads, tile, direct=False):
    torch.manual_seed(96)
    q = (torch.randn(3, heads, 512, device="cuda") * 0.2).to(dtype)
    s = (torch.randn(19, 1, 512, device="cuda") * 0.2).to(dtype)
    c = (torch.randn(23, 1, 512, device="cuda") * 0.2 + 0.1).to(dtype)
    si = torch.randint(19, (3, 128), device="cuda", dtype=torch.int32)
    ci = torch.randint(23, (3, 65), device="cuda", dtype=torch.int32)
    si[:, ::7], ci[:, ::5] = -1, -1
    si[2], ci[2] = -1, -1
    sl = torch.tensor([128, 0, 128], device="cuda", dtype=torch.int32)
    cl = torch.tensor([65, 65, 65], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
    sinks = torch.randn(heads, device="cuda")
    sinks[0], sinks[1] = torch.inf, -torch.inf
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper._impl._tuning = _SparseMlaTuning(
        family=family,
        tile_size_q=tile,
        split_kv=3,
        gather_issue_warps=4,
        offset_cache="coalesced",
        fuse_epilogue=True,
        direct_inputs=direct,
    )
    wrapper.plan(
        q.device,
        2,
        heads,
        max_seq_len_q=2,
        packed_query=True,
        q_data_type=dtype,
        max_topk=128,
        max_extra_topk=65,
        has_sinks=True,
        return_lse=True,
    )
    out = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty(3, heads, device="cuda")
    kwargs = dict(
        swa_indices=si,
        compressed_indices=ci,
        swa_topk_lens=sl,
        compressed_topk_lens=cl,
        qo_indptr=offsets,
        sinks=sinks,
        out=out,
        lse=lse,
    )

    def check(current_lse, compressed_scale=1.0):
        expected, expected_lse, bound = sparse_mla_reference(
            q,
            s,
            c,
            si,
            ci,
            swa_topk_lens=sl,
            compressed_topk_lens=cl,
            sinks=sinks,
            compressed_kv_scale=compressed_scale,
            return_fp8_error_bound=True,
        )
        if dtype == torch.float8_e4m3fn:
            assert ((out.double() - expected).abs() <= bound).all()
        else:
            torch.testing.assert_close(out.double(), expected, atol=8e-4, rtol=0.02)
        torch.testing.assert_close(
            current_lse.double(), expected_lse, atol=2e-4, rtol=1e-4
        )

    wrapper.run(**prepare_fixture(wrapper, q, s, c, **kwargs))
    assert wrapper._impl._state["last_fused_epilogue"]
    assert wrapper._impl._state["last_direct_inputs"] == direct
    check(lse)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(**prepare_fixture(wrapper, q, s, c, validate=False, **kwargs))
    cl[1] = 33
    si[0] = si[0].roll(1)
    graph.replay()
    check(lse)
    assert torch.count_nonzero(out[2]) == 0 and torch.isneginf(lse[2]).all()

    unaligned = torch.empty(3 * heads + 1, device="cuda")[1:].view(3, heads)
    wrapper.run(**prepare_fixture(wrapper, q, s, c, **{**kwargs, "lse": unaligned}))
    assert not wrapper._impl._state["last_fused_epilogue"]
    check(unaligned)
    cs = torch.tensor(1.75 if dtype == torch.float8_e4m3fn else 1.0, device="cuda")
    wrapper.run(**prepare_fixture(wrapper, q, s, c, compressed_kv_scale=cs, **kwargs))
    assert not wrapper._impl._state["last_fused_epilogue"]
    check(lse, cs.item())


@pytest.mark.parametrize("family,heads,tile", [("swap", 24, 16), ("keep", 96, 64)])
def test_direct_split_and_fallbacks(family, heads, tile):
    test_fused_split_reduction_and_fallbacks(
        torch.float8_e4m3fn, family, heads, tile, True
    )
