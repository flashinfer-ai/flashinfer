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
from flashinfer.attention.prims_ts.sparse_mla_decode import (
    BatchSparseMLADecodePagedTSWrapper,
)
from flashinfer.experimental.prims_ts_sparse_mla.policy import _SparseMlaTuning
from flashinfer.testing.sparse_mla import sparse_mla_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Requires SM100/SM103",
)


@pytest.mark.parametrize("family", ["swap", "keep", "2cta"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("page_s,page_c", [(1, 1), (256, 2)])
def test_sparse_wrapper_sources_tails_sinks_and_graph(
    family,
    dtype,
    page_s,
    page_c,
    gather_issue_warps=1,
    offset_cache="strided",
    fuse_epilogue=False,
    direct_inputs=False,
):
    torch.manual_seed(57)
    heads = 16 if family == "swap" else 64 if family == "keep" else 128
    q = (torch.randn(2, 2, heads, 512, device="cuda") * 0.2).to(dtype)
    # Padding creates real page strides that differ between the two sources.
    swa = (torch.randn(3, page_s + 1, 512, device="cuda") * 0.2 - 0.1).to(dtype)[
        :, :page_s
    ]
    compressed = (torch.randn(5, page_c + 2, 512, device="cuda") * 0.2 + 0.1).to(dtype)[
        :, :page_c
    ]
    si = torch.randint(0, 3 * page_s, (2, 2, 128), device="cuda", dtype=torch.int32)
    ci = torch.randint(0, 5 * page_c, (2, 2, 65), device="cuda", dtype=torch.int32)
    si[..., ::7] = -1
    ci[..., ::5] = -1
    sl = torch.tensor([[128, 0], [0, 13]], device="cuda", dtype=torch.int32)
    cl = torch.tensor([[13, 65], [0, 0]], device="cuda", dtype=torch.int32)
    sinks = torch.randn(heads, device="cuda", dtype=torch.float32)
    sinks[0] = torch.inf
    sinks[1] = -torch.inf
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper._impl._tuning = _SparseMlaTuning(
        family=family,
        gather_issue_warps=gather_issue_warps,
        offset_cache=offset_cache,
        fuse_epilogue=fuse_epilogue,
        direct_inputs=direct_inputs,
    )
    wrapper.plan(
        q.device,
        2,
        heads,
        max_topk=128,
        max_extra_topk=65,
        max_seq_len_q=2,
        q_data_type=dtype,
        has_sinks=True,
        return_lse=True,
    )
    out = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:-1], device="cuda", dtype=torch.float32)
    qscale = torch.tensor(1.25 if dtype == torch.float8_e4m3fn else 1.0, device="cuda")
    kvscale = torch.tensor(1.5 if dtype == torch.float8_e4m3fn else 1.0, device="cuda")
    kwargs = dict(
        swa_indices=si,
        compressed_indices=ci,
        swa_topk_lens=sl,
        compressed_topk_lens=cl,
        sinks=sinks,
        q_scale=qscale,
        swa_kv_scale=kvscale,
        out=out,
        lse=lse,
    )

    def check():
        expected, expected_lse, error_bound = sparse_mla_reference(
            q,
            swa,
            compressed,
            si,
            ci,
            swa_topk_lens=sl,
            compressed_topk_lens=cl,
            sinks=sinks,
            q_scale=qscale.item(),
            swa_kv_scale=kvscale.item(),
            compressed_kv_scale=kvscale.item(),
            return_fp8_error_bound=True,
        )
        atol = 1.5e-3 if dtype == torch.float8_e4m3fn else 8e-4
        if dtype == torch.float8_e4m3fn:
            error = (out.double() - expected).abs()
            assert (error <= error_bound).all(), (error - error_bound).max().item()
        else:
            torch.testing.assert_close(out.double(), expected, atol=atol, rtol=0.01)
        torch.testing.assert_close(lse.double(), expected_lse, atol=2e-4, rtol=1e-4)
        assert torch.count_nonzero(out[1, 0]) == 0
        assert torch.isneginf(lse[1, 0]).all()

    wrapper.run(**prepare_fixture(wrapper, q, swa, compressed, **kwargs))
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(
            **prepare_fixture(wrapper, q, swa, compressed, validate=False, **kwargs)
        )
    si.copy_(si.roll(3, -1))
    cl[0, 1] = 17
    sinks[2] += 0.5
    if dtype == torch.float8_e4m3fn:
        qscale.fill_(0.75)
        kvscale.fill_(2.0)
    graph.replay()
    torch.cuda.synchronize()
    check()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("warps", [1, 2, 4])
@pytest.mark.parametrize("cache", ["quad", "global", "coalesced"])
def test_m16_gather_profiles(dtype, warps, cache):
    test_sparse_wrapper_sources_tails_sinks_and_graph("swap", dtype, 1, 2, warps, cache)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_keep_gather_profile(dtype):
    test_sparse_wrapper_sources_tails_sinks_and_graph(
        "keep", dtype, 256, 2, 4, "coalesced"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("family", ["swap", "keep"])
def test_fused_gather_profile_device_scales(dtype, family):
    test_sparse_wrapper_sources_tails_sinks_and_graph(
        family, dtype, 1, 2, 4, "coalesced", True
    )


@pytest.mark.parametrize("family", ["swap", "keep"])
def test_direct_gather_profile_device_scales(family):
    test_sparse_wrapper_sources_tails_sinks_and_graph(
        family, torch.float8_e4m3fn, 1, 2, 4, "coalesced", True, True
    )
