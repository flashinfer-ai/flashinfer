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

PROFILES = [
    ("swap", 8, 6, 1, "nonpersistent", "gmem_separate"),
    ("swap", 16, 24, 3, "nonpersistent", "gmem_separate"),
    ("swap", 32, 48, 2, "nonpersistent", "gmem_separate"),
    ("swap", 32, 48, 1, "static", "gmem_separate"),
    ("swap", 16, 16, 2, "nonpersistent", "cluster"),
    ("swap", 16, 16, 1, "clc", "gmem_separate"),
    ("keep", 64, 96, 3, "nonpersistent", "gmem_separate"),
    ("keep", 64, 64, 1, "clc", "gmem_separate"),
    ("2cta", 128, 96, 3, "nonpersistent", "gmem_separate"),
    ("2cta", 128, 128, 1, "persistent", "gmem_separate"),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("family,tile,heads,splits,scheduler,reduction", PROFILES)
def test_sparse_schedules_packed_and_independent_scales(
    dtype, family, tile, heads, splits, scheduler, reduction, head_dim_ctas=1
):
    torch.manual_seed(71)
    if scheduler == "persistent":
        scheduler = "static" if dtype == torch.float8_e4m3fn else "clc"
    many_rows = scheduler in ("static", "clc")
    n = (
        (
            40
            if family == "keep"
            else torch.cuda.get_device_properties(0).multi_processor_count + 5
        )
        if many_rows
        else 4
    )
    q = (torch.randn(n, heads, 512, device="cuda") * 0.125).to(dtype)
    swa = (torch.randn(257, 1, 512, device="cuda") * 0.125 - 0.05).to(dtype)
    cmp = (torch.randn(641, 1, 512, device="cuda") * 0.125 + 0.05).to(dtype)
    si = torch.randint(0, 257, (n, 128), device="cuda", dtype=torch.int32)
    ci = torch.randint(0, 641, (n, 513), device="cuda", dtype=torch.int32)
    si[:, ::13] = -1
    ci[:, ::17] = -1
    sl = (torch.arange(n, device="cuda", dtype=torch.int32) * 31) % 129
    cl = (torch.arange(n, device="cuda", dtype=torch.int32) * 173) % 514
    cl[-1] = 513
    offsets = torch.tensor([0, 0, n // 2, n], device="cuda", dtype=torch.int32)
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper._impl._tuning = _SparseMlaTuning(
        family, tile, splits, scheduler, reduction, head_dim_ctas
    )
    wrapper.plan(
        q.device,
        3,
        heads,
        max_topk=128,
        max_extra_topk=513,
        max_seq_len_q=n,
        packed_query=True,
        q_data_type=dtype,
        return_lse=True,
    )
    source_s = torch.tensor(1.0, device="cuda")
    source_c = torch.tensor(2.0 if dtype == torch.float8_e4m3fn else 1.0, device="cuda")
    kwargs = dict(
        swa_indices=si,
        compressed_indices=ci,
        swa_topk_lens=sl,
        compressed_topk_lens=cl,
        qo_indptr=offsets,
        swa_kv_scale=source_s,
        compressed_kv_scale=source_c,
    )
    out, lse = wrapper.run(**prepare_fixture(wrapper, q, swa, cmp, **kwargs))
    expected, expected_lse, bound = sparse_mla_reference(
        q,
        swa,
        cmp,
        si,
        ci,
        swa_topk_lens=sl,
        compressed_topk_lens=cl,
        swa_kv_scale=source_s.item(),
        compressed_kv_scale=source_c.item(),
        return_fp8_error_bound=True,
    )
    if dtype == torch.float8_e4m3fn:
        assert ((out.double() - expected).abs() <= bound).all()
    else:
        torch.testing.assert_close(out.double(), expected, rtol=0.02, atol=8e-4)
    torch.testing.assert_close(lse.double(), expected_lse, rtol=1e-4, atol=2e-4)
    assert torch.count_nonzero(out[0]) == 0
    assert torch.isneginf(lse[0]).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("head_dim_ctas", [2, 4])
@pytest.mark.parametrize("family,tile,heads", [("swap", 16, 24), ("keep", 64, 64)])
def test_sparse_value_decomposition(dtype, head_dim_ctas, family, tile, heads):
    test_sparse_schedules_packed_and_independent_scales(
        dtype,
        family,
        tile,
        heads,
        3,
        "nonpersistent",
        "gmem_separate",
        head_dim_ctas,
    )
