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


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Requires Blackwell",
)
@pytest.mark.parametrize("schedule", ["static", "clc", "cluster"])
def test_tuned_m16_schedules(schedule):
    torch.manual_seed(101)
    n = (
        4
        if schedule == "cluster"
        else torch.cuda.get_device_properties(0).multi_processor_count + 5
    )
    q = (torch.randn(n, 1, 16, 512, device="cuda") * 0.125).to(torch.float8_e4m3fn)
    s = (torch.randn(257, 1, 512, device="cuda") * 0.125).to(q.dtype)
    c = (torch.randn(769, 1, 512, device="cuda") * 0.125).to(q.dtype)
    si = torch.randint(0, 257, (n, 1, 128), device="cuda", dtype=torch.int32)
    ci = torch.randint(0, 769, (n, 1, 513), device="cuda", dtype=torch.int32)
    si[..., ::7] = -1
    ci[..., ::13] = -1
    sl = (torch.arange(n, device="cuda", dtype=torch.int32) * 17 % 129).view(n, 1)
    cl = (torch.arange(n, device="cuda", dtype=torch.int32) * 137 % 514).view(n, 1)
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper._impl._tuning = _SparseMlaTuning(
        family="swap",
        tile_size_q=16,
        split_kv=2 if schedule == "cluster" else 1,
        scheduler="nonpersistent" if schedule == "cluster" else schedule,
        reduction="cluster" if schedule == "cluster" else "gmem_separate",
        gather_issue_warps=4,
        offset_cache="quad",
    )
    wrapper.plan(
        q.device,
        n,
        16,
        max_topk=128,
        max_extra_topk=513,
        q_data_type=q.dtype,
        return_lse=True,
    )
    out, lse = wrapper.run(
        **prepare_fixture(
            wrapper,
            q,
            s,
            c,
            swa_indices=si,
            compressed_indices=ci,
            swa_topk_lens=sl,
            compressed_topk_lens=cl,
        )
    )
    expected, expected_lse, bound = sparse_mla_reference(
        q,
        s,
        c,
        si,
        ci,
        swa_topk_lens=sl,
        compressed_topk_lens=cl,
        return_fp8_error_bound=True,
    )
    assert ((out.double() - expected).abs() <= bound).all()
    torch.testing.assert_close(lse.double(), expected_lse, rtol=1e-4, atol=2e-4)
