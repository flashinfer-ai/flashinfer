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

"""Guard grouped-head normalization, including partial CTAs and live metadata."""

import math
import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0")
from flashinfer.experimental.prims_ts_sparse_mla.runtime import _compile_finish

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Requires SM100/SM103",
)


@pytest.mark.parametrize("independent", [False, True])
def test_sparse_finish_partial_cta_live_sources(independent):
    torch.manual_seed(719)
    rows, heads = 5, 3  # Fifteen heads leave a partial four-warp CTA.
    ps = torch.randn(rows, heads, 512, device="cuda").to(torch.bfloat16)
    pc = torch.randn_like(ps)
    ls = torch.randn(rows, heads, device="cuda") * 4
    lc = torch.randn_like(ls) * 4
    ns = torch.tensor([0, 7, 0, 9, 2], dtype=torch.int32, device="cuda")
    nc = torch.tensor([0, 0, 3, 9, 1], dtype=torch.int32, device="cuda")
    ps[ns == 0] = torch.nan
    pc[nc == 0] = torch.nan
    sinks = torch.tensor([torch.inf, -torch.inf, 0.25], device="cuda")
    # Four-byte aligned storage is part of the standalone helper contract.
    storage = torch.full((ps.numel() + 4,), torch.nan, dtype=ps.dtype, device="cuda")
    out = storage[2:-2].view_as(ps)
    lse = torch.empty_like(ls)
    fn = _compile_finish(0, heads, independent)

    def run():
        fn(ps, pc, ls, lc, ns, nc, sinks, out, lse)

    def check():
        sl = ls.double().masked_fill(ns[:, None] == 0, -torch.inf) * math.log(2)
        cl = lc.double().masked_fill(nc[:, None] == 0, -torch.inf) * math.log(2)
        if not independent:
            cl.fill_(-torch.inf)
        expected_lse = torch.logaddexp(sl, cl)
        denom = torch.logaddexp(expected_lse, sinks.double())
        ws = torch.nan_to_num((sl - denom).exp(), nan=0.0)
        wc = torch.nan_to_num((cl - denom).exp(), nan=0.0)
        a = torch.nan_to_num(ps.double()) * ws[..., None]
        b = torch.nan_to_num(pc.double()) * wc[..., None]
        expected = a + b
        budget = expected.abs() * 2**-8 + (a.abs() + b.abs()) * 2**-18 + 1e-7
        assert torch.isfinite(out).all()
        assert ((out.double() - expected).abs() <= budget).all()
        torch.testing.assert_close(lse.double(), expected_lse, atol=2e-5, rtol=2e-5)
        assert torch.isnan(storage[:2]).all() and torch.isnan(storage[-2:]).all()

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Change source presence and sinks without recompiling or recapturing.
    ns[4] = 0
    nc[3] = 0
    sinks[2] = -1.5
    graph.replay()
    torch.cuda.synchronize()
    check()
