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

import math
import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0")
from flashinfer.experimental.prims_ts_sparse_mla.runtime import _compile_sparse_reduce

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Requires SM100/SM103",
)


@pytest.mark.parametrize("splits", [2, 3, 17, 33, 65, 128])
@pytest.mark.parametrize("direct", [False, True])
def test_sparse_split_reduction_forward_error(splits, direct):
    torch.manual_seed(113)
    rows, heads, storage_heads = 3, 7, 8
    partial = torch.randn(rows, storage_heads, splits, 512, device="cuda").to(
        torch.bfloat16
    )
    storage = torch.full(
        (partial.numel() + 2,), torch.nan, device="cuda", dtype=torch.bfloat16
    )
    aligned4 = storage[2:].view_as(partial)
    aligned4.copy_(partial)
    partial = aligned4
    stats = torch.randn(rows, storage_heads, splits, device="cuda") * 4
    stats[0] = -torch.inf
    partial[0] = torch.nan
    stats[1, :, ::3] = -torch.inf
    partial[1, :, ::3] = torch.nan
    stats[2, :, :] = torch.linspace(-100, 100, splits, device="cuda")[None, :]
    counts = torch.tensor([0, 1, 1], device="cuda", dtype=torch.int32)
    sinks = torch.randn(heads, device="cuda")
    sinks[0], sinks[1] = torch.inf, -torch.inf
    logs = stats[:, :heads].double() * math.log(2)
    expected_lse = logs.logsumexp(-1)
    denominator = torch.logaddexp(expected_lse, sinks.double())
    weights = (logs - denominator[..., None]).exp()
    weights[0] = 0
    expected = (torch.nan_to_num(partial[:, :heads].double()) * weights[..., None]).sum(
        -2
    )
    out_storage = torch.full(
        (rows * heads * 512 + 2,), torch.nan, device="cuda", dtype=torch.bfloat16
    )
    out = out_storage[2:].view(rows, heads, 512)
    lse = torch.empty(rows, heads, device="cuda")
    # Compare to unrounded FP64: two independently rounded BF16 answers can
    # differ by one ULP when FP32 arithmetic lands on opposite sides of a tie.
    sensitivity = (
        torch.nan_to_num(partial[:, :heads].double()).abs() * weights[..., None]
    ).sum(-2)
    budget = expected.abs() * 2**-8 + (2 * splits + 16) * 2**-23 * sensitivity + 1e-7
    fn = _compile_sparse_reduce(0, heads, storage_heads, splits, direct)
    fn(partial, stats, counts, sinks, out, lse)
    assert torch.isfinite(out).all()
    assert ((out.double() - expected).abs() <= budget).all()
    torch.testing.assert_close(lse.double(), expected_lse, atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(out[0]) == 0
    assert torch.isneginf(lse[0]).all()
