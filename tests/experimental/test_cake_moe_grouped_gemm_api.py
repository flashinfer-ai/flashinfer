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

flashinfer public-API tests of the Cake MoE grouped GEMM package.

The stable entry point ``flashinfer.grouped_mm.grouped_mm_bf16(..., backend="cake")``
routes the forward projection to ``flashinfer.experimental.cake_moe_grouped_gemm``.
The tests here check that route against the per-group fp64 reference (including the
rows past ``m_indptr[-1]``, which belong to no group) and that it rejects the options
the Cake backend does not take.  The backend tests of the package live in
``test_cake_moe_grouped_gemm.py``, which imports the package only through
``tests.test_helpers.cake_package``.
"""

import math

import pytest
import torch

from flashinfer.experimental.cake_moe_grouped_gemm import cake_backend as cb
from flashinfer.experimental.cake_moe_grouped_gemm.cake_backend import (
    SUPPORTED_COMPUTE_CAPABILITIES,
)

ATOL = RTOL = 1e-2  # bf16 outputs

# Group size lists: an empty group, a one-row group and sizes that are not tile
# multiples (the same lists as the backend suite).
RAGGED_E4 = [300, 0, 1, 700]
RAGGED_E6 = [1, 257, 511, 4099, 0, 33]


def _require_program(op, out_dtype=None):
    if not torch.cuda.is_available():
        pytest.skip("the ragged grouped GEMM programs require an SM100/SM103/SM107 GPU")
    device = torch.device("cuda", 0)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))
    if arch is None:
        pytest.skip("the ragged grouped GEMM programs require an SM100/SM103/SM107 GPU")
    if not cb.generated_program_available(device, op, out_dtype):
        pytest.skip(
            f"no generated ragged grouped GEMM program for {op} registered for {arch}"
        )
    return device


def _offs(sizes, device):
    return torch.tensor(sizes, dtype=torch.int32, device=device).cumsum(
        0, dtype=torch.int32
    )


def _m_indptr(sizes, device):
    ends = [sum(sizes[: i + 1]) for i in range(len(sizes))]
    return torch.tensor([0] + ends, dtype=torch.int32, device=device)


def _inputs(sizes, n, k, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    sum_m = sum(sizes)
    x = torch.randn(sum_m, k, generator=gen, device=device, dtype=torch.float32)
    w = torch.randn(len(sizes), n, k, generator=gen, device=device, dtype=torch.float32)
    g = torch.randn(sum_m, n, generator=gen, device=device, dtype=torch.float32)
    x = (x / math.sqrt(k)).to(torch.bfloat16)
    w = (w / math.sqrt(k)).to(torch.bfloat16)
    g = g.to(torch.bfloat16)
    return x, w, g, _offs(sizes, device)


def _bounds(sizes):
    ends = [sum(sizes[: i + 1]) for i in range(len(sizes))]
    return list(zip([0] + ends[:-1], ends, strict=True))


def _reference_fwd(x, w, sizes):
    """Per-group fp64 reference of the forward projection ``Y = X @ W[e].T`` in fp32."""
    out = torch.zeros(x.shape[0], w.shape[1], dtype=torch.float32, device=x.device)
    for e, (s, t) in enumerate(_bounds(sizes)):
        if t > s:
            out[s:t] = (x[s:t].double() @ w[e].double().T).float()
    return out


def _assert_bf16_close(out, ref):
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)


def test_grouped_mm_bf16_cake_backend_matches_reference():
    device = _require_program("fwd")
    from flashinfer.grouped_mm import grouped_mm_bf16

    x, w, _g, _offs = _inputs(RAGGED_E6, 1024, 512, device)
    out = grouped_mm_bf16(x, w, _m_indptr(RAGGED_E6, device), backend="cake")
    assert out.shape == (sum(RAGGED_E6), 1024) and out.dtype == torch.bfloat16
    _assert_bf16_close(out, _reference_fwd(x, w, RAGGED_E6))
    # rows past m_indptr[-1] belong to no group and are left untouched
    padded_x = torch.cat([x, torch.zeros(37, 512, dtype=x.dtype, device=device)])
    out = torch.full(
        (sum(RAGGED_E6) + 37, 1024), float("nan"), dtype=torch.bfloat16, device=device
    )
    grouped_mm_bf16(padded_x, w, _m_indptr(RAGGED_E6, device), out=out, backend="cake")
    _assert_bf16_close(out[: sum(RAGGED_E6)], _reference_fwd(x, w, RAGGED_E6))
    assert torch.isnan(out[sum(RAGGED_E6) :]).all()


def test_grouped_mm_bf16_cake_backend_rejects_unsupported_options():
    device = _require_program("fwd")
    from flashinfer.grouped_mm import grouped_mm_bf16

    x, w, _g, _offs = _inputs(RAGGED_E4, 512, 256, device)
    m_indptr = _m_indptr(RAGGED_E4, device)
    with pytest.raises(ValueError):
        grouped_mm_bf16(x, w, m_indptr, out_dtype=torch.float32, backend="cake")
    with pytest.raises(ValueError):
        grouped_mm_bf16(x, w, m_indptr, backend="cake", tactic=0)
