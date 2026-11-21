# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``gemv_bf16_fp4_gather``: gathered NVFP4 rows, scattered output."""

import pytest
import torch

from flashinfer.gemm.gemm_bf16_fp4_gather import gemv_bf16_fp4_gather
from flashinfer.utils import get_compute_capability

_E2M1_VALUES_FP32 = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)  # fmt: skip


@pytest.fixture(autouse=True)
def _require_sm12x():
    if get_compute_capability(torch.device("cuda"))[0] != 12:
        pytest.skip("gemv_bf16_fp4_gather is SM12x only")


def _random_rows(num_rows, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    b = torch.randint(
        0, 256, (num_rows, k // 2), dtype=torch.uint8, device="cuda", generator=g
    )
    sf = torch.rand(num_rows, k // 16, device="cuda", generator=g) * 1.5 + 0.25
    return b, sf.to(torch.float8_e4m3fn)


def _dequant(b, sf, alpha):
    lut = torch.tensor(_E2M1_VALUES_FP32, device=b.device)
    codes = b.long()
    w = torch.stack([lut[codes & 0xF], lut[codes >> 4]], dim=-1).reshape(b.shape[0], -1)
    w = w * sf.float().repeat_interleave(16, dim=1)
    return w if alpha is None else w * alpha.float()


def _reference(a, b, sf, alpha, ids, n_cols, offset):
    logits = a.float() @ _dequant(b, sf, alpha).t()
    out = torch.full((a.shape[0], n_cols), float("-inf"), device=a.device)
    for m in range(a.shape[0]):
        keep = ids[m] >= 0
        rows = ids[m][keep].long()
        out[m, offset + rows] = logits[m, rows]
    return out


def _check(out, ref, out_dtype):
    finite = torch.isfinite(ref)
    assert torch.equal(finite, torch.isfinite(out.float()))
    tol = 1e-2 if out_dtype != torch.float32 else 1e-4
    scale = ref[finite].abs().max()
    torch.testing.assert_close(
        out.float()[finite] / scale, ref[finite] / scale, atol=tol, rtol=0
    )


@pytest.mark.parametrize("k", [2048, 2560, 1056, 2080])
@pytest.mark.parametrize("m", [1, 4, 5, 32])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_gather_scatter_matches_reference(k, m, ids_dtype, out_dtype):
    num_rows, slots, offset = 3000, 517, 101
    b, sf = _random_rows(num_rows, k, seed=k + m)
    alpha = torch.tensor([0.0125], device="cuda")
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    ids = torch.stack(
        [torch.randperm(num_rows, device="cuda")[:slots] for _ in range(m)]
    ).to(ids_dtype)
    ids[0, :7] = -1
    n_cols = offset + num_rows + 5
    out = torch.full((m, n_cols), float("-inf"), device="cuda", dtype=out_dtype)
    gemv_bf16_fp4_gather(a, b, sf, alpha, ids, out, out_col_offset=offset)
    _check(out, _reference(a, b, sf, alpha, ids, n_cols, offset), out_dtype)


def test_alpha_none_duplicates_and_strided_out():
    k, num_rows = 2048, 1000
    b, sf = _random_rows(num_rows, k, seed=7)
    a = torch.randn(3, k, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, num_rows, (3, 300), device="cuda")
    ids[:, 1] = ids[:, 0]
    big = torch.full((3, 2 * num_rows), float("-inf"), device="cuda")
    out = big[:, : num_rows + 10]  # row stride != row length
    gemv_bf16_fp4_gather(a, b, sf, None, ids, out)
    _check(out, _reference(a, b, sf, None, ids, num_rows + 10, 0), torch.float32)
    assert torch.isinf(big[:, num_rows + 10 :]).all()


def test_rows_from_fp4_quantize():
    from flashinfer import fp4_quantize

    k, num_rows = 2560, 2048
    w = torch.randn(num_rows, k, device="cuda", dtype=torch.bfloat16) * 0.02
    global_sf = (448.0 * 6.0 / w.float().abs().max()).reshape(1)
    b, sf = fp4_quantize(w, global_sf, is_sf_swizzled_layout=False)
    sf = sf.view(torch.float8_e4m3fn).reshape(num_rows, k // 16)
    alpha = (1.0 / global_sf).float()
    a = torch.randn(4, k, device="cuda", dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(num_rows, device="cuda")[:256] for _ in range(4)])
    out = torch.full((4, num_rows), float("-inf"), device="cuda")
    gemv_bf16_fp4_gather(a, b, sf, alpha, ids, out)
    _check(out, _reference(a, b, sf, alpha, ids, num_rows, 0), torch.float32)


def test_cuda_graph_replay():
    k, num_rows = 2560, 4000
    b, sf = _random_rows(num_rows, k, seed=3)
    a = torch.randn(4, k, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, num_rows, (4, 1024), device="cuda")
    out = torch.full((4, num_rows), float("-inf"), device="cuda", dtype=torch.bfloat16)
    gemv_bf16_fp4_gather(a, b, sf, None, ids, out)  # compile outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemv_bf16_fp4_gather(a, b, sf, None, ids, out)
    out.fill_(float("-inf"))
    a.copy_(torch.randn_like(a))
    ids.copy_(torch.randint(0, num_rows, ids.shape, device="cuda"))
    graph.replay()
    _check(out, _reference(a, b, sf, None, ids, num_rows, 0), torch.bfloat16)


def test_rejects_bad_shapes():
    b, sf = _random_rows(10, 64, seed=0)
    a = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16)
    ids = torch.zeros(1, 4, dtype=torch.int64, device="cuda")
    with pytest.raises(ValueError):
        gemv_bf16_fp4_gather(a, b, sf, None, ids, torch.empty(1, 5, device="cuda"))
    with pytest.raises(ValueError):
        gemv_bf16_fp4_gather(
            a.float(), b, sf, None, ids, torch.empty(1, 10, device="cuda")
        )
