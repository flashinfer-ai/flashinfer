"""Reference correctness test for the gemv_bf16_fp4_gather trace API."""

import pytest
import torch

from tests.trace.reference_utils import _cc, _check


@pytest.mark.parametrize("M", [1, 5])
def test_gemv_bf16_fp4_gather_reference_correctness(M):
    if _cc()[0] != 12:
        pytest.skip("gemv_bf16_fp4_gather requires SM120/121")
    from flashinfer import gemv_bf16_fp4_gather
    from flashinfer.trace.templates.gemm import gemv_bf16_fp4_gather_trace as tpl

    inputs = tpl.init(M=M, J=1024, K=2048, R=8192)
    ref_out = inputs["out"].clone()
    api_out = gemv_bf16_fp4_gather(**inputs)
    torch.cuda.synchronize()
    tpl.reference(**{**inputs, "out": ref_out})
    _check(tpl, ref_out, api_out, rtol=1e-2, atol=1e-2, max_mismatch_pct=0.0)


def test_gemv_bf16_fp4_gather_init_reproduces_const_axes():
    if _cc()[0] != 12:
        pytest.skip("gemv_bf16_fp4_gather requires SM120/121")
    from flashinfer.trace.template import _render_init_source
    from flashinfer.trace.templates.gemm import gemv_bf16_fp4_gather_trace as tpl

    namespace = {}
    exec(_render_init_source(tpl.init, {"K": 2048, "R": 3000, "N": 3200}), namespace)
    inputs = namespace[tpl.init.__name__](M=2, J=64)
    assert inputs["a"].shape == (2, 2048)
    assert inputs["b"].shape == (3000, 1024)
    assert inputs["out"].shape == (2, 3200)
    assert inputs["out_col_offset"] == 200


def test_gemv_bf16_fp4_gather_reference_empty_selection():
    from flashinfer.trace.templates.gemm import gemv_bf16_fp4_gather_trace as tpl

    k, num_rows = 64, 4
    b = torch.randint(0, 256, (num_rows, k // 2), dtype=torch.uint8)
    b_descale = torch.ones(num_rows, k // 16).to(torch.float8_e4m3fn)
    a = torch.randn(2, k, dtype=torch.bfloat16)
    row_ids = torch.tensor([[-1, -1], [1, -1]])
    out = torch.full((2, num_rows), float("-inf"))
    tpl.reference(a, b, b_descale, None, row_ids, out)
    assert torch.isinf(out[0]).all()
    assert torch.isfinite(out[1, 1]) and torch.isinf(out[1, [0, 2, 3]]).all()
    empty = torch.full((2, num_rows), float("-inf"))
    tpl.reference(a, b, b_descale, None, row_ids[:, :0], empty)
    assert torch.isinf(empty).all()
