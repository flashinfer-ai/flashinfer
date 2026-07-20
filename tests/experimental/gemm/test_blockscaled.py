# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""gemm.blockscaled: NVFP4/MXFP8 dense block-scaled GEMM.

Curated from b12x tests/test_gemm_stack.py: flashinfer-core cuDNN oracle
parity (NVFP4), grouped MXFP8 per-batch scale correctness, and CUDA-graph
replay. Exhaustive tile/support-matrix sweeps stay in the b12x repo.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer.experimental.sm12x._lib import dense_gemm as dense_module
from flashinfer.experimental.sm12x._lib.intrinsics import quantize_grouped_nvfp4_torch
from flashinfer.experimental.sm12x._lib.utils import convert_sf_from_mma_layout
from flashinfer.experimental.sm12x.gemm import blockscaled
from flashinfer.experimental.sm12x.gemm._shared.wo_mxfp8 import (
    dequantize_mxfp8_rows_torch,
    pack_fp8_block_scaled_weight_mxfp8,
    quantize_mxfp8_rows_torch,
)

from ..conftest import require_sm12x


def _require_cudnn_fp4_oracle():
    """flashinfer core mm_fp4 (cuDNN backend) as the correctness oracle.

    Tests may import both core and experimental; only package code is bound
    by the isolation rule.
    """
    try:
        from flashinfer.gemm import mm_fp4
        from flashinfer.gemm.gemm_base import (
            CUDNN_AVAILABLE,
            _check_cudnn_fp4_availability,
        )
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"flashinfer core mm_fp4 oracle unavailable: {exc}")
    if not CUDNN_AVAILABLE:
        pytest.skip("cuDNN Python bindings not installed")
    try:
        _check_cudnn_fp4_availability()
    except RuntimeError as exc:
        pytest.skip(f"cuDNN FP4 not available: {exc}")
    return mm_fp4


def _make_quantized_operand(
    shape: tuple[int, int, int],
    *,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    source = torch.randn(shape, device="cuda", dtype=dtype) / 4
    row_counts = torch.full(
        (shape[0],), shape[1], dtype=torch.int32, device=source.device
    )
    tensor_amax = source.abs().max().to(torch.float32)
    global_scale = torch.tensor(
        [torch.finfo(torch.float8_e4m3fn).max * 6.0 / tensor_amax],
        dtype=torch.float32,
        device=source.device,
    )
    packed, scales = quantize_grouped_nvfp4_torch(source, row_counts, global_scale)
    return (packed, scales), global_scale


def _mm_nvfp4(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    lhs_scale: torch.Tensor,
    rhs_scale: torch.Tensor,
    *,
    c_dtype: str = "bfloat16",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    alpha = (1.0 / (lhs_scale[0] * rhs_scale[0])).view(1)
    return blockscaled.mm(
        lhs,
        rhs,
        out=out,
        alpha=alpha,
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype=c_dtype,
        sf_vec_size=16,
    )


@pytest.mark.parametrize(
    ("m", "n", "k", "c_dtype"),
    [
        (128, 128, 128, "bfloat16"),
        (256, 512, 128, "bfloat16"),
        (512, 256, 256, "bfloat16"),
        (128, 128, 128, "float16"),
    ],
)
def test_mm_nvfp4_matches_flashinfer_cudnn(m, n, k, c_dtype) -> None:
    require_sm12x()
    mm_fp4 = _require_cudnn_fp4_oracle()
    torch.manual_seed(42)

    lhs, lhs_scale = _make_quantized_operand((1, m, k), dtype=torch.bfloat16)
    rhs, rhs_scale = _make_quantized_operand((1, n, k), dtype=torch.bfloat16)
    alpha = (1.0 / (lhs_scale[0] * rhs_scale[0])).view(1)

    actual = _mm_nvfp4(lhs, rhs, lhs_scale, rhs_scale, c_dtype=c_dtype)

    packed_a, sfa = lhs
    packed_b, sfb = rhs
    oracle = mm_fp4(
        packed_a[:, :, 0].contiguous(),
        packed_b[:, :, 0].contiguous().T,
        convert_sf_from_mma_layout(sfa, m=m, k=k, num_groups=1),
        convert_sf_from_mma_layout(sfb, m=n, k=k, num_groups=1).T,
        alpha,
        torch.bfloat16 if c_dtype == "bfloat16" else torch.float16,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="cudnn",
        use_nvfp4=True,
    )

    torch.testing.assert_close(actual[:, :, 0], oracle, rtol=0, atol=0)


def test_mm_mxfp8_grouped_batches_use_their_own_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_sm12x()
    torch.manual_seed(29)

    # Real grouped WO-A geometry; force the shape-gated BK64 specialization so
    # this compact test covers its packed-scale address arithmetic for L>1.
    m, n, k = 64, 1024, 512
    groups = 4
    group_multipliers = torch.tensor(
        [1.0, 2.0, 4.0, 0.5], device="cuda", dtype=torch.bfloat16
    ).view(1, 1, groups)
    a = torch.randn((m, k, groups), device="cuda", dtype=torch.bfloat16) / 4
    a_q = quantize_mxfp8_rows_torch(a * group_multipliers)
    b_values = (
        torch.randn((groups * n, k), device="cuda", dtype=torch.bfloat16) / 32
    ).to(torch.float8_e4m3fn)
    b_scales = (
        torch.tensor([1.0, 2.0, 4.0, 0.5], device="cuda", dtype=torch.float32)
        .view(groups, 1, 1)
        .expand(groups, n // 128, k // 128)
        .reshape(groups * (n // 128), k // 128)
        .contiguous()
    )
    b_q = pack_fp8_block_scaled_weight_mxfp8(
        b_values, b_scales, m=n, k=k, num_groups=groups
    )
    assert not torch.equal(a_q.scale_rows[0], a_q.scale_rows[1])
    assert not torch.equal(b_q.scale_rows[0], b_q.scale_rows[1])

    monkeypatch.setattr(dense_module, "_select_mxfp8_tile_k", lambda *_: 64)
    out = blockscaled.mm(
        (a_q.values, a_q.scale_mma),
        (b_q.values, b_q.scale_mma),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
        mma_tiler_mn=(128, 128),
        expected_m=2048,
        sfb_k_replicated=True,
    )
    a_deq = dequantize_mxfp8_rows_torch(a_q.values, a_q.scale_rows).to(torch.bfloat16)
    b_deq = dequantize_mxfp8_rows_torch(b_q.values, b_q.scale_rows).to(torch.bfloat16)
    ref = torch.einsum("mkl,nkl->mnl", a_deq, b_deq).to(torch.bfloat16)

    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_mm_pair_replays_under_cuda_graph() -> None:
    require_sm12x()
    torch.manual_seed(1234)

    gate_m, gate_n, gate_k = 32, 2048, 512
    down_m, down_n, down_k = 32, 1024, 2048

    gate_lhs, gate_ls = _make_quantized_operand(
        (1, gate_m, gate_k), dtype=torch.bfloat16
    )
    gate_rhs, gate_rs = _make_quantized_operand(
        (1, gate_n, gate_k), dtype=torch.bfloat16
    )
    down_lhs, down_ls = _make_quantized_operand(
        (1, down_m, down_k), dtype=torch.bfloat16
    )
    down_rhs, down_rs = _make_quantized_operand(
        (1, down_n, down_k), dtype=torch.bfloat16
    )

    eager_gate = _mm_nvfp4(gate_lhs, gate_rhs, gate_ls, gate_rs)
    eager_down = _mm_nvfp4(down_lhs, down_rhs, down_ls, down_rs)
    torch.cuda.synchronize()

    graph_gate = torch.empty_like(eager_gate)
    graph_down = torch.empty_like(eager_down)

    # Prime compiled kernels before capture, matching the serving warmup path.
    _mm_nvfp4(gate_lhs, gate_rhs, gate_ls, gate_rs)
    _mm_nvfp4(down_lhs, down_rhs, down_ls, down_rs)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _mm_nvfp4(gate_lhs, gate_rhs, gate_ls, gate_rs, out=graph_gate)
        _mm_nvfp4(down_lhs, down_rhs, down_ls, down_rs, out=graph_down)

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_gate, eager_gate, rtol=0, atol=0)
    torch.testing.assert_close(graph_down, eager_down, rtol=0, atol=0)
