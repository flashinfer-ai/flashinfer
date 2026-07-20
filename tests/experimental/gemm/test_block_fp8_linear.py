# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""gemm.block_fp8_linear: quantized-reference parity + the planned lifecycle
(plan -> bind -> run) replaying under CUDA-graph capture.

Curated from b12x tests/test_gemm_block_fp8_linear.py; kernel-reuse and
regime-policy tests stay in the b12x repo.
"""

from __future__ import annotations

import torch

from flashinfer.experimental.sm12x.gemm import block_fp8_linear as bfl
from flashinfer.experimental.sm12x.gemm._shared.wo_mxfp8 import (
    dequantize_mxfp8_rows_torch,
)

from ..conftest import require_sm12x


def _make_block_fp8_weight(
    out_features: int, in_features: int
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = (
        torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16)
        / 8
    ).to(torch.float8_e4m3fn)
    scale_u8 = (
        torch.arange(
            (out_features // 128) * (in_features // 128),
            device="cuda",
            dtype=torch.int32,
        )
        % 3
        + 126
    ).to(torch.uint8)
    scale = scale_u8.view(torch.float8_e8m0fnu).reshape(
        out_features // 128, in_features // 128
    )
    return weight, scale


def _reference(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
    x_q = bfl.quantize_input(x)
    w_q = bfl.pack_weight(weight, scale)
    x_deq = dequantize_mxfp8_rows_torch(x_q.values, x_q.scale_rows)
    w_deq = dequantize_mxfp8_rows_torch(w_q.weight.values, w_q.weight.scale_rows)
    return x_deq @ w_deq.T


def test_run_matches_quantized_reference() -> None:
    require_sm12x()
    torch.manual_seed(20260523)

    tokens, in_features, out_features = 7, 256, 384
    x = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = bfl.pack_weight(weight, scale)

    actual = bfl.run(x, packed)
    expected = _reference(x, weight, scale)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual.float(), expected.to(actual.dtype).float(), rtol=0, atol=0
    )


def test_plan_bind_run_replays_under_cuda_graph() -> None:
    require_sm12x()
    torch.manual_seed(20260526)

    tokens, in_features, out_features = 1, 128, 256
    x = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = bfl.pack_weight(weight, scale)

    plan = bfl.plan(
        bfl.Caps(
            device=x.device,
            max_tokens=tokens,
            in_features=in_features,
            out_features=out_features,
            output_dtype=x.dtype,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=x.device)
        for shape, dtype in plan.shapes_and_dtypes()
    )
    output = torch.empty((tokens, out_features, 1), dtype=x.dtype, device=x.device)
    binding = bfl.bind(
        plan, scratch=scratch, source=x, packed_weight=packed, output=output
    )

    def run_once() -> torch.Tensor:
        return bfl.run(binding=binding)

    eager = run_once().clone()
    torch.cuda.synchronize()

    run_once()  # warm before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run_once()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, eager, rtol=0, atol=0)
