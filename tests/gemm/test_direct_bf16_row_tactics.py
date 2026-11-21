# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="CuTe DSL low-M BF16 requires SM100/SM103/SM107",
)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (3, 96, 5120),
        (8, 64, 8192),
        (16, 96, 5120),
        (24, 128, 4096),
        (32, 256, 8192),
        (7, 1, 768),
    ],
)
def test_narrow_projection_row_tactics(m, n, k):
    from flashinfer.gemm.kernels.dense_bf16_gemm_direct import (
        autotune_tactics,
        default_tactic,
        run_direct_dense,
    )

    torch.manual_seed(123)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).T
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    default = default_tactic(m, n, k)
    candidates = [
        t
        for t in autotune_tactics(m, n, k)
        if t.rows_per_block < default.rows_per_block
    ]
    assert candidates
    for tactic in candidates:
        run_direct_dense(a, b, out, False, tactic)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_direct_dense(a, b, out, False, tactic)
        # Independent rows plus changed inputs catch row-ownership/indexing bugs.
        a.add_(0.125)
        b.add_(0.25)
        out.fill_(float("nan"))
        graph.replay()
        ref = (a.double() @ b.double()).bfloat16()
        torch.testing.assert_close(out, ref, rtol=0.008, atol=0.005)
