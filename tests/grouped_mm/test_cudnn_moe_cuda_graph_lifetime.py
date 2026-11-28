# Regression test: a cuDNN MoE grouped GEMM graph that was executed inside a
# captured CUDA graph must stay alive after its builder's lru_cache drops it.
#
# The captured CUDA graph can depend on resources that are freed together with
# the cuDNN graph: a host buffer that the replay copies to the device and, with
# cudnn-frontend 1.33, the plans' kernel code.  Without the fix (and with
# cudnn-frontend's own retention off, as below), evicting the cuDNN graph frees
# them, and the next replay fails with a CUDA error or crashes, so the scenario
# runs in a subprocess and the test checks its exit status and output.
#
# cudnn-frontend >= 1.31 has its own retention, on by default, so the scenario
# runs with CUDNN_CUDA_GRAPH_RETENTION=none (1.32+) to leave FlashInfer's own
# retention as the only one.
#
# The block-scaled operands are synthetic and every block scale is one, so the
# order of the scales in their swizzled layout does not matter: the test only
# compares each replay with the eager result.

import os
import subprocess
import sys
import textwrap

import pytest

from .conftest import (
    requires_cudnn_moe,
    requires_cudnn_moe_block_scale,
    requires_grouped_mm_bf16_cc,
    requires_grouped_mm_fp4_cc,
    requires_grouped_mm_mxfp8_cc,
)

_SCENARIO = textwrap.dedent(
    """
    import sys

    import torch
    from flashinfer.grouped_mm import grouped_mm_bf16, grouped_mm_fp4, grouped_mm_mxfp8
    from flashinfer.grouped_mm.cudnn import (
        _cudnn_moe_block_scale_grouped_gemm_plan_count,
        _cudnn_moe_grouped_gemm_plan_count,
    )
    from flashinfer.grouped_mm.cudnn.core import (
        _build_cudnn_moe_block_scale_grouped_gemm_graph,
        _build_cudnn_moe_grouped_gemm_graph,
    )

    kind = sys.argv[1]
    dev = torch.device("cuda")
    torch.manual_seed(0)
    if kind == "bf16":
        num_experts, n, k = 8, 1024, 2048
        seg_lens = [8, 16, 4, 12, 8, 0, 4, 12]
    else:
        # The shapes of test_non_uniform_distribution in the block-scaled tests:
        # token and N counts in multiples of 128 need no scale padding.
        num_experts, n, k = 4, 256, 512
        seg_lens = [256, 128, 384, 256]
    rows = sum(seg_lens)
    m_indptr = torch.tensor(
        [0, *torch.tensor(seg_lens).cumsum(0).tolist()], dtype=torch.int32, device=dev
    )

    if kind == "bf16":
        a = torch.randn(rows, k, device=dev, dtype=torch.bfloat16)
        b = torch.randn(num_experts, n, k, device=dev, dtype=torch.bfloat16) * 0.05
        builder = _build_cudnn_moe_grouped_gemm_graph
        plans = _cudnn_moe_grouped_gemm_plan_count(a, b, m_indptr)

        def run(out, tactic):
            grouped_mm_bf16(a, b, m_indptr, out=out, tactic=tactic)

    elif kind == "mxfp8":
        a_q = torch.randn(rows, k, device=dev).to(torch.float8_e4m3fn)
        b_q = torch.randn(num_experts, n, k, device=dev).to(torch.float8_e4m3fn)
        # E8M0 127 is a block scale of 1.
        a_s = torch.full((rows, k // 32), 127, dtype=torch.uint8, device=dev)
        b_s = torch.full((num_experts, n, k // 32), 127, dtype=torch.uint8, device=dev)
        builder = _build_cudnn_moe_block_scale_grouped_gemm_graph
        plans = _cudnn_moe_block_scale_grouped_gemm_plan_count(
            a_q, b_q, a_s, b_s, m_indptr, block_size=32
        )

        def run(out, tactic):
            grouped_mm_mxfp8(a_q, b_q, a_s, b_s, m_indptr, out=out, tactic=tactic)

    elif kind == "fp4":  # packed E2M1 bytes, E4M3 block scales of 1, alpha 1
        a_q = torch.randint(0, 256, (rows, k // 2), dtype=torch.uint8, device=dev)
        b_q = torch.randint(
            0, 256, (num_experts, n, k // 2), dtype=torch.uint8, device=dev
        )
        a_s = torch.ones(rows, k // 16, device=dev).to(torch.float8_e4m3fn)
        b_s = torch.ones(num_experts, n, k // 16, device=dev).to(torch.float8_e4m3fn)
        alpha = torch.ones(1, dtype=torch.float32, device=dev)
        builder = _build_cudnn_moe_block_scale_grouped_gemm_graph
        plans = _cudnn_moe_block_scale_grouped_gemm_plan_count(
            a_q, b_q, a_s, b_s, m_indptr, alpha=alpha, block_size=16
        )

        def run(out, tactic):
            grouped_mm_fp4(
                a_q,
                b_q,
                a_s,
                b_s,
                m_indptr,
                alpha=alpha,
                out=out,
                block_size=16,
                tactic=tactic,
            )

    else:
        raise ValueError(f"unknown kind {kind!r}")

    # Capture one CUDA graph per cuDNN plan.
    captured = []
    for tactic in range(plans):
        out = torch.empty(rows, n, device=dev, dtype=torch.bfloat16)
        run(out, tactic)  # build outside capture
        torch.cuda.synchronize()
        ref = out.clone()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run(out, tactic)
        captured.append((tactic, g, out, ref))

    assert captured, f"{kind}: cuDNN offered no execution plan"

    # Evict the graphs, as more than 1024 distinct token counts do in serving.
    builder.cache_clear()
    torch.cuda.synchronize()

    for tactic, g, out, ref in captured:
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref), f"{kind} tactic {tactic}: replay result changed"
    print(f"OK {len(captured)} captured graphs replayed correctly")
    """
)


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param("bf16", marks=[requires_grouped_mm_bf16_cc, requires_cudnn_moe]),
        pytest.param(
            "mxfp8",
            marks=[requires_grouped_mm_mxfp8_cc, requires_cudnn_moe_block_scale],
        ),
        pytest.param(
            "fp4", marks=[requires_grouped_mm_fp4_cc, requires_cudnn_moe_block_scale]
        ),
    ],
)
def test_cudnn_moe_grouped_gemm_graph_outlives_cache_eviction(kind):
    env = dict(os.environ, CUDNN_CUDA_GRAPH_RETENTION="none")
    proc = subprocess.run(
        [sys.executable, "-c", _SCENARIO, kind],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, (
        f"subprocess exited with {proc.returncode}\n"
        f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-4000:]}"
    )
