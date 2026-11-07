# Regression test: a cuDNN MoE grouped GEMM graph that was executed inside a
# captured CUDA graph must stay alive after the
# _build_cudnn_moe_grouped_gemm_graph lru_cache drops it.
#
# Replaying the captured CUDA graph copies a host buffer held by the graph's
# execution plan to the device.  Without the fix, evicting the graph frees that
# buffer and the next replay fails (cudaErrorIllegalInstruction), so the
# scenario runs in a subprocess and the test checks its exit status and output.
#
# cudnn-frontend >= 1.31 also keeps the plan alive by default, so the scenario
# runs with CUDNN_CUDA_GRAPH_RETENTION=none (1.32+) to leave FlashInfer's own
# retention as the only one.

import os
import subprocess
import sys
import textwrap

from .conftest import requires_cudnn_moe, requires_grouped_mm_bf16_cc

_SCENARIO = textwrap.dedent(
    """
    import torch
    from flashinfer.grouped_mm import grouped_mm_bf16
    from flashinfer.grouped_mm.cudnn import _cudnn_moe_grouped_gemm_plan_count
    from flashinfer.grouped_mm.cudnn.core import _build_cudnn_moe_grouped_gemm_graph

    dev = torch.device("cuda")
    torch.manual_seed(0)
    num_experts, n, k = 8, 1024, 2048
    seg_lens = [8, 16, 4, 12, 8, 0, 4, 12]
    a = torch.randn(sum(seg_lens), k, device=dev, dtype=torch.bfloat16)
    b = torch.randn(num_experts, n, k, device=dev, dtype=torch.bfloat16) * 0.05
    m_indptr = torch.tensor(
        [0, *torch.tensor(seg_lens).cumsum(0).tolist()], dtype=torch.int32, device=dev
    )

    # Capture one CUDA graph per cuDNN plan.
    captured = []
    for tactic in range(_cudnn_moe_grouped_gemm_plan_count(a, b, m_indptr)):
        out = torch.empty(a.shape[0], n, device=dev, dtype=torch.bfloat16)
        grouped_mm_bf16(a, b, m_indptr, out=out, tactic=tactic)  # build outside capture
        torch.cuda.synchronize()
        ref = out.clone()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            grouped_mm_bf16(a, b, m_indptr, out=out, tactic=tactic)
        captured.append((tactic, g, out, ref))

    # Evict the graphs, as more than 1024 distinct token counts do in serving.
    _build_cudnn_moe_grouped_gemm_graph.cache_clear()
    torch.cuda.synchronize()

    for tactic, g, out, ref in captured:
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref), f"tactic {tactic}: replay result changed"
    print(f"OK {len(captured)} captured graphs replayed correctly")
    """
)


@requires_grouped_mm_bf16_cc
@requires_cudnn_moe
def test_cudnn_moe_grouped_gemm_graph_outlives_cache_eviction():
    env = dict(os.environ, CUDNN_CUDA_GRAPH_RETENTION="none")
    proc = subprocess.run(
        [sys.executable, "-c", _SCENARIO],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, (
        f"subprocess exited with {proc.returncode}\n"
        f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-4000:]}"
    )
