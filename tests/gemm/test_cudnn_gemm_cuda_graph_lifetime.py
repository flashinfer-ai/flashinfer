# Regression test: a cuDNN GEMM graph that was executed inside a captured CUDA
# graph must stay alive after the build_cudnn_gemm_* lru_caches drop it.
#
# cuDNN frees a plan's runtime-compiled kernels with the last reference to the
# graph, while CUDA graph replay keeps launching them.  Without the fix, evicting
# the graph and then building other plans makes the next replay a use-after-free
# (segfault in cuGraphLaunch), so the scenario runs in a subprocess and the test
# checks its exit status and output.

import subprocess
import sys
import textwrap

import pytest
import torch

_SCENARIO = textwrap.dedent(
    """
    import sys
    import torch
    import flashinfer.gemm.gemm_base as gb

    evict = sys.argv[1]
    dev = torch.device("cuda")
    torch.manual_seed(0)
    one = torch.ones((), dtype=torch.float32, device=dev)
    ws = torch.zeros(64 << 20, dtype=torch.uint8, device=dev)

    def fp8(*shape):
        return (torch.randn(*shape, device=dev) * 0.1).to(torch.float8_e4m3fn)

    def gemm(a, b, out, tactic):
        gb._cudnn_gemm_fp8(ws, a, b, one, one, out, torch.bfloat16, tactic=tactic)

    def num_plans(a, b):
        cdt = gb._torch_data_type_to_cudnn_data_type
        graph = gb.build_cudnn_gemm_fp8_graph(
            a.shape, a.stride(), b.shape, b.stride(),
            cdt(a.dtype), cdt(b.dtype), cdt(torch.bfloat16), a.device, tactic=0,
        )
        return graph.get_execution_plan_count()

    m, n, k = 16, 12288, 2048
    B = fp8(n, k).t().unsqueeze(0)  # (1, K, N), column-major
    A = fp8(1, m, k)

    # Capture one CUDA graph per cuDNN plan, so that at least one of them uses
    # a runtime-compiled engine on any GPU that has one.
    captured = []
    for tactic in range(num_plans(A, B)):
        out = torch.empty(1, m, n, device=dev, dtype=torch.bfloat16)
        gemm(A, B, out, tactic)  # build the plan outside capture
        torch.cuda.synchronize()
        ref = out.clone()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gemm(A, B, out, tactic)
        captured.append((tactic, g, out, ref))

    if evict == "clear":
        gb.clear_cudnn_graph_cache()
    else:  # overflow the LRU with new shapes, as varying prefill lengths do
        for i in range(2100):
            mm = m + 1 + i
            o = torch.empty(1, mm, n, device=dev, dtype=torch.bfloat16)
            gemm(fp8(1, mm, k), B, o, -1)
    torch.cuda.synchronize()

    # Build plans for another shape so that freed kernel memory gets reused.
    B2 = fp8(9216, k).t().unsqueeze(0)
    for mm in range(1, 9):
        x = fp8(1, mm, k)
        o = torch.empty(1, mm, 9216, device=dev, dtype=torch.bfloat16)
        for tactic in range(min(num_plans(x, B2), 8)):
            gemm(x, B2, o, tactic)
    torch.cuda.synchronize()

    for tactic, g, out, ref in captured:
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref), f"tactic {tactic}: replay result changed"
    print(f"OK {len(captured)} captured graphs replayed correctly")
    """
)


def _skip_reason():
    if not torch.cuda.is_available():
        return "CUDA is not available"
    import flashinfer.gemm.gemm_base as gb

    if not gb.CUDNN_AVAILABLE:
        return "cuDNN is not available"
    if torch.cuda.get_device_capability() < (8, 9):
        return "FP8 GEMM requires SM89+"
    return None


@pytest.mark.parametrize("evict", ["clear", "overflow"])
def test_cudnn_fp8_gemm_graph_outlives_cache_eviction(evict):
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    proc = subprocess.run(
        [sys.executable, "-c", _SCENARIO, evict],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, (
        f"subprocess exited with {proc.returncode}\n"
        f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-4000:]}"
    )
