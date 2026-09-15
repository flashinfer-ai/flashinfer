"""Captured Split-K work clears its workspace once and remains replay-safe."""

import re
from pathlib import Path

import pytest
import torch

from flashinfer.gemm.gemm_svdquant import DEFAULT_WORKSPACE_SIZE
from flashinfer.gemm.svdquant_sm120_cutlass import get_nvfp4_svdquant_sm120_module

from .test_nvfp4_svdquant_gemm import _make_gemm_problem, _sqnr_db


def _graph_node_types(dot: str) -> list[str]:
    """Read CUDA graph node labels, excluding occurrences in kernel names."""
    labels = re.findall(r'\blabel\s*=\s*"((?:\\.|[^"\\])*)"', dot, flags=re.DOTALL)
    return [
        match.group(1).upper()
        for label in labels
        if (
            match := re.match(
                r"\s*\{?\s*(MEMSET|KERNEL|MEMCPY)\b", label, re.IGNORECASE
            )
        )
    ]


@pytest.mark.parametrize("enable_pdl", (False, True), ids=("pdl-off", "pdl-on"))
@pytest.mark.parametrize("use_bias", (False, True), ids=("no-bias", "bias"))
@pytest.mark.parametrize("tactic", (16, 73), ids=("splitk-k128", "splitk-k256-swap"))
def test_sm120_splitk_initializes_workspace_once_and_replays_fresh_inputs(
    tmp_path: Path, tactic: int, use_bias: bool, enable_pdl: bool
) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    m, n, k, rank = 129, 256, 512, 32
    module = get_nvfp4_svdquant_sm120_module()
    assert module.nvfp4_svdquant_gemm_can_implement(m, n, k, rank, tactic)
    torch.manual_seed(20260914)
    problem = _make_gemm_problem(
        m, n, k, rank=rank, quant_backend="cute-dsl", residual_backend="b12x"
    )
    workspace_bytes = max(
        DEFAULT_WORKSPACE_SIZE,
        int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic)),
    )
    workspace = torch.zeros(workspace_bytes, dtype=torch.uint8, device="cuda")
    guard_elements = 128 * n
    storage = torch.full(
        (2 * guard_elements + m * n,), -2048, dtype=torch.bfloat16, device="cuda"
    )
    output = storage[guard_elements:-guard_elements].view(m, n)
    bias = problem["bias"] if use_bias else None

    def launch() -> None:
        module.nvfp4_svdquant_gemm(
            problem["xq"],
            problem["wq"],
            problem["x_sf_flat"],
            problem["w_sf_flat"],
            problem["alpha"],
            problem["d"],
            problem["l1_scaled"],
            bias,
            output,
            workspace,
            tactic,
            enable_pdl,
        )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            launch()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    keep_graph = hasattr(torch.cuda.CUDAGraph, "raw_cuda_graph")
    graph = (
        torch.cuda.CUDAGraph(keep_graph=True) if keep_graph else torch.cuda.CUDAGraph()
    )
    graph.enable_debug_mode()
    with torch.cuda.graph(graph):
        launch()
    dot_path = tmp_path / f"splitk-{tactic}-pdl-{enable_pdl}.dot"
    graph.debug_dump(str(dot_path))
    if keep_graph:
        graph.instantiate()

    keys = ("xq", "wq", "x_sf_flat", "w_sf_flat", "alpha", "d", "l1_scaled", "bias")
    addresses = [problem[key].data_ptr() for key in keys] + [
        workspace.data_ptr(),
        output.data_ptr(),
    ]
    previous = output.clone()
    # Keep the workspace left by each replay. Resetting it here would mask a
    # missing initialization or a stale Split-K reduction/semaphore state.
    for replay in range(3):
        torch.manual_seed(20260915 + replay)
        fresh = _make_gemm_problem(
            m, n, k, rank=rank, quant_backend="cute-dsl", residual_backend="b12x"
        )
        for key in keys:
            problem[key].copy_(fresh[key])
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert bool(torch.isfinite(output).all()), f"nonfinite replay {replay}"
        expected = fresh["ref_bias"] if use_bias else fresh["ref"]
        assert _sqnr_db(expected, output.float()) > 40.0, f"inaccurate replay {replay}"
        assert not torch.equal(previous, output), "fresh operands had no effect"
        assert bool((storage[:guard_elements] == -2048).all()), "leading guard changed"
        assert bool((storage[-guard_elements:] == -2048).all()), (
            "trailing guard changed"
        )
        assert addresses == [problem[key].data_ptr() for key in keys] + [
            workspace.data_ptr(),
            output.data_ptr(),
        ]
        previous.copy_(output)

    node_types = _graph_node_types(dot_path.read_text(encoding="utf-8"))
    assert node_types.count("KERNEL") == 1, f"expected one GEMM kernel: {dot_path}"
    assert node_types.count("MEMSET") == 1, (
        f"expected one workspace initialization, found {node_types.count('MEMSET')}: {dot_path}"
    )
