"""
Benchmark: Dynamic shape FP4 GEMM vs per-shape graph building.

This demonstrates how cuDNN dynamic shape support can eliminate the
per-shape graph build overhead for FlashInfer FP4 GEMM.

Usage:
    pytest tests/gemm/test_fp4_dynamic_shape_benchmark.py -v -s

Requires: Blackwell GPU (sm100+) and cuDNN >= 9.18.0
"""

import time

import pytest
import torch

cudnn = pytest.importorskip("cudnn")

from flashinfer.gemm.gemm_base import UIDs  # noqa: E402


# ============================================================================
# Helpers
# ============================================================================

INDESTRUCTIBLE_128x4_BLOCK_M_N = 128
INDESTRUCTIBLE_128x4_BLOCK_K = 4


def div_up(a, b):
    return (a + b - 1) // b


def calculate_block_scale_dims(m, n, k, block_size=16):
    block_scale_dim_m = (
        div_up(m, INDESTRUCTIBLE_128x4_BLOCK_M_N) * INDESTRUCTIBLE_128x4_BLOCK_M_N
    )
    block_scale_dim_n = (
        div_up(n, INDESTRUCTIBLE_128x4_BLOCK_M_N) * INDESTRUCTIBLE_128x4_BLOCK_M_N
    )
    block_scale_dim_k = (
        div_up(div_up(k, block_size), INDESTRUCTIBLE_128x4_BLOCK_K)
        * INDESTRUCTIBLE_128x4_BLOCK_K
    )
    return block_scale_dim_m, block_scale_dim_n, block_scale_dim_k


A_UID = UIDs.A_UID.value
B_UID = UIDs.B_UID.value
BLOCK_DESCALE_A_UID = UIDs.BLOCK_DESCALE_A_UID.value
BLOCK_DESCALE_B_UID = UIDs.BLOCK_DESCALE_B_UID.value
O_UID = UIDs.O_UID.value


# ============================================================================
# Graph builders
# ============================================================================


def _build_fp4_graph(handle, b, m, n, k, block_size=16, dynamic=False):
    """Build a cuDNN FP4 GEMM graph, optionally with dynamic shape support."""
    bs_m, bs_n, bs_k = calculate_block_scale_dims(m, n, k, block_size)

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
        is_dynamic_shape_enabled=dynamic,
    )

    A = graph.tensor(
        name="A",
        uid=A_UID,
        dim=[b, m, k],
        stride=[m * k, k, 1],
        data_type=cudnn.data_type.FP4_E2M1,
    )
    SF_A = graph.tensor(
        name="SF_A",
        uid=BLOCK_DESCALE_A_UID,
        dim=[b, bs_m, bs_k],
        stride=[bs_m * bs_k, bs_k, 1],
        data_type=cudnn.data_type.FP8_E4M3,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    dequant_a = graph.block_scale_dequantize(
        A, SF_A, block_size=[1, block_size], name="dequant_a"
    )

    B = graph.tensor(
        name="B",
        uid=B_UID,
        dim=[b, k, n],
        stride=[n * k, 1, k],
        data_type=cudnn.data_type.FP4_E2M1,
    )
    SF_B = graph.tensor(
        name="SF_B",
        uid=BLOCK_DESCALE_B_UID,
        dim=[b, bs_k, bs_n],
        stride=[bs_n * bs_k, 1, bs_k],
        data_type=cudnn.data_type.FP8_E4M3,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    dequant_b = graph.block_scale_dequantize(
        B, SF_B, block_size=[block_size, 1], name="dequant_b"
    )

    C = graph.matmul(
        dequant_a, dequant_b, compute_data_type=cudnn.data_type.FLOAT, name="matmul"
    )
    C.set_uid(O_UID).set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    return graph


def execute_dynamic(graph, handle, b, m, n, k, block_size=16):
    """Execute a dynamic-shape graph with specific runtime shapes."""
    bs_m, bs_n, bs_k = calculate_block_scale_dims(m, n, k, block_size)

    A_gpu = torch.randint(0, 256, (b, m, k // 2), dtype=torch.uint8, device="cuda")
    SF_A_gpu = torch.ones((b, bs_m, bs_k), dtype=torch.float8_e4m3fn, device="cuda")
    B_gpu = torch.randint(0, 256, (b, k // 2, n), dtype=torch.uint8, device="cuda")
    SF_B_gpu = torch.ones((b, bs_k, bs_n), dtype=torch.float8_e4m3fn, device="cuda")
    C_gpu = torch.empty((b, m, n), dtype=torch.bfloat16, device="cuda")

    override_uids = [A_UID, BLOCK_DESCALE_A_UID, B_UID, BLOCK_DESCALE_B_UID, O_UID]
    override_shapes = [
        [b, m, k],
        [b, bs_m, bs_k],
        [b, k, n],
        [b, bs_k, bs_n],
        [b, m, n],
    ]
    override_strides = [
        [m * k, k, 1],
        [bs_m * bs_k, bs_k, 1],
        [n * k, 1, k],
        [bs_n * bs_k, 1, bs_k],
        [m * n, n, 1],
    ]

    variant_pack = {
        A_UID: A_gpu,
        BLOCK_DESCALE_A_UID: SF_A_gpu,
        B_UID: B_gpu,
        BLOCK_DESCALE_B_UID: SF_B_gpu,
        O_UID: C_gpu,
    }

    workspace = torch.empty(
        max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda"
    )

    graph.execute_plan_at_index(
        variant_pack,
        workspace,
        0,
        handle=handle,
        override_uids=override_uids,
        override_shapes=override_shapes,
        override_strides=override_strides,
    )
    torch.cuda.synchronize()
    return C_gpu


# ============================================================================
# Tests
# ============================================================================


@pytest.fixture
def cudnn_handle():
    handle = cudnn.create_handle()
    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=handle, stream=stream)
    return handle


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="Requires Blackwell GPU (sm100+)",
)
@pytest.mark.skipif(
    cudnn.backend_version() < 91800,
    reason="Requires cuDNN >= 9.18.0 for dynamic shapes",
)
class TestFP4DynamicShapeBenchmark:
    """Benchmark dynamic vs static FP4 GEMM graph building."""

    N = 7168
    K = 2048
    BLOCK_SIZE = 16
    # 128-aligned M values (required for dynamic shapes with F8_128x4)
    M_VALUES = [128, 256, 384, 512, 640, 768, 896, 1024]

    def test_dynamic_shape_executes_correctly(self, cudnn_handle):
        """Verify dynamic shape graph produces valid output for all M values."""
        b = 1
        max_m = max(self.M_VALUES)
        graph = _build_fp4_graph(
            cudnn_handle, b, max_m, self.N, self.K, self.BLOCK_SIZE, dynamic=True
        )

        for m in self.M_VALUES:
            result = execute_dynamic(
                graph, cudnn_handle, b, m, self.N, self.K, self.BLOCK_SIZE
            )
            assert result.shape == (b, m, self.N)
            assert not torch.isnan(result).any(), f"NaN in output for M={m}"

    def test_dynamic_faster_than_static(self, cudnn_handle):
        """Dynamic shape graph build should be faster than N static builds."""
        b = 1
        max_m = max(self.M_VALUES)

        # Static: build separate graph per shape
        t0 = time.perf_counter()
        for m in self.M_VALUES:
            _build_fp4_graph(
                cudnn_handle, b, m, self.N, self.K, self.BLOCK_SIZE, dynamic=False
            )
        static_time = time.perf_counter() - t0

        # Dynamic: build one graph
        t0 = time.perf_counter()
        _build_fp4_graph(
            cudnn_handle, b, max_m, self.N, self.K, self.BLOCK_SIZE, dynamic=True
        )
        dynamic_time = time.perf_counter() - t0

        print(f"\nStatic ({len(self.M_VALUES)} graphs): {1000 * static_time:.0f}ms")
        print(f"Dynamic (1 graph): {1000 * dynamic_time:.0f}ms")
        print(f"Speedup: {static_time / dynamic_time:.1f}x")

        # Dynamic build of 1 graph should be faster than static build of N graphs
        assert dynamic_time < static_time, (
            f"Dynamic ({dynamic_time:.3f}s) should be faster than "
            f"static ({static_time:.3f}s)"
        )
