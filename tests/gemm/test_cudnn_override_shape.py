"""
Tests for cuDNN GEMM operations using is_override_shape_enabled API.

A single cuDNN graph is compiled once with a "cache shape" (large M).
At execution time, the actual M is passed via override_shapes / override_strides,
so no graph rebuild is triggered for varying M values.

Requires:
  - CUDA compute capability == SM100 / SM103
  - cuDNN frontend >= 1.20 / backend_version >= 92100
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer.gemm.gemm_base import (
    CUDNN_AVAILABLE,
    build_cudnn_gemm_bf16_graph_override_shape,
    execute_cudnn_gemm_bf16_graph_override_shape,
    build_cudnn_gemm_fp4_graph_override_shape,
    execute_cudnn_gemm_fp4_graph_override_shape,
    build_cudnn_gemm_mxfp8_graph_override_shape,
    execute_cudnn_gemm_mxfp8_graph_override_shape,
    is_cudnn_override_shape_available,
    _calculate_block_scale_dims,
)
from flashinfer.utils import get_compute_capability
from flashinfer.fp4_quantization import fp4_quantize
from flashinfer.fp8_quantization import mxfp8_quantize


def _skip_if_no_cudnn():
    if not CUDNN_AVAILABLE:
        pytest.skip("cuDNN not available")


def _skip_if_override_shape_not_supported():
    if not CUDNN_AVAILABLE:
        pytest.skip("cuDNN not available")
    if not is_cudnn_override_shape_available():
        pytest.skip(
            "cuDNN override-shape requires higher version of cuDNN backend and frontend"
        )


def _skip_if_not_sm100_or_sm103():
    major, minor = get_compute_capability(torch.device("cuda"))
    if major * 10 + minor not in [100, 103]:
        pytest.skip("override-shape GEMM requires SM100 or SM103")


# ============================================================================
# BF16 GEMM with override_shape
# ============================================================================


class TestCudnnBf16OverrideShape:
    """Single compiled plan handles multiple M dimensions for BF16 GEMM."""

    @pytest.mark.skipif(
        not CUDNN_AVAILABLE,
        reason="cuDNN not available",
    )
    @pytest.mark.parametrize(
        "cache_m,dynamic_ms",
        [
            (2048, [1, 4, 16, 32, 64, 128, 512, 1024, 2048]),
            (4096, [1, 8, 64, 256, 1024, 4096]),
        ],
    )
    @pytest.mark.parametrize("n", [1024, 2048])
    @pytest.mark.parametrize("k", [1024, 2048])
    def test_bf16_override_shape_dynamic_m(self, cache_m, dynamic_ms, n, k):
        _skip_if_no_cudnn()
        _skip_if_override_shape_not_supported()
        _skip_if_not_sm100_or_sm103()

        from flashinfer.gemm.gemm_base import _torch_data_type_to_cudnn_data_type

        device = torch.device("cuda")
        in_dtype = torch.bfloat16
        out_dtype = torch.bfloat16

        # Build graph once with cache_m
        graph = build_cudnn_gemm_bf16_graph_override_shape(
            batch=1,
            n=n,
            k=k,
            o_type=_torch_data_type_to_cudnn_data_type(out_dtype),
            device=device,
            bias_is_not_none=False,
            cache_m=cache_m,
            is_a_k_major=True,
            is_b_k_major=True,
        )

        workspace = torch.empty(
            graph.get_workspace_size(), dtype=torch.uint8, device=device
        )

        b = torch.randn(1, n, k, dtype=in_dtype, device=device).transpose(1, 2)

        for m in dynamic_ms:
            a = torch.randn(1, m, k, dtype=in_dtype, device=device)
            out = torch.empty(1, m, n, dtype=out_dtype, device=device)
            ref = torch.bmm(a.float(), b.float()).to(out_dtype)

            execute_cudnn_gemm_bf16_graph_override_shape(
                graph, a, b, None, out, workspace, tactic=0
            )
            torch.cuda.synchronize()

            assert torch.allclose(ref, out, rtol=5e-2, atol=5e-2), (
                f"BF16 override_shape failed for m={m}, n={n}, k={k}: "
                f"max_abs_err={(ref - out).abs().max().item():.4f}, "
                f"max_rel_err={((ref - out).abs() / (ref.abs() + 1e-8)).max().item():.4f}"
            )


# ============================================================================
# NVFP4 GEMM with override_shape
# ============================================================================


class TestCudnnNVFp4OverrideShape:
    """Single compiled plan handles multiple M dimensions for NVFP4 GEMM."""

    @pytest.mark.skipif(
        not CUDNN_AVAILABLE,
        reason="cuDNN not available",
    )
    @pytest.mark.parametrize(
        "cache_m,dynamic_ms",
        [
            (2048, [1, 4, 16, 32, 64, 128, 512, 1024, 2048]),
            (4096, [1, 8, 64, 256, 1024, 4096]),
        ],
    )
    @pytest.mark.parametrize("n", [1024, 2048])
    @pytest.mark.parametrize("k", [1024, 2048])
    def test_nvfp4_override_shape_dynamic_m(self, cache_m, dynamic_ms, n, k):
        _skip_if_no_cudnn()
        _skip_if_override_shape_not_supported()
        _skip_if_not_sm100_or_sm103()

        import cudnn

        if cudnn.backend_version() < 91002:
            pytest.skip("FP4 requires cuDNN backend >= 91002")
        from flashinfer.gemm.gemm_base import _torch_data_type_to_cudnn_data_type

        device = torch.device("cuda")
        block_size = 16
        out_dtype = torch.bfloat16

        # Compute block scale dims using cache_m
        _, block_scale_dim_n, block_scale_dim_k = _calculate_block_scale_dims(
            cache_m, n, k, block_size
        )

        # Build graph once with cache_m
        graph = build_cudnn_gemm_fp4_graph_override_shape(
            batch=1,
            n=n,
            k=k,
            ab_type=cudnn.data_type.FP4_E2M1,
            o_type=_torch_data_type_to_cudnn_data_type(out_dtype),
            block_size=block_size,
            device=device,
            alpha_is_not_none=False,
            use_nvfp4=True,
            cache_m=cache_m,
        )

        workspace = torch.empty(
            graph.get_workspace_size(),
            dtype=torch.uint8,
            device=device,
        )

        global_sf = torch.tensor(1.0, dtype=torch.float32, device=device)

        # B is fixed across all dynamic_ms
        b_bf16 = torch.empty([1, n, k], device="cuda", dtype=torch.bfloat16).uniform_(
            -5.0, 5.0
        )
        b_packed, b_scale = fp4_quantize(b_bf16, global_sf)

        b_bf16 = b_bf16.transpose(1, 2)
        b_packed = b_packed.transpose(1, 2)
        b_scale = b_scale.unsqueeze(0).transpose(1, 2)

        for m in dynamic_ms:
            block_scale_dim_m, _, _ = _calculate_block_scale_dims(m, n, k, block_size)

            a_bf16 = torch.empty(
                [1, m, k], device="cuda", dtype=torch.bfloat16
            ).uniform_(-5.0, 5.0)
            a_packed, a_scale = fp4_quantize(a_bf16, global_sf)

            a_scale = a_scale.unsqueeze(0)

            # Execute with cached graph (override_shape)
            out = torch.empty(1, m, n, dtype=out_dtype, device=device)
            execute_cudnn_gemm_fp4_graph_override_shape(
                graph,
                a_packed,
                b_packed,
                a_scale,
                b_scale,
                alpha=None,
                c_final=out,
                workspace_buffer=workspace,
                tactic=0,
            )
            torch.cuda.synchronize()

            ref = torch.bmm(a_bf16, b_bf16).to(out_dtype)

            min_cos_sim = 0.9
            cos_sim = F.cosine_similarity(ref.reshape(-1), out.reshape(-1), dim=0)
            assert cos_sim > min_cos_sim, (
                f"Cosine similarity {cos_sim:.4f} is too low (expected > {min_cos_sim})"
            )


# ============================================================================
# MXFP8 GEMM with override_shape
# ============================================================================


class TestCudnnMXFp8OverrideShape:
    """Single compiled plan handles multiple M dimensions for MXFP8 GEMM."""

    @pytest.mark.skipif(
        not CUDNN_AVAILABLE,
        reason="cuDNN not available",
    )
    @pytest.mark.parametrize(
        "cache_m,dynamic_ms",
        [
            (2048, [1, 4, 16, 32, 64, 128, 512, 1024, 2048]),
            (4096, [1, 8, 64, 256, 1024, 4096]),
        ],
    )
    @pytest.mark.parametrize("n", [1024, 2048])
    @pytest.mark.parametrize("k", [1024, 2048])
    def test_mxfp8_override_shape_dynamic_m(self, cache_m, dynamic_ms, n, k):
        _skip_if_no_cudnn()
        _skip_if_override_shape_not_supported()
        _skip_if_not_sm100_or_sm103()

        import cudnn
        from flashinfer.gemm.gemm_base import _torch_data_type_to_cudnn_data_type

        device = torch.device("cuda")
        block_size = 32
        out_dtype = torch.bfloat16

        # Compute block scale dims using cache_m
        _, block_scale_dim_n, block_scale_dim_k = _calculate_block_scale_dims(
            cache_m, n, k, block_size
        )

        # Build graph once with cache_m
        graph = build_cudnn_gemm_mxfp8_graph_override_shape(
            batch=1,
            n=n,
            k=k,
            a_type=cudnn.data_type.FP8_E4M3,
            b_type=cudnn.data_type.FP8_E4M3,
            o_type=_torch_data_type_to_cudnn_data_type(out_dtype),
            block_size=block_size,
            device=device,
            cache_m=cache_m,
        )

        workspace = torch.empty(
            graph.get_workspace_size(),
            dtype=torch.uint8,
            device=device,
        )

        # B is fixed across all dynamic_ms
        b_bf16 = torch.empty([1, n, k], device="cuda", dtype=torch.bfloat16).uniform_(
            -5.0, 5.0
        )
        b, b_scale = mxfp8_quantize(b_bf16, True)

        b_bf16 = b_bf16.transpose(1, 2)
        b = b.transpose(1, 2)
        b_scale = b_scale.reshape((-1, block_scale_dim_n, block_scale_dim_k)).transpose(
            1, 2
        )

        for m in dynamic_ms:
            block_scale_dim_m, _, _ = _calculate_block_scale_dims(m, n, k, block_size)

            a_bf16 = torch.empty(
                [1, m, k], device="cuda", dtype=torch.bfloat16
            ).uniform_(-5.0, 5.0)
            a, a_scale = mxfp8_quantize(a_bf16, True)

            a_scale = a_scale.reshape((-1, block_scale_dim_m, block_scale_dim_k))

            # Execute with cached graph (override_shape)
            out = torch.empty(1, m, n, dtype=out_dtype, device=device)
            execute_cudnn_gemm_mxfp8_graph_override_shape(
                graph,
                a,
                b,
                a_scale,
                b_scale,
                c_final=out,
                workspace_buffer=workspace,
                tactic=0,
            )
            torch.cuda.synchronize()

            ref = torch.bmm(a_bf16, b_bf16).to(out_dtype)

            min_cos_sim = 0.9
            cos_sim = F.cosine_similarity(ref.reshape(-1), out.reshape(-1), dim=0)
            assert cos_sim > min_cos_sim, (
                f"Cosine similarity {cos_sim:.4f} is too low (expected > {min_cos_sim})"
            )
